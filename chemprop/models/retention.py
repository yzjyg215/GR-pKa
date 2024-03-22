import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function


class MultiBondFastRetention(nn.Module):
    """
    A :class:`MultiBondFastretention` is the bond level self-retention block (Fastformer) in the message passing phase.
    """

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MultiBondFastRetention, self).__init__()
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.dropout = args.dropout
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.gamma = 1 - 2 ** (-5)

        self.num_heads = args.num_heads
        self.ret_size = self.hidden_size // self.num_heads
        self.scale_factor = self.ret_size ** -0.5
        self.weight_alpha = nn.Parameter(
            torch.randn(self.ret_size))
        self.weight_beta = nn.Parameter(torch.randn(self.ret_size))
        self.weight_r = nn.Linear(self.ret_size, self.ret_size, bias=False)

        self.W_b_q = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_b_k = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_b_v = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_b_o = nn.Linear(
            self.num_heads * self.ret_size, self.hidden_size)
        #self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        self.norm=nn.GroupNorm(8,self.hidden_size,affine=True)

    def forward(self, message, b_scope):
        """
        Calculate the bond-level retention of a molecule with Fastformer in each message passing step.

        :param message: Hidden states (messages) of a batch of molecules
        :param b_scope: List of tuples indicating (start_bond_index, num_bonds) for each molecule
        :return: A PyTorch tensor of shape :code:`(batch_num_bonds, hidden_size)` containing the hidden states of a batch of molecules.
        """
        bond_vecs = []
        r = 0
        for i, (b_start, b_size) in enumerate(b_scope):  # b_scope: (1, num_bonds)
            if i == 0:
                bond_vecs.append(self.cached_zero_vector)
            cur_bond_message = message.narrow(
                0, b_start, b_size)  # num_bonds x hidden (1 mol)
            cur_bond_message_size = cur_bond_message.size()

            # (num_bonds, num_head, ret_size)
            b_q = self.W_b_q(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.ret_size)
            b_k = self.W_b_k(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.ret_size)
            b_v = self.W_b_v(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.ret_size)
            b_q = b_q.transpose(0, 1)  # (num_head, num_bonds, ret_size)
            b_k = b_k.transpose(0, 1)  # (num_head, num_bonds, ret_size)
            b_v = b_v.transpose(0, 1)  # (num_head, num_bonds, ret_size)
            h, n, d = b_q.shape

            # Caculate the global query
            # (num_head, num_bonds, ret_size)
            alpha_weight = torch.mul(
                b_q, self.weight_alpha) * self.scale_factor
            
            # (num_head, num_bonds, ret_size)
            alpha_weight = F.softmax(alpha_weight, dim=-1)
            # (num_head, num_bonds, ret_size)
            global_query = torch.mul(alpha_weight, b_q)
            # (num_head, ret_size)
            global_query = torch.sum(global_query, dim=1)

            # Model the interaction between global query vector and the key vector
            # (num_head, num_bonds, ret_size)
            repeat_global_query = einops.repeat(
                global_query, 'h d -> h copy d', copy=n)
            # (num_head, num_atoms, ret_size)
            p = torch.mul(repeat_global_query, b_k)
            # (num_head, num_bonds, ret_size)
            beta_weight = torch.mul(p, self.weight_beta) * self.scale_factor
            # (num_head, num_bonds, ret_size)
            # beta_weight = F.softmax(beta_weight, dim=-1)
            r = torch.matmul(b_k, b_v) + (self.gamma) ** b_size * r
            beta_weight = beta_weight + torch.matmul(repeat_global_query, r) * (self.gamma) ** (i + 1)
            # (num_head, num_bonds, ret_size)
            global_key = torch.mul(beta_weight, p)
            global_key = torch.sum(global_key, dim=1)  # (num_head, ret_size)

            # key-value
            key_value_interaction = torch.einsum(
                'hd,hnd->hnd', global_key, b_v)  # (num_head, num_bonds, ret_size)
            key_value_interaction_out = self.weight_r(
                key_value_interaction)  # (num_head, num_bonds, ret_size)
            ret_b_h = key_value_interaction_out + \
                b_q  # (num_head, num_bonds, ret_size)

            ret_b_h = self.act_func(ret_b_h)
            ret_b_h = self.dropout_layer(ret_b_h)
            # (num_bonds, num_head, ret_size)
            ret_b_h = ret_b_h.transpose(0, 1).contiguous()
            # (num_bonds, hidden_size)
            ret_b_h = ret_b_h.view(
                cur_bond_message_size[0], self.num_heads * self.ret_size)
            ret_b_h = self.W_b_o(ret_b_h)  # (num_bonds, hidden_size)

            assert ret_b_h.size() == cur_bond_message_size

            ret_b_h = ret_b_h.unsqueeze(dim=0)
            ret_b_h = self.norm(ret_b_h)
            # ret_b_h = group_norm(ret_b_h, gamma=1, beta=1e-5, G=2)
            ret_b_h = (ret_b_h).squeeze(dim=0)

            bond_vecs.extend(ret_b_h)

        bond_vecs = torch.stack(bond_vecs, dim=0)
        return bond_vecs


class MultiBondRetention(nn.Module):
    """
    A :class:`MultiBondretention` is the bond level self-retention block (Transformer) in the message passing phase.
    """

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MultiBondRetention, self).__init__()
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.dropout = args.dropout
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.num_heads = args.num_heads
        self.ret_size = self.hidden_size // self.num_heads
        self.scale_factor = self.ret_size ** -0.5
        self.gamma = 1 - 2 ** (-5)

        self.W_b_q = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_b_k = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_b_v = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_b_o = nn.Linear(
            self.num_heads * self.ret_size, self.hidden_size)
        #self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        self.norm=nn.GroupNorm(8,self.hidden_size,affine=True)

    def forward(self, message, b_scope):
        """
        Calculate the bond-level retention of a molecule with Transformer in each message passing step.

        :param message: Hidden states (messages) of a batch of molecules
        :param b_scope: List of tuples indicating (start_bond_index, num_bonds) for each molecule
        :return: A PyTorch tensor of shape :code:`(batch_num_bonds, hidden_size)` containing the hidden states of a batch of molecules.
        """
        bond_vecs = []
        r = 0
        for i, (b_start, b_size) in enumerate(b_scope):  # b_scope: (1, num_bonds)
            if i == 0:
                bond_vecs.append(self.cached_zero_vector)

            cur_bond_message = message.narrow(
                0, b_start, b_size)  # num_bonds x hidden
            cur_bond_message_size = cur_bond_message.size()
            # (num_bonds, num_head, ret_size)
            b_q = self.W_b_q(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.ret_size)
            b_k = self.W_b_k(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.ret_size)
            b_v = self.W_b_v(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.ret_size)
            b_q = b_q.transpose(0, 1)  # (num_head, num_bonds, ret_size)
            # (num_head, ret_size, num_bonds)
            b_k = b_k.transpose(0, 1).transpose(1, 2)
            b_v = b_v.transpose(0, 1)  # (num_head, num_bonds, ret_size)


            # (num_head, num_bonds, num_bonds)
            ret_b_w = torch.matmul(b_q, b_k)

            r = torch.matmul(b_k, b_v) + (self.gamma) ** b_size * r
            # (num_head, num_bonds, num_bonds)
            # ret_b_w = F.softmax(ret_b_w * self.scale_factor, dim=2)
            ret_b_w = ret_b_w * self.scale_factor
            # (num_head, num_bonds, ret_size)
            ret_b_h = torch.matmul(ret_b_w, b_v) + torch.matmul(b_q, r) * (self.gamma) ** (i + 1)
            ret_b_h = self.act_func(ret_b_h)
            ret_b_h = self.dropout_layer(ret_b_h)

            # (num_bonds, num_head, ret_size)
            ret_b_h = ret_b_h.transpose(0, 1).contiguous()
            # (num_bonds, hidden_size)
            ret_b_h = ret_b_h.view(
                cur_bond_message_size[0], self.num_heads * self.ret_size)
            ret_b_h = self.W_b_o(ret_b_h)  # (num_bonds, hidden_size)
            assert ret_b_h.size() == cur_bond_message_size
            ret_b_h = cur_bond_message + self.dropout_layer(ret_b_h)

            ret_b_h = ret_b_h.unsqueeze(dim=0)
            ret_b_h = self.norm(ret_b_h)
            # ret_b_h = group_norm(ret_b_h, gamma=1, beta=1e-5, G=2)
            ret_b_h = (ret_b_h).squeeze(dim=0)

            bond_vecs.extend(ret_b_h)

        bond_vecs = torch.stack(bond_vecs, dim=0)
        return bond_vecs


class MultiAtomRetention(nn.Module):
    """
    A :class:`MultiAtomretention` is the atom level self-retention block (Transformer) in the readout phase.
    """

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MultiAtomRetention, self).__init__()
        self.atom_retention = args.atom_retention
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.dropout = args.dropout
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.normalize_matrices = args.normalize_matrices
        self.distance = args.distance
        self.adjacency = args.adjacency
        self.coulomb = args.coulomb
        self.device = args.device
        self.num_heads = args.num_heads
        self.ret_size = self.hidden_size // self.num_heads
        self.scale_factor = self.ret_size ** -0.5
        self.f_scale = args.f_scale
        self.gamma = 1 - 2 ** (-5)

        self.W_a_q = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_a_k = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_a_v = nn.Linear(
            self.hidden_size, self.num_heads * self.ret_size, bias=False)
        self.W_a_o = nn.Linear(
            self.num_heads * self.ret_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

    def forward(self, cur_hiddens, i, f_adj, f_dist, f_clb, viz_dir=None):
        """
        Calculate the atom-level retention of a molecule with Transformer in the readout phase.

        :param cur_hiddens: Hidden states of a molecule
        :param i: An atom index to a list of atom matrices.
        :param f_adj: The adjacency matrix of the molecule.
        :param f_dist: The distance matrix of the molecule.
        :param f_clb: The coulomb matrix of the molecule.
        :return: A PyTorch tensor of shape :code:`(num_atoms, hidden_size)` containing the hidden states of a molecules.
        """
        # cur_hidden (1 mol): num_atoms x hidden_size
        cur_hiddens_size = cur_hiddens.size()

        # (num_atoms, num_head, ret_size)
        a_q = self.W_a_q(cur_hiddens).view(
            cur_hiddens_size[0], self.num_heads, self.ret_size)
        # (num_atoms, num_head, ret_size)
        a_k = self.W_a_k(cur_hiddens).view(
            cur_hiddens_size[0], self.num_heads, self.ret_size)
        # (num_atoms, num_head, ret_size)
        a_v = self.W_a_v(cur_hiddens).view(
            cur_hiddens_size[0], self.num_heads, self.ret_size)
        a_q = a_q.transpose(0, 1)  # (num_head, num_atoms, ret_size)
        # (num_head, ret_size, num_atoms)
        a_k = a_k.transpose(0, 1).transpose(1, 2)
        a_v = a_v.transpose(0, 1)  # (num_head, num_atoms, ret_size)

        ret_a_w = torch.matmul(a_q, a_k)  # (num_head, num_atoms, num_atoms)

        if self.adjacency:
            mol_adj = torch.Tensor(f_adj[i]).to(
                self.device)  # (num_atoms, num_atoms)
            ret_a_w[0] = ret_a_w[0] + self.f_scale * \
                mol_adj  # (num_head, num_atoms, num_atoms)
            ret_a_w[1] = ret_a_w[1] + self.f_scale * \
                mol_adj  # (num_head, num_atoms, num_atoms)

        if self.distance:
            mol_dist = torch.Tensor(f_dist[i]).to(
                self.device)  # (num_atoms, num_atoms)
            if self.normalize_matrices:
                mol_dist = F.softmax(mol_dist, dim=1)

            ret_a_w[2] = ret_a_w[2] + self.f_scale * \
                mol_dist  # (num_head, num_atoms, num_atoms)
            ret_a_w[3] = ret_a_w[3] + self.f_scale * \
                mol_dist  # (num_head, num_atoms, num_atoms)

        if self.coulomb:
            mol_clb = torch.Tensor(f_clb[i]).to(
                self.device)  # (num_atoms, num_atoms)
            if self.normalize_matrices:
                mol_clb = F.softmax(mol_clb, dim=1)

            ret_a_w[4] = ret_a_w[4] + self.f_scale * \
                mol_clb  # (num_head, num_atoms, num_atoms)
            ret_a_w[5] = ret_a_w[5] + self.f_scale * \
                mol_clb  # (num_head, num_atoms, num_atoms)

        # (num_head, num_atoms, num_atoms)
        # ret_a_w = F.softmax(ret_a_w * self.scale_factor, dim=2)
        ret_a_w = ret_a_w * (self.gamma ** self.scale_factor)
        ret_a_h = torch.matmul(ret_a_w, a_v)  # (num_head, num_atoms, ret_size)
        ret_a_h = self.act_func(ret_a_h)
        ret_a_h = self.dropout_layer(ret_a_h)

        # (num_atom, num_head, ret_size)
        ret_a_h = ret_a_h.transpose(0, 1).contiguous()
        # (num_atom, hidden_size)
        ret_a_h = ret_a_h.view(
            cur_hiddens_size[0], self.num_heads * self.ret_size)
        ret_a_h = self.W_a_o(ret_a_h)  # (num_atoms, hidden_size)
        assert ret_a_h.size() == cur_hiddens_size
        ret_a_h = cur_hiddens + self.dropout_layer(ret_a_h)

        ret_a_h = ret_a_h.unsqueeze(dim=0)
        ret_a_h = self.norm(ret_a_h)
        # ret_a_h = group_norm(ret_a_h, gamma=1, beta=1e-5, G=2)

        mol_vec = (ret_a_h).squeeze(dim=0)  # (num_atoms, hidden_size)

        return mol_vec, torch.mean(ret_a_w, axis=0)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, dropout):
        """
        :param dropout: the dropout ratio.
        """
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, original, retention):
        """Apply residual connection to any sublayer with the same size."""
        return original + self.dropout(retention)
