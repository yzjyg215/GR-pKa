# GR-pKa: A message-passing neural network with retention mechanism for pKa prediction

## Introduction

This repository provides codes and materials associated with the manuscript [GR-pKa: A message-passing neural network with retention mechanism for pKa prediction].

- **GR-pKa** is a novel pka prediction method which takes advantages of multi-fidelity learning, quantum mechanical (QM) properties and retention mechanism.
The prediction process consisted of three steps: molecular graph construction and featurization, message passing with the retention mechanism, and pKa prediction.

We acknowledge the paper [Liu et al (2023). ABT-MPNN: an atom-bond transformer-based message-passing neural network for molecular property prediction. J Cheminform 2023;15(1):29.](https://doi.org/10.1186/s13321-023-00698-9) and the [Chemprop](https://github.com/chemprop/chemprop) repository ([version 1.2.0](https://github.com/chemprop/chemprop/releases/tag/v1.2.0)) which this code leveraged and built on top of.

## Overview 
- ```GR_pKa/```: the source codes of GR-pKa.
- ```features/```: the molecular QM features of all datasets.
- ```data/```: the pre-training dataset, fine-tuning dataset, E-pKa dataset used in GR-pKa.
- ```model/```: the pre-trained model weights of GR-pKa.
- ```prediction/```: the prediction results on SAMPL6, SAMPL7, and E-pKa datasets by GR-pKa.

## Dependencies

```
cuda >= 8.0 + cuDNN
python>=3.6
flask>=1.1.2
gunicorn>=20.0.4
hyperopt>=0.2.3
matplotlib>=3.1.3
numpy>=1.18.1
pandas>=1.0.3
pandas-flavor>=0.2.0
pip>=20.0.2
pytorch>=1.4.0
rdkit>=2020.03.1.0
scipy>=1.4.1
tensorboardX>=2.0
torchvision>=0.5.0
tqdm>=4.45.0
einops>=0.3.2
seaborn>=0.11.1
```
## Data and QM features

The data file must be be a CSV file with a header row. For example:

```
smiles,pKa(or pKb)
O=C(NCO)c1ccccc1,13.05
O=C(O)C(F)c1ccccc1,2.45
...
```

Data sets used in our study are available in the `data` directory of this repository.
QM features used in our study are available in the `features` directory of this repository.

## Featurization(the first step)

**To save adjacency / distance / Coulomb matrices for a dataset, run:**

```
python save_atom_features.py --data_path <path> --save_dir <dir> --adjacency --coulomb --distance
```

where `<path>` is the path to a CSV file containing a dataset, and `<dir>` is the directory where inter-atomic matrices will be saved. To generate adjacency, distance, Coulomb matrices, specify `--adjacency`, `--distance`, `--coulomb` flags.

For example:

```
python save_atom_features.py --data_path data/pre-training/pretrain_pka.csv --save_dir features/ --adjacency --coulomb --distance
```

## Training

To train a GR-pKa model, run:

```
python train.py --data_path <path> --dataset_type <regression> --save_dir <dir> --bond_fast_retention --atom_retention --adjacency --adjacency_path <adj_path> --distance --distance_path <dist_path> --coulomb --coulomb_path <clb_path> --normalize_matrices --features_path <QM_features_path> --no_features_scaling
```

**Notes:**

- `<path>` is the path to a CSV file containing a dataset.
- `<dir>` is the directory where model checkpoints will be saved.
- To use bond retention in the message passing phase, add `--bond_fast_retention`
- To use atom retention in the readout phase, add `--atom_retention`
- Specify `--adjacency` to add adjacency matrix and `<adj_path>` is the path to a npz file containing the saved adjacency matrices of a dataset.
- Specify `--distance` to add distance matrix and `<dist_path>` is the path to a npz file containing the saved distance matrices of a dataset.
- Specify `--coulomb` to add Coulomb matrix and `<clb_path>` is the path to a npz file containing the saved coulomb matrices of a dataset.
- `<QM_features_path>` is the path to a csv file containing the QM features of a dataset.
- Specify `--normalize_matrices` to normalize inter-atomic matrices.

A full list of available command-line arguments can be found in `GR_pKa/args.py`

## Predicting

To load a trained model and make predictions, run `predict.py` and specify:

- `--test_path <path>` Path to the data to predict on.
- `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
- `--preds_path` Path where a pickle file containing the predictions will be saved.
- `--adjacency_path <adj_path>` Path to a npz file containing the saved adjacency matrices
- `--distance_path <dist_path>` Path to a npz file containing the saved distance matrices
- `--coulomb_path <clb_path>` Path to a npz file containing the saved coulomb matrices
- `--features_path <molf_path>` Path to a csv file containing the QM features.
- `--normalize_matrices` and `--no_features_scaling` also must be specified if used in training.

For example:

```
python predict.py --test_path data/E-pKa/pka_E-pKa.csv --checkpoint_path model/pka_best.pt --preds_path prediction/pred.csv --adjacency_path xxx/adj.npz --distance_path xxx/dist.npz --coulomb_path xxx/clb.npz --features_path features/E-pKa/pka_E-pKa_features.csv --normalize_matrices --no_features_scaling
```
