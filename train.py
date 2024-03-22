"""Trains a model on a dataset."""

from GR_pKa.args import TrainArgs
from GR_pKa.train import cross_validate
from GR_pKa.train import run_training


if __name__ == '__main__':
    cross_validate(args=TrainArgs().parse_args(), train_func=run_training)
