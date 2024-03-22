"""Loads a trained model checkpoint and makes predictions on a dataset."""

from GR_pKa.args import PredictArgs
from GR_pKa.train import make_predictions


if __name__ == "__main__":
    make_predictions(args=PredictArgs().parse_args())
