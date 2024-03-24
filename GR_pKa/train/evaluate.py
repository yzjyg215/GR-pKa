from collections import defaultdict
import logging
from typing import Dict, List

from .predict import predict
from GR_pKa.data import MoleculeDataLoader, StandardScaler
from GR_pKa.models import MoleculeModel
from GR_pKa.utils import get_metric_func
import numpy as np
import pandas as pd


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metrics: List[str],
                         dataset_type: str,
                         r2_total_new=None,
                         mae_total_new=None,
                         mse_total_new=None,
                         save:bool=True,
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

    info = logger.info if logger is not None else print

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = defaultdict(list)
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                for metric in metrics:
                    results[metric].append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        for metric, metric_func in metric_to_func.items():
            if dataset_type == 'multiclass':
                results[metric].append(metric_func(valid_targets[i], valid_preds[i],
                                                   labels=list(range(len(valid_preds[i][0])))))
            else:
                results[metric].append(metric_func(
                    valid_targets[i], valid_preds[i]))
        r2_epoch=r2_score(valid_targets[i], valid_preds[i])
        rmse_epoch=np.sqrt(mean_squared_error(valid_targets[i], valid_preds[i]))
        mse_epoch=mean_squared_error(valid_targets[i], valid_preds[i])
        mae_epoch=mean_absolute_error(valid_targets[i], valid_preds[i])
        try:
            r2_total_new.append(r2_epoch)
            mae_total_new.append(mae_epoch)
            mse_total_new.append(mse_epoch)
        except:
            print('test')
        print('R2_score:'+ str(r2_epoch))
        print('Rmse:'+ str(rmse_epoch))
        print('MSE:'+ str(mse_epoch))
        print('MAE:'+ str(mae_epoch))

    results = dict(results)

    if save:
        np.save('./preds.npy',valid_preds[i])
        np.save('./targets.npy',valid_targets[i])
    #hard_preds = [1 if p > 0.5 else 0 for p in valid_preds[i]]
    
    return results


def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             dataset_type: str,
             r2_total=None,
             mae_total=None,
             mse_total=None,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`~GR_pKa.models.model.MoleculeModel`.
    :param data_loader: A :class:`~GR_pKa.data.data.MoleculeDataLoader`.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param scaler: A :class:`~GR_pKa.features.scaler.StandardScaler` object fit on the training targets.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    preds = predict(
        model=model,
        data_loader=data_loader,
        scaler=scaler
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        num_tasks=num_tasks,
        metrics=metrics,
        r2_total_new=r2_total,
        mae_total_new=mae_total,
        mse_total_new=mse_total,
        dataset_type=dataset_type,
        logger=logger
    )

    return results
