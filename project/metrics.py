from enum import Enum
from typing import Dict, List, Union

import numpy as np

from project.common import save_dict
from project.logger import get_logger

logger = get_logger(__name__)


def get_precision(y_pred: np.array, y_true: np.array):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tp / (tp + fp)


def get_acc(y_pred: np.array, y_true: np.array):
    return (y_pred == y_true).mean()


def get_recall(y_pred: np.array, y_true: np.array):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    t = (y_true == 1).sum()
    return tp / t


class MetricsName(str, Enum):
    PRECISION = 'precision'
    RECALL = 'recall'
    ACCURACY = 'accuracy'


METRICS = {
    MetricsName.RECALL: get_recall,
    MetricsName.PRECISION: get_precision,
    MetricsName.ACCURACY: get_acc,
}

def log_metrics(save_path: str, metrics: Union[Dict, List]):
    logger.info(f'metrics - {metrics}')
    save_dict(save_path, metrics)
