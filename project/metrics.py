import numpy as np

from enum import Enum
from typing import Dict, List, Union
from functools import partial
from sklearn.metrics import precision_score, recall_score

from project.common import save_dict
from project.logger import get_logger


logger = get_logger(__name__)


def get_acc(y_pred: np.array, y_true: np.array):
    return (y_pred == y_true).mean()


class MetricsName(str, Enum):
    PRECISION = 'precision'
    RECALL = 'recall'
    ACCURACY = 'accuracy'


METRICS = {
    MetricsName.RECALL: partial(recall_score, average='weighted'),
    MetricsName.PRECISION: partial(precision_score, average='weighted'),
    MetricsName.ACCURACY: get_acc,
}


def log_metrics(save_path: str, metrics: Union[Dict, List]):
    logger.info(f'metrics - {metrics}')
    save_dict(save_path, metrics)


def log_params(params: Dict, save_path: str = 'parameters.json'):
    save_dict(metrics=params, filename=save_path)


def load_params(params: Dict, save_path: str = 'parameters.json'):
    save_dict(metrics=params, filename=save_path)
