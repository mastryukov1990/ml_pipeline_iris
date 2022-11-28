from enum import Enum
from typing import List

import numpy as np
import attr

from project.config_from_file import ConfigFromArgs
from project.train_model.config import ModelNames


def get_precision(y_pred: np.array, y_true: np.array):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tp / (tp + fp)


def get_recall(y_pred: np.array, y_true: np.array):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    t = (y_true == 1).sum()
    return tp / t


class MetricsName(str, Enum):
    PRECISION = 'precision'
    RECALL = 'recall'


METRICS = {
    MetricsName.RECALL: get_recall,
    MetricsName.PRECISION: get_precision,
}


@attr.s
class EvalModelsConfig(ConfigFromArgs):
    SECTION = 'eval_model'

    model_name: ModelNames = attr.ib()
    metrics: List[MetricsName] = attr.ib(default=[MetricsName.PRECISION])
