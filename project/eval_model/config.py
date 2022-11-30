from enum import Enum
from typing import List

import numpy as np
import attr

from project.config_from_file import ConfigFromArgs
from project.metrics import MetricsName
from project.train_model.config import ModelNames



@attr.s
class EvalModelsConfig(ConfigFromArgs):
    SECTION = 'eval_model'

    model_name: ModelNames = attr.ib()
    metrics: List[MetricsName] = attr.ib(default=[MetricsName.PRECISION])
