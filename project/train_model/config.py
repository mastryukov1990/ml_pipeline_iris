from typing import List

import attr


from project.config_from_file import ConfigFromArgs
from project.metrics import MetricsName
from project.models import ModelNames

TRAIN_MODEL = 'train_model'



@attr.s
class TrainModelsConfig(ConfigFromArgs):
    SECTION = TRAIN_MODEL
    model_name: ModelNames = attr.ib()
    metrics: List[MetricsName] = attr.ib(default=[MetricsName.PRECISION])

