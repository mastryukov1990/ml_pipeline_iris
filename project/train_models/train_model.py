from typing import Union, Dict, List

import pandas as pd

from project.common import save_dict
from project.constants import TRAIN_DATASET, TARGET_COLUMN, TRAIN_METRICS
from project.train_models.config import TrainModelsConfig, Model


def log_metrics(save_path: str,metrics: Union[Dict, List]):
    save_dict(save_path, metrics)


def train_model(config: TrainModelsConfig):
    train_df = pd.read_csv(TRAIN_DATASET)
    model = Model(config.model_name)

    model.fit(y=train_df[TARGET_COLUMN], x=train_df.drop(TARGET_COLUMN), )
    scores = model.get_score(train_df[TARGET_COLUMN], train_df.drop(TARGET_COLUMN))

    log_metrics(TRAIN_METRICS, scores)
    model.log_model()
