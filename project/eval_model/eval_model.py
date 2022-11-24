from typing import Union, Dict, List

import pandas as pd

from project.common import save_dict
from project.constants import TARGET_COLUMN, TRAIN_METRICS, TEST_DATASET, TEST_METRICS
from project.eval_model.config import EvalModelsConfig, METRICS
from project.train_models.config import TrainModelsConfig, Model


def log_metrics(save_path: str, metrics: Union[Dict, List]):
    save_dict(save_path, metrics)


def eval_model(config: EvalModelsConfig):
    test_df = pd.read_csv(TEST_DATASET)

    model = Model(config.model_name)
    model.load_model()

    preds = model.predict(test_df.drop(TARGET_COLUMN, axis=1))

    test_scores = {
        metric: METRICS[metric](preds, test_df[TARGET_COLUMN])
        for metric in config.metrics
    }

    log_metrics(TEST_METRICS, test_scores)
