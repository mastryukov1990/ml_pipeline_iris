import argparse
from typing import Union, Dict, List

import pandas as pd

from project.common import config_path_parser, create_parent_folder
from project.constants import TasksList, TARGET_COLUMN
from project.eval_model.config import EvalModelsConfig
from project.metrics import METRICS, MetricsName, log_metrics
from project.models import Model


def eval_model(config: EvalModelsConfig):
    test_df = pd.read_csv(TasksList.TEST_DATASET)

    model = Model(config.model_name)
    model.load_model()

    predictions = model.predict(test_df.drop(TARGET_COLUMN, axis=1))

    test_scores = {
        metric: METRICS[metric](predictions, test_df[TARGET_COLUMN])
        for metric in config.metrics
    }

    create_parent_folder(TasksList.TEST_METRICS)
    log_metrics(TasksList.TEST_METRICS, test_scores)


def main():
    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)
    parser.add_argument("--metrics", nargs='+', type=MetricsName,)

    config = EvalModelsConfig.from_args(parser.parse_args())

    eval_model(config=config)


if __name__ == "__main__":
    main()
