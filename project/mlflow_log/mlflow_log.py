import argparse

import mlflow

from project.common import load_dict, config_path_parser
from project.config_from_file import read_yaml
from project.constants import TasksList
from project.mlflow_log.config import MlflowLogConfig
from project.mlflow_tools import mlflow_set_experiment


def mlflow_log(config: MlflowLogConfig):
    mlflow_set_experiment(config.experiment_name, config.tracking_uri)

    train_metrics = load_dict(TasksList.TRAIN_METRICS)
    test_metrics = load_dict(TasksList.TEST_METRICS)
    params = {k: v for section, params in read_yaml('params.yaml').items() for k, v in params.items()}

    with mlflow.start_run(run_name=f"train_{config.run_name}"):
        mlflow.log_params(params)
        mlflow.log_metrics(train_metrics)

    with mlflow.start_run(run_name=f"test_{config.run_name}"):
        mlflow.log_params(params)
        mlflow.log_metrics(test_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)

    config = MlflowLogConfig.from_args(parser.parse_args())

    mlflow_log(config=config)


if __name__ == "__main__":
    main()
