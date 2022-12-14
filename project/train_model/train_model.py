import argparse

import pandas as pd

from project.common import  config_path_parser, create_parent_folder
from project.constants import TasksList, TARGET_COLUMN
from project.metrics import METRICS, log_metrics
from project.models import Model, ModelNames
from project.train_model.config import TrainModelsConfig


def train_model(config: TrainModelsConfig):
    train_df = pd.read_csv(TasksList.TRAIN_DATASET)
    model = Model(config.model_name)

    y, x = train_df[TARGET_COLUMN].values.reshape(-1, 1), train_df.drop(TARGET_COLUMN, axis=1)
    model.fit(y=y, x=x)

    predictions = model.predict(train_df.drop(TARGET_COLUMN, axis=1))
    train_scores = {
        metric: METRICS[metric](predictions, train_df[TARGET_COLUMN])
        for metric in config.metrics
    }

    create_parent_folder(TasksList.TRAIN_METRICS)
    create_parent_folder(TasksList.MODEL_SAVE_PATH)


    log_metrics(TasksList.TRAIN_METRICS, train_scores)
    model.log_model()



def main():
    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)

    parser.add_argument("--model-name", type=ModelNames)

    config = TrainModelsConfig.from_args(parser.parse_args())

    train_model(config=config)


if __name__ == "__main__":
    main()
