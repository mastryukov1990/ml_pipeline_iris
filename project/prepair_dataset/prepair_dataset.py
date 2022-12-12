import argparse
import numpy as np
import pandas as pd

from typing import Tuple

from project.common import config_path_parser, save_csv
from project.constants import TasksList
from project.prepair_dataset.config import PrepareDatasetConfig


def split_train_test(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = np.random.rand(df.shape[0])
    train_mask = mask > test_ratio
    return df[train_mask], df[np.logical_not(train_mask)]


def get_datasets(config: PrepareDatasetConfig):
    joined_df = pd.read_csv(TasksList.JOINED)
    train_df, test_df = split_train_test(joined_df, config.ratio)

    save_csv(TasksList.TRAIN_DATASET, train_df)
    save_csv(TasksList.TEST_DATASET, test_df)



def main():
    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)
    parser.add_argument("--ratio", type=float)

    config = PrepareDatasetConfig.from_args(parser.parse_args())
    get_datasets(config=config)


if __name__ == "__main__":
    main()
