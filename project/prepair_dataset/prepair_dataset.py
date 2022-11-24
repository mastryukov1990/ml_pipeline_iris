import argparse
import numpy as np
import pandas as pd

from typing import Tuple

from project.common import config_path_parser
from project.constants import JOINED, TRAIN_DATASET, TEST_DATASET
from project.prepair_dataset.config import PrepareDatasetConfig


def split_train_test(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = np.random.rand(df.shape[0])
    train_mask = mask > test_ratio
    return df[train_mask], df[np.logical_not(train_mask)]


def get_datasets(config: PrepareDatasetConfig):
    joined_df = pd.read_csv(JOINED)
    train_df, test_df = split_train_test(joined_df, config.ratio)

    train_df.to_csv(TRAIN_DATASET, index=False)
    test_df.to_csv(TEST_DATASET, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)
    parser.add_argument("--ratio", type=float, default=0.1)

    config = PrepareDatasetConfig.from_args(parser.parse_args())
    get_datasets(config=config)


if __name__ == "__main__":
    main()
