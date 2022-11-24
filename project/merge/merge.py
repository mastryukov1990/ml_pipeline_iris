import argparse
from enum import Enum

import pandas as pd

from project.common import config_path_parser
from project.constants import FEATURES_IRIS_PREPARED, INDEX_COLUMN, TARGET_PREPARED, JOINED
from project.merge.config import JoinedConfig, HowMerge


def join2df(df_features: pd.DataFrame, df_target: pd.DataFrame, how: HowMerge = HowMerge.Right) -> pd.DataFrame:
    print(df_features.columns, df_target.columns)
    return pd.merge(left=df_target, right=df_features, how=how, on=INDEX_COLUMN)


def join(config: JoinedConfig):
    df_features = pd.read_csv(FEATURES_IRIS_PREPARED, index_col=INDEX_COLUMN)
    df_target = pd.read_csv(TARGET_PREPARED, index_col=INDEX_COLUMN)

    joined_df = join2df(df_features=df_features, df_target=df_target, how=config.how)

    joined_df.to_csv(JOINED)


def main():
    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)
    parser.add_argument("--how", type=HowMerge, default=HowMerge.Outer)

    config = JoinedConfig.from_args(parser.parse_args())

    join(config=config)


if __name__ == "__main__":
    main()
