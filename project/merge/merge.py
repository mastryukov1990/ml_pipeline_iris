import argparse

import pandas as pd

from project.common import config_path_parser, save_csv
from project.constants import TasksList, INDEX_COLUMN
from project.merge.config import JoinedConfig, HowMerge


def join2df(df_features: pd.DataFrame, df_target: pd.DataFrame, how: HowMerge) -> pd.DataFrame:
    return pd.merge(left=df_target, right=df_features, how=how, on=INDEX_COLUMN)


def join(config: JoinedConfig):
    df_features = pd.read_csv(TasksList.FEATURES_IRIS_PREPARED, index_col=INDEX_COLUMN)
    df_target = pd.read_csv(TasksList.TARGET_PREPARED, index_col=INDEX_COLUMN)

    joined_df = join2df(df_features=df_features, df_target=df_target, how=config.how)

    save_csv(TasksList.JOINED, joined_df)



def main():
    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)
    parser.add_argument("--how", type=HowMerge)

    config = JoinedConfig.from_args(parser.parse_args())

    join(config=config)


if __name__ == "__main__":
    main()
