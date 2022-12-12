import argparse
from enum import Enum

import pandas as pd

from project.common import config_path_parser, get_logger, save_csv
from project.constants import TasksList
from project.prepair_target.config import PrepareTargetsConfig, TargetGroup

TARGETS = {
    TargetGroup.EASY: [0, 1, ],
    TargetGroup.ALL: [0, 1, 2],
}

logger = get_logger(__name__)


def get_feature_group_from_df(group_name: TargetGroup, df: pd.DataFrame) -> pd.DataFrame:
    return df[df['target'].isin(TARGETS.get(group_name))]


def prepare_targets(config: PrepareTargetsConfig):
    target_df = pd.read_csv(TasksList.TARGET_RAW)
    target_df = get_feature_group_from_df(config.target_group, target_df)
    logger.info(f'target shape - {target_df.shape}')

    save_csv(TasksList.TARGET_PREPARED, target_df)


def main():
    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)

    parser.add_argument("--target-group", type=TargetGroup,)

    config = PrepareTargetsConfig.from_args(parser.parse_args())

    prepare_targets(config=config)


if __name__ == "__main__":
    main()
