import argparse
from enum import Enum

import pandas as pd

from project.common import config_path_parser, get_logger, save_csv
from project.constants import TasksList
from project.prepair_features.config import PrepareFeaturesConfig, FEATURES, FeatureGroup


logger = get_logger(__name__)


def get_feature_group_from_df(group_name: FeatureGroup, df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES[group_name]]


def prepare_features(config: PrepareFeaturesConfig):
    features_df = pd.read_csv(TasksList.FEATURES_IRIS_RAW)
    features_df = get_feature_group_from_df(config.features_group, features_df)

    logger.info(f'features shape - {features_df.shape}')

    save_csv(TasksList.FEATURES_IRIS_PREPARED, features_df)


def main():
    logger.info('Start prepare features')

    parser = argparse.ArgumentParser()
    parser = config_path_parser(parser)
    parser.add_argument("--features-group", type=FeatureGroup, default=FeatureGroup.SEPAL_FEATURES_NAME)

    config = PrepareFeaturesConfig.from_args(parser.parse_args())
    logger.info(f'config - {config}')
    prepare_features(config=config)

    logger.info('Finished prepare features')


if __name__ == "__main__":
    main()
