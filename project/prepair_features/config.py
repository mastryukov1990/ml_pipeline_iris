from enum import Enum

import attr

from project.config_from_file import ConfigFromFile, ConfigFromArgs
from project.constants import INDEX_COLUMN

PREPARE_FEATURES = 'prepare_features'


class FeatureGroup(str, Enum):
    SEPAL_FEATURES_NAME = 'sepal_group'
    PETAL_FEATURES_NAME = 'petal_group'
    ALL_FEATURES_NAME = 'all'


FEATURES = {
    FeatureGroup.PETAL_FEATURES_NAME: ['petal_len', 'petal_wid', INDEX_COLUMN],
    FeatureGroup.SEPAL_FEATURES_NAME: ['sepal_len', 'sepal_wid', INDEX_COLUMN],
    FeatureGroup.ALL_FEATURES_NAME: ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', INDEX_COLUMN],
}


@attr.s
class PrepareFeaturesConfig(ConfigFromFile, ConfigFromArgs):
    features_group: FeatureGroup = attr.ib()

    @classmethod
    def from_args(cls, args):
        return cls(
            features_group=args.features_group
        ) if not args.config_path else cls.from_file(args.config_path, PREPARE_FEATURES)
