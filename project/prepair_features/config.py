from enum import Enum

import attr

from project.config_from_file import ConfigFromArgs
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
class PrepareFeaturesConfig(ConfigFromArgs):
    SECTION = 'prepare_features'

    features_group: FeatureGroup = attr.ib()
