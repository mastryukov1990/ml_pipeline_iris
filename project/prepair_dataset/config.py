import attr

from project.config_from_file import ConfigFromArgs

PREPARE_DATASET = 'prepare_dataset'


@attr.s
class PrepareDatasetConfig(ConfigFromArgs):
    SECTION = PREPARE_DATASET
    ratio: float = attr.ib()
