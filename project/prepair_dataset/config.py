from enum import Enum

import attr

from project.config_from_file import ConfigFromFile, ConfigFromArgs
from project.constants import INDEX_COLUMN

PREPARE_DATASET = 'prepare_dataset'


@attr.s
class PrepareDatasetConfig(ConfigFromFile, ConfigFromArgs):
    ratio: float = attr.ib(default=0.1)

    @classmethod
    def from_args(cls, args):
        return cls(
            ratio=args.ratio
        ) if not args.config_path else cls.from_file(args.config_path, PREPARE_DATASET)
