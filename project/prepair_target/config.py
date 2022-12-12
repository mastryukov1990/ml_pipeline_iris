from enum import Enum

import attr

from project.config_from_file import ConfigFromArgs

PREPARE_TARGET_SECTION = 'prepare_target'


class TargetGroup(str, Enum):
    EASY = 'target_easy'
    ALL = 'all'


@attr.s
class PrepareTargetsConfig(ConfigFromArgs):
    SECTION = 'prepare_target'
    target_group: TargetGroup = attr.ib(default=TargetGroup.EASY)
