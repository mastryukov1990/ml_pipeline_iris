from enum import Enum

import attr

from project.config_from_file import ConfigFromArgs

JOINED = 'joined'


class HowMerge(str, Enum):
    Left = 'left'
    Right = 'right'
    Outer = 'outer'
    Inner = 'inner'

@attr.s
class JoinedConfig(ConfigFromArgs):
    SECTION = JOINED
    how: HowMerge = attr.ib()
