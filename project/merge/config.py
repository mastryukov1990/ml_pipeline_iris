from enum import Enum

import attr

from project.config_from_file import ConfigFromFile, ConfigFromArgs

JOINED = 'joined'


class HowMerge(str, Enum):
    Left = 'left'
    Right = 'right'
    Outer = 'outer'


@attr.s
class JoinedConfig(ConfigFromFile, ConfigFromArgs):
    how: HowMerge = attr.ib(default=HowMerge.Outer)

    @classmethod
    def from_args(cls, args):
        return cls(
            how=args.how
        ) if not args.config_path else cls.from_file(args.config_path, JOINED)