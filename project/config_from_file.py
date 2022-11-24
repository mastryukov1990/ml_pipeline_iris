import abc

import yaml


def read_yaml(filename: str) -> dict:
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


class ConfigFromDict:
    SECTION: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class DataFromFile:
    SECTION: str

    @classmethod
    def from_file(cls, path_to_yaml: str, section: str):
        return read_yaml(path_to_yaml)[section]


class ConfigFromArgs(DataFromFile, ConfigFromDict):

    @classmethod
    def from_args(cls, args):
        data = {}

        if args.config_path:
            data = cls.from_file(args.config_path, section=cls.SECTION)

        args_dict = vars(args)

        return cls.from_dict(
            **{
                k: args_dict[k] if k in args_dict and args_dict[k] else v
                for k, v in data.items()
            }
        )
