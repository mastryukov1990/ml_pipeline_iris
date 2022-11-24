import argparse
import logging
import sys
import json

from typing import Union, Dict, List


def config_path_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--config-path', type=str, default="")
    return parser


def get_logger(name: str):
    logger = logging.Logger(name)
    logger.setLevel('INFO')

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def save_dict(filename: str, metrics: Union[Dict, List]):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
