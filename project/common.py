import argparse
import json
import os
from pathlib import Path

from typing import Union, Dict, List

import pandas as pd

from project.logger import get_logger

logger = get_logger(__name__)


def config_path_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--config-path', type=str, default="params.yaml")
    return parser


def save_dict(filename: str, metrics: Union[Dict, List]):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


def create_parent_folder(filename: str):
    parent_folder = Path(filename).parent

    if not os.path.exists(parent_folder):
        logger.info(f'create parent folder - {parent_folder}')
        os.makedirs(parent_folder)


def save_csv(save_path: str, df: pd.DataFrame):
    create_parent_folder(save_path)
    df.to_csv(save_path, index=False)
