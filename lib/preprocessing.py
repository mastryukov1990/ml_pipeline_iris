import json
import os
import random

import numpy as np
import yaml
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IRIS_TARGET_NAMES = [
    'setosa', 'versicolor', 'virginica'
]
IRIS_FEATURE_NAMES = [
    'sepal_length', 'sepal_width',
    'petal_length', 'petal_width'
]


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def preprocessing():
    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['preprocessing']

    iris = datasets.load_iris()
    task_dir = 'data/preprocessing'

    features = [IRIS_FEATURE_NAMES.index(name) for name in config['features']]
    x = iris['data'][:, features].tolist()
    y = iris['target'].tolist()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=config['test_size'])

    save_data = {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y,
    }

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_dict(save_data, os.path.join(task_dir, 'data.json'))


if __name__ == '__main__':
    preprocessing()
