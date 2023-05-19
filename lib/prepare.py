import json
import os
import random
import numpy as np
import pandas as pd
import yaml
import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split

import json
import os
import pickle
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from lib.train import save_dict

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('lexatref')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FEATURES = {
    'sl': 'sepal length (cm)',
    'sw': 'sepal width (cm)',
    'pl': 'petal length (cm)',
    'pw': 'petal width (cm)',
}

def prepare():

    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['prepare']
    task_dir = 'data/prepare'

    iris = datasets.load_iris(as_frame=True)

    features = []
    for feature in config['features']:
        features.append(FEATURES[feature])

    x = iris['data'][features].values.tolist()
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

    params = {}
    for i in params_data.values():
        params.update(i)

    params['run_type'] = 'prepare'

    mlflow.log_params(params)
    mlflow.log_artifact(os.path.join(task_dir, "data.json"))


if __name__ == '__main__':
    prepare()