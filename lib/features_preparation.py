import json
import os
import random
import numpy as np
import pandas as pd
import yaml
import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('litvinov_ivan_experiments')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def features_preparation():
    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['features_preparation']
    task_dir = 'data/features_preparation'

    iris = datasets.load_iris(as_frame=True)

    features = [iris.columns[i] for i in range(4) if int(config['features'][i])]

    x = iris['data'][features].values.tolist()
    y = iris['target'].tolist()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=config['test_size'])

    save_data = {
        'train_x': list(train_x),
        'test_x': list(test_x),
        'train_y': list(train_y),
        'test_y': list(test_y),
    }

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_dict(save_data, os.path.join(task_dir, 'data.json'))

    params = {}
    for i in params_data.values():
        params.update(i)

    params['run_type'] = 'features_preparation'

    mlflow.log_params(params)
    mlflow.log_artifact(os.path.join(task_dir, "data.json"))


if __name__ == '__main__':
    features_preparation()