import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn import datasets
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow

mlflow.set_tracking_uri('http://51.250.18.36:90')
mlflow.set_experiment('mipt_test_train_size')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def train_model(x, y):
    model = DecisionTreeClassifier()
    model.fit(x, y)
    return model


def train():
    with open('params.yaml', 'rb') as f:
        data = yaml.safe_load(f)

    config = data['train']

    iris = datasets.load_iris()
    task_dir = 'data/train'

    x = iris['data'].tolist()
    y = iris['target'].tolist()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=config['test_size'])

    model = train_model(train_x, train_y)

    metrics = {}
    metrics['precision'] = precision_score(train_y, model.predict(train_x), average='micro')

    save_data = {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y,
    }

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_dict(save_data, os.path.join(task_dir, 'data.json'))
    save_dict(metrics, os.path.join(task_dir, 'metrics.json'))

    sns.heatmap(pd.DataFrame(train_x).corr())
    plt.savefig('data/train/heatmap.png')

    with open('data/train/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    params = {}
    for i in data.values():
        params.update(i)

    params['run_type'] = 'train'
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)


if __name__ == '__main__':
    train()
