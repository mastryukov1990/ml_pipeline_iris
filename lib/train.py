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

from lib.common import save_dict
from lib.mlflow_log import mlflow_log

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def train_model(x, y):
    model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    model.fit(x, y)
    return model


def train():
    with open('params.yaml', 'rb') as f:
        data = yaml.safe_load(f)

    config = data['train']
    task_dir = 'data/train'

    iris = datasets.load_iris()
    x = iris['data'].tolist()
    y = iris['target'].tolist()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=config['test_size'], random_state=RANDOM_SEED)

    model = train_model(train_x, train_y)

    metrics = {'precision': precision_score(train_y, model.predict(train_x), average='micro')}

    save_data = {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y,
    }

    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    save_dict(save_data, os.path.join(task_dir, 'data.json'))
    save_dict(metrics, os.path.join(task_dir, 'metrics.json'))

    sns.heatmap(pd.DataFrame(train_x).corr())
    plt.savefig('data/train/heatmap.png')

    with open('data/train/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    mlflow_log(experiment_name= 'change_test_size',metrics=metrics, run_name='train', params={'test_size': config['test_size']})


if __name__ == '__main__':
    train()
