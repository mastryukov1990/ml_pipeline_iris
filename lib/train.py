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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
import mlflow

from lib.preprocessing import IRIS_TARGET_NAMES


mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('pokroy')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

METRICS = {
    'recall': partial(recall_score, average='macro'),
    'precision': partial(precision_score, average='macro'),
    'accuracy': accuracy_score,
}

MODELS = {
    'DecisionTree': DecisionTreeClassifier,
    'RandomForest': RandomForestClassifier,
    'NaiveBayes': GaussianNB,
    'SVM': SVC,
    'KNN': KNeighborsClassifier
}


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def train_model(model_class, x, y):
    if model_class in MODELS:
        model = MODELS[model_class]()
    else:
        raise ValueError(f'Unknown model - {model_class}')
    model.fit(x, y)
    return model


def train():
    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['train']
    task_dir = 'data/train'

    data = load_dict('data/preprocessing/data.json')

    model = train_model(config['model'], data['train_x'], data['train_y'])

    preds = model.predict(data['train_x'])

    metrics = {}
    for metric_name in params_data['eval']['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['train_y'], preds)

    report = classification_report(data['train_y'], preds,
                                   target_names=IRIS_TARGET_NAMES,
                                   output_dict=True)

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_dict(metrics, os.path.join(task_dir, 'metrics.json'))

    save_dict(report, os.path.join(task_dir, 'classification_report.json'))

    sns.heatmap(pd.DataFrame(data['train_x']).corr())

    plt.savefig('data/train/heatmap.png')

    with open('data/train/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    params = {}
    for i in params_data.values():
        params.update(i)

    params['run_type'] = 'train'

    print(f'train params - {params}')
    print(f'train metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact('data/train/heatmap.png')
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_artifact(os.path.join(task_dir, 'classification_report.json'))



if __name__ == '__main__':
    train()
