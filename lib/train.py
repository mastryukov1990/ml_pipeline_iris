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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import mlflow

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('litvinov_ivan_experiments')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

METRICS = {
    'recall': partial(recall_score, average='macro'),
    'precision': partial(precision_score, average='macro'),
    'accuracy': accuracy_score,
}

MODELS_MASK = {
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'knn': KNeighborsClassifier(),
    'logistic_regression': LogisticRegression(),
    'svm': LinearSVC(),
    'catboost': CatBoostClassifier(),
}


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def train_model(x, y, model_name):
    model = MODELS_MASK[model_name]
    model.fit(x, y)
    return model


def train():
    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['train']
    task_dir = 'data/train'

    data = load_dict('data/features_preparation/data.json')
    model = train_model(data['train_x'], data['train_y'], config['model'])

    preds = model.predict(data['train_x'])

    metrics = {}
    for metric_name in params_data['eval']['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['train_y'], preds)

    cls_report = classification_report(data['train_y'], preds, output_dict=True)

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_dict(metrics, os.path.join(task_dir, 'metrics.json'))
    save_dict(cls_report, os.path.join(task_dir, 'cls_report.json'))

    sns.heatmap(pd.DataFrame(data['train_x']).corr())

    plt.savefig('data/train/heatmap.png')

    if config['model'] == 'catboost':
        model.save_model(os.path.join(task_dir, "catboost_model",))
        mlflow.catboost.log_model(model, "catboost_model")
    else:
        with open('data/train/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        mlflow.sklearn.log_model(model, "model.pkl")

    params = {}
    for i in params_data.values():
        params.update(i)

    params['run_type'] = 'train'

    print(f'train params - {params}')
    print(f'train metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(os.path.join(task_dir, "heatmap.png",))
    mlflow.log_artifact(os.path.join(task_dir, "cls_report.json"))


if __name__ == '__main__':
    train()
