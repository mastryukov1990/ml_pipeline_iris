import os
import pickle
import random
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import mlflow
from preprocessing import save_dict, get_data

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('mmkuznecov')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

METRICS = {
    'recall': partial(recall_score, average='macro'),
    'precision': partial(precision_score, average='macro'),
    'accuracy': accuracy_score
}
    
def get_model(model_type):

    if model_type == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif model_type == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()

    return model


def train_model(x, y, model_type):

    model = get_model(model_type)

    model.fit(x, y)
    return model


def train():
    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['train']

    task_dir = 'data/train'

    train_x, test_x, train_y, test_y = get_data()

    model = train_model(train_x, train_y, config['model_type'])

    preds = model.predict(train_x)

    metrics = {}
    for metric_name in params_data['eval']['metrics']:
        metrics[metric_name] = METRICS[metric_name](train_y, preds)

    report = classification_report(train_y, preds)

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

    with open('data/train/classification_report.txt', 'w') as f:
        f.write(report)

    params = {}
    for i in params_data.values():
        params.update(i)

    params['run_type'] = 'train'

    print(f'train params - {params}')
    print(f'train metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact('data/train/heatmap.png')
    mlflow.log_artifact('data/train/classification_report.txt')
    mlflow.sklearn.log_model(model, 'logged_model')


if __name__ == '__main__':
    train()
