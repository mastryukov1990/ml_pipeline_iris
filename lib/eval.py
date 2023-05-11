import os.path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import mlflow

from lib.train import load_dict, save_dict, METRICS
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

def eval():
    with open('params.yaml', 'r') as f:
        params_data = yaml.safe_load(f)

    config = params_data['eval']

    if params_data['train']['model'] == 'catboost':
        model = CatBoostClassifier()
        model.load('data/train/catboost_model')
    else:
        with open('data/train/model.pkl', 'rb') as f:
            model = pickle.load(f)

    data = load_dict('data/features_preparation/data.json')
    preds = model.predict(data['test_x'])

    if not os.path.exists('data/eval'):
        os.mkdir('data/eval')

    task_dir = 'data/eval'

    metrics = {}
    for metric_name in config['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['test_y'], preds)

    cls_report = classification_report(data['test_y'], preds, output_dict=True)

    save_dict(metrics, 'data/metrics.json')
    save_dict(cls_report, os.path.join(task_dir, 'cls_report.json'))

    sns.heatmap(pd.DataFrame(data['test_x']).corr())
    plt.savefig('data/eval/heatmap.png')

    params = {'run_type': 'eval'}
    for i in params_data.values():
        params.update(i)

    print(f'eval params - {params}')
    print(f'eval metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    if params_data['train']['model'] == 'catboost':
        mlflow.catboost.log_model(model, "catboost_model")
    else:
        mlflow.sklearn.log_model(model, "model.pkl")

    mlflow.log_artifact(os.path.join(task_dir, "heatmap.png"))
    mlflow.log_artifact(os.path.join(task_dir, "cls_report.json"))


if __name__ == '__main__':
    eval()
