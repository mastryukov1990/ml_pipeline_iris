import os.path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import mlflow
from sklearn.metrics import classification_report

from lib.train import load_dict, save_dict, METRICS


def eval():
    with open('params.yaml', 'r') as f:
        params_data = yaml.safe_load(f)

    config = params_data['eval']
    with open('data/train/model.pkl', 'rb') as f:
        model = pickle.load(f)

    data = load_dict('data/train/data.json')
    preds = model.predict(data['test_x'])

    if not os.path.exists('data/eval'):
        os.mkdir('data/eval')

    metrics = {}
    for metric_name in config['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['test_y'], preds)

    save_dict(metrics, 'data/metrics.json')

    sns.heatmap(pd.DataFrame(data['test_x']).corr())
    plt.savefig('data/eval/heatmap.png')

    params = {'run_type': 'eval'}
    for i in params_data.values():
        params.update(i)

    print(f'eval params - {params}')
    print(f'eval metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact('data/train/heatmap.png')
    mlflow.sklearn.log_model(model, "my_model")

    clsf_report = pd.DataFrame(classification_report(data['test_y'], preds, output_dict=True))
    clsf_report.to_json('classification_report.json')
    mlflow.log_artifact('classification_report.json')

if __name__ == '__main__':
    eval()
