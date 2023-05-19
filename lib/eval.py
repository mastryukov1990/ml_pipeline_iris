import os.path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import mlflow
from sklearn.metrics import classification_report

from lib.preprocessing import IRIS_TARGET_NAMES
from lib.train import load_dict, save_dict, METRICS


def eval():
    with open('params.yaml', 'r') as f:
        params_data = yaml.safe_load(f)

    config = params_data['eval']
    with open('data/train/model.pkl', 'rb') as f:
        model = pickle.load(f)

    data = load_dict('data/preprocessing/data.json')
    preds = model.predict(data['test_x'])

    if not os.path.exists('data/eval'):
        os.mkdir('data/eval')

    metrics = {}
    for metric_name in config['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['test_y'], preds)

    report = classification_report(data['test_y'], preds,
                                   target_names=IRIS_TARGET_NAMES,
                                   output_dict=True)

    save_dict(metrics, 'data/metrics.json')

    save_dict(report, 'data/eval/classification_report.json')

    sns.heatmap(pd.DataFrame(data['test_x']).corr())
    plt.savefig('data/eval/heatmap.png')

    params = {'run_type': 'eval'}
    for i in params_data.values():
        params.update(i)

    print(f'eval params - {params}')
    print(f'eval metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact('data/eval/heatmap.png')
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_artifact('data/eval/classification_report.json')


if __name__ == '__main__':
    eval()
