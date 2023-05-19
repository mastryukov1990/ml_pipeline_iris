import os.path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import mlflow

from sklearn.metrics import classification_report

from lib.train import save_dict, load_dict, METRICS

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('lexatref')

def eval():
    with open('params.yaml', 'r') as f:
        params_data = yaml.safe_load(f)

    config = params_data['eval']
    task_dir = 'data/eval'

    with open('data/train/model.pkl', 'rb') as f:
        model = pickle.load(f)

    data = load_dict('data/prepare/data.json')
    preds = model.predict(data['test_x'])

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    metrics = {}
    for metric_name in config['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['test_y'], preds)

    class_report = classification_report(data['test_y'], preds, output_dict=True)

    save_dict(metrics, 'data/metrics.json')
    save_dict(class_report, os.path.join(task_dir, 'class_report.json'))

    sns.heatmap(pd.DataFrame(data['test_x']).corr())
    plt.savefig(os.path.join(task_dir, 'heatmap.png'))

    params = {'run_type': 'eval'}
    for i in params_data.values():
        params.update(i)

    print(f'eval params - {params}')
    print(f'eval metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(os.path.join(task_dir, "heatmap.png",))
    mlflow.log_artifact(os.path.join(task_dir, "class_report.json"))


if __name__ == '__main__':
    eval()
