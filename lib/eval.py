import os.path
import pickle
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score

from lib.common import load_dict, save_dict

METRICS = {
    'recall': partial(recall_score, average='micro'),
    'precision': partial(precision_score, average='micro'),
    'accuracy': accuracy_score,
}


def eval():
    with open('params.yaml', 'r') as f:
        data = yaml.safe_load(f)

    config = data['eval']
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


if __name__ == '__main__':
    eval()
