import os.path

from sklearn import datasets
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

import mlflow

import seaborn as sns
import matplotlib.pyplot as plt

import argparse



def main(model_name: str, run_name: str):
    features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']

    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data)
    iris_df.columns = features
    iris_df['target'] = iris.target

    x = iris.data
    y = iris.target

    test_size = 0.8

    train_x, test_x, train_y, test_y = train_test_split(x[:, :], y, test_size=test_size, random_state=1)

    if model_name == 'LogisticRegressionCV':
        model = LogisticRegressionCV(random_state=1)

    if model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(random_state=1)

    model.fit(train_x, train_y)

    pred_test_y = model.predict(test_x)

    pred_train_y = model.predict(train_x)
    recal_train = recall_score(pred_train_y, train_y, average='weighted')

    prec_train = precision_score(pred_train_y, train_y, average='weighted')

    model_params = {k: v for k, v in model.__dict__.items() if isinstance(v, (float, str, int))}

    recal = recall_score(test_y, pred_test_y, average='weighted')

    prec = precision_score(test_y, pred_test_y, average='weighted')

    sns.heatmap(iris_df.corr())

    if not os.path.exists('data'):
        os.mkdir('data')

    plt.savefig(f'data/corr.png')

    mlflow.set_tracking_uri('http://84.201.128.89:90/')

    mlflow.set_experiment('seminar-admastryukov')

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(
            {
                'prec': prec,
                'recal': recal,
            }
        )
        mlflow.log_params(
            {
                'model_name': model_name,
                'num_features': len(train_x[0]),
                'test_size': test_size,
            } | model_params
        )

        mlflow.log_artifact('data/corr.png')
        mlflow.sklearn.log_model(model, 'model')

        with mlflow.start_run(run_name='full_model_params_train', nested=True):
            mlflow.log_metrics(
                {
                    'recal': recal_train,
                    "prec": prec_train,
                }
            )
            mlflow.sklearn.log_model(model, 'model')
            mlflow.log_params(
                {
                    'model_name': model_name,
                    'num_features': len(train_x[0]),
                    'test_size': test_size,
                } | model_params
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='model name', default = 'LogisticRegressionCV')
    parser.add_argument('--run-name', help='run name', default = 'test')

    args = parser.parse_args()
    main(args.model_name,  run_name = args.run_name)
