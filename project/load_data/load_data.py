import logging
import sys

from sklearn import datasets
import pandas as pd

from project.common import get_logger
from project.constants import FEATURES_IRIS_RAW, TARGET_RAW, INDEX_COLUMN


logger = get_logger(__name__)


def save_raw():
    logger.info(f'Start load data')
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(iris.data)
    iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
    iris_df[INDEX_COLUMN] = iris_df.index
    iris_df.dropna(how="all", inplace=True)
    iris_df.to_csv(FEATURES_IRIS_RAW, index=False)

    target_df = pd.DataFrame(iris.target)
    target_df[INDEX_COLUMN] = target_df.index
    target_df.columns = ['target', 'index']
    target_df.dropna(how="all", inplace=True)
    target_df.to_csv(TARGET_RAW, index=False)

    logger.info(f'Finish load data')



if __name__ == '__main__':
    save_raw()
