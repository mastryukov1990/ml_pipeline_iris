import logging
import sys

from sklearn import datasets
import pandas as pd

from project.common import get_logger, create_parent_folder
from project.constants import TasksList, INDEX_COLUMN

logger = get_logger(__name__)


def save_raw():
    logger.info(f'Start load data')
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(iris.data)
    iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
    iris_df[INDEX_COLUMN] = iris_df.index
    iris_df.dropna(how="all", inplace=True)

    create_parent_folder(TasksList.FEATURES_IRIS_RAW)
    iris_df.to_csv(TasksList.FEATURES_IRIS_RAW, index=False)

    target_df = pd.DataFrame(iris.target)
    target_df[INDEX_COLUMN] = target_df.index
    target_df.columns = ['target', 'index']
    target_df.dropna(how="all", inplace=True)

    create_parent_folder(TasksList.TARGET_RAW)
    target_df.to_csv(TasksList.TARGET_RAW, index=False)

    logger.info(f'Finish load data')



if __name__ == '__main__':
    save_raw()
