import os

DATA_DIR = 'data'


def get_save_dir(path: str) -> str:
    return os.path.join(DATA_DIR, path)


INDEX_COLUMN = 'index'
TARGET_COLUMN = 'target'

FEATURES_IRIS_RAW = get_save_dir('features_iris.csv')
TARGET_RAW = get_save_dir('target.csv')

FEATURES_IRIS_PREPARED = get_save_dir('features_iris_prepared.csv')
TARGET_PREPARED = get_save_dir('target_prepared.csv')

JOINED = get_save_dir('joined.csv')

TRAIN_DATASET = get_save_dir('train_dataset.csv')
TEST_DATASET = get_save_dir('test_dataset.csv')

MODEL_SAVE_PATH = get_save_dir('model')

TRAIN_METRICS = get_save_dir('train_metrics.json')
TEST_METRICS = get_save_dir('test_metrics.json')
