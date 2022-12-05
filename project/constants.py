import os

DATA_DIR = 'data'

INDEX_COLUMN = 'index'
TARGET_COLUMN = 'target'
TRACKING_URI = 'localhost:6666'


def get_save_dir(path: str) -> str:
    return os.path.join(DATA_DIR, path)


class TasksList:
    METRIC_FOLDER = 'metrics'


    LOAD_RAW_FOLDER_NAME = 'load_raw'
    LOAD_RAW_FOLDER_PATH = os.path.join(DATA_DIR, LOAD_RAW_FOLDER_NAME)
    FEATURES_IRIS_RAW = os.path.join(LOAD_RAW_FOLDER_PATH, 'features_iris.csv')
    TARGET_RAW = os.path.join(LOAD_RAW_FOLDER_PATH, 'target.csv')


    PREPARE_FEATURES_FOLDER_NAME = 'prepare_features'
    PREPARE_FEATURES_FOLDER_PATH = os.path.join(DATA_DIR, PREPARE_FEATURES_FOLDER_NAME)
    FEATURES_IRIS_PREPARED = os.path.join(PREPARE_FEATURES_FOLDER_PATH, 'features_iris_prepared.csv')

    PREPARE_TARGETS_FOLDER_NAME = 'prepare_targets'
    PREPARE_TARGETS_FOLDER_PATH = os.path.join(DATA_DIR, PREPARE_TARGETS_FOLDER_NAME)
    TARGET_PREPARED = os.path.join(PREPARE_TARGETS_FOLDER_PATH, 'target_prepared.csv')

    JOINED_FOLDER_NAME = 'joined'
    JOINED_FOLDER_PATH = os.path.join(DATA_DIR, JOINED_FOLDER_NAME)
    JOINED = os.path.join(JOINED_FOLDER_PATH, 'joined.csv')

    PREPARE_DATASET_FOLDER_NAME = 'prepare_dataset'
    JOINED_FOLDER_PATH = os.path.join(DATA_DIR, PREPARE_DATASET_FOLDER_NAME)
    TRAIN_DATASET = os.path.join(JOINED_FOLDER_PATH, 'train_dataset.csv')
    TEST_DATASET = os.path.join(JOINED_FOLDER_PATH, 'test_dataset.csv')

    TRAIN_MODEL_FOLDER_NAME = 'train_model'
    TRAIN_MODEL_FOLDER_PATH = os.path.join(DATA_DIR, TRAIN_MODEL_FOLDER_NAME)
    MODEL_SAVE_PATH = os.path.join(TRAIN_MODEL_FOLDER_PATH, 'model.pkl')
    TRAIN_METRICS = os.path.join(DATA_DIR, METRIC_FOLDER, 'train_metrics.json')

    EVAL_MODEL_FOLDER_NAME = 'eval_model'
    EVAL_MODEL_FOLDER_PATH = os.path.join(DATA_DIR, EVAL_MODEL_FOLDER_NAME)
    TEST_METRICS = os.path.join(DATA_DIR, METRIC_FOLDER, 'test_metrics.json')
