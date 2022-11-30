from enum import Enum
from typing import List

import attr
import pickle

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from project.config_from_file import ConfigFromArgs
from project.constants import TasksList
from project.eval_model.config import MetricsName

TRAIN_MODEL = 'train_model'


class ModelNames(str, Enum):
    RANDOM_FOREST = 'random_forest'
    DECISION_TREE_MODEL = 'decision_tree'



def get_model(model_name: ModelNames) -> ClassifierMixin:
    return {
        ModelNames.RANDOM_FOREST: RandomForestClassifier,
        ModelNames.DECISION_TREE_MODEL: DecisionTreeClassifier,
    }[model_name]()


class Model:
    def __init__(self, name: ModelNames):
        self.model = get_model(name)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def log_model(self):
        pickle.dump(self.model, open(TasksList.MODEL_SAVE_PATH, 'wb'))

    def load_model(self):
        self.model = pickle.load(open(TasksList.MODEL_SAVE_PATH, 'rb'))

    def get_score(self, x, y):
        return {'acc': self.model.score(x, y)}


@attr.s
class TrainModelsConfig(ConfigFromArgs):
    SECTION = TRAIN_MODEL
    model_name: ModelNames = attr.ib()
    metrics: List[MetricsName] = attr.ib(default=[MetricsName.PRECISION])

