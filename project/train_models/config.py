from enum import Enum
from typing import Dict, Any, Union, Type

import attr
import pickle

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from abc import ABCMeta

from project.config_from_file import ConfigFromArgs
from project.constants import MODEL_SAVE_PATH


class ModelNames(str, Enum):
    LOGISTIC_REGRESSION = 'logistic_regression'
    RIDGE_REGRESSION = 'ridge_regression'
    DECISION_TREE_MODEL = 'decision_tree'
    SVC = 'svc'



def get_model(model_name: ModelNames) -> ClassifierMixin:
    return {
        ModelNames.RIDGE_REGRESSION: RidgeClassifier,
        ModelNames.LOGISTIC_REGRESSION: LogisticRegression,
        ModelNames.DECISION_TREE_MODEL: DecisionTreeClassifier,
        ModelNames.SVC: SVC,
    }[model_name]()


class Model:
    def __init__(self, name: ModelNames):
        self.model = get_model(name)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def log_model(self):
        pickle.dump(self.model, open(MODEL_SAVE_PATH, 'wb'))

    def load_model(self):
        self.model = pickle.load(open(MODEL_SAVE_PATH, 'rb'))

    def get_score(self, x, y):
        return self.model.score(x, y)


@attr.s
class TrainModelsConfig(ConfigFromArgs):
    model_name: ModelNames = attr.ib(default=ModelNames.LOGISTIC_REGRESSION)
