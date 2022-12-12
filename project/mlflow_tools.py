import mlflow

from project.logger import get_logger

logger = get_logger(__name__)


def mlflow_set_experiment(
        experiment_name: str,
        tracking_uri: str,
):
    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.get_experiment_by_name(experiment_name):  # проверяем наличие эксперимента с таким именем
        mlflow.set_experiment(experiment_name)  # если есть то сетапим эксперимент
    else:
        mlflow.create_experiment(experiment_name)  # создаем новый эксперимент
        mlflow.set_experiment(experiment_name)

