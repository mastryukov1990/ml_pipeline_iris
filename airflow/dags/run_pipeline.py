from airflow import models
from airflow.decorators import task

from project.load_data.load_data import save_raw
from project.prepair_features.prepair_features import main as prepare_features_main
from project.prepair_target.prepair_target import main as prepare_targets_main
from project.merge.merge import main as merge_main


@task(task_id="load_data")
def load_data_operator():
    save_raw()


@task(task_id="prepare_features")
def prepare_features_operator():
    prepare_features_main()


@task(task_id="prepare_targets")
def prepare_targets_operator():
    prepare_targets_main()


@task(task_id="merge")
def merge_operator():
    merge_main()


def get_data_dag(dag_id: str = 'test'):
    with models.DAG(
            dag_id=dag_id,
    ) as dag:
        load_data_operator() >> [prepare_features_operator(), prepare_targets_operator()] >> merge_operator()
