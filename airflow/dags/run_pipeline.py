import os

from airflow import models
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
import datetime

DAG_ID = 'test'


def get_config_value(task_id: str, key: str):
    return f'{{{{task_instance.xcom_pull(task_ids="{task_id}", key="{key}")}}}}'


class Key:
    TASK_INSTANCE_KEY = 'ti'  # xcom object
    # https://airflow.apache.org/docs/apache-airflow/stable/concepts/operators.html#reserved-params-keyword
    PARAMS = 'params'
    DAG_RUN = 'dag_run'


class ConfigPusher:
    DEFAULT_CONFIG = {'command': 'dvc repro', 'MLFLOW_TRACKING_URI': 'http://51.250.108.121:90/'}
    SERVICES = []

    def prepare_default(self, dag_run) -> dict:
        return {}

    def finalize_config(self, config) -> dict:
        return {}

    def __call__(self, **kwargs):
        task_instance = kwargs[Key.TASK_INSTANCE_KEY]
        dag_run = kwargs[Key.DAG_RUN]
        dag_config = kwargs.get(Key.PARAMS, {})

        default_config = self.DEFAULT_CONFIG.copy()
        default_config.update(self.prepare_default(dag_run))
        config = {
            key: dag_config[key] if key in dag_config else default_value
            for key, default_value in default_config.items()
        }
        config.update(self.finalize_config(config))

        for key, value in sorted(config.items()):
            task_instance.xcom_push(key=key, value=value)


def get_data_dag(dag_id: str = 'test'):
    with models.DAG(
            dag_id=dag_id,
            start_date=datetime.datetime(2022, 4, 1),
            schedule='@once'
    ) as dag:
        config_operator = PythonOperator(
            task_id='config',
            python_callable=ConfigPusher(),
            do_xcom_push=True,
        )

        load_data_operator = DockerOperator(
            task_id='load_data',
            image='tolkkk/irisr_simpe',
            docker_url="unix://var/run/docker.sock",
            network_mode="bridge",
            command=get_config_value('config', 'command'),
            environment={
                'MLFLOW_TRACKING_URI': get_config_value('config','MLFLOW_TRACKING_URI'),
            }

        ),


        config_operator >> load_data_operator


globals()[DAG_ID] = get_data_dag(DAG_ID)
