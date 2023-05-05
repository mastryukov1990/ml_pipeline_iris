import pandas as pd
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

    def prepare_default(self, dag_run) -> dict:
        return {}

    def finalize_config(self, config) -> dict:
        return {}

    def __call__(self, **kwargs):
        task_instance = kwargs[Key.TASK_INSTANCE_KEY]
        dag_config = kwargs.get(Key.PARAMS, {})

        default_config = self.DEFAULT_CONFIG.copy()
        config = {
            key: dag_config[key] if key in dag_config else default_value
            for key, default_value in default_config.items()
        }
        for key, value in sorted(config.items()):
            task_instance.xcom_push(key=key, value=value)


def get_data_dag(dag_id: str = 'test', image='tolkkk/irisr_simpe'):
    with models.DAG(
            dag_id=dag_id,
            start_date=datetime.datetime(2022, 4, 1),
            schedule='@once'
    ):
        config_operator = PythonOperator(
            task_id='config',
            python_callable=ConfigPusher(),
            do_xcom_push=True,
        )

        load_data_operator = DockerOperator(
            task_id='run_pipeline',
            image=image,
            docker_url="unix://var/run/docker.sock",
            network_mode="bridge",
            command=get_config_value('config', 'command'),
            environment={
                'MLFLOW_TRACKING_URI': get_config_value('config', 'MLFLOW_TRACKING_URI'),
            }

        ),

        config_operator >> load_data_operator


def get_custom_dags_df():
    SHEET_ID = '17T1BDxLedpHHo7q4c7I7EPfnx0TXWc0T-Q6UWNCCZD8'
    SHEET_NAME = 'airflow'

    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    return pd.read_csv(url)


def create_custom_pipelines():
    df = get_custom_dags_df()
    for i, row in df.iterrows():
        globals()[row['dag_id']] = get_data_dag(row['dag_id'], row['container'])


globals()[DAG_ID] = get_data_dag(DAG_ID)
create_custom_pipelines()
