from airflow import models
from airflow.providers.docker.operators.docker import DockerOperator
import datetime
# from project.load_data.load_data import save_raw
# from project.prepair_features.prepair_features import main as prepare_features_main
# from project.prepair_target.prepair_target import main as prepare_targets_main
# from project.merge.merge import main as merge_main


DAG_ID = 'test'

def get_data_dag(dag_id: str = 'test'):
    with models.DAG(
        dag_id=dag_id,
        start_date= datetime.datetime(2022, 4, 1),
    ) as dag:
        load_data_operator = DockerOperator(
            task_id='load_data',
            image='iris_docker:latest',

            # cmd='PYTHONPATH="." python project/load_data/load_data.py --config-path params.yaml'
        )

        load_data_operator

globals()[DAG_ID] = get_data_dag(DAG_ID)