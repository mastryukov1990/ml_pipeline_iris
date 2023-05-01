from airflow.dags.run_pipeline import get_data_dag
import pandas as pd


def get_custom_dags_df():
    SHEET_ID = '17T1BDxLedpHHo7q4c7I7EPfnx0TXWc0T-Q6UWNCCZD8'
    SHEET_NAME = 'airflow'

    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    return pd.read_csv(url)


def create_custom_pipelines():
    df = get_custom_dags_df()
    for row in df.iterrows():
        globals()[row['dag_id']] = get_data_dag(row['dag_id'], row['container'])


create_custom_pipelines()
