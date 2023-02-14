import os

import mlflow

MLFLOW_TRACKING_URI = 'MLFLOW_TRACKING_URI'
os.environ[MLFLOW_TRACKING_URI] = 'http://51.250.18.36:90/'


def mlflow_log(experiment_name: str, params = None, metrics= None, run_name=None, artifact_dir:str = None ):
    mlflow.set_tracking_uri(os.environ[MLFLOW_TRACKING_URI])
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)

        if metrics:
            mlflow.log_metrics(metrics)

        if artifact_dir:
            mlflow.log_artifacts(artifact_dir)
