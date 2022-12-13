import os

import attr


from project.config_from_file import ConfigFromArgs

MLFLOW_LOG = 'mlflow_log'
MLFLOW_TRACKING_URI = 'MLFLOW_TRACKING_URI'



@attr.s
class MlflowLogConfig(ConfigFromArgs):
    SECTION = MLFLOW_LOG
    experiment_name: str = attr.ib()
    run_name: str = attr.ib()
    tracking_uri: str = attr.ib(default=os.environ[MLFLOW_TRACKING_URI])
