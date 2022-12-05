import attr


from project.config_from_file import ConfigFromArgs

MLFLOW_LOG = 'mlflow_log'



@attr.s
class MlflowLogConfig(ConfigFromArgs):
    SECTION = MLFLOW_LOG
    experiment_name: str = attr.ib()
    run_name: str = attr.ib()
    tracking_uri: str = attr.ib()

