# Как поднять MLFLOW.

##  Quick start
1. Зайдите на сервер и скачайте репозиторий
2. Запустите создайте файлик в этом репозитории `web-variables.env`. В нем надо указать ряд переменных окружения:
    - AWS_ACCESS_KEY_ID=<access_key>
    - AWS_SECRET_ACCESS_KEY=<secret_access_key>
    - AWS_DEFAULT_REGION=<aws region, например us-west-2>
    - _MLFLOW_SERVER_ARTIFACT_DESTINATION=<путь до  артифактов>
    - _MLFLOW_SERVER_ARTIFACT_ROOT=<путь внутри сервиса для артифактов >
    - _MLFLOW_SERVER_SERVE_ARTIFACTS=<поддержка артифактов mlflow> 
    - MLFLOW_S3_ENDPOINT_URL=<если логирование в s3, то указать кастомный урл>
3. В командной строке: docker compose up 

## Переменный окружения запуска 5 сценария
Все переменные обязательны:
    - AWS_ACCESS_KEY_ID=<access_key>
    - AWS_SECRET_ACCESS_KEY=<secret_access_key>
    - AWS_DEFAULT_REGION=<aws region, например us-west-2>
    - _MLFLOW_SERVER_ARTIFACT_DESTINATION=s3://<bucker-name>
    - _MLFLOW_SERVER_ARTIFACT_ROOT=mlflow-artifacts:/mlartifacts
    - _MLFLOW_SERVER_SERVE_ARTIFACTS=true
    - MLFLOW_S3_ENDPOINT_URL=<если логирование в s3, то указать кастомный урл>

