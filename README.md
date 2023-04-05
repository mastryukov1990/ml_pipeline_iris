# Задача
В репозитории рассмотрена задача по класификации цветков ириса. Стандартно у нас есть 3 класа и 4 поля фичей. 

# Используемые инструменты
Для запусков экспериментов рассмотренны: 
- Jupyter
- Python
- DVC
- MLflow
- Airflow

## Jupyter
Ссылка на ноутбук

## Python 
[Тренировка](https://github.com/mastryukov1990/ml_pipeline_iris/blob/main/lib/train.py), [валидация](https://github.com/mastryukov1990/ml_pipeline_iris/blob/main/lib/eval.py), [параметры запуска](https://github.com/mastryukov1990/ml_pipeline_iris/blob/main/params.yaml)

## DVC
[Настройка пайплайна](https://github.com/mastryukov1990/ml_pipeline_iris/blob/main/dvc.yaml), [параметры запуска](https://github.com/mastryukov1990/ml_pipeline_iris/blob/main/params.yaml), [логи](https://github.com/mastryukov1990/ml_pipeline_iris/blob/main/dvc.lock)

Команды для запуска
```
dvc dag # проверить пайплайн

dvc repro # запуск эксперимента

dvc push # отправить данные в хранилище

dvc pull # скачать данные

dvc exp run # запуск эксперимента
```
