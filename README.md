На занятии рассмотрим, как создать воспроизводимые эксперименты с помощью разных инструментов. 
Мы будем это делать на примере решения задачи про классификацию цветков ириса.

# Задача
У нас есть 3 класса цветков и 4 поля фичей из датасета в sklearn (link).
Надо построить и залогировать модель машинного обучения для экспериментов на этих данных.

# Используемые инструменты
Для запусков экспериментов рассмотрены: 
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
