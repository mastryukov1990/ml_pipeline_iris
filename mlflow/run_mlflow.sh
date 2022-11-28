#!/usr/bin/env bash


sudo docker run -it --rm --name mlflow -p 5000:5000 \
    -v .:/app mlflow-basis:latest

# start a mlflow server
# host 0.0.0.0: allow all remote access
mlflow server --file-store ./mlruns --host 0.0.0.0