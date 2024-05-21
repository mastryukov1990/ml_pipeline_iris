FROM python:3.9-slim

RUN pip3 install --upgrade "pip==22.3"

WORKDIR app

COPY ./requirements.txt  $WORKDIR/

RUN pip3 install --no-cache-dir -r $WORKDIR/requirements.txt

COPY . $WORKDIR


RUN PYTHONPATH="$WORKDIR:$PYTHONPATH"