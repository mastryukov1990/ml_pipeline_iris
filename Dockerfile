FROM python:3.9

RUN pip3 install --upgrade "pip==22.3"

WORKDIR app

COPY ./requirements.txt  $WORKDIR/

RUN pip3 install --no-cache-dir -r $WORKDIR/requirements.txt

RUN git config --global user.email "lexatref@gmail.com"
RUN git config --global user.name "lexatref"

COPY . $WORKDIR


RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    jupyter nbextension enable code_prettify/code_prettify && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable ExecuteTime && \
    jupyter nbextension enable freeze/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable  execute_time/ExecuteTime && \
    python3.9 -m ipykernel.kernelspec

RUN PYTHONPATH="$WORKDIR:$PYTHONPATH"
