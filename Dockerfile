FROM python:3.9

RUN pip3 install --upgrade "pip==22.3"

COPY ./requirements.txt  $PROJECT_ROOT/

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . $PROJECT_ROOT


RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    jupyter nbextension enable code_prettify/code_prettify && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable ExecuteTime && \
    jupyter nbextension enable freeze/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable  execute_time/ExecuteTime && \
    python3.9 -m ipykernel.kernelspec

RUN PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
