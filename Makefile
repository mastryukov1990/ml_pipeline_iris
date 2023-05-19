IMAGE:=avagapov/aaa_iris_pipeline

build:
	@docker build -t ${IMAGE} .

run:
	@docker run -it ${IMAGE} dvc repro