IMAGE ?= heart-disease-api
TAG ?= 1.0
SKLEARN_VERSION := $(shell python -c "import sklearn; print(sklearn.__version__)")

.PHONY: setup data eda train test lint run docker-build docker-run smoke compose-up

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-dev.txt

data:
	python scripts/download_data.py

eda:
	python -m src.eda

train:
	python -m src.train

test:
	pytest --cov=src --cov-report=term-missing

lint:
	flake8 src tests scripts

run:
	uvicorn src.api:app --reload --port 8000

docker-build:
	docker build --build-arg SKLEARN_VERSION=$(SKLEARN_VERSION) -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE):$(TAG)

smoke:
	bash scripts/smoke_test.sh

compose-up:
	SKLEARN_VERSION=$(SKLEARN_VERSION) docker compose up --build
