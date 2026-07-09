.PHONY: setup train-fast train evaluate api test lint format docker-build docker-run clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .[dev]

train-fast:
	python -m gtsrb_classifier.train --fast --model-name cnn

train:
	python -m gtsrb_classifier.train --model-name resnet18 --epochs 8 --batch-size 128

evaluate:
	python -m gtsrb_classifier.evaluate --checkpoint models/gtsrb_resnet18_best.pt

api:
	uvicorn gtsrb_classifier.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

docker-build:
	docker build -t gtsrb-classifier:latest .

docker-run:
	docker run --rm -p 8000:8000 -v "$$(pwd)/models:/app/models" gtsrb-classifier:latest

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
