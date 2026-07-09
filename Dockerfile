FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=models/gtsrb_resnet18_best.pt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "gtsrb_classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
