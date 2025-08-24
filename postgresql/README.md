# mini-rag

This is a minimal implementation of the RAG model using PostgreSQL.

## Requirements

- Python 3.10

### Install Dependencies

```bash
sudo apt update
sudo apt install libpq-dev gcc python3-dev
```

### Install Python using MiniConda

1- Download and install MiniConda from [MiniConda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)

2- Create a new environment using the following command:

```bash
conda create -n mini-rag python=3.10
```

3- Activate the environment:

```bash
conda activate mini-rag
```

### (Optional) Setup you command line interface for better readability

```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

## Installation

### Install the required packages

```bash
pip install -r requirements.txt
```

### Setup the environment variables

```bash
cp .env.example .env
```

### Run Alembic Migration

```bash
alembic upgrade head
```

Set your environment variables in the `.env` file. Like `GROQ_API_KEY` value.

## Run Docker Compose Services

```bash
cd docker
sudo cp .env.example .env
```

- update `.env` with your credentials

```bash
cd docker
sudo docker compose up -d
```

## Run the FastAPI server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

## Celery (Development Mode)

For development, you can run Celery services manually instead of using Docker:

To Run the **Celery worker**, you need to run the following command in a separate terminal:

```bash
python -m celery -A celery_app worker --queues=default,mail_service_queue --loglevel=info
```

Run **Celery worker** With default and file_processing_queue Queue

```bash
python -m celery -A celery_app worker --queues=default,file_processing_queue --loglevel=info
```

Run **Celery worker** With default, file_processing_queue and data_indexing_queue Queue

```bash
python -m celery -A celery_app worker --queues=default,file_processing_queue,data_indexing_queue --loglevel=info
```

```bash
python -m celery -A celery_app worker --queues=default,file_processing_queue,data_indexing_queue,maintenance_queue --loglevel=info
```

To run the **Beat scheduler**, you can run the following command in a separate terminal:

```bash
python -m celery -A celery_app beat --loglevel=info
```

To Run **Flower Dashboard**, you can run the following command in a separate terminal:

```bash
python -m celery -A celery_app flower --conf=flower_config.py
```

open your browser and go to `http://localhost:5555` to see the dashboard.
