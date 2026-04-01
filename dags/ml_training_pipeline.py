"""Airflow DAG for continuous training and model registration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow import DAG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "models" / "metrics.json"
DEFAULT_THRESHOLD = 0.12


def check_data_available() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data is missing: {DATA_PATH}")


def choose_next_task(**_: object) -> str:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file is missing: {METRICS_PATH}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    f1 = float(metrics["f1"])
    return "register_model" if f1 >= DEFAULT_THRESHOLD else "stop_pipeline"


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2026, 4, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "ct", "lab5"],
) as dag:
    data_check = PythonOperator(
        task_id="data_check",
        python_callable=check_data_available,
    )

    prepare = BashOperator(
        task_id="prepare",
        cwd=str(PROJECT_ROOT),
        bash_command="python -m dvc repro prepare",
    )

    train = BashOperator(
        task_id="train",
        cwd=str(PROJECT_ROOT),
        bash_command="python -m dvc repro train",
    )

    branch = BranchPythonOperator(
        task_id="branch_on_quality",
        python_callable=choose_next_task,
    )

    register_model = BashOperator(
        task_id="register_model",
        cwd=str(PROJECT_ROOT),
        bash_command=(
            "python src/register_model.py "
            "--metrics-path data/models/metrics.json "
            "--model-name twitter-hate-speech-classifier "
            "--stage Staging"
        ),
    )

    stop_pipeline = EmptyOperator(task_id="stop_pipeline")

    data_check >> prepare >> train >> branch
    branch >> register_model
    branch >> stop_pipeline
