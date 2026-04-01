"""Register the latest successful model in MLflow Model Registry."""

import argparse
import json
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlflow_utils import get_registry_uri, get_tracking_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register model in MLflow Registry")
    parser.add_argument(
        "--metrics-path",
        default="data/models/metrics.json",
        help="Path to metrics JSON produced by training.",
    )
    parser.add_argument(
        "--model-name",
        default="twitter-hate-speech-classifier",
        help="Registered model name in MLflow.",
    )
    parser.add_argument(
        "--stage",
        default="Staging",
        help="Target alias stage label for the registered version.",
    )
    return parser.parse_args()


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)


def main() -> None:
    args = parse_args()
    metrics_path = PROJECT_ROOT / args.metrics_path
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_id = metrics.get("run_id")
    if not run_id:
        raise ValueError("metrics.json does not contain run_id required for registry.")

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_registry_uri(get_registry_uri())
    client = MlflowClient()
    ensure_registered_model(client, args.model_name)

    source = f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri=source, name=args.model_name)
    client.set_registered_model_alias(
        args.model_name, args.stage.lower(), version.version
    )

    print(f"Registered model: {args.model_name}")
    print(f"Version: {version.version}")
    print(f"Alias set: {args.stage.lower()}")


if __name__ == "__main__":
    main()
