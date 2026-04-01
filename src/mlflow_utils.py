"""Shared MLflow configuration helpers for training and orchestration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKEND_STORE = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").resolve()


def get_tracking_uri() -> str:
    """Use a SQLite backend so Model Registry is available locally."""
    return DEFAULT_BACKEND_STORE


def get_registry_uri() -> str:
    """Registry shares the same backend store as tracking for this lab."""
    return DEFAULT_BACKEND_STORE


def get_artifact_root() -> Path:
    """Artifact root used by the local MLflow server."""
    return DEFAULT_ARTIFACT_ROOT
