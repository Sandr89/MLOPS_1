import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data" / "models"


def artifact_dir() -> Path:
    return Path(os.getenv("ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))


def test_artifacts_exist():
    base_dir = artifact_dir()
    expected_files = [
        base_dir / "model.joblib",
        base_dir / "metrics.json",
        base_dir / "confusion_matrix.png",
    ]

    for path in expected_files:
        assert path.exists(), f"Required artifact not found: {path}"


def test_metrics_file_contains_required_values():
    metrics_path = artifact_dir() / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for key in ("accuracy", "f1", "accuracy_test", "f1_test"):
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} must be numeric"


def test_quality_gate_f1():
    threshold = float(os.getenv("F1_THRESHOLD", "0.12"))
    metrics_path = artifact_dir() / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    f1 = float(metrics["f1"])

    assert f1 >= threshold, f"Quality Gate not passed: f1={f1:.4f} < {threshold:.2f}"
