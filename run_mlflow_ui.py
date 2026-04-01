"""Run MLflow UI with a SQLite backend and local artifacts."""
import subprocess
import sys
from src.mlflow_utils import get_artifact_root, get_tracking_uri

print("MLflow UI: http://127.0.0.1:5000")
print(f"Using backend store: {get_tracking_uri()}")
print(f"Using artifacts: {get_artifact_root()}")
subprocess.run(
    [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        get_tracking_uri(),
        "--default-artifact-root",
        get_artifact_root().as_uri(),
    ]
)
