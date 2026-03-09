"""Run MLflow UI with correct file URI for Windows paths."""
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent
mlruns = (project_root / "mlruns").resolve()
uri = f"file:///{mlruns.as_posix()}"

print("MLflow UI: http://127.0.0.1:5000")
print(f"Using: {uri}")
subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri])
