@echo off
cd /d "%~dp0"
echo MLflow UI: http://127.0.0.1:5000
"%~dp0venv\Scripts\python.exe" "%~dp0run_mlflow_ui.py"
pause
