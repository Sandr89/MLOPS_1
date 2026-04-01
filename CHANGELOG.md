# Changelog

## Lab 5

- Added a multi-stage `Dockerfile` and `.dockerignore` for reproducible ML and Airflow environments.
- Added `docker-compose.yml` for local Apache Airflow orchestration with scheduler and webserver.
- Added `dags/ml_training_pipeline.py` for continuous training: data check, prepare, train, branch, and model registration.
- Added `src/register_model.py` to register the trained model in MLflow Model Registry and assign the `staging` alias.
- Added `tests/test_dag.py` and `.github/workflows/main.yaml` for DAG integrity checks and Docker build validation in CI.
- Switched MLflow lab setup to a SQLite backend to support the Model Registry workflow locally.
