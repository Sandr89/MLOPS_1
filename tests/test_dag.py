from pathlib import Path

from airflow.models import DagBag

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dag_import_and_structure():
    dag_bag = DagBag(
        dag_folder=str(PROJECT_ROOT / "dags"),
        include_examples=False,
    )

    assert (
        len(dag_bag.import_errors) == 0
    ), f"DAG import errors: {dag_bag.import_errors}"

    dag = dag_bag.dags.get("ml_training_pipeline")
    assert dag is not None, "Expected DAG 'ml_training_pipeline' to be loaded"

    expected_tasks = {
        "data_check",
        "prepare",
        "train",
        "branch_on_quality",
        "register_model",
        "stop_pipeline",
    }
    assert expected_tasks.issubset({task.task_id for task in dag.tasks})
