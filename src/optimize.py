"""
Optuna HPO stage with Hydra config and MLflow nested runs.
Uses data/prepared/train.csv for optimization and data/prepared/test.csv for final evaluation.
"""
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import hydra
import mlflow
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_prepared_data(prepared_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prepared_path = PROJECT_ROOT / prepared_dir
    train_csv = prepared_path / "train.csv"
    test_csv = prepared_path / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Prepared data not found in {prepared_path}. Run prepare stage first.")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")
    return train_df, test_df


def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=seed, max_iter=1000, **params)
    raise ValueError(f"Unsupported model type: {model_type}")


def suggest_params(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    search_space = cfg.model.search_space
    if cfg.model.type == "random_forest":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                int(search_space.n_estimators.low),
                int(search_space.n_estimators.high),
                step=int(search_space.n_estimators.step),
            ),
            "max_depth": trial.suggest_int(
                "max_depth", int(search_space.max_depth.low), int(search_space.max_depth.high)
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                int(search_space.min_samples_split.low),
                int(search_space.min_samples_split.high),
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                int(search_space.min_samples_leaf.low),
                int(search_space.min_samples_leaf.high),
            ),
            "max_features": trial.suggest_categorical(
                "max_features", list(search_space.max_features.choices)
            ),
        }

    solver = trial.suggest_categorical("solver", list(search_space.solver.choices))
    params = {
        "C": trial.suggest_float(
            "C",
            float(search_space.C.low),
            float(search_space.C.high),
            log=bool(search_space.C.log),
        ),
        "solver": solver,
        "class_weight": trial.suggest_categorical(
            "class_weight", list(search_space.class_weight.choices)
        ),
    }
    if solver == "liblinear":
        params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
    return params


def fit_and_score(
    train_text: pd.Series,
    train_y: pd.Series,
    valid_text: pd.Series,
    valid_y: pd.Series,
    cfg: DictConfig,
    params: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        max_features=int(cfg.model.max_features),
        ngram_range=tuple(cfg.model.ngram_range),
    )
    x_train_vec = vectorizer.fit_transform(train_text)
    x_valid_vec = vectorizer.transform(valid_text)

    model = build_model(cfg.model.type, params, int(cfg.seed))
    model.fit(x_train_vec, train_y)

    train_pred = model.predict(x_train_vec)
    valid_pred = model.predict(x_valid_vec)
    metrics = {
        "accuracy_train": float(accuracy_score(train_y, train_pred)),
        "accuracy_valid": float(accuracy_score(valid_y, valid_pred)),
        "f1_train": float(f1_score(train_y, train_pred, zero_division=0)),
        "f1_valid": float(f1_score(valid_y, valid_pred, zero_division=0)),
    }
    return metrics, {"model": model, "vectorizer": vectorizer}


def objective(trial: optuna.Trial, cfg: DictConfig, train_df: pd.DataFrame) -> float:
    start_time = time.perf_counter()
    train_split, valid_split = train_test_split(
        train_df,
        test_size=float(cfg.hpo.validation_size),
        stratify=train_df["label"],
        random_state=int(cfg.seed),
    )
    params = suggest_params(trial, cfg)

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        metrics, _ = fit_and_score(
            train_split["text"],
            train_split["label"],
            valid_split["text"],
            valid_split["label"],
            cfg,
            params,
        )
        duration = time.perf_counter() - start_time
        loggable_params = {k: ("none" if v is None else v) for k, v in params.items()}
        mlflow.log_params(loggable_params)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_metric("duration_sec", duration)
        mlflow.set_tag("trial_number", trial.number)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("seed", cfg.seed)
        mlflow.set_tag("git_commit", get_git_commit())

    return metrics[f"{cfg.metric}_valid"]


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    train_df, test_df = load_prepared_data(cfg.paths.prepared_dir)

    tracking_uri = cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    if cfg.hpo.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=int(cfg.seed))
    elif cfg.hpo.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=int(cfg.seed))
    else:
        raise ValueError(f"Unsupported sampler: {cfg.hpo.sampler}")

    with mlflow.start_run(run_name=f"study_{cfg.model.type}_{cfg.hpo.sampler}") as parent_run:
        mlflow.log_param("model_type", cfg.model.type)
        mlflow.log_param("sampler", cfg.hpo.sampler)
        mlflow.log_param("metric", cfg.metric)
        mlflow.log_param("n_trials", int(cfg.hpo.n_trials))
        mlflow.log_param("seed", int(cfg.seed))
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")

        study = optuna.create_study(
            direction=str(cfg.hpo.direction),
            sampler=sampler,
            study_name=str(cfg.hpo.study_name),
        )
        study.optimize(lambda trial: objective(trial, cfg, train_df), n_trials=int(cfg.hpo.n_trials))

        best_params = dict(study.best_params)
        report_dir = PROJECT_ROOT / str(cfg.paths.report_dir)
        model_dir = PROJECT_ROOT / str(cfg.paths.model_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        best_params_path = report_dir / f"best_params_{cfg.model.type}_{cfg.hpo.sampler}.json"
        trials_path = report_dir / f"study_results_{cfg.model.type}_{cfg.hpo.sampler}.json"

        final_metrics, bundle = fit_and_score(
            train_df["text"],
            train_df["label"],
            test_df["text"],
            test_df["label"],
            cfg,
            best_params,
        )

        best_model_path = model_dir / f"best_model_{cfg.model.type}_{cfg.hpo.sampler}.joblib"
        joblib.dump(bundle, best_model_path)

        save_json(
            best_params_path,
            {
                "model_type": cfg.model.type,
                "sampler": cfg.hpo.sampler,
                "metric": cfg.metric,
                "best_value": study.best_value,
                "best_params": best_params,
            },
        )
        save_json(
            trials_path,
            {
                "study_name": study.study_name,
                "best_value": study.best_value,
                "best_trial": study.best_trial.number,
                "n_trials": len(study.trials),
                "trial_values": [t.value for t in study.trials if t.value is not None],
            },
        )

        mlflow.log_metric("best_validation_value", float(study.best_value))
        mlflow.log_metric("accuracy_test", final_metrics["accuracy_valid"])
        mlflow.log_metric("f1_test", final_metrics["f1_valid"])
        mlflow.log_artifact(str(best_params_path), "reports")
        mlflow.log_artifact(str(trials_path), "reports")
        mlflow.log_artifact(str(best_model_path), "models")
        mlflow.log_dict(best_params, "best_params.json")

        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation {cfg.metric}: {study.best_value:.4f}")
        print(f"Final test F1: {final_metrics['f1_valid']:.4f}")
        print(f"Best model saved to {best_model_path}")
        print(f"Parent run id: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()