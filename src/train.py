"""
Training stage: prepared data -> model + MLflow.
Input: data/prepared/train.csv, data/prepared/test.csv
Output: data/models/, MLflow artifacts.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlflow_utils import get_registry_uri, get_tracking_uri


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    save_path: str = "feature_importance.png",
):
    """Plot top N feature importances and save to file."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[-top_n:][::-1]
    top_features = [
        feature_names[i] if i < len(feature_names) else f"f{i}" for i in indices
    ]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importance (words)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """Plot a confusion matrix for CI reports and save to file."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Twitter Hate Speech classifier")
    parser.add_argument("prepared_dir", type=str, help="Path to data/prepared")
    parser.add_argument(
        "model_dir", type=str, help="Path to save model (e.g. data/models)"
    )
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    prepared_path = Path(args.prepared_dir)
    model_path = Path(args.model_dir)
    train_csv = prepared_path / "train.csv"
    test_csv = prepared_path / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Prepared data not found in {prepared_path}. Run prepare stage first."
        )

    model_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")
    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    model.fit(X_train_vec, y_train)

    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)

    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred, zero_division=0)
    f1_test = f1_score(y_test, y_test_pred, zero_division=0)

    model_file = model_path / "model.joblib"
    metrics_file = model_path / "metrics.json"
    confusion_matrix_file = model_path / "confusion_matrix.png"
    feature_importance_file = model_path / "feature_importance.png"

    joblib.dump({"model": model, "vectorizer": vectorizer}, model_file)

    metrics = {
        "accuracy_train": float(accuracy_train),
        "accuracy_test": float(accuracy_test),
        "f1_train": float(f1_train),
        "f1_test": float(f1_test),
        "accuracy": float(accuracy_test),
        "f1": float(f1_test),
        "model_path": str(model_file.as_posix()),
    }
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    feature_names = vectorizer.get_feature_names_out()
    plot_feature_importance(
        model, list(feature_names), save_path=str(feature_importance_file)
    )
    plot_confusion_matrix(y_test, y_test_pred, save_path=str(confusion_matrix_file))

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_registry_uri(get_registry_uri())
    mlflow.set_experiment("Twitter_Hate_Speech_V19")
    with mlflow.start_run() as run:
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)

        mlflow.log_metric("accuracy_train", accuracy_train)
        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.log_metric("f1_train", f1_train)
        mlflow.log_metric("f1_test", f1_test)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(metrics_file), "metrics")
        mlflow.log_artifact(str(feature_importance_file), "feature_importance")
        mlflow.log_artifact(str(confusion_matrix_file), "evaluation")

        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("author", "Variant19")
        mlflow.set_tag("dataset", "twitter_hate_speech")
        metrics["run_id"] = run.info.run_id
        metrics["experiment_name"] = "Twitter_Hate_Speech_V19"
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Accuracy train: {accuracy_train:.4f}, test: {accuracy_test:.4f}")
    print(f"F1 train: {f1_train:.4f}, test: {f1_test:.4f}")
    print(f"Model saved to {model_file}")
    print(f"Metrics saved to {metrics_file}")
    print(f"Confusion matrix saved to {confusion_matrix_file}")
    print(f"MLflow run id: {metrics['run_id']}")


if __name__ == "__main__":
    main()
