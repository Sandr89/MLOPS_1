"""
Training script for Twitter Hate Speech classification.
Variant 19 - MLOps Lab 1.
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import clean_tweet, get_label_column, get_text_column


def load_data(data_path: Path) -> Tuple[pd.Series, pd.Series]:
    """Load and preprocess data."""
    df = pd.read_csv(data_path)
    text_col = get_text_column(df)
    label_col = get_label_column(df)

    df[text_col] = df[text_col].astype(str).apply(clean_tweet)
    X = df[text_col]
    y = df[label_col]

    return X, y


def plot_feature_importance(model, feature_names: list, top_n: int = 20, save_path: str = "feature_importance.png"):
    """Plot top N feature importances and save to file."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[-top_n:][::-1]
    top_features = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in indices]
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


def main():
    parser = argparse.ArgumentParser(description="Train Twitter Hate Speech classifier")
    parser.add_argument("--max_features", type=int, default=5000, help="Max features for TF-IDF")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees for RandomForest")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth for RandomForest")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction (0.2 = 80%% train / 20%% test)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to train.csv")
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else (PROJECT_ROOT / "data" / "raw" / "train.csv")
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Place train.csv in data/raw/ or use --data_path")
        sys.exit(1)

    X, y = load_data(data_path)

    # Поділ train.csv на 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

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

    # Feature importance plot
    feature_names = vectorizer.get_feature_names_out()
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    plot_path = artifact_dir / "feature_importance.png"
    plot_feature_importance(model, feature_names, save_path=str(plot_path))

    # MLflow - explicit path so runs always go to project mlruns
    mlruns_path = (PROJECT_ROOT / "mlruns").resolve()
    mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")
    mlflow.set_experiment("Twitter_Hate_Speech_V19")
    with mlflow.start_run():
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)

        mlflow.log_metric("accuracy_train", accuracy_train)
        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.log_metric("f1_train", f1_train)
        mlflow.log_metric("f1_test", f1_test)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(plot_path), "feature_importance")

        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("author", "Variant19")
        mlflow.set_tag("dataset", "twitter_hate_speech")

    print(f"Accuracy train: {accuracy_train:.4f}, test: {accuracy_test:.4f}")
    print(f"F1 train: {f1_train:.4f}, test: {f1_test:.4f}")
    print("Run: mlflow ui to view results")


if __name__ == "__main__":
    main()
