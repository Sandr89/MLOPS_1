"""
Data preparation stage: raw CSV -> cleaned train/test splits.
Input: data/raw/train.csv
Output: data/prepared/train.csv, data/prepared/test.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import clean_tweet, get_label_column, get_text_column


def main():
    parser = argparse.ArgumentParser(description="Prepare raw data for training")
    parser.add_argument(
        "input_csv", type=str, help="Path to raw CSV (e.g. data/raw/train.csv)"
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory (e.g. data/prepared)"
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    text_col = get_text_column(df)
    label_col = get_label_column(df)

    df[text_col] = df[text_col].astype(str).apply(clean_tweet)
    X = df[[text_col]].rename(columns={text_col: "text"})
    y = df[label_col]
    X["label"] = y

    X_train, X_test = train_test_split(
        X, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    train_out = output_path / "train.csv"
    test_out = output_path / "test.csv"
    X_train.to_csv(train_out, index=False)
    X_test.to_csv(test_out, index=False)

    print(f"Saved {len(X_train)} train, {len(X_test)} test -> {output_path}")


if __name__ == "__main__":
    main()
