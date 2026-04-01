import os
from pathlib import Path

import pandas as pd

from src.preprocess import get_label_column, get_text_column

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def resolve_data_path() -> Path:
    return Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))


def test_raw_data_exists():
    data_path = resolve_data_path()
    assert data_path.exists(), f"Data not found: {data_path}"


def test_data_schema_basic():
    df = pd.read_csv(resolve_data_path())
    text_col = get_text_column(df)
    label_col = get_label_column(df)

    assert text_col in df.columns
    assert label_col in df.columns
    assert df[label_col].notna().all(), f"{label_col} contains missing values"
    assert df.shape[0] >= 50, "Too few rows for a stable smoke experiment"


def test_label_has_multiple_classes():
    df = pd.read_csv(resolve_data_path())
    label_col = get_label_column(df)
    assert (
        df[label_col].nunique() >= 2
    ), "Label column must contain at least two classes"
