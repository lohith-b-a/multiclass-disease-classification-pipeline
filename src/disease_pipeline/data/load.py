from pathlib import Path
import pandas as pd


def load_raw_data(data_path: str, target_column: str = "disease") -> pd.DataFrame:
    """
    Load raw dataset from CSV and check basic validity.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    return df