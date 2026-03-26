from pathlib import Path
import pandas as pd


def load_processed_data(path: str = "data/processed"):
    """
    Load processed train, validation, and test datasets.
    """
    data_path = Path(path)

    train = pd.read_csv(data_path / "train.csv")
    val = pd.read_csv(data_path / "val.csv")
    test = pd.read_csv(data_path / "test.csv")

    return train, val, test


def split_features_target(df: pd.DataFrame, target_column: str = "disease_encoded"):
    """
    Split dataframe into features and target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y