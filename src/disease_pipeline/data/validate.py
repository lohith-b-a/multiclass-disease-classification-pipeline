import pandas as pd


def get_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    return [col for col in df.columns if col != target_column]


def validate_raw_dataset(df: pd.DataFrame, target_column: str = "disease") -> dict:
    feature_columns = get_feature_columns(df, target_column)

    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    non_binary_columns = [
        col for col in feature_columns
        if not set(df[col].dropna().unique()).issubset({0, 1})
    ]

    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_column": target_column,
        "n_features": len(feature_columns),
        "n_classes": int(df[target_column].nunique()),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "non_binary_columns": non_binary_columns,
    }

    return report