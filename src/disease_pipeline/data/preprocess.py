from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def filter_rare_classes(
    df: pd.DataFrame,
    target_column: str = "disease",
    min_samples: int = 20
) -> pd.DataFrame:
    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    return df[df[target_column].isin(valid_classes)].copy()


def encode_target(
    df: pd.DataFrame,
    target_column: str = "disease",
    encoded_column: str = "disease_encoded"
):
    label_encoder = LabelEncoder()
    df = df.copy()
    df[encoded_column] = label_encoder.fit_transform(df[target_column])

    mapping_df = pd.DataFrame({
        "disease": label_encoder.classes_,
        "encoded_label": range(len(label_encoder.classes_))
    })

    return df, label_encoder, mapping_df


def build_feature_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    encoded_column: str = "disease_encoded"
):
    X = df[feature_columns].copy()
    y = df[encoded_column].copy()

    if not set(pd.unique(X.values.ravel())).issubset({0, 1}):
        raise ValueError("Non-binary values found in features.")

    X = X.astype("uint8")
    return X, y


def stratified_split(X, y, seed: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_splits(
    X_train, X_val, X_test, y_train, y_val, y_test, output_dir: str
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    val_df = X_val.copy()
    test_df = X_test.copy()

    train_df["disease_encoded"] = y_train
    val_df["disease_encoded"] = y_val
    test_df["disease_encoded"] = y_test

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)