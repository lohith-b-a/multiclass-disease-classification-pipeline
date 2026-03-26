from pathlib import Path
import yaml

from disease_pipeline.utils.seed import set_seed
from disease_pipeline.data.load import load_raw_data
from disease_pipeline.data.validate import validate_raw_dataset, get_feature_columns
from disease_pipeline.data.preprocess import (
    filter_rare_classes,
    encode_target,
    build_feature_target,
    stratified_split,
    save_processed_splits,
)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/preprocessing.yaml"):
    config = load_config(config_path)

    seed = config["seed"]
    raw_data_path = config["data"]["raw_data_path"]
    target_column = config["data"]["target_column"]
    encoded_target_column = config["data"]["encoded_target_column"]
    processed_output_dir = config["data"]["processed_output_dir"]
    min_samples = config["preprocessing"]["min_samples_per_class"]
    label_mapping_path = config["artifacts"]["label_mapping_path"]

    set_seed(seed)

    df = load_raw_data(raw_data_path, target_column=target_column)

    report = validate_raw_dataset(df, target_column=target_column)
    print("Validation report:")
    print(report)

    feature_columns = get_feature_columns(df, target_column)

    df_filtered = filter_rare_classes(
        df,
        target_column=target_column,
        min_samples=min_samples,
    )

    df_encoded, label_encoder, mapping_df = encode_target(
        df_filtered,
        target_column=target_column,
        encoded_column=encoded_target_column,
    )

    X, y = build_feature_target(
        df_encoded,
        feature_columns=feature_columns,
        encoded_column=encoded_target_column,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, seed=seed)

    save_processed_splits(
        X_train, X_val, X_test, y_train, y_val, y_test,
        output_dir=processed_output_dir
    )

    label_mapping_file = Path(label_mapping_path)
    label_mapping_file.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(label_mapping_file, index=False)

    print("Preprocessing completed successfully.")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")


if __name__ == "__main__":
    main()