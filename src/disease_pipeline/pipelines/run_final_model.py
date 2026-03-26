import json
from pathlib import Path

import joblib
import yaml
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

from disease_pipeline.models.train import load_processed_data, split_features_target
from disease_pipeline.models.weights import compute_sample_weights
from disease_pipeline.models.save import save_model


def load_config(config_path: str = "configs/final_model.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/final_model.yaml"):
    config = load_config(config_path)

    processed_data_dir = config["data"]["processed_data_dir"]
    model_name = config["model"]["name"]
    final_test_results_path = config["artifacts"]["final_test_results_path"]
    metadata_path = config["artifacts"]["metadata_path"]

    train_df, val_df, test_df = load_processed_data(processed_data_dir)

    X_train, y_train = split_features_target(
        train_df, target_column=config["data"]["target_column"]
    )
    X_val, y_val = split_features_target(
        val_df, target_column=config["data"]["target_column"]
    )
    X_test, y_test = split_features_target(
        test_df, target_column=config["data"]["target_column"]
    )

    sample_weights = None
    if config["training"]["use_class_weights"]:
        sample_weights, class_weights = compute_sample_weights(y_train)
        print("Class weights computed for final model training.")

    model = XGBClassifier(
        n_estimators=config["xgboost"]["n_estimators"],
        max_depth=config["xgboost"]["max_depth"],
        learning_rate=config["xgboost"]["learning_rate"],
        subsample=config["xgboost"]["subsample"],
        colsample_bytree=config["xgboost"]["colsample_bytree"],
        min_child_weight=config["xgboost"]["min_child_weight"],
        gamma=config["xgboost"]["gamma"],
        reg_alpha=config["xgboost"]["reg_alpha"],
        reg_lambda=config["xgboost"]["reg_lambda"],
        random_state=config["xgboost"]["random_state"],
        eval_metric=config["xgboost"]["eval_metric"],
    )

    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted")
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")

    print("\nValidation Results")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Weighted F1: {val_f1_weighted:.4f}")
    print(f"Macro F1: {val_f1_macro:.4f}")

    save_model(model, model_name)

    metadata = {
        "best_model_name": model_name,
        "selection_metric": config["training"]["selection_metric"],
        "validation_accuracy": round(float(val_accuracy), 6),
        "validation_weighted_f1": round(float(val_f1_weighted), 6),
        "validation_macro_f1": round(float(val_f1_macro), 6),
    }

    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata: {metadata_file}")

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

    test_results = {
        "best_model_name": model_name,
        "Accuracy": round(float(test_accuracy), 6),
        "Weighted F1": round(float(test_f1_weighted), 6),
        "Macro F1": round(float(test_f1_macro), 6),
    }

    test_results_file = Path(final_test_results_path)
    test_results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_results_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print("\nFinal Test Results")
    print(test_results)
    print(f"\nSaved: {test_results_file}")


if __name__ == "__main__":
    main()