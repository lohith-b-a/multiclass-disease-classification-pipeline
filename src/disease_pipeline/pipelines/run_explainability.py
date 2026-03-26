

from pathlib import Path
import json
import joblib
import pandas as pd

from disease_pipeline.models.train import load_processed_data, split_features_target
from disease_pipeline.models.save import make_safe_model_name


def load_best_model_metadata(metadata_path: str = "models/best_model_metadata.json"):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    

    metadata = load_best_model_metadata()
    best_model_name = metadata["best_model_name"]

    safe_model_name = make_safe_model_name(best_model_name)
    model_path = Path("models") / f"{safe_model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = joblib.load(model_path)

    train_df, _, _ = load_processed_data("data/processed")
    X_train, _ = split_features_target(train_df, target_column="disease_encoded")

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Loaded model does not support feature importances.")

    feature_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    output_dir = Path("results/explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "feature_importance.csv"
    feature_importance_df.to_csv(csv_path, index=False)

    print("\nTop 20 Important Features:")
    print(feature_importance_df.head(20))

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()