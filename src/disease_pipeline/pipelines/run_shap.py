from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from disease_pipeline.models.train import load_processed_data, split_features_target
from disease_pipeline.models.save import make_safe_model_name


def load_best_model_metadata(metadata_path: str = "models/best_model_metadata.json"):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(sample_size: int = 500):
    metadata = load_best_model_metadata()
    best_model_name = metadata["best_model_name"]

    safe_model_name = make_safe_model_name(best_model_name)
    model_path = Path("models") / f"{safe_model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = joblib.load(model_path)

    _, _, test_df = load_processed_data("data/processed")
    X_test, _ = split_features_target(test_df, target_column="disease_encoded")

    if len(X_test) > sample_size:
        X_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_test.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    output_dir = Path("results/explainability/shap")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sampled input used for SHAP
    X_sample.to_csv(output_dir / "shap_sample.csv", index=False)

    X_sample = X_sample.copy()
    X_sample.columns = [
        c.replace("symptom_", "").replace("_", " ").title()
        for c in X_sample.columns
    ]

    plt.figure(figsize=(16, 8))

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=20,
        plot_size=None
    )

    fig = plt.gcf()

    for ax in fig.axes:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")
            label.set_fontsize(8)

    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.25)
    plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary plot saved: {output_dir / 'shap_summary.png'}")
    print(f"SHAP sample saved: {output_dir / 'shap_sample.csv'}")


if __name__ == "__main__":
    main()