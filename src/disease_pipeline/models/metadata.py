import json
from pathlib import Path


def save_best_model_metadata(
    model_name: str,
    selection_metric: str,
    best_score: float,
    output_path: str = "models/best_model_metadata.json"
) -> Path:
    """
    Save metadata for the selected best model.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "best_model_name": model_name,
        "selection_metric": selection_metric,
        "best_score": round(float(best_score), 6),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Best model metadata saved: {path}")
    return path