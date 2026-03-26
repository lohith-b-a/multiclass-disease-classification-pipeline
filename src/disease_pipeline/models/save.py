from pathlib import Path
import re
import joblib


def make_safe_model_name(model_name: str) -> str:
    """
    Convert model name into a clean file-safe name.
    Example:
    'XGBoost (Class Weights)' -> 'xgboost_class_weights'
    """
    safe_name = model_name.lower()
    safe_name = re.sub(r"[()]", "", safe_name)
    safe_name = re.sub(r"[^a-z0-9]+", "_", safe_name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")
    return safe_name


def save_model(model, model_name: str, output_dir: str = "models"):
    """
    Save trained model to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = make_safe_model_name(model_name)
    file_path = output_path / f"{safe_name}.joblib"

    joblib.dump(model, file_path)

    print(f"Model saved: {file_path}")

    return file_path