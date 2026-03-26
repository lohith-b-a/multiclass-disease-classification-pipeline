import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    model_name: str,
    sample_weight=None
):
    """
    Train a model and evaluate it on the validation set.
    Returns:
        result (dict): evaluation metrics
        trained_model: fitted model
        y_val_pred: validation predictions
    """
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_val_pred)
    f1_weighted = f1_score(y_val, y_val_pred, average="weighted")
    f1_macro = f1_score(y_val, y_val_pred, average="macro")

    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    result = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Weighted F1": f1_weighted,
        "Macro F1": f1_macro,
    }

    return result, model, y_val_pred


def results_to_dataframe(results_list: list[dict]) -> pd.DataFrame:
    results_df = pd.DataFrame(results_list)
    return results_df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)