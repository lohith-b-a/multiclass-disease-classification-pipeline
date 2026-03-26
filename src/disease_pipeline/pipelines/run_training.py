import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from disease_pipeline.models.train import load_processed_data, split_features_target
from disease_pipeline.models.evaluate import evaluate_model, results_to_dataframe
from disease_pipeline.models.save import save_model
from disease_pipeline.models.metadata import save_best_model_metadata
from disease_pipeline.models.weights import compute_sample_weights

def main():
    train_df, val_df, test_df = load_processed_data("data/processed")

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    sample_weights, class_weights = compute_sample_weights(y_train)
    print("\nComputed class weights for imbalanced training data.")

    results_list = []
    trained_models = {}

    dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy_results, dummy_trained_model, _ = evaluate_model(
        dummy_model,
        X_train, y_train,
        X_val, y_val,
        "Dummy Classifier"
    )
    results_list.append(dummy_results)
    trained_models["Dummy Classifier"] = dummy_trained_model

    logistic_model = LogisticRegression(
        max_iter=2600,
        class_weight="balanced",
        random_state=42
    )
    logistic_results, logistic_trained_model, _ = evaluate_model(
        logistic_model,
        X_train, y_train,
        X_val, y_val,
        "Logistic Regression (Balanced)"
    )
    results_list.append(logistic_results)
    trained_models["Logistic Regression (Balanced)"] = logistic_trained_model

    svm_model = LinearSVC(
        C=1,
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    )
    svm_results, svm_trained_model, _ = evaluate_model(
        svm_model,
        X_train, y_train,
        X_val, y_val,
        "Linear SVM (C=1, Balanced)"
    )
    results_list.append(svm_results)
    trained_models["Linear SVM (C=1, Balanced)"] = svm_trained_model

    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1
        ))
    ])
    knn_results, knn_trained_model, _ = evaluate_model(
        knn_model,
        X_train, y_train,
        X_val, y_val,
        "KNN (k=5, distance-weighted)"
    )
    results_list.append(knn_results)
    trained_models["KNN (k=5, distance-weighted)"] = knn_trained_model

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    rf_results, rf_trained_model, _ = evaluate_model(
        rf_model,
        X_train, y_train,
        X_val, y_val,
        "Random Forest (300 trees)"
    )
    results_list.append(rf_results)
    trained_models["Random Forest (300 trees)"] = rf_trained_model

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    xgb_results, xgb_trained_model, _ = evaluate_model(
        xgb_model,
        X_train, y_train,
        X_val, y_val,
        "XGBoost"
    )
    results_list.append(xgb_results)
    trained_models["XGBoost"] = xgb_trained_model

    mlp_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ))
])
    mlp_results, mlp_trained_model, _ = evaluate_model(
        mlp_model,
        X_train, y_train,
        X_val, y_val,
        "MLP Classifier"
    )
    results_list.append(mlp_results)
    trained_models["MLP Classifier"] = mlp_trained_model

    xgb_weighted_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    xgb_weighted_results, xgb_weighted_trained_model, _ = evaluate_model(
        xgb_weighted_model,
        X_train, y_train,
        X_val, y_val,
        "XGBoost (Class Weights)",
        sample_weight=sample_weights
    )
    results_list.append(xgb_weighted_results)
    trained_models["XGBoost (Class Weights)"] = xgb_weighted_trained_model

    results_df = results_to_dataframe(results_list)

    print("\nModel comparison:")
    print(results_df)

    results_df.to_csv("data/processed/training_results.csv", index=False)
    print("\nSaved: data/processed/training_results.csv")

    best_model_name = results_df.iloc[0]["Model"]
    best_macro_f1 = results_df.iloc[0]["Macro F1"]
    best_model = trained_models[best_model_name]

    print(f"\nBest model selected by Macro F1: {best_model_name}")
    print(f"Best Macro F1: {best_macro_f1:.4f}")

    save_model(best_model, best_model_name)

    save_best_model_metadata(
        model_name=best_model_name,
        selection_metric="Macro F1",
        best_score=best_macro_f1
    )


if __name__ == "__main__":
    main()