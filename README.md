# Multiclass Disease Classification Pipeline

## Overview

This project implements a reproducible machine learning pipeline for multiclass disease classification using symptom-based features.

The work began as an exploratory research workflow developed across multiple notebooks and was subsequently refactored into a modular, reusable, and production-oriented pipeline. The final selected model is a tuned XGBoost classifier with class weighting, chosen based on Macro F1-score to address class imbalance in the dataset.

---

## Objectives

- Build a fully reproducible end-to-end machine learning pipeline  
- Separate experimental research from production-ready code  
- Handle multiclass class imbalance effectively  
- Establish a consistent and robust evaluation framework  
- Perform systematic model comparison and hyperparameter tuning  
- Provide interpretable outputs using feature importance and SHAP  

---

## Pipeline Architecture

The pipeline follows a structured sequence of stages:
Raw Data → Validation → Preprocessing → Feature Engineering → Model Training → Evaluation → Explainability


### Design Principles

- **Modularity**: Each stage is independently executable  
- **Reproducibility**: Configuration-driven via YAML files  
- **Scalability**: Easily extendable to new models and datasets  
- **Separation of concerns**: Distinct research and production layers  

---

## Final Model

| Component         | Description                         |
|------------------|-------------------------------------|
| Model            | XGBoost Classifier                  |
| Enhancement      | Class weighting                     |
| Tuning           | Randomized hyperparameter search    |
| Selection Metric | Macro F1-score                      |

The model was selected to ensure balanced predictive performance across all classes in an imbalanced dataset.

## Final Model Performance

- Accuracy: 0.6543  
- Weighted F1: 0.6636  
- Macro F1: 0.5325

---

## Evaluation Metric

In multiclass imbalanced classification problems, accuracy can be misleading. This project uses Macro F1-score as the primary evaluation metric:

F1_macro = (1/N) * Σ F1_i


This metric assigns equal importance to all classes, ensuring that minority classes are not overlooked during evaluation.

---

## Project Structure
multiclass-disease-classification-pipeline/
├── configs/
├── data/
│ ├── raw/
│ └── processed/
├── models/
├── notebooks/
├── results/
│ └── explainability/
├── src/
│ └── disease_pipeline/
├── run_preprocessing.py
├── run_training.py
├── run_final_model.py
├── run_explainability.py
├── run_shap.py
├── requirements.txt
├── requirements-lock.txt
├── .env.example
├── .gitignore
└── README.md


---

## Notebooks

The `notebooks/` directory contains the research and experimentation phase of the project, including:

- Exploratory Data Analysis (EDA)  
- Model experimentation and comparison  
- Class imbalance analysis  
- Hyperparameter tuning  
- Intermediate findings for thesis discussion  

These notebooks are not used for production execution. The pipeline implemented in `src/` serves as the primary and reproducible workflow.

---

## Installation

git clone https://github.com/lohith-b-a/multiclass-disease-classification-pipeline

cd multiclass-disease-classification-pipeline
pip install -r requirements.txt


---

## Usage

### Preprocessing
PYTHONPATH=src python run_preprocessing.py

### Model Training and Comparison
PYTHONPATH=src python run_training.py


### Final Model Training
PYTHONPATH=src python run_final_model.py


### Feature Importance
PYTHONPATH=src python run_explainability.py

### SHAP Explainability
PYTHONPATH=src python run_shap.py


---

## Outputs

### Processed Data (`data/processed/`)

- train.csv  
- val.csv  
- test.csv  
- disease_label_mapping.csv  
- training_results.csv  
- final_test_results.json  

---

### Model Artifacts (`models/`)

- Final tuned model  
- Best model metadata  

---

### Explainability Outputs (`results/explainability/`)

- feature_importance.csv  
- shap/shap_summary.png  
- shap/shap_sample.csv  

---

## Methodology

### Data Processing

- Data validation  
- Rare-class filtering  
- Label encoding  
- Stratified train/validation/test split  

### Modeling

Models evaluated:

- Dummy Classifier  
- Logistic Regression  
- Linear SVM  
- KNN  
- Random Forest  
- XGBoost  
- XGBoost with class weighting  
- MLP  

### Imbalance Handling

Multiple strategies were explored. Class weighting was selected based on comparative performance.

### Hyperparameter Tuning

Conducted in notebooks and integrated into the pipeline.

### Explainability

- Feature importance (global)  
- SHAP analysis (global and local)  

---

## Reproducibility

- Fixed random seeds  
- YAML-based configuration  
- Modular pipeline design  
- Clear separation between research and production code  

---

## Thesis Contribution

This project demonstrates:

- Transition from notebook-based experimentation to a structured pipeline  
- Effective handling of multiclass imbalance  
- Integration of explainable AI in healthcare  
- Application of industry-standard machine learning engineering practices  

---

## Future Work

- Unit testing  
- Logging and monitoring  
- Dockerization  
- CI/CD pipelines  
- MLflow experiment tracking  
- DVC for data versioning  

---

## Author

Lohith Basavanahalli Anjinappa
