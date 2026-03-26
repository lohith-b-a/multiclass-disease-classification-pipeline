# Dataset Overview

## Data Source
The dataset used in this study is sourced from Kaggle:
"Community Healthcare Multi-Symptom Disease Dataset".

## Data Description
The dataset consists of 100,000 patient records and 175 columns.

- 174 features representing symptoms
- 1 target variable representing disease

## Feature Description
All feature columns are binary encoded:
- 1 → symptom present
- 0 → symptom absent

Feature names follow the pattern:
- symptom_<name>
- rare_symptom_<name>

## Target Variable
- Column name: disease
- Type: categorical
- Total classes: 46

This represents a multi-class classification problem.

## Data Quality Assessment
- No missing values
- No duplicate records
- Features are consistent and structured

## Class Distribution
The dataset is imbalanced, with some diseases having significantly more samples than others (e.g., Asthma vs Migraine).

## Leakage Inspection
No features directly encode or reveal the target variable.
Thus, no immediate data leakage is identified.

## Bias and Limitations
- The dataset appears to be synthetic or highly structured
- Lack of missing values and noise reduces real-world realism
- Symptoms are perfectly encoded, which may not reflect clinical variability
- Results may not generalize directly to real-world healthcare settings

## Research Use Justification
The dataset is suitable for:
- comparative evaluation of classification algorithms
- controlled experimentation without data noise

However, findings should be interpreted as performance under ideal conditions.