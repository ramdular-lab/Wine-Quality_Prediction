
# Wine Quality Classification

## Overview
This project analyzes red and white wine datasets to predict wine quality. The focus is on **multi-class classification** with three categories: `low`, `medium`, and `high`. The pipeline includes data preprocessing, train-test split with stratification, model training, hyperparameter tuning, and evaluation.

## Datasets
- **Red wine:** `winequality-red.csv`  
- **White wine:** `winequality-white.csv`  
- Combined into a single dataframe with a `wine_type` column (0=red, 1=white).

## Features
- 11 chemical properties:  
  `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`,  
  `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`  
- `wine_type` (red/white)  
- Target variable:  
  - `quality` (numeric, 3–9)  
  - `quality_label` (categorical: low / medium / high)

## Preprocessing
- Checked for missing values and duplicates.  
- Combined red and white wines.  
- Created `quality_label` for classification.  
- No feature scaling applied (tree-based model used).  
- **All features retained**; no columns dropped.

## Models Implemented
- **Logistic Regression** – baseline model  
- **RandomForestClassifier** – tuned via **GridSearchCV** with 5-fold cross-validation  
  - Hyperparameters: `n_estimators=100`, `max_depth=30`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`, `class_weight='balanced'`

## Evaluation
- **Metrics:** accuracy, macro F1-score, precision, recall, F1 per class (`low`, `medium`, `high`)  
- **Visualization:** confusion matrix  
- **Results:**  
  - Accuracy: 0.65  
  - Macro F1-score: 0.63  
  - Performance is balanced across all classes.

## How to Run
1. Clone the repository and download the datasets.  
2. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
````

3. Run the Jupyter notebook step by step:

   * Data loading and preprocessing
   * Train-test split
   * Model training (Logistic Regression → RandomForest)
   * Evaluation and visualization

## Notes

* Tree-based models like RandomForest **do not require feature scaling**.
* GridSearchCV was used for hyperparameter tuning of RandomForest.
* All features were retained; removing low-importance or low-correlation features did not improve performance significantly.

```

