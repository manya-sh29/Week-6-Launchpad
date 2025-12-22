# MODEL INTERPRETATION — Day 4

## 1. Overview

This document summarizes the model explainability and error analysis for the RandomForest model trained on the Titanic dataset (target: `survived`) as part of Day 4 tasks.



## 2. Hyperparameter Tuning Results

**Optuna best parameters:**

```json
{
    "n_estimators": 449,
    "max_depth": 4,
    "min_samples_split": 7,
    "min_samples_leaf": 5
}
```

* **Optuna best accuracy:** 0.8288
* **GridSearch baseline accuracy:** 0.8147

**Bias / Variance Analysis:**

| Metric         | Value  |
| -------------- | ------ |
| Train Accuracy | 0.8553 |
| Test Accuracy  | 0.8101 |
| Bias           | 0.1447 |
| Variance       | 0.0453 |

> The model shows low variance and moderate bias, indicating slight underfitting but good generalization.



## 3. Feature Importance

The **feature importance chart** shows the most influential features in predicting survival:

1. **Sex** — strongest predictor
2. **Pclass** — second most important
3. **Fare** — moderately important
4. Other features (Age, SibSp, Parch, Embarked) — less influence

> Features like Name, Ticket, and Cabin were dropped since they contain text/ID values and do not help the model.

*Plot saved as: `evaluation/plots/feature_importance.png`*



## 4. SHAP Analysis

**SHAP summary plot** indicates:

* **Positive SHAP values** → increase probability of survival
* **Negative SHAP values** → decrease probability of survival
* Sex (female) has strong positive impact
* Higher Pclass (3rd class) has negative impact

*Plot saved as: `evaluation/plots/shap_summary.png`*

> SHAP confirms that the model relies on the most meaningful features and provides insight into individual predictions.



## 5. Error Analysis

**Error correlation heatmap** shows relationships between misclassified samples and input features:

* Most errors occur in mid-range Fare and Age values.
* Certain combinations of Pclass and Sex lead to misclassification.

*Plot saved as: `evaluation/plots/error_heatmap.png`*

> This helps identify patterns where the model struggles and can guide future feature engineering.



## 6. Summary

* RandomForest model performs well with **~81% test accuracy**.
* **Sex and Pclass** are the most important features.
* SHAP and error analysis provide **clear insights for model explainability**.
* Day 4 tasks — **hyperparameter tuning, explainability, and error analysis** — are fully completed.
