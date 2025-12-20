# Day 3 - Model Comparison

This file summarizes all tasks performed in Day-3: Model Building, Training, Evaluation, and Documentation.

## 1. Dataset Information
- Number of Samples: 892
- Data Source: `src/data/processed/final.csv` (processed dataset from Day-2)


## 2. Feature-Target Split
- **X (Features):** All columns except target  
- **y (Target):** Target column 


## 3. Models Defined
- Logistic Regression  
- Random Forest  
- Neural Network (MLPClassifier)  
- XGBoost   


## 4. Cross-Validation
- **Method:** 5-Fold Stratified Cross Validation  
- **Purpose:** Reduce overfitting and get reliable evaluation metrics  
- Metrics collected per fold: Accuracy, Precision, Recall, F1 Score, ROC-AUC


## 5. Model Performance Metrics

| Model Name          | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| LogisticRegression  | 0.7486   | 0.7683    | 0.7486 | 0.7252   | 0.8648  |
| RandomForest        | 0.8328   | 0.8392    | 0.8328 | 0.8270   | 0.8786  |
| NeuralNetwork       | 0.6026   | 0.6439    | 0.6026 | 0.6052   | 0.6586  |
| XGBoost             | 0.8013   | 0.8008    | 0.8013 | 0.8004   | 0.8641  |

> **Best Model Selected:** RandomForest  


## 6. Observations
- RandomForest demonstrates the most balanced and highest overall performance.  
- NeuralNetwork performed poorly, likely due to limited data and high feature dimensionality.  
- LogisticRegression shows decent performance, but may benefit from feature scaling.  
- XGBoost performs well but F1 is slightly lower than RandomForest.  


## 7. Model Saving & Evaluation Artifacts
- Best model saved as: `models/best_model.pkl`  
- Metrics saved as: `evaluation/metrics.json`  
- Confusion matrix saved as: `evaluation/confusion_matrix.png`  


