## Feature Engineering Documentation  
Day 2 — Feature Engineering & Feature Selection

#Overview
This document describes the feature engineering pipeline implemented in `build_features.py`.  
The pipeline transforms the processed Titanic dataset into a machine-learning-ready format by performing feature cleaning, encoding, feature generation, scaling, train-test splitting, and feature selection.


#Objective
- Prepare clean and meaningful features from processed data  
- Generate additional informative features  
- Normalize numerical features  
- Apply feature selection  
- Produce final `X_train`, `X_test`, `y_train`, and `y_test` datasets.


#Input Data
- Input File: `data/processed/final.csv`


#Feature Engineering Steps

Feature engineering is performed to convert processed data into a machine-learning-ready format.  
The following steps are applied in sequence:

1. Feature Cleaning
Irrelevant, identifier-based, and high-cardinality features are removed to reduce noise and dimensionality.

2. Categorical Feature Encoding
Categorical variables are converted into numerical representations using appropriate encoding techniques to make them compatible with machine learning algorithms.

3. Feature Generation
New features are created from existing attributes to capture hidden patterns, relationships, and interaction effects that improve model learning.

4. Train–Test Splitting
The dataset is split into training and testing sets to ensure unbiased model evaluation, while preserving class distribution.

5. Feature Scaling
Numerical features are standardized to ensure uniform scale and stable model convergence.

6. Feature Selection
Relevant features are selected using statistical and model-based techniques to reduce redundancy and improve generalization.

7. Final Dataset Preparation
The final optimized feature set is saved for consistent use in model training and evaluation.

