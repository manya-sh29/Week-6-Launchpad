import os
import json
import optuna
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


os.makedirs("tuning", exist_ok=True)
RANDOM_STATE = 42


df = pd.read_csv("data/processed/final.csv")

TARGET_COL = None
for col in df.columns:
    if col.lower() == "survived":
        TARGET_COL = col
        break
if TARGET_COL is None:
    raise ValueError("Target column 'survived' not found")


DROP_COLS = ["Name", "Ticket", "Cabin"]
for col in DROP_COLS:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


cat_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    if col != TARGET_COL:
        df[col] = le.fit_transform(df[col].astype(str))


X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    return cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)


grid_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    grid_params,
    cv=5,
    scoring="accuracy",
)
grid.fit(X_train, y_train)


final_model = RandomForestClassifier(
    **study.best_params,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)


train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

bias_variance = {
    "train_accuracy": round(train_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "bias": round(1 - train_acc, 4),
    "variance": round(train_acc - test_acc, 4),
}


errors = (y_test != y_test_pred).astype(int)
error_df = X_test.copy()
error_df["error"] = errors

kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
error_df["cluster"] = kmeans.fit_predict(error_df.drop("error", axis=1))


results = {
    "optuna_best_params": study.best_params,
    "optuna_best_score": round(study.best_value, 4),
    "grid_best_params": grid.best_params_,
    "grid_best_score": round(grid.best_score_, 4),
    "bias_variance_analysis": bias_variance,
}

with open("tuning/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Day 4 tuning completed")
print("tuning/results.json generated successfully")
