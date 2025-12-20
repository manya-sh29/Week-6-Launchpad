import pandas as pd
import pathlib as Path
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

data_path = "src/data/processed/final.csv"

df = pd.read_csv(data_path)

if 'target' in df.columns:
    target_col = 'target'
else:
    target_col = df.columns[1]  

X = df.drop(target_col, axis=1)
y = df[target_col]

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna(X[col].mode()[0])
    else:
        X[col] = X[col].fillna(X[col].mean())

mask = y.notna()
X = X[mask]
y = y[mask]

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"Features: {X.shape}, Target: {y.name}")

models = {
    "LogisticRegression": LogisticRegression(max_iter=3000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    acc_list, prec_list, rec_list, f1_list, roc_list = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        try:
            if len(np.unique(y_test)) > 2:
                y_prob = model.predict_proba(X_test)
                roc_list.append(roc_auc_score(y_test, y_prob, multi_class='ovr'))
            else:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_list.append(roc_auc_score(y_test, y_prob))
        except:
            roc_list.append(0)

        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_list.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_list.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    results[name] = {
        "Accuracy": float(np.mean(acc_list)),
        "Precision": float(np.mean(prec_list)),
        "Recall": float(np.mean(rec_list)),
        "F1": float(np.mean(f1_list)),
        "ROC_AUC": float(np.mean(roc_list))
    }

best_model_name = max(results, key=lambda x: results[x]["F1"])
best_model = models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")

best_model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

os.makedirs("evaluation", exist_ok=True)
with open("evaluation/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

y_pred_final = best_model.predict(X)
cm = confusion_matrix(y, y_pred_final)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("evaluation/confusion_matrix.png")
plt.close()

print("Day-3 Training Pipeline Completed Successfully")
