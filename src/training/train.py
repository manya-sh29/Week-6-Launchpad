import json
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

X_train = pd.read_csv("../features/X_train.csv")
y_train = pd.read_csv("../features/y_train.csv").values.ravel()

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    avg_scores = {metric: np.mean(scores[f"test_{metric}"]) for metric in scoring}
    results[name] = avg_scores

    if avg_scores["f1"] > best_score:
        best_score = avg_scores["f1"]
        best_model = model

best_model.fit(X_train, y_train)

joblib.dump(best_model, "../models/best_model.pkl")

with open("../evaluation/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PLOTS_DIR = "../evaluation/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

y_pred = best_model.predict(X_train)
cm = confusion_matrix(y_train, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix - Best Model")
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.close()