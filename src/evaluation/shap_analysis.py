import os
import json
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


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
    X, y, test_size=0.2, random_state=42, stratify=y
)


with open("tuning/results.json", "r") as f:
    results = json.load(f)

best_params = results["optuna_best_params"]
final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)


explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_vals_to_plot = shap_values[1]  
else:
    shap_vals_to_plot = shap_values

os.makedirs("evaluation/plots", exist_ok=True)

plt.figure()
shap.summary_plot(shap_vals_to_plot, X_test, show=False)
plt.tight_layout()
plt.savefig("evaluation/plots/shap_summary.png", dpi=300)
plt.close()

feature_importances = pd.Series(final_model.feature_importances_, index=X_train.columns)
plt.figure(figsize=(8,6))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("evaluation/plots/feature_importance.png", dpi=300)
plt.close()


y_pred = final_model.predict(X_test)
errors = (y_test != y_pred).astype(int)

error_df = X_test.copy()
error_df["error"] = errors

plt.figure(figsize=(10,8))
sns.heatmap(error_df.corr(), annot=True, cmap="coolwarm")
plt.title("Error Correlation Heatmap")
plt.tight_layout()
plt.savefig("evaluation/plots/error_heatmap.png", dpi=300)
plt.close()

print("SHAP analysis completed")
print("Plots saved in evaluation/plots/")
