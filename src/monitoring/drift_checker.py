import pandas as pd
from scipy.stats import ks_2samp
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "src", "data","processed", "final.csv")
PRED_LOG_PATH = os.path.join(BASE_DIR, "prediction_logs.csv")
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "best_model.pkl")


def safe_read_csv(path, name):
    if not os.path.exists(path):
        print(f"[WARNING] {name} not found at: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Unable to read {name}: {e}")
        return None


def load_model(path):
    if not os.path.exists(path):
        print(f"[WARNING] Model not found at: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[ERROR] Unable to load model: {e}")
        return None


def check_drift():
    model = load_model(MODEL_PATH)
    train_df = safe_read_csv(TRAIN_DATA_PATH, "Train CSV")
    pred_df = safe_read_csv(PRED_LOG_PATH, "Prediction Logs")

    if model is None or train_df is None or pred_df is None:
        print("\n[INFO] Drift check skipped due to missing resources.\n")
        return

    if not hasattr(model, "feature_names_in_"):
        print("[WARNING] Model has no feature_names_in_")
        return

    features = model.feature_names_in_

    print("\n----- DATA DRIFT REPORT -----\n")

    for col in features:
        if col not in train_df.columns or col not in pred_df.columns:
            print(f"{col} : SKIPPED (column missing)")
            continue

        try:
            _, p_value = ks_2samp(train_df[col], pred_df[col])
            status = "DRIFT" if p_value < 0.05 else "NO_DRIFT"
            print(f"{col} : {status} (p-value = {p_value:.4f})")
        except Exception as e:
            print(f"{col} : ERROR ({e})")


if __name__ == "__main__":
    check_drift()
