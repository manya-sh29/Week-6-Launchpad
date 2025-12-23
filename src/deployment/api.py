import uuid
import json
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "src/models/best_model.pkl"
FEATURE_LIST_PATH = "src/features/feature_list.json"
LOG_PATH = "prediction_logs.csv"
MODEL_VERSION = "v1.0"

model = joblib.load(MODEL_PATH)

with open(FEATURE_LIST_PATH, "r") as f:
    feature_meta = json.load(f)

FEATURES = feature_meta["selected_features"]
app = FastAPI(
    title="Titanic Survival API",
    version=MODEL_VERSION
)
class PassengerInput(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: PassengerInput):
    request_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    X = pd.DataFrame([data.features])
    X = X.reindex(columns=FEATURES, fill_value=0)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    log_data = X.copy()

    log_data["prediction"] = prediction
    log_data["probability"] = probability
    log_data["request_id"] = request_id
    log_data["model_version"] = MODEL_VERSION
    log_data["timestamp"] = timestamp

    log_data.to_csv(
        LOG_PATH,
        mode="a",
        header=not pd.io.common.file_exists(LOG_PATH),
        index=False
    )
    return {
        "request_id": request_id,
        "prediction": prediction,
        "survival_probability": probability,
        "model_version": MODEL_VERSION
    }