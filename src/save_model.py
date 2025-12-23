import joblib
from training.train import model  
joblib.dump(model, "models/model_v1.pkl")

print("Model saved successfully at models/model_v1.pkl")
