import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_selector import select_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final.csv")
FEATURE_DIR = os.path.join(BASE_DIR, "features")
FEATURE_LIST_PATH = os.path.join(FEATURE_DIR, "feature_list.json")

os.makedirs(FEATURE_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Processed data not found")

    return pd.read_csv(DATA_PATH)

def engineer_features(df):
    df = df.drop(columns=["Name", "Ticket", "PassengerId", "Cabin"], errors="ignore")
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["IsChild"] = (df["Age"] < 12).astype(int)
    df["IsElderly"] = (df["Age"] > 60).astype(int)
    df["IsUpperClass"] = (df["Pclass"] == 1).astype(int)
    df["IsLowerClass"] = (df["Pclass"] == 3).astype(int)
    df["FemaleUpperClass"] = df["Sex"] * df["IsUpperClass"]
    df["MaleLowerClass"] = (1 - df["Sex"]) * df["IsLowerClass"]
    df["HighFare"] = (df["Fare"] > df["Fare"].median()).astype(int)

    return df

def split_and_scale(df):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test

def save_feature_list(all_features, selected_features):
    feature_metadata = {
        "total_features": len(all_features),
        "selected_features_count": len(selected_features),
        "all_features": all_features,
        "selected_features": selected_features
    }

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_metadata, f, indent=4)

def main():
    df = load_data()
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_and_scale(df)

    selected_features = select_features(X_train, y_train)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    X_train.to_csv(os.path.join(FEATURE_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(FEATURE_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(FEATURE_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(FEATURE_DIR, "y_test.csv"), index=False)

    save_feature_list(
        all_features=list(X_train.columns),
        selected_features=selected_features
    )

    print("Feature engineering pipeline completed successfully")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

if __name__ == "__main__":
    main()