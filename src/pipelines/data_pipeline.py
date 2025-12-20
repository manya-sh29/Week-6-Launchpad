import os
import pandas as pd
import numpy as np
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(SRC_DIR, "data", "raw", "dataset.csv")
PROCESSED_DATA_DIR = os.path.join(SRC_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "final.csv")

def load_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("train.csv not found in data/raw")

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Data loaded with shape: {df.shape}")
    return df

def handle_missing_values(df):
    # Numerical columns
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    # Categorical columns
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    # Drop Cabin (too many missing values)
    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    return df

def remove_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Duplicates removed: {before - after}")
    return df


def handle_outliers(df):
    numeric_cols = ["Age", "Fare"]

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower, upper)

    return df


def save_processed_data(df):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned data saved to {OUTPUT_PATH}")

def main():
    df = load_data()
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    save_processed_data(df)

if __name__ == "__main__":
    main()