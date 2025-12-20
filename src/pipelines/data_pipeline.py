import pandas as pd
import numpy as np
from scipy import stats
import os


RAW_FILE = "src/data/raw/dataset.csv"
PROCESSED_FILE = "src/data/processed/final.csv"


def load_data():
    print("Step 1: Loading dataset")
    df = pd.read_csv(RAW_FILE)
    return df


def clean_data(df):
    print("Step 2: Cleaning dataset")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing values
    for column in df.columns:
        if df[column].dtype == "int64" or df[column].dtype == "float64":
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    # Remove outliers using Z-score (only numeric columns)
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    z_scores = np.abs(stats.zscore(df[numeric_columns]))
    df = df[(z_scores < 3).all(axis=1)]

    return df


def save_data(df):
    print("Step 3: Saving cleaned dataset")

    # Create folder if not exists
    os.makedirs("src/data/processed", exist_ok=True)

    # Save CSV
    df.to_csv(PROCESSED_FILE, index=False)

def main():
    df = load_data()
    df_cleaned = clean_data(df)
    save_data(df_cleaned)
    print("Data pipeline completed successfully ")

# Run program
if __name__ == "__main__":
    main()
