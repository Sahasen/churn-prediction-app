"""
preprocessing.py
Data loading, cleaning, and feature engineering for the
Bank Customer Churn prediction project.
"""

import pandas as pd
import numpy as np


RAW_COLUMNS_TO_DROP = ["CustomerId", "Surname", "Year"]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw European bank customer CSV file."""
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the derived features requested in the project spec:
    - Balance-to-Salary ratio
    - Product density indicator
    - Engagement-product interaction
    - Age-tenure interaction feature
    """
    df = df.copy()

    # Balance-to-Salary ratio (avoid divide by zero)
    df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"].replace(0, np.nan)
    df["BalanceSalaryRatio"] = df["BalanceSalaryRatio"].fillna(0)

    # Product density indicator: products relative to tenure (longer tenure, fewer
    # products per year = lower "density" of adoption)
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # Engagement-product interaction: active members with more products are the
    # most "engaged"; inactive members with many products may be at higher risk
    df["EngagementProductInteraction"] = df["IsActiveMember"] * df["NumOfProducts"]

    # Age-Tenure interaction: captures relationship length relative to customer age
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

    # Zero balance flag - common strong churn signal in this dataset
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and drop non-informative identifier columns."""
    df = df.copy()

    # Handle missing values (none present in current data, but kept for robustness)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Drop non-informative / identifier columns
    drop_cols = [c for c in RAW_COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Geography and Gender."""
    df = df.copy()
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    return df


def prepare_dataset(path: str):
    """
    Full pipeline: load -> clean -> engineer features -> encode.
    Returns the model-ready dataframe (features + target) and the
    list of feature column names (excluding target).
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)

    target_col = "Exited"
    feature_cols = [c for c in df.columns if c != target_col]

    return df, feature_cols, target_col


if __name__ == "__main__":
    df, features, target = prepare_dataset("data/European_Bank.csv")
    print("Shape after processing:", df.shape)
    print("Feature columns:", features)
    print(df.head())
