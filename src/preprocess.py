"""
Data preprocessing module.
Loads raw data (DVC-tracked), performs feature engineering,
and splits into train/test sets.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_data(raw_path: str) -> pd.DataFrame:
    """Load raw dataset from DVC-tracked path."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Data not found at {raw_path}. Run 'dvc pull' first."
        )
    return pd.read_csv(raw_path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and engineer features from raw data.
    Returns feature matrix X and target vector y.
    """
    df = df.copy()

    # Drop duplicates and rows with excessive missing values
    df = df.drop_duplicates()
    df = df.dropna(thresh=len(df.columns) * 0.7)

    # Separate target (assumes last column is target)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target if categorical
    if y.dtype == "object":
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=target_col)

    # Fill remaining numeric NaN with median
    X = X.fillna(X.median(numeric_only=True))

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y


def split_and_save(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> None:
    """Split data and save to processed directory."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    print(f"Features:  {X_train.shape[1]}")


def main():
    """Run full preprocessing pipeline."""
    params = load_params()
    data_params = params["data"]

    print("Loading raw data...")
    df = load_data(data_params["raw_path"])
    print(f"Raw data shape: {df.shape}")

    print("Preprocessing...")
    X, y = preprocess(df)

    print("Splitting and saving...")
    split_and_save(
        X, y,
        test_size=data_params["test_size"],
        random_state=data_params["random_state"],
    )
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
