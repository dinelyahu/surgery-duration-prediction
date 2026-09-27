"""
Surgery Duration Prediction
Public-safe training script based on the modeling work from the capstone project.

The original clinical dataset is intentionally NOT included in this repository.
Expected target column: "Total Surgery Time"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COL = "Total Surgery Time"

# Direct identifiers / fields intentionally excluded from modeling.
IDENTIFIER_COLUMNS = [
    "Patient",
    "Medical Record",
    "Case Number",
    "Surgeon Name",
    "Anesthetist",
    "Procedure Code",
]


def prepare_data(df: pd.DataFrame):
    """Clean, encode and split the dataset for regression."""
    df = df.copy()

    df.drop(
        columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns],
        inplace=True,
        errors="ignore",
    )

    if TARGET_COL not in df.columns:
        raise ValueError(f'Column "{TARGET_COL}" not found in data.')

    # Missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    object_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in object_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    df = pd.get_dummies(df, columns=object_cols, drop_first=True)

    # Remove target outliers using the IQR rule
    q1 = df[TARGET_COL].quantile(0.25)
    q3 = df[TARGET_COL].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[df[TARGET_COL].between(lower, upper)]

    # Remove highly correlated predictors
    corr = df.corr(numeric_only=True).abs()
    upper_triangle = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )
    correlated = [
        col
        for col in upper_triangle.columns
        if col != TARGET_COL and any(upper_triangle[col] > 0.95)
    ]
    df.drop(columns=correlated, inplace=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def build_model():
    """Create the stacking ensemble used in the project experiments."""
    lgb_1 = LGBMRegressor(
        learning_rate=0.05,
        max_depth=8,
        n_estimators=300,
        random_state=42,
    )
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        random_state=42,
    )
    lgb_2 = LGBMRegressor(
        n_estimators=200,
        random_state=42,
    )
    meta_model = xgb.XGBRegressor(
        n_estimators=100,
        random_state=42,
    )

    return StackingRegressor(
        estimators=[
            ("lgb", lgb_1),
            ("xgb", xgb_model),
            ("lgb2", lgb_2),
        ],
        final_estimator=meta_model,
        passthrough=False,
    )


def main(data_path: str, output_path: str):
    df = pd.read_csv(data_path, low_memory=False)
    X_train, X_test, y_train, y_test = prepare_data(df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Stacking Regressor MAE: {mae:.2f} minutes")

    dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": list(X_train.columns),
        },
        output_path,
    )
    print(f"Saved model bundle to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to an authorized, de-identified CSV dataset")
    parser.add_argument(
        "--output",
        default="surgery_duration_model.joblib",
        help="Output model path",
    )
    args = parser.parse_args()
    main(args.data, args.output)
