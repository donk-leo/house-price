"""Experiment 003: semantic cleaning, stable house features, and log skew.

This keeps the same Ridge/CV protocol as the earlier experiments so the score
change can be attributed to feature processing rather than a new evaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from exp002_semantic_ridge import (
    CATEGORICAL_CODE_COLUMNS,
    NONE_COLUMNS,
    ZERO_COLUMNS,
)


EXCLUDED_FROM_LOG = {
    "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "GarageYrBlt",
    "YrSold", "MoSold", "MSSubClass",
}


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=["Id"], errors="ignore").copy()
    for column in NONE_COLUMNS:
        if column in result:
            result[column] = result[column].fillna("None")
    for column in ZERO_COLUMNS:
        if column in result:
            result[column] = result[column].fillna(0)

    # Features that express total usable space, facilities, and house age.
    result["TotalSF"] = result[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].sum(axis=1)
    result["TotalBathrooms"] = (
        result["FullBath"] + 0.5 * result["HalfBath"]
        + result["BsmtFullBath"] + 0.5 * result["BsmtHalfBath"]
    )
    result["TotalPorchSF"] = result[
        ["OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch", "WoodDeckSF"]
    ].sum(axis=1)
    result["HouseAge"] = (result["YrSold"] - result["YearBuilt"]).clip(lower=0)
    result["RemodAge"] = (result["YrSold"] - result["YearRemodAdd"]).clip(lower=0)
    result["GarageAge"] = (result["YrSold"] - result["GarageYrBlt"]).clip(lower=0)
    result["TotalOverallScore"] = result["OverallQual"] * result["OverallCond"]

    for column in CATEGORICAL_CODE_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string")
    return result


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    log_features = [
        column for column in numeric_features
        if column not in EXCLUDED_FROM_LOG and X[column].min() >= 0
        and X[column].skew() > 0.75
    ]
    linear_features = [column for column in numeric_features if column not in log_features]
    numeric_linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    numeric_log = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric_linear", numeric_linear, linear_features),
        ("numeric_log", numeric_log, log_features),
        ("categorical", categorical, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=12.0)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run House Prices exp-003")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[1] / "data" / "raw")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "outputs" / "submissions" / "exp003_feature_engineering_ridge.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    X = prepare_features(train.drop(columns=["SalePrice"]))
    X_test = prepare_features(test)
    y = np.log1p(train["SalePrice"])
    pipeline = build_pipeline(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"CV log-RMSE: {scores.mean():.6f} +/- {scores.std():.6f}")
    print("Fold scores: " + ", ".join(f"{score:.6f}" for score in scores))
    pipeline.fit(X, y)
    predictions = np.maximum(0.0, np.expm1(pipeline.predict(X_test)))
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved submission: {args.output}")
    print(f"Rows: {len(submission)}, positive predictions: {(predictions > 0).all()}")


if __name__ == "__main__":
    main()
