"""First reproducible baseline for Kaggle House Prices.

The competition metric is RMSE on log(SalePrice), so the model is trained on
log1p(SalePrice) and predictions are converted back with expm1 for submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=12.0)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the House Prices Ridge baseline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "raw",
        help="Directory containing train.csv, test.csv, and sample_submission.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1]
        / "outputs"
        / "submissions"
        / "baseline_ridge.csv",
    )
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    X = train.drop(columns=["SalePrice"])
    y = np.log1p(train["SalePrice"])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = build_pipeline(X)
    scores = -cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    print(f"CV log-RMSE: {scores.mean():.6f} +/- {scores.std():.6f}")
    print("Fold scores: " + ", ".join(f"{score:.6f}" for score in scores))

    pipeline.fit(X, y)
    predictions = np.maximum(0.0, np.expm1(pipeline.predict(test)))
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved submission: {args.output}")
    print(f"Rows: {len(submission)}, positive predictions: {(predictions > 0).all()}")


if __name__ == "__main__":
    main()
