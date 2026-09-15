"""Experiment 005: two-branch stacking (Ridge + XGBoost/tree fallback).

Nested cross-validation evaluates the complete stack without fitting the
meta-model on its own validation rows.  The target is log1p(SalePrice).

XGBoost is used when installed.  In environments where it is unavailable,
sklearn HistGradientBoostingRegressor is used so the stacking workflow can be
validated locally; the script prints which model was selected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from exp002_semantic_ridge import CATEGORICAL_CODE_COLUMNS, NONE_COLUMNS, ZERO_COLUMNS

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    XGBRegressor = None
    HAS_XGBOOST = False


EXCLUDED_FROM_LOG = {
    "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "GarageYrBlt",
    "YrSold", "MoSold", "MSSubClass",
}

# Explicit feature groups for the two first-level models.
# A small overlap on the strongest drivers is intentional: Ridge learns their
# smooth additive trend, while XGBoost learns thresholds and interactions.
LINEAR_NUMERIC_FEATURES = [
    "OverallQual", "OverallCond", "GrLivArea", "TotalSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "TotalBathrooms", "GarageCars", "GarageArea",
    "YearBuilt", "YearRemodAdd", "GarageYrBlt", "HouseAge", "RemodAge",
    "GarageAge", "LotFrontage", "LotArea", "TotalPorchSF", "BedroomAbvGr",
    "TotRmsAbvGrd", "FullBath", "HalfBath", "Fireplaces", "MasVnrArea",
]
LINEAR_CATEGORICAL_FEATURES = [
    "Neighborhood", "MSZoning", "HouseStyle", "BldgType", "Foundation",
    "CentralAir", "KitchenQual", "ExterQual", "BsmtQual", "GarageFinish",
    "SaleCondition", "SaleType", "PavedDrive",
]
TREE_NUMERIC_FEATURES = [
    "OverallQual", "OverallCond", "GrLivArea", "TotalSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "LotArea", "YearBuilt", "YearRemodAdd",
    "GarageCars", "GarageArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBathrooms", "TotalOverallScore", "OverallQual2",
    "OverallQual_GrLivArea", "OverallQual_TotalSF", "PoolArea", "MiscVal",
    "OverallQual_TotalBath", "HouseAge", "RemodAge", "GarageAge",
]
TREE_CATEGORICAL_FEATURES = [
    "Neighborhood", "MSSubClass", "MSZoning", "KitchenQual", "ExterQual",
    "BsmtQual", "BsmtExposure", "Functional", "HouseStyle", "Exterior1st",
    "Exterior2nd", "SaleCondition", "GarageType", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageFinish",
]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=["Id"], errors="ignore").copy()
    for column in NONE_COLUMNS:
        if column in result:
            result[column] = result[column].fillna("None")
    for column in ZERO_COLUMNS:
        if column in result:
            result[column] = result[column].fillna(0)

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
    result["OverallQual2"] = result["OverallQual"] ** 2
    result["OverallQual_GrLivArea"] = result["OverallQual"] * result["GrLivArea"]
    result["OverallQual_TotalSF"] = result["OverallQual"] * result["TotalSF"]

    for column in CATEGORICAL_CODE_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string")
    return result


def select_branch_features(
    X: pd.DataFrame,
    numeric_names: list[str],
    categorical_names: list[str],
) -> pd.DataFrame:
    """Select an explicit branch subset while preserving dataframe dtypes."""
    names = list(dict.fromkeys(
        name for name in numeric_names + categorical_names if name in X.columns
    ))
    return X.loc[:, names].copy()


def make_ridge_pipeline(X: pd.DataFrame, alpha: float) -> Pipeline:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    log_features = [
        c for c in numeric
        if c not in EXCLUDED_FROM_LOG and X[c].min() >= 0 and X[c].skew() > 0.75
    ]
    linear_features = [c for c in numeric if c not in log_features]
    linear_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    log_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("linear_num", linear_num, linear_features),
        ("log_num", log_num, log_features),
        ("cat", cat, categorical),
    ])
    return Pipeline([("preprocessor", prep), ("model", Ridge(alpha=alpha))])


def make_tree_pipeline(
    X: pd.DataFrame,
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
) -> Pipeline:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    # Dense output is required by HistGradientBoosting and accepted by XGBoost.
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", cat_encoder),
        ]), categorical),
    ])
    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective="reg:squarederror",
            random_state=42, n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=12, l2_regularization=1.0, random_state=42,
        )
    return Pipeline([("preprocessor", prep), ("model", model)])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def make_tree_params(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "n_estimators": args.xgb_n_estimators,
        "learning_rate": args.xgb_learning_rate,
        "max_depth": args.xgb_max_depth,
        "min_child_weight": args.xgb_min_child_weight,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
        "reg_alpha": args.xgb_reg_alpha,
        "reg_lambda": args.xgb_reg_lambda,
    }


def generate_base_oof(
    X_lin: pd.DataFrame,
    X_tree: pd.DataFrame,
    y: np.ndarray,
    cv: KFold,
    ridge_alpha: float,
    tree_params: dict[str, float | int],
) -> np.ndarray:
    oof = np.zeros((len(y), 2), dtype=float)
    for train_idx, valid_idx in cv.split(X_lin):
        lin = make_ridge_pipeline(X_lin.iloc[train_idx], ridge_alpha)
        tree = make_tree_pipeline(X_tree.iloc[train_idx], **tree_params)
        lin.fit(X_lin.iloc[train_idx], y[train_idx])
        tree.fit(X_tree.iloc[train_idx], y[train_idx])
        oof[valid_idx, 0] = lin.predict(X_lin.iloc[valid_idx])
        oof[valid_idx, 1] = tree.predict(X_tree.iloc[valid_idx])
    return oof


def nested_stacking_cv(
    X_lin: pd.DataFrame,
    X_tree: pd.DataFrame,
    y: np.ndarray,
    *,
    outer_splits: int,
    inner_splits: int,
    ridge_alpha: float,
    meta_alpha: float,
    tree_params: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return leakage-free outer-fold predictions for both branches and stack."""
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    branch_predictions = np.zeros((len(y), 2), dtype=float)
    stacked_predictions = np.zeros(len(y), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X_lin), start=1):
        X_lin_train = X_lin.iloc[train_idx]
        X_tree_train = X_tree.iloc[train_idx]
        y_train = y[train_idx]
        inner_cv = KFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=100 + fold,
        )
        inner_oof = generate_base_oof(
            X_lin_train,
            X_tree_train,
            y_train,
            inner_cv,
            ridge_alpha,
            tree_params,
        )
        meta = Ridge(alpha=meta_alpha).fit(inner_oof, y_train)

        lin = make_ridge_pipeline(X_lin_train, ridge_alpha)
        tree = make_tree_pipeline(X_tree_train, **tree_params)
        lin.fit(X_lin_train, y_train)
        tree.fit(X_tree_train, y_train)
        branch_predictions[valid_idx, 0] = lin.predict(X_lin.iloc[valid_idx])
        branch_predictions[valid_idx, 1] = tree.predict(X_tree.iloc[valid_idx])
        stacked_predictions[valid_idx] = meta.predict(branch_predictions[valid_idx])

        print(
            f"Outer fold {fold}: "
            f"Ridge={rmse(y[valid_idx], branch_predictions[valid_idx, 0]):.6f}, "
            f"Tree={rmse(y[valid_idx], branch_predictions[valid_idx, 1]):.6f}, "
            f"Stack={rmse(y[valid_idx], stacked_predictions[valid_idx]):.6f}"
        )
    return branch_predictions, stacked_predictions

def main() -> None:
    parser = argparse.ArgumentParser(description="Run House Prices exp-005 stacking")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[1] / "data" / "raw")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "outputs" / "submissions" / "exp005_stacking_ridge_tree.csv")
    parser.add_argument("--ridge-alpha", type=float, default=6.1)
    parser.add_argument("--meta-alpha", type=float, default=1.0)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--xgb-n-estimators", type=int, default=860)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-min-child-weight", type=float, default=3.0)
    parser.add_argument("--xgb-subsample", type=float, default=0.75)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.01)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    args = parser.parse_args()

    if args.ridge_alpha < 0 or args.meta_alpha < 0:
        parser.error("Ridge alpha values must be non-negative")
    if args.outer_splits < 2 or args.inner_splits < 2:
        parser.error("Cross-validation split counts must be at least 2")
    if not 0 < args.xgb_subsample <= 1 or not 0 < args.xgb_colsample_bytree <= 1:
        parser.error("XGBoost sampling ratios must be in (0, 1]")

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    X = prepare_features(train.drop(columns=["SalePrice"]))
    X_test = prepare_features(test)
    y = np.log1p(train["SalePrice"].to_numpy())
    tree_params = make_tree_params(args)

    X_lin = select_branch_features(X, LINEAR_NUMERIC_FEATURES, LINEAR_CATEGORICAL_FEATURES)
    X_test_lin = select_branch_features(X_test, LINEAR_NUMERIC_FEATURES, LINEAR_CATEGORICAL_FEATURES)
    X_tree = select_branch_features(X, TREE_NUMERIC_FEATURES, TREE_CATEGORICAL_FEATURES)
    X_test_tree = select_branch_features(X_test, TREE_NUMERIC_FEATURES, TREE_CATEGORICAL_FEATURES)

    print("Tree model: " + ("XGBRegressor" if HAS_XGBOOST else "HistGradientBoostingRegressor (XGBoost unavailable)"))
    print(f"Linear branch features: {X_lin.shape[1]}; tree branch features: {X_tree.shape[1]}")
    print(f"Ridge alpha: {args.ridge_alpha}; meta Ridge alpha: {args.meta_alpha}")
    nested_branches, nested_stack = nested_stacking_cv(
        X_lin,
        X_tree,
        y,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        ridge_alpha=args.ridge_alpha,
        meta_alpha=args.meta_alpha,
        tree_params=tree_params,
    )
    print(f"Nested OOF Ridge log-RMSE: {rmse(y, nested_branches[:, 0]):.6f}")
    print(f"Nested OOF tree log-RMSE: {rmse(y, nested_branches[:, 1]):.6f}")
    print(f"Nested OOF stacked log-RMSE: {rmse(y, nested_stack):.6f}")

    # Train the final meta-model on full-data OOF predictions, then refit both
    # base branches on all rows for test prediction.
    full_oof_cv = KFold(n_splits=args.outer_splits, shuffle=True, random_state=42)
    full_oof = generate_base_oof(
        X_lin,
        X_tree,
        y,
        full_oof_cv,
        args.ridge_alpha,
        tree_params,
    )
    meta = Ridge(alpha=args.meta_alpha).fit(full_oof, y)
    print(f"Meta coefficients: Ridge={meta.coef_[0]:.6f}, Tree={meta.coef_[1]:.6f}, intercept={meta.intercept_:.6f}")

    lin = make_ridge_pipeline(X_lin, args.ridge_alpha)
    tree = make_tree_pipeline(X_tree, **tree_params)
    lin.fit(X_lin, y)
    tree.fit(X_tree, y)
    test_meta = np.column_stack([lin.predict(X_test_lin), tree.predict(X_test_tree)])
    predictions = np.maximum(0.0, np.expm1(meta.predict(test_meta)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": test["Id"], "SalePrice": predictions}).to_csv(args.output, index=False)
    print(f"Saved local artifact: {args.output}")


if __name__ == "__main__":
    main()
