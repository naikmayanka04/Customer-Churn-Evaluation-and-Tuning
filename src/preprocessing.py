from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE


def split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) based on dtype."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Unfitted ColumnTransformer; column lists are read from X's schema only."""
    numeric, categorical = split_column_types(X)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # unseen categories at predict time become all-zeros instead of an error
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def build_dummy_baseline(X: pd.DataFrame) -> Pipeline:
    """Majority-class predictor: the floor any real model must beat."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )


def build_logistic_baseline(X: pd.DataFrame) -> Pipeline:
    """Plain logistic regression with default regularisation (C=1.0).

    No class weighting and no tuning on purpose: this is the Milestone 01 reference point.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def build_logistic_class_weighted(X: pd.DataFrame) -> Pipeline:
    """Logistic regression with class_weight='balanced'.

    scikit-learn reweights the loss inversely proportional to class frequency, so
    mistakes on the minority (churn) class are penalised more. This addresses
    imbalance WITHOUT duplicating or discarding any rows.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def build_random_forest_class_weighted(X: pd.DataFrame) -> Pipeline:
    """Random forest with class_weight='balanced_subsample'.

    Uses the same preprocessing as the logistic models (scaling is unnecessary for
    a tree-based model, but keeping it identical isolates the comparison in
    Milestone 03 to the ALGORITHM, not the feature preparation). Depth and leaf
    size are intentionally modest defaults, not tuned, to keep this a fair
    "reasonable default vs. reasonable default" comparison rather than a
    tuning exercise.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
