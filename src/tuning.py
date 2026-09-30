from __future__ import annotations

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import RANDOM_STATE

# Logistic regression: regularisation strength, the parameter most responsible for
# controlling over/underfitting in a linear model. class_weight="balanced" is fixed
# (not searched) because Milestones 02-03 already established it as the right way to
# handle this dataset's imbalance; this grid tunes the model itself, not whether to
# handle imbalance. Penalty type is left at scikit-learn's default (L2) rather than
# searched, since only one value would be searched anyway and newer scikit-learn
# versions deprecate passing `penalty` explicitly alongside `C`.
LOGISTIC_PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
}

# Random forest: tree count, depth, and leaf size, the parameters most
# responsible for the overfitting gap observed in Milestone 03.
RANDOM_FOREST_PARAM_GRID = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [4, 8, 12, None],
    "model__min_samples_leaf": [1, 5, 10],
}

PARAM_GRIDS = {
    "logistic_regression": LOGISTIC_PARAM_GRID,
    "random_forest": RANDOM_FOREST_PARAM_GRID,
}


def build_grid_search(pipeline, model_key: str, scoring: str = "f1", cv_folds: int = 5) -> GridSearchCV:
    """GridSearchCV over the pipeline, using the grid registered for model_key.

    Stratified CV because the target is imbalanced (see Milestone 02).
    scoring="f1" by default: on this imbalanced problem, accuracy would let
    GridSearchCV pick parameters that favour the majority class.
    """
    if model_key not in PARAM_GRIDS:
        raise KeyError(f"No param grid registered for '{model_key}'. Known: {list(PARAM_GRIDS)}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRIDS[model_key],
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
