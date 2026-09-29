import json

import matplotlib

matplotlib.use("Agg")

from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, RANDOM_STATE, ROOT, TEST_SIZE
from .data_preparation import clean, load_raw, make_xy
from .evaluation import cross_validate_on_train, evaluate_on, plot_confusion_matrix, plot_roc
from .preprocessing import build_logistic_class_weighted, build_random_forest_class_weighted

MILESTONE03_METRICS_PATH = ROOT / "reports" / "milestone03_metrics.json"


def main() -> None:
    raw = load_raw()
    df, _ = clean(raw)
    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "logistic_regression": build_logistic_class_weighted(X_train),
        "random_forest": build_random_forest_class_weighted(X_train),
    }

    results = {"random_state": RANDOM_STATE, "models": {}}

    for name, model in models.items():
        print(f"\n=== {name} ===")
        cv = cross_validate_on_train(model, X_train, y_train)
        print("CV (train only):\n", cv.round(4))

        model.fit(X_train, y_train)
        train_metrics, _, _ = evaluate_on(model, X_train, y_train)
        test_metrics, y_pred, y_proba = evaluate_on(model, X_test, y_test)
        print("Train metrics:", {k: round(v, 4) for k, v in train_metrics.items()})
        print("Test metrics :", {k: round(v, 4) for k, v in test_metrics.items()})

        plot_confusion_matrix(
            y_test, y_pred, f"{name}: confusion matrix (test)",
            FIGURES_DIR / f"milestone03_{name}_confusion_matrix.png",
        )
        plot_roc(
            y_test, y_proba, f"{name}: ROC curve (test)",
            FIGURES_DIR / f"milestone03_{name}_roc_curve.png",
        )

        results["models"][name] = {
            "cv_train": cv.round(4).to_dict(),
            "train": {k: round(float(v), 4) for k, v in train_metrics.items()},
            "test": {k: round(float(v), 4) for k, v in test_metrics.items()},
            "train_test_gap_f1": round(float(train_metrics["f1"] - test_metrics["f1"]), 4),
        }

    MILESTONE03_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MILESTONE03_METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {MILESTONE03_METRICS_PATH}")


if __name__ == "__main__":
    main()