import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split

from .config import FIGURES_DIR, RANDOM_STATE, ROOT, TEST_SIZE
from .data_preparation import clean, load_raw, make_xy
from .preprocessing import build_logistic_class_weighted, build_random_forest_class_weighted
from .tuning import build_grid_search

MILESTONE04_METRICS_PATH = ROOT / "reports" / "milestone04_metrics.json"

CV_SCORING_FOR_SEARCH = "f1"   # what GridSearchCV optimises for (imbalanced target, see tuning.py)


def main() -> None:
    raw = load_raw()
    df, _ = clean(raw)
    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "logistic_regression": build_logistic_class_weighted(X_train),
        "random_forest": build_random_forest_class_weighted(X_train),
    }

    results = {"random_state": RANDOM_STATE, "cv_scoring_for_search": CV_SCORING_FOR_SEARCH, "models": {}}

    # 1. Baseline cross_val_score, BEFORE tuning (exact requested usage)
    print("=== Baseline cross_val_score (untuned, class-weighted models) ===")
    baseline_cv = {}
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
        baseline_cv[name] = {"roc_auc_mean": float(scores.mean()), "roc_auc_std": float(scores.std())}
        print(f"{name}: ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    # 2. GridSearchCV per model type
    print("\n=== GridSearchCV (per model type) ===")
    tuned = {}
    for name, pipeline in candidates.items():
        search = build_grid_search(pipeline, model_key=name, scoring=CV_SCORING_FOR_SEARCH)
        search.fit(X_train, y_train)
        tuned[name] = search
        print(f"{name}: best_params={search.best_params_}, best_{CV_SCORING_FOR_SEARCH}={search.best_score_:.4f}")

    # 3. Select the tuned model with the higher GridSearchCV score
    best_name = max(tuned, key=lambda k: tuned[k].best_score_)
    best_search = tuned[best_name]
    best_model = best_search.best_estimator_
    print(f"\nSelected model: {best_name} (best CV {CV_SCORING_FOR_SEARCH}={best_search.best_score_:.4f})")

    # 4. Final evaluation on the held-out test set (used once)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    report_text = classification_report(y_test, y_pred, target_names=["Stayed", "Churned"], zero_division=0)
    report_dict = classification_report(
        y_test, y_pred, target_names=["Stayed", "Churned"], zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    print("\n=== Final test-set evaluation:", best_name, "===")
    print(report_text)
    print("Confusion matrix:\n", cm)
    print("ROC-AUC:", round(auc, 4))

    # ROC curve, explicit plotting code
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Milestone 04: ROC curve (tuned model, test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "milestone04_roc_curve.png", dpi=150)
    plt.close()

    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Milestone 04: confusion matrix ({best_name}, test)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "milestone04_confusion_matrix.png", dpi=150)
    plt.close(fig)

    results["baseline_cross_val_score_roc_auc"] = baseline_cv
    for name, search in tuned.items():
        results["models"][name] = {
            "best_params": search.best_params_,
            f"best_cv_{CV_SCORING_FOR_SEARCH}": round(float(search.best_score_), 4),
        }
    results["selected_model"] = best_name
    results["test_classification_report"] = report_dict
    results["test_confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    results["test_roc_auc"] = round(float(auc), 4)

    MILESTONE04_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MILESTONE04_METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {MILESTONE04_METRICS_PATH}")


if __name__ == "__main__":
    main()
