"""
STEP 4 - Final evaluation of every model on the TEST set (done only once).

Run:     python src/evaluate.py
Creates: results/test_metrics.csv       the main results table
         results/test_predictions.csv   every test row with each model's score (used in error analysis)
         figures/eval_01_roc_curves.png, eval_02_pr_curves.png, eval_03_confusion_matrices.png

Nothing here changes a model or a threshold. The thresholds were fixed on the validation
set in train.py, so these test results are a fair estimate of real-world performance.
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from config import (FIGURES_DIR, MODEL_COLORS, MODEL_NAMES, MODELS_DIR, PROCESSED_DIR,
                    RESULTS_DIR, TARGET)

REAL_MODELS = ["logistic_regression", "random_forest", "xgboost"]   # the dummy is left out of the plots
GREY = "#52514e"


def compute_metrics(y_true, probabilities, threshold):
    """All the numbers for one model. A person is flagged when their score is >= the threshold.

    precision   = of the people we flagged, how many really have diabetes
    recall      = of the people with diabetes, how many we flagged
    specificity = of the people without diabetes, how many we correctly did not flag
    f1          = one number that balances precision and recall
    roc_auc     = how well the scores rank people (0.5 = random guessing, 1 = perfect)
    pr_auc      = like ROC-AUC, but focused on the diabetes class (better when classes are imbalanced)
    """
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 4),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def plot_roc_curves(y_test, probabilities):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in REAL_MODELS:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities[name])
        auc = roc_auc_score(y_test, probabilities[name])
        ax.plot(false_positive_rate, true_positive_rate, lw=2, color=MODEL_COLORS[name],
                label=f"{MODEL_NAMES[name]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(FIGURES_DIR / "eval_01_roc_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(y_test, probabilities):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in REAL_MODELS:
        precision, recall, _ = precision_recall_curve(y_test, probabilities[name])
        pr_auc = average_precision_score(y_test, probabilities[name])
        ax.plot(recall, precision, lw=2, color=MODEL_COLORS[name],
                label=f"{MODEL_NAMES[name]} (PR-AUC = {pr_auc:.3f})")
    # A random model's precision equals the share of people with diabetes, so that is the baseline here.
    prevalence = y_test.mean()
    ax.axhline(prevalence, ls="--", lw=1, color=GREY, label=f"Random guess (= prevalence {prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.set_title("Precision-recall curves on the test set")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(FIGURES_DIR / "eval_02_pr_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(y_test, probabilities, thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for ax, name in zip(axes, REAL_MODELS):
        predictions = (probabilities[name] >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, display_labels=["No diabetes", "Diabetes"],
            cmap="Blues", colorbar=False, values_format=",", ax=ax)
        ax.set_title(f"{MODEL_NAMES[name]}\n(threshold = {thresholds[name]:.2f})")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_03_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def main():
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].to_numpy()
    print(f"Test set: {len(test):,} rows, {y_test.sum():,} with diabetes ({y_test.mean():.2%})")

    probabilities = {}
    thresholds = {}
    rows = []
    for name in MODEL_NAMES:   # the dummy first, then the three real models
        saved = joblib.load(MODELS_DIR / f"{name}.joblib")
        probabilities[name] = saved["pipeline"].predict_proba(X_test[saved["features"]])[:, 1]
        thresholds[name] = saved["threshold"]

        row = {"model": MODEL_NAMES[name]}
        row.update(compute_metrics(y_test, probabilities[name], thresholds[name]))
        rows.append(row)

    results = pd.DataFrame(rows).round(4)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    print("\nTest results:")
    print(results.to_string(index=False))

    # Keep every test row with each model's score, for the error-analysis notebook.
    predictions = test.copy()
    for name in REAL_MODELS:
        predictions[f"proba_{name}"] = probabilities[name].round(5)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    plot_roc_curves(y_test, probabilities)
    plot_pr_curves(y_test, probabilities)
    plot_confusion_matrices(y_test, probabilities, thresholds)
    print("\nSaved the tables in results/ and the figures in figures/")


if __name__ == "__main__":
    main()
