"""Step 4: final, one-time evaluation of every saved model on the untouched TEST set.

Run:  python src/evaluate.py
Output:
  results/test_metrics.csv          main comparison table
  results/test_bootstrap_ci.csv     95% confidence intervals (ROC-AUC, PR-AUC, recall)
  results/test_predictions.csv      per-row probabilities (input for error analysis)
  figures/eval_01_roc_curves.png, eval_02_pr_curves.png, eval_03_confusion_matrices.png

Nothing in this script changes a model or a threshold. Thresholds were fixed on the
validation set in train.py, so the test set gives an unbiased estimate.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from config import (BOOTSTRAP_ROUNDS, FIGURES_DIR, MODEL_COLORS, MODEL_NAMES, MODELS_DIR,
                    PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR, TARGET)

TEXT_MUTED = "#52514e"
TRAINED_MODELS = [m for m in MODEL_NAMES if m != "dummy"]


def load_models() -> dict:
    return {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in MODEL_NAMES}


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
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


def bootstrap_intervals(y_true: np.ndarray, probabilities: dict, thresholds: dict) -> pd.DataFrame:
    """95% CIs by resampling the test set with replacement.

    All models are scored on the SAME resample each round, so we can also give a CI for the
    AUC difference between XGBoost and Logistic Regression (is the gain real or just noise?).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    scores = {name: {"roc_auc": [], "pr_auc": [], "recall": []} for name in probabilities}
    auc_difference = []

    for _ in range(BOOTSTRAP_ROUNDS):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        round_auc = {}
        for name, proba in probabilities.items():
            p_sample = proba[idx]
            round_auc[name] = roc_auc_score(y_sample, p_sample)
            scores[name]["roc_auc"].append(round_auc[name])
            scores[name]["pr_auc"].append(average_precision_score(y_sample, p_sample))
            scores[name]["recall"].append(recall_score(y_sample, (p_sample >= thresholds[name]).astype(int)))
        auc_difference.append(round_auc["xgboost"] - round_auc["logistic_regression"])

    rows = []
    for name, metric_scores in scores.items():
        for metric, values in metric_scores.items():
            low, high = np.percentile(values, [2.5, 97.5])
            rows.append({"model": MODEL_NAMES[name], "metric": metric,
                         "ci_low": round(low, 4), "ci_high": round(high, 4)})
    low, high = np.percentile(auc_difference, [2.5, 97.5])
    rows.append({"model": "XGBoost minus Logistic Regression", "metric": "roc_auc_difference",
                 "ci_low": round(low, 4), "ci_high": round(high, 4)})
    return pd.DataFrame(rows)


def plot_roc_curves(y_true: np.ndarray, probabilities: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS[name],
                label=f"{MODEL_NAMES[name]} (AUC = {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=TEXT_MUTED, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(FIGURES_DIR / "eval_01_roc_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(y_true: np.ndarray, probabilities: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, proba in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, lw=2, color=MODEL_COLORS[name],
                label=f"{MODEL_NAMES[name]} (PR-AUC = {average_precision_score(y_true, proba):.3f})")
    # A random model's precision equals the positive share, so that is the baseline here (not 0.5).
    prevalence = y_true.mean()
    ax.axhline(prevalence, ls="--", lw=1, color=TEXT_MUTED, label=f"Random guess (= prevalence {prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.set_title("Precision-recall curves on the test set")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(FIGURES_DIR / "eval_02_pr_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(y_true: np.ndarray, probabilities: dict, thresholds: dict) -> None:
    fig, axes = plt.subplots(1, len(probabilities), figsize=(4.2 * len(probabilities), 3.8))
    for ax, (name, proba) in zip(axes, probabilities.items()):
        predictions = (proba >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_true, predictions, display_labels=["No diabetes", "Diabetes"],
            cmap="Blues", colorbar=False, values_format=",", ax=ax)
        ax.set_title(f"{MODEL_NAMES[name]}\n(threshold = {thresholds[name]:.2f})")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_03_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].to_numpy()
    print(f"Test set: {len(test):,} rows, {y_test.sum():,} positives ({y_test.mean():.2%})")

    models = load_models()
    probabilities = {name: bundle["pipeline"].predict_proba(X_test[bundle["features"]])[:, 1]
                     for name, bundle in models.items()}
    thresholds = {name: bundle["threshold"] for name, bundle in models.items()}

    metrics = pd.DataFrame([{"model": MODEL_NAMES[name],
                             **compute_metrics(y_test, probabilities[name], thresholds[name])}
                            for name in MODEL_NAMES])
    metrics = metrics.round(4)
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    print("\nTest-set metrics:")
    print(metrics.to_string(index=False))

    trained_probabilities = {name: probabilities[name] for name in TRAINED_MODELS}
    print(f"\nBootstrapping {BOOTSTRAP_ROUNDS} resamples for 95% confidence intervals...")
    intervals = bootstrap_intervals(y_test, trained_probabilities, thresholds)
    intervals.to_csv(RESULTS_DIR / "test_bootstrap_ci.csv", index=False)
    print(intervals.to_string(index=False))

    # Saved for the error-analysis notebook: original features + true label + each model's probability.
    predictions = test.copy()
    for name in TRAINED_MODELS:
        predictions[f"proba_{name}"] = probabilities[name].round(5)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    plot_roc_curves(y_test, trained_probabilities)
    plot_pr_curves(y_test, trained_probabilities)
    plot_confusion_matrices(y_test, trained_probabilities, thresholds)
    print(f"\nSaved tables to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
