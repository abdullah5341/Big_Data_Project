"""Step 5: explain the deployed model with SHAP and cross-check against the other models.

Run:  python src/explain.py
Output:
  figures/xai_01_shap_importance_bar.png     global importance (mean |SHAP|)
  figures/xai_02_shap_beeswarm.png           direction + spread of each feature's effect
  figures/xai_03_waterfall_<case>.png        one true positive, one false negative, one false positive
  results/feature_importance_comparison.csv  SHAP vs Logistic Regression vs Random Forest rankings

How to read SHAP values here: XGBoost works in log-odds. Each SHAP value is how much one feature
pushed this person's log-odds of diabetes up (+) or down (-) from the average prediction.
The values add up exactly to the model's output, which is why SHAP is easy to defend.
SHAP explains the MODEL, not biology: it shows association, not causation.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import (DEPLOYED_MODEL, FIGURES_DIR, MODEL_NAMES, MODELS_DIR, PROCESSED_DIR,
                    RANDOM_STATE, RESULTS_DIR, SHAP_SAMPLE_SIZE, TARGET)


def load_test_sample(features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    sample = test.sample(n=min(SHAP_SAMPLE_SIZE, len(test)), random_state=RANDOM_STATE)
    return sample[features].reset_index(drop=True), sample[TARGET].reset_index(drop=True)


def compute_shap(bundle: dict, X: pd.DataFrame) -> shap.Explanation:
    # TreeExplainer computes exact SHAP values for tree models (fast, no approximation).
    explainer = shap.TreeExplainer(bundle["pipeline"].named_steps["model"])
    return explainer(X)


def save_current_figure(name: str) -> None:
    plt.gcf().savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close("all")


def plot_global(explanation: shap.Explanation) -> None:
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title("Global feature importance: mean |SHAP value| (XGBoost)")
    save_current_figure("xai_01_shap_importance_bar")

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("How each feature pushes predicted diabetes risk (XGBoost)")
    save_current_figure("xai_02_shap_beeswarm")


def pick_example_cases(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    """Choose clear, representative cases for local explanations:
    - true positive  : diabetic person the model is most confident about
    - false negative : diabetic person the model scored LOWEST (the most worrying miss)
    - false positive : non-diabetic person the model scored HIGHEST
    """
    predicted = probabilities >= threshold
    positives, negatives = y_true.to_numpy() == 1, y_true.to_numpy() == 0
    candidates = {
        "true_positive": np.where(positives & predicted)[0],
        "false_negative": np.where(positives & ~predicted)[0],
        "false_positive": np.where(negatives & predicted)[0],
    }
    pick_highest = {"true_positive": True, "false_negative": False, "false_positive": True}
    chosen = {}
    for case, rows in candidates.items():
        if len(rows) == 0:
            continue
        order = np.argsort(probabilities[rows])
        chosen[case] = int(rows[order[-1]] if pick_highest[case] else rows[order[0]])
    return chosen


def plot_local(explanation: shap.Explanation, cases: dict[str, int], probabilities: np.ndarray) -> None:
    for case, row in cases.items():
        shap.plots.waterfall(explanation[row], max_display=12, show=False)
        plt.title(f"{case.replace('_', ' ').title()}: predicted risk = {probabilities[row]:.2f}")
        save_current_figure(f"xai_03_waterfall_{case}")


def compare_with_other_models(explanation: shap.Explanation, features: list[str]) -> pd.DataFrame:
    """Do different models and methods agree on what matters? Agreement increases trust."""
    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")["pipeline"].named_steps["model"]
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")["pipeline"].named_steps["model"]

    table = pd.DataFrame({
        "feature": features,
        "xgb_mean_abs_shap": np.abs(explanation.values).mean(axis=0),
        # Inputs were standardised, so coefficient sizes are comparable across features.
        "lr_standardised_coef": lr.coef_[0],
        "rf_impurity_importance": rf.feature_importances_,
    })
    table["rank_xgb_shap"] = table["xgb_mean_abs_shap"].rank(ascending=False).astype(int)
    table["rank_lr_abs_coef"] = table["lr_standardised_coef"].abs().rank(ascending=False).astype(int)
    table["rank_rf_impurity"] = table["rf_impurity_importance"].rank(ascending=False).astype(int)
    return table.sort_values("rank_xgb_shap").round(4)


def main() -> None:
    bundle = joblib.load(MODELS_DIR / f"{DEPLOYED_MODEL}.joblib")
    features = bundle["features"]
    X, y = load_test_sample(features)
    probabilities = bundle["pipeline"].predict_proba(X)[:, 1]
    print(f"Explaining {MODEL_NAMES[DEPLOYED_MODEL]} on {len(X):,} test rows ({y.mean():.1%} positive)")

    explanation = compute_shap(bundle, X)
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_global(explanation)

    cases = pick_example_cases(y, probabilities, bundle["threshold"])
    plot_local(explanation, cases, probabilities)
    for case, row in cases.items():
        print(f"{case}: sample row {row}, predicted risk {probabilities[row]:.3f}")

    comparison = compare_with_other_models(explanation, features)
    RESULTS_DIR.mkdir(exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "feature_importance_comparison.csv", index=False)
    print("\nFeature importance comparison (sorted by XGBoost SHAP):")
    print(comparison.to_string(index=False))

    rank_columns = ["rank_xgb_shap", "rank_lr_abs_coef", "rank_rf_impurity"]
    print("\nSpearman correlation between the three rankings (1 = identical order):")
    print(comparison[rank_columns].corr(method="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
