"""
STEP 5 - Explain the XGBoost model with SHAP.

Run:     python src/explain.py
Creates: figures/xai_01_shap_importance_bar.png   which features matter most overall
         figures/xai_02_shap_beeswarm.png         whether high or low values push the risk up or down
         figures/xai_03_waterfall_<case>.png      why the model scored three individual people as it did
         results/shap_importance.csv              average importance of each feature

How to read a SHAP value: it is how much one answer pushed this person's score up (+) or
down (-) compared with the average person. All the pushes add up to the model's output.
SHAP explains what the MODEL has learned. It shows associations, not causes.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import (DEPLOYED_MODEL, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,
                    RESULTS_DIR, SHAP_SAMPLE_SIZE, TARGET)


def save_figure(name):
    plt.gcf().savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close("all")


def pick_example_people(y_true, probabilities, threshold):
    """Pick three clear examples for the individual (waterfall) plots."""
    y_true = y_true.to_numpy()
    flagged = probabilities >= threshold

    # True positive = has diabetes and was flagged. We take the one with the HIGHEST score.
    true_positives = np.where((y_true == 1) & flagged)[0]
    # False negative = has diabetes but was missed. We take the LOWEST score (the worst miss).
    false_negatives = np.where((y_true == 1) & ~flagged)[0]
    # False positive = no diabetes but was flagged. We take the one with the HIGHEST score.
    false_positives = np.where((y_true == 0) & flagged)[0]

    return {
        "true_positive": true_positives[np.argmax(probabilities[true_positives])],
        "false_negative": false_negatives[np.argmin(probabilities[false_negatives])],
        "false_positive": false_positives[np.argmax(probabilities[false_positives])],
    }


def main():
    saved = joblib.load(MODELS_DIR / f"{DEPLOYED_MODEL}.joblib")
    features = saved["features"]

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    sample = test.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)
    X = sample[features].reset_index(drop=True)
    y = sample[TARGET].reset_index(drop=True)
    probabilities = saved["pipeline"].predict_proba(X)[:, 1]
    print(f"Explaining XGBoost on {len(X):,} test rows")

    # TreeExplainer calculates exact SHAP values for tree models, and it is fast.
    xgb_model = saved["pipeline"].named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X)

    FIGURES_DIR.mkdir(exist_ok=True)

    # 1) Overall importance: the average size of each feature's push.
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Global feature importance: mean |SHAP value| (XGBoost)")
    save_figure("xai_01_shap_importance_bar")

    # 2) Beeswarm: one dot per person. Red = high value of the feature, blue = low value.
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("How each feature pushes predicted diabetes risk (XGBoost)")
    save_figure("xai_02_shap_beeswarm")

    # 3) Waterfall plots: why the model gave three individual people their score.
    examples = pick_example_people(y, probabilities, saved["threshold"])
    for case, row in examples.items():
        shap.plots.waterfall(shap_values[row], max_display=12, show=False)
        plt.title(f"{case.replace('_', ' ').title()}: predicted risk = {probabilities[row]:.2f}")
        save_figure(f"xai_03_waterfall_{case}")
        print(f"{case}: sample row {row}, predicted risk {probabilities[row]:.3f}")

    # Average importance of each feature, saved for the report.
    importance = pd.Series(np.abs(shap_values.values).mean(axis=0), index=features)
    importance = importance.sort_values(ascending=False).round(4)
    RESULTS_DIR.mkdir(exist_ok=True)
    importance.to_csv(RESULTS_DIR / "shap_importance.csv", header=["mean_abs_shap"], index_label="feature")
    print("\nMean |SHAP| per feature (most important first):")
    print(importance.to_string())


if __name__ == "__main__":
    main()
