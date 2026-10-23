"""
STEP 6 - Feature selection: how many survey questions does the model really need?

Run:     python src/feature_selection.py      (after train.py)
Creates: results/feature_selection_ranking.csv   importance of every feature
         results/feature_selection_curve.csv     validation ROC-AUC when keeping the top 3, top 5, ... features
         results/feature_selection_test.csv      final check: all features vs the selected ones
         figures/fs_01_auc_vs_num_features.png

Three steps, each using a different part of the data, so there is no leakage:
1. RANK the features using the TRAINING data only (permutation importance, 5 folds).
   Permutation importance = shuffle one feature's values and see how much ROC-AUC drops.
   A big drop means the model really needs that feature.
2. CHOOSE how many to keep using the VALIDATION data: the smallest number of features
   whose ROC-AUC is within 0.002 of using all of them.
3. CHECK that choice once on the TEST data.
XGBoost keeps the settings found in train.py; only the list of features changes.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (CV_FOLDS, DEPLOYED_MODEL, FEATURE_SELECTION_AUC_TOLERANCE, FEATURE_SUBSET_SIZES,
                    FIGURES_DIR, MODEL_COLORS, MODELS_DIR, PERMUTATION_REPEATS, PROCESSED_DIR,
                    RANDOM_STATE, RESULTS_DIR, TARGET)
from evaluate import compute_metrics
from train import choose_threshold, load_split


def rank_features(model, X_train, y_train):
    """Step 1: average permutation importance over 5 folds of the TRAINING data."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    importance_per_fold = []
    for fit_rows, check_rows in folds.split(X_train, y_train):
        # Train on 4 folds, then measure importance on the 5th fold, which the model has not seen.
        fold_model = clone(model).fit(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
        result = permutation_importance(fold_model, X_train.iloc[check_rows], y_train.iloc[check_rows],
                                        scoring="roc_auc", n_repeats=PERMUTATION_REPEATS,
                                        random_state=RANDOM_STATE, n_jobs=-1)
        importance_per_fold.append(result.importances_mean)

    ranking = pd.DataFrame({
        "feature": X_train.columns,
        "auc_drop_mean": np.mean(importance_per_fold, axis=0),
        "auc_drop_std_across_folds": np.std(importance_per_fold, axis=0),
    })
    ranking = ranking.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking.round(5)


def score_top_k(model, ranking, X_train, y_train, X_val, y_val):
    """Step 2: retrain using only the top-k features, for several k, and score each on VALIDATION."""
    sizes = FEATURE_SUBSET_SIZES + [len(ranking)]   # the last size means "all features"
    rows = []
    for k in sizes:
        top_k = list(ranking["feature"].head(k))
        k_model = clone(model).fit(X_train[top_k], y_train)
        val_scores = k_model.predict_proba(X_val[top_k])[:, 1]
        rows.append({
            "num_features": k,
            "val_roc_auc": roc_auc_score(y_val, val_scores),
            "val_pr_auc": average_precision_score(y_val, val_scores),
            "features": ", ".join(top_k),
        })
        print(f"top {k:>2} features: validation ROC-AUC {rows[-1]['val_roc_auc']:.4f}")
    return pd.DataFrame(rows)


def choose_number_of_features(curve):
    """The smallest k whose validation ROC-AUC is within the tolerance of using all features."""
    all_features_auc = curve["val_roc_auc"].iloc[-1]
    good_enough = curve[curve["val_roc_auc"] >= all_features_auc - FEATURE_SELECTION_AUC_TOLERANCE]
    return int(good_enough["num_features"].min())


def check_on_test(model, feature_sets, X_train, y_train, X_val, y_val):
    """Step 3: train, choose the threshold on validation, then score once on TEST."""
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].to_numpy()

    rows = []
    for label, features in feature_sets.items():
        set_model = clone(model).fit(X_train[features], y_train)
        threshold = choose_threshold(y_val, set_model.predict_proba(X_val[features])[:, 1])
        row = {"feature_set": label, "num_features": len(features)}
        row.update(compute_metrics(y_test, set_model.predict_proba(X_test[features])[:, 1], threshold))
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def plot_curve(curve, chosen_k):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(curve["num_features"], curve["val_roc_auc"], marker="o", markersize=7, lw=2,
            color=MODEL_COLORS[DEPLOYED_MODEL], label="Validation ROC-AUC")

    # Grey band = "as good as all features" (within the tolerance).
    all_features_auc = curve["val_roc_auc"].iloc[-1]
    ax.axhspan(all_features_auc - FEATURE_SELECTION_AUC_TOLERANCE, all_features_auc, color="#e9e8e4",
               label=f"Within {FEATURE_SELECTION_AUC_TOLERANCE} of all features")

    chosen_auc = curve.loc[curve["num_features"] == chosen_k, "val_roc_auc"].iloc[0]
    ax.annotate(f"selected: {chosen_k} features", (chosen_k, chosen_auc), xytext=(10, -28),
                textcoords="offset points", color="#52514e", arrowprops={"arrowstyle": "-", "color": "#52514e"})
    ax.set_xticks(curve["num_features"])
    ax.set_xlabel("Number of features (top-k by permutation importance on train)")
    ax.set_ylabel("Validation ROC-AUC (XGBoost)")
    ax.set_title("How many features does the model need?")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    fig.savefig(FIGURES_DIR / "fs_01_auc_vs_num_features.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    # The tuned XGBoost pipeline; clone() copies its settings without the training.
    model = joblib.load(MODELS_DIR / f"{DEPLOYED_MODEL}.joblib")["pipeline"]
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print("Step 1: ranking the features (permutation importance, 5 folds of train)...")
    ranking = rank_features(model, X_train, y_train)
    print(ranking.to_string(index=False))

    print("\nStep 2: keeping only the top-k features, scored on validation...")
    curve = score_top_k(model, ranking, X_train, y_train, X_val, y_val)
    chosen_k = choose_number_of_features(curve)
    selected = list(ranking["feature"].head(chosen_k))
    print(f"\nSelected {chosen_k} features: {selected}")

    print("\nStep 3: checking once on test...")
    feature_sets = {"all features": list(X_train.columns), "selected": selected}
    comparison = check_on_test(model, feature_sets, X_train, y_train, X_val, y_val)
    print(comparison.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    ranking.to_csv(RESULTS_DIR / "feature_selection_ranking.csv", index=False)
    curve.round(4).to_csv(RESULTS_DIR / "feature_selection_curve.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "feature_selection_test.csv", index=False)
    plot_curve(curve, chosen_k)
    print("\nSaved the results in results/ and the figure in figures/")


if __name__ == "__main__":
    main()
