"""Step 5b: feature selection — how many survey questions does the model actually need?

Run:  python src/feature_selection.py   (after train.py)
Output:
  results/feature_selection_ranking.csv   permutation importance of every feature (train CV)
  results/feature_selection_curve.csv     validation performance for each subset size
  results/feature_selection_test.csv      final test comparison: all features vs selected subset
  figures/fs_01_auc_vs_num_features.png

Leakage-safe design (each data part has one job):
1. RANK features with permutation importance inside 5-fold CV on the TRAINING set only.
   Permutation importance = how much ROC-AUC drops when one feature's values are shuffled.
2. CHOOSE the subset size on the VALIDATION set: the smallest subset whose validation ROC-AUC
   is within FEATURE_SELECTION_AUC_TOLERANCE of the full model.
3. CONFIRM once on the TEST set. The choice is fixed before the test set is looked at.
XGBoost keeps the hyperparameters tuned in train.py, so only the feature set changes.
"""
import time

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


def rank_features_on_train(pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Average permutation importance over the held-out part of each CV fold (training data only)."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_importances = []
    for fit_idx, held_out_idx in folds.split(X, y):
        model = clone(pipeline).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        result = permutation_importance(model, X.iloc[held_out_idx], y.iloc[held_out_idx],
                                        scoring="roc_auc", n_repeats=PERMUTATION_REPEATS,
                                        random_state=RANDOM_STATE, n_jobs=-1)
        fold_importances.append(result.importances_mean)
    importances = np.array(fold_importances)
    ranking = pd.DataFrame({"feature": X.columns,
                            "auc_drop_mean": importances.mean(axis=0),
                            "auc_drop_std_across_folds": importances.std(axis=0)})
    ranking = ranking.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking.round(5)


def evaluate_subsets(pipeline, ranking, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """Retrain on the top-k features for several k and score each on the validation set."""
    sizes = sorted(set(FEATURE_SUBSET_SIZES) | {len(ranking)})
    rows = []
    for k in sizes:
        features = ranking["feature"].head(k).tolist()
        start = time.perf_counter()
        model = clone(pipeline).fit(X_train[features], y_train)
        fit_seconds = time.perf_counter() - start
        proba = model.predict_proba(X_val[features])[:, 1]
        rows.append({"num_features": k, "val_roc_auc": roc_auc_score(y_val, proba),
                     "val_pr_auc": average_precision_score(y_val, proba),
                     "fit_seconds": round(fit_seconds, 2), "features": features})
        print(f"k={k:>2}: val ROC-AUC {rows[-1]['val_roc_auc']:.4f} | PR-AUC {rows[-1]['val_pr_auc']:.4f}")
    return pd.DataFrame(rows)


def choose_subset_size(curve: pd.DataFrame) -> int:
    full_auc = curve.loc[curve["num_features"].idxmax(), "val_roc_auc"]
    good_enough = curve[curve["val_roc_auc"] >= full_auc - FEATURE_SELECTION_AUC_TOLERANCE]
    return int(good_enough["num_features"].min())


def confirm_on_test(pipeline, feature_sets: dict, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """Final check: threshold from validation, metrics on test, for the full and selected feature sets."""
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].to_numpy()
    rows = []
    for label, features in feature_sets.items():
        model = clone(pipeline).fit(X_train[features], y_train)
        threshold = choose_threshold(y_val, model.predict_proba(X_val[features])[:, 1])
        metrics = compute_metrics(y_test, model.predict_proba(X_test[features])[:, 1], threshold)
        rows.append({"feature_set": label, "num_features": len(features), **metrics})
    return pd.DataFrame(rows).round(4)


def plot_curve(curve: pd.DataFrame, chosen_k: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(curve["num_features"], curve["val_roc_auc"], marker="o", markersize=7, lw=2,
            color=MODEL_COLORS[DEPLOYED_MODEL], label="Validation ROC-AUC")
    full_auc = curve["val_roc_auc"].iloc[-1]
    ax.axhspan(full_auc - FEATURE_SELECTION_AUC_TOLERANCE, full_auc, color="#e9e8e4",
               label=f"Within {FEATURE_SELECTION_AUC_TOLERANCE} of all features")
    chosen_auc = curve.loc[curve["num_features"] == chosen_k, "val_roc_auc"].iloc[0]
    ax.annotate(f"selected: {chosen_k} features", (chosen_k, chosen_auc), xytext=(10, -28),
                textcoords="offset points", color="#52514e", arrowprops={"arrowstyle": "-", "color": "#52514e"})
    ax.set_xticks(curve["num_features"])
    ax.set_xlabel("Number of features (top-k by permutation importance on train)")
    ax.set_ylabel("Validation ROC-AUC (XGBoost)")
    ax.set_title("How many features does the model need?")
    ax.legend(frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)
    fig.savefig(FIGURES_DIR / "fs_01_auc_vs_num_features.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    pipeline = joblib.load(MODELS_DIR / f"{DEPLOYED_MODEL}.joblib")["pipeline"]
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print("1) Ranking features with permutation importance (5-fold CV on train)...")
    ranking = rank_features_on_train(pipeline, X_train, y_train)
    print(ranking.to_string(index=False))

    print("\n2) Scoring top-k subsets on validation...")
    curve = evaluate_subsets(pipeline, ranking, X_train, y_train, X_val, y_val)
    chosen_k = choose_subset_size(curve)
    selected = ranking["feature"].head(chosen_k).tolist()
    print(f"\nSelected {chosen_k} features (smallest set within {FEATURE_SELECTION_AUC_TOLERANCE} "
          f"validation AUC of all features): {selected}")

    print("\n3) Confirming once on test...")
    comparison = confirm_on_test(pipeline, {"all features": list(X_train.columns), "selected": selected},
                                 X_train, y_train, X_val, y_val)
    print(comparison.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    ranking.to_csv(RESULTS_DIR / "feature_selection_ranking.csv", index=False)
    curve.assign(features=curve["features"].map(", ".join)).round(4).to_csv(
        RESULTS_DIR / "feature_selection_curve.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "feature_selection_test.csv", index=False)
    plot_curve(curve, chosen_k)
    print(f"\nSaved results to {RESULTS_DIR} and figure to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
