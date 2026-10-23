"""
STEP 3 - Train and tune the three models, then save them.

Run:     python src/train.py      (about 15 minutes; Random Forest takes the longest)
Creates: models/<model name>.joblib and results/training_summary.csv

Which data is used for what:
- TRAIN (70%):      training the models and choosing their settings (5-fold cross-validation)
- VALIDATION (15%): choosing each model's decision threshold
- TEST (15%):       NOT used here at all. It is used once, in evaluate.py.

Class imbalance (only about 15% have diabetes) is handled with class weights:
mistakes on people with diabetes count more during training. No fake rows are created.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (CV_FOLDS, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR,
                    SEARCH_ITERATIONS, TARGET, TUNING_METRIC)


def load_split(name):
    """Load train, val or test, and separate the inputs (X) from the answer we predict (y)."""
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_models(class_ratio):
    """Create the three models, each with the range of settings the search may try.

    Each model sits inside a Pipeline. "model__C" means "setting C of the pipeline step
    called model". randint / uniform / loguniform describe a range that random values
    are picked from (loguniform spreads them evenly over 0.001, 0.01, 0.1, ...).
    """
    logistic_regression = Pipeline([
        # The scaler is inside the pipeline, so during cross-validation it only learns
        # from the training folds (no leakage).
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    logistic_regression_settings = {
        "model__C": loguniform(1e-3, 1e1),   # smaller C = simpler model
    }

    random_forest = Pipeline([
        ("model", RandomForestClassifier(class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    random_forest_settings = {
        "model__n_estimators": randint(150, 351),    # number of trees
        "model__max_depth": randint(8, 21),          # how deep a tree may grow
        "model__min_samples_leaf": randint(5, 51),   # fewest people allowed in a leaf
        "model__max_features": ["sqrt", 0.5],        # how many features each split may look at
    }

    xgboost = Pipeline([
        ("model", XGBClassifier(
            # XGBoost's version of class weights: a person with diabetes counts
            # class_ratio (about 5.5) times as much as a person without.
            scale_pos_weight=class_ratio,
            tree_method="hist", eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    xgboost_settings = {
        "model__n_estimators": randint(200, 601),      # number of trees, built one after another
        "model__max_depth": randint(3, 8),
        "model__learning_rate": loguniform(0.02, 0.2),  # how big a correction each new tree makes
        "model__subsample": uniform(0.7, 0.3),          # share of people each tree sees (0.7 to 1.0)
        "model__colsample_bytree": uniform(0.6, 0.4),   # share of features each tree sees (0.6 to 1.0)
        "model__min_child_weight": randint(1, 11),
    }

    return {
        "logistic_regression": (logistic_regression, logistic_regression_settings),
        "random_forest": (random_forest, random_forest_settings),
        "xgboost": (xgboost, xgboost_settings),
    }


def tune(name, pipeline, settings, X_train, y_train):
    """Try random combinations of settings, scoring each with 5-fold cross-validation on TRAIN."""
    search = RandomizedSearchCV(
        pipeline,
        settings,
        n_iter=SEARCH_ITERATIONS[name],
        scoring=TUNING_METRIC,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,   # at the end, train the best combination once more on ALL training data
    )
    search.fit(X_train, y_train)
    return search


def choose_threshold(y_val, val_probabilities):
    """Find the score cut-off with the best F1 on the VALIDATION set.

    Why not just use 0.5? Class weights push the model's scores upwards, so 0.5 would
    flag too many people. We choose the cut-off on data the model was not trained on.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, val_probabilities)
    # precision and recall have one extra value at the end that has no threshold, so drop it.
    precision = precision[:-1]
    recall = recall[:-1]
    # F1 = 2PR / (P + R). np.clip stops a division by zero when P and R are both 0.
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_index = np.argmax(f1)
    return float(thresholds[best_index])


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    # How many people without diabetes there are for each person with diabetes.
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train: {len(X_train):,} rows   Validation: {len(X_val):,} rows   "
          f"Class ratio: {class_ratio:.2f}")

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # A reference model that always predicts the most common answer ("no diabetes").
    # It shows why accuracy alone is misleading.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    joblib.dump({"pipeline": dummy, "threshold": 0.5, "features": list(X_train.columns)},
                MODELS_DIR / "dummy.joblib")

    summary_rows = []
    for name, (pipeline, settings) in build_models(class_ratio).items():
        print(f"\n=== {name} ===")
        search = tune(name, pipeline, settings, X_train, y_train)
        best_model = search.best_estimator_

        val_probabilities = best_model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_probabilities)

        # Remove the "model__" prefix and round the numbers so the settings are easy to read.
        best_params = {}
        for setting, value in search.best_params_.items():
            if isinstance(value, float):
                value = round(float(value), 4)
            best_params[setting.replace("model__", "")] = value

        row = {
            "model": name,
            "cv_roc_auc_mean": round(float(search.best_score_), 4),
            "cv_roc_auc_std": round(float(search.cv_results_["std_test_score"][search.best_index_]), 4),
            "val_roc_auc": round(roc_auc_score(y_val, val_probabilities), 4),
            "val_pr_auc": round(average_precision_score(y_val, val_probabilities), 4),
            "threshold": round(threshold, 4),
            "fit_seconds": round(search.refit_time_, 2),   # time to train the final model once
            "best_params": best_params,
        }
        summary_rows.append(row)
        print(row)

        # Save the model together with its threshold, so the other scripts and the app
        # use exactly the same model and the same cut-off.
        joblib.dump({"pipeline": best_model, "threshold": threshold,
                     "features": list(X_train.columns), "best_params": best_params},
                    MODELS_DIR / f"{name}.joblib")

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "training_summary.csv", index=False)
    print("\nSaved the models in models/ and the summary in results/training_summary.csv")


if __name__ == "__main__":
    main()
