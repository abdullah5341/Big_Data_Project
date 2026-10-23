"""Step 3: train, tune and save the three models (plus a dummy reference).

Run:  python src/train.py
Output: models/<name>.joblib and results/training_summary.csv

Data use (leakage rules):
- TRAIN (70%): fitting + hyperparameter search with stratified 5-fold cross-validation.
- VALIDATION (15%): choosing each model's decision threshold and comparing models.
- TEST (15%): NOT touched here. It is used once, in evaluate.py.

Class imbalance is handled with class weights (no synthetic rows), so there is
nothing that could leak synthetic data into validation or test.
"""
import time

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


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    return df.drop(columns=[TARGET]), df[TARGET]


def build_candidates(negative_to_positive_ratio: float) -> dict:
    """Each model: an untrained pipeline plus the hyperparameter ranges to search."""
    return {
        "logistic_regression": (
            # Scaling lives INSIDE the pipeline, so in each CV fold it is fitted on that fold's training part only.
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
            ]),
            {"model__C": loguniform(1e-3, 1e1)},  # smaller C = stronger regularisation
        ),
        "random_forest": (
            Pipeline([
                ("model", RandomForestClassifier(class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=RANDOM_STATE)),
            ]),
            {
                "model__n_estimators": randint(150, 351),
                "model__max_depth": randint(8, 21),
                "model__min_samples_leaf": randint(5, 51),
                "model__max_features": ["sqrt", 0.5],
            },
        ),
        "xgboost": (
            Pipeline([
                ("model", XGBClassifier(
                    # Up-weights the minority class by the class ratio (the XGBoost equivalent of class_weight).
                    scale_pos_weight=negative_to_positive_ratio,
                    tree_method="hist", eval_metric="logloss",
                    n_jobs=-1, random_state=RANDOM_STATE)),
            ]),
            {
                "model__n_estimators": randint(200, 601),
                "model__max_depth": randint(3, 8),
                "model__learning_rate": loguniform(0.02, 0.2),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__min_child_weight": randint(1, 11),
            },
        ),
    }


def tune(name: str, pipeline: Pipeline, search_space: dict, X: pd.DataFrame, y: pd.Series) -> RandomizedSearchCV:
    """Randomised search with stratified CV on the training data only."""
    search = RandomizedSearchCV(
        pipeline, search_space, n_iter=SEARCH_ITERATIONS[name], scoring=TUNING_METRIC,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=1, refit=True)
    search.fit(X, y)
    return search


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Pick the probability cut-off that maximises F1 on the VALIDATION set.

    0.5 is only a default. With class weights the probabilities are shifted,
    so we choose the cut-off from data the model was not trained on.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    # The last precision/recall pair has no threshold, so drop it before matching them up.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def time_final_fit(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Refit the chosen configuration once on the full training set and time it (used for the scalability discussion)."""
    start = time.perf_counter()
    pipeline.fit(X, y)
    return time.perf_counter() - start


def main() -> None:
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train: {len(X_train):,} rows | Validation: {len(X_val):,} rows | "
          f"negative:positive ratio = {ratio:.2f}")

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_rows = []

    # Reference model: always predicts the majority class. Shows why accuracy alone is misleading.
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    joblib.dump({"pipeline": dummy, "threshold": 0.5, "features": list(X_train.columns)},
                MODELS_DIR / "dummy.joblib")

    for name, (pipeline, search_space) in build_candidates(ratio).items():
        print(f"\n=== {name} ===")
        search_start = time.perf_counter()
        search = tune(name, pipeline, search_space, X_train, y_train)
        search_seconds = time.perf_counter() - search_start

        best_pipeline = search.best_estimator_
        fit_seconds = time_final_fit(best_pipeline, X_train, y_train)

        val_probabilities = best_pipeline.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_probabilities)
        # Convert numpy numbers to plain Python so the CSV and printout stay readable.
        best_params = {k.replace("model__", ""): round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                       for k, v in search.best_params_.items()}

        row = {
            "model": name,
            "cv_roc_auc_mean": round(float(search.best_score_), 4),
            "cv_roc_auc_std": round(float(search.cv_results_["std_test_score"][search.best_index_]), 4),
            "val_roc_auc": round(roc_auc_score(y_val, val_probabilities), 4),
            "val_pr_auc": round(average_precision_score(y_val, val_probabilities), 4),
            "threshold": round(threshold, 4),
            "fit_seconds": round(fit_seconds, 2),
            "search_seconds": round(search_seconds, 1),
            "best_params": best_params,
        }
        summary_rows.append(row)
        print({k: v for k, v in row.items() if k != "best_params"})
        print("best params:", best_params)

        joblib.dump({"pipeline": best_pipeline, "threshold": threshold,
                     "features": list(X_train.columns), "best_params": best_params},
                    MODELS_DIR / f"{name}.joblib")

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "training_summary.csv", index=False)
    print(f"\nSaved models to {MODELS_DIR} and summary to {RESULTS_DIR / 'training_summary.csv'}")


if __name__ == "__main__":
    main()
