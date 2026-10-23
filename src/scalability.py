"""Step 6: scalability benchmark — evidence for the Big Data platform discussion (CLO3).

Run:  python src/scalability.py
Output: results/scalability_benchmark.csv and figures/scale_01_training_time.png

Question answered: does a single-machine Python stack cope with this dataset, and how do
training time, prediction throughput and memory grow as the data grows?
We train each model (with its tuned settings) on 25%, 50% and 100% of the training set,
plus 2x and 4x versions made by resampling rows with replacement. The synthetic sizes are
used ONLY to measure speed and memory, never to report accuracy.
"""
import time
import tracemalloc

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone

from config import (FIGURES_DIR, MODEL_COLORS, MODEL_NAMES, MODELS_DIR, PROCESSED_DIR,
                    RANDOM_STATE, RESULTS_DIR, TARGET)

SIZE_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]
BENCHMARKED_MODELS = ["logistic_regression", "random_forest", "xgboost"]


def make_training_set(train: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Subsample (<1x) or resample with replacement (>1x) to the requested size."""
    n_rows = int(len(train) * multiplier)
    return train.sample(n=n_rows, replace=multiplier > 1, random_state=RANDOM_STATE)


def benchmark(pipeline, X: pd.DataFrame, y: pd.Series, X_score: pd.DataFrame) -> dict:
    model = clone(pipeline)  # same tuned hyperparameters, untrained
    tracemalloc.start()
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - start
    # tracemalloc sees Python-level allocations only (not native threads), so treat it as a lower bound.
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    start = time.perf_counter()
    model.predict_proba(X_score)
    predict_seconds = time.perf_counter() - start
    return {"fit_seconds": round(fit_seconds, 2),
            "peak_python_memory_mb": round(peak_bytes / 1e6, 1),
            "predictions_per_second": int(len(X_score) / predict_seconds)}


def plot_training_time(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in BENCHMARKED_MODELS:
        part = results[results["model"] == MODEL_NAMES[name]]
        ax.plot(part["rows"], part["fit_seconds"], marker="o", markersize=7, lw=2,
                color=MODEL_COLORS[name], label=MODEL_NAMES[name])
    ax.axvline(results.loc[results["multiplier"] == 1.0, "rows"].iloc[0], ls="--", lw=1, color="#52514e")
    ax.text(results.loc[results["multiplier"] == 1.0, "rows"].iloc[0], ax.get_ylim()[1] * 0.95,
            " actual training set", color="#52514e", va="top")
    ax.set_xlabel("Training rows")
    ax.set_ylabel("Training time (seconds)")
    ax.set_title("Training time vs data size (single laptop)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)
    fig.savefig(FIGURES_DIR / "scale_01_training_time.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    X_score = pd.read_csv(PROCESSED_DIR / "test.csv").drop(columns=[TARGET])
    print(f"Training set: {len(train):,} rows, "
          f"{train.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory as a pandas DataFrame")

    rows = []
    for name in BENCHMARKED_MODELS:
        pipeline = joblib.load(MODELS_DIR / f"{name}.joblib")["pipeline"]
        for multiplier in SIZE_MULTIPLIERS:
            data = make_training_set(train, multiplier)
            result = benchmark(pipeline, data.drop(columns=[TARGET]), data[TARGET], X_score)
            row = {"model": MODEL_NAMES[name], "multiplier": multiplier, "rows": len(data), **result}
            rows.append(row)
            print(row)

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "scalability_benchmark.csv", index=False)
    plot_training_time(results)
    print(f"\nSaved {RESULTS_DIR / 'scalability_benchmark.csv'} and figure")


if __name__ == "__main__":
    main()
