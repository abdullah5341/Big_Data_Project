"""
STEP 7 - Scalability test: can one laptop handle this data, and how does training time grow?

Run:     python src/scalability.py
Creates: results/scalability_benchmark.csv and figures/scale_01_training_time.png

Each model (with its tuned settings) is trained on 25%, 50% and 100% of the training data,
and we time it. We also measure how many predictions per second it makes.
This is the evidence for the Big Data platform discussion in the report.
"""
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone

from config import (FIGURES_DIR, MODEL_COLORS, MODEL_NAMES, MODELS_DIR, PROCESSED_DIR,
                    RANDOM_STATE, RESULTS_DIR, TARGET)

DATA_FRACTIONS = [0.25, 0.5, 1.0]
MODELS = ["logistic_regression", "random_forest", "xgboost"]


def plot_training_time(results):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in MODELS:
        model_rows = results[results["model"] == MODEL_NAMES[name]]
        ax.plot(model_rows["rows"], model_rows["fit_seconds"], marker="o", markersize=7, lw=2,
                color=MODEL_COLORS[name], label=MODEL_NAMES[name])
    ax.set_xlabel("Training rows")
    ax.set_ylabel("Training time (seconds)")
    ax.set_title("Training time vs data size (single laptop)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value / 1000:,.0f}k"))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    fig.savefig(FIGURES_DIR / "scale_01_training_time.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "test.csv").drop(columns=[TARGET])
    memory_mb = train.memory_usage(deep=True).sum() / 1e6
    print(f"Training data: {len(train):,} rows, {memory_mb:.1f} MB in memory")

    rows = []
    for name in MODELS:
        tuned_pipeline = joblib.load(MODELS_DIR / f"{name}.joblib")["pipeline"]
        for fraction in DATA_FRACTIONS:
            part = train.sample(frac=fraction, random_state=RANDOM_STATE)
            model = clone(tuned_pipeline)   # same settings, but not trained yet

            start = time.perf_counter()
            model.fit(part.drop(columns=[TARGET]), part[TARGET])
            fit_seconds = time.perf_counter() - start

            start = time.perf_counter()
            model.predict_proba(X_test)
            predict_seconds = time.perf_counter() - start

            rows.append({
                "model": MODEL_NAMES[name],
                "fraction": fraction,
                "rows": len(part),
                "fit_seconds": round(fit_seconds, 2),
                "predictions_per_second": int(len(X_test) / predict_seconds),
            })
            print(rows[-1])

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "scalability_benchmark.csv", index=False)
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_training_time(results)
    print("\nSaved results/scalability_benchmark.csv and the figure")


if __name__ == "__main__":
    main()
