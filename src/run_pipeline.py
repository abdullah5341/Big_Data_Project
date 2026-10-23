"""Runs the whole offline pipeline in order, so all results can be reproduced with one command.

Run:  python src/run_pipeline.py            (about 15-20 minutes; hyperparameter search dominates)
The two notebooks (01_eda, 03_error_analysis) are run separately in Jupyter.
"""
import time

import data_loader
import evaluate
import explain
import make_architecture_diagram
import preprocess
import scalability
import train

STEPS = [
    ("Download data + quality report", lambda: data_loader.print_quality_report(data_loader.download_dataset())),
    ("Deduplicate, engineer features, split", preprocess.main),
    ("Train and tune models", train.main),
    ("Evaluate once on the test set", evaluate.main),
    ("SHAP explanations", explain.main),
    ("Scalability benchmark", scalability.main),
    ("Architecture diagram", make_architecture_diagram.main),
]


def main() -> None:
    for number, (title, step) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 70}\nStep {number}/{len(STEPS)}: {title}\n{'=' * 70}")
        start = time.perf_counter()
        step()
        print(f"-- done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
