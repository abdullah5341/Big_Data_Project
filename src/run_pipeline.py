"""
Runs all the steps in order, so every result can be recreated with one command.

Run:  python src/run_pipeline.py      (about 15-20 minutes; tuning Random Forest takes the longest)
The two notebooks (01_eda and 02_error_analysis) are opened and run separately in Jupyter.
"""
import data_loader
import evaluate
import explain
import feature_selection
import make_architecture_diagram
import preprocess
import scalability
import train

STEPS = [
    ("Step 1: download the data and check its quality", data_loader.main),
    ("Step 2: remove duplicates, add features, split", preprocess.main),
    ("Step 3: train and tune the three models", train.main),
    ("Step 4: evaluate once on the test set", evaluate.main),
    ("Step 5: SHAP explanations", explain.main),
    ("Step 6: feature selection", feature_selection.main),
    ("Step 7: scalability test", scalability.main),
    ("Step 8: architecture diagram", make_architecture_diagram.main),
]


def main():
    for title, run_step in STEPS:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        run_step()
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
