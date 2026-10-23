"""
STEP 1 - Download the dataset and check its quality.

Run:     python src/data_loader.py
Creates: data/raw/cdc_diabetes_health_indicators.csv
Prints:  number of rows, missing values, duplicates, class balance and value ranges
"""
import pandas as pd
from ucimlrepo import fetch_ucirepo

from config import ID_COLUMN, RAW_DIR, RAW_FILE, TARGET, UCI_DATASET_ID


def download_dataset():
    """Download the data from the UCI repository and save it as one CSV file."""
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    # UCI gives the features and the target separately, so we join them into one table.
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)
    return df


def load_dataset():
    """Read the saved CSV (download it first if it is not there). Other scripts use this."""
    if RAW_FILE.exists():
        return pd.read_csv(RAW_FILE)
    return download_dataset()


def print_quality_report(df):
    """Print the basic facts about the data that we need for the report."""
    print(f"Rows: {len(df):,}   Features: {df.shape[1] - 1}   Target: {TARGET}")
    print(f"Missing values: {df.isna().sum().sum()}")

    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates:,} ({duplicates / len(df):.1%})")

    print("\nTarget distribution:")
    counts = df[TARGET].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label}: {count:,} ({count / len(df):.1%})")

    print("\nSmallest and largest value of each column:")
    print(df.agg(["min", "max"]).T)


def main():
    df = download_dataset()
    print_quality_report(df)


if __name__ == "__main__":
    main()
