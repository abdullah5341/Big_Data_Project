"""Step 1: download the CDC Diabetes Health Indicators dataset from UCI and run basic checks.

Run:  python src/data_loader.py
Output: data/raw/cdc_diabetes_health_indicators.csv and a printed data-quality summary.
"""
import pandas as pd
from ucimlrepo import fetch_ucirepo

from config import ID_COLUMN, RAW_DIR, RAW_FILE, TARGET, UCI_DATASET_ID


def download_dataset() -> pd.DataFrame:
    """Fetch the dataset from UCI, join features and target into one table, and cache it."""
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # The ID identifies a respondent and carries no health information.
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)
    return df


def load_dataset() -> pd.DataFrame:
    """Load the cached CSV, downloading it first if needed. Other scripts call this."""
    if RAW_FILE.exists():
        return pd.read_csv(RAW_FILE)
    return download_dataset()


def print_quality_report(df: pd.DataFrame) -> None:
    """Print the facts we must verify and report: size, missing values, duplicates, class balance, ranges."""
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Feature columns ({df.shape[1] - 1}): {[c for c in df.columns if c != TARGET]}")
    print(f"\nMissing values (total): {int(df.isna().sum().sum())}")
    print(f"Exact duplicate rows: {int(df.duplicated().sum()):,} ({df.duplicated().mean():.1%} of rows)")

    print(f"\nTarget '{TARGET}' distribution:")
    for label, n in df[TARGET].value_counts().sort_index().items():
        print(f"  {label}: {n:,} ({n / len(df):.1%})")

    print("\nValue ranges (min / max) per column:")
    print(df.agg(["min", "max"]).T.to_string())


if __name__ == "__main__":
    print_quality_report(download_dataset())
