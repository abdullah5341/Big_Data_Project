"""Step 2: clean the data, engineer features, and create a leakage-safe train/validation/test split.

Run:  python src/preprocess.py
Output: data/processed/{train,val,test}.csv and results/split_summary.csv

Leakage rules followed here:
- Duplicates are removed BEFORE splitting, so no identical row can sit in both train and test.
- Only row-wise feature engineering happens here (each new value uses one row's own answers),
  so nothing is learned from other rows or from the test set.
- Anything that is *fitted* (scaling, class weights, feature selection, thresholds) happens later,
  inside the model pipeline, on training data only.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (BMI_BINS, BMI_LABELS, COMORBIDITY_COLUMNS, PROCESSED_DIR, RANDOM_STATE,
                    RESULTS_DIR, TARGET, TEST_SIZE, VAL_SIZE)
from data_loader import load_dataset


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and report how this changes the class balance."""
    before_rate = df[TARGET].mean()
    deduplicated = df.drop_duplicates().reset_index(drop=True)
    removed = len(df) - len(deduplicated)
    print(f"Removed {removed:,} duplicate rows: {len(df):,} -> {len(deduplicated):,}")
    print(f"Positive class share: {before_rate:.2%} -> {deduplicated[TARGET].mean():.2%}")
    return deduplicated


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add two clinically meaningful features. Both are calculated from a single row, so no leakage."""
    df = df.copy()
    # Groups raw BMI into the WHO bands that clinicians actually use.
    df["BMICategory"] = pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False).astype(int)
    # Multiple chronic conditions together are a stronger risk signal than any one alone.
    df["ComorbidityCount"] = df[COMORBIDITY_COLUMNS].sum(axis=1)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split. Stratifying keeps the same diabetes share in every part."""
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)
    # VAL_SIZE is a share of the whole dataset, so rescale it to a share of the remaining rows.
    val_share_of_remaining = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_share_of_remaining, stratify=train_val[TARGET], random_state=RANDOM_STATE)
    return train, val, test


def check_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if any identical row appears in more than one split."""
    columns = list(train.columns)
    for name_a, a, name_b, b in [("train", train, "val", val), ("train", train, "test", test),
                                 ("val", val, "test", test)]:
        shared = a.merge(b, on=columns, how="inner")
        assert shared.empty, f"{len(shared)} identical rows shared between {name_a} and {name_b}"
    print("Overlap check passed: no identical rows shared between splits.")


def summarise_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = pd.DataFrame({
        "rows": {name: len(part) for name, part in splits.items()},
        "positives": {name: int(part[TARGET].sum()) for name, part in splits.items()},
        "positive_share": {name: round(part[TARGET].mean(), 4) for name, part in splits.items()},
    })
    print("\nSplit summary:")
    print(summary.to_string())
    return summary


def main() -> None:
    df = remove_duplicates(load_dataset())
    df = add_engineered_features(df)
    train, val, test = split_data(df)
    check_no_overlap(train, val, test)

    splits = {"train": train, "val": val, "test": test}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    summarise_splits(splits).to_csv(RESULTS_DIR / "split_summary.csv")


if __name__ == "__main__":
    main()
