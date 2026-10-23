"""
STEP 2 - Clean the data, add two new features, and split it into train / validation / test.

Run:     python src/preprocess.py
Creates: data/processed/train.csv, val.csv, test.csv and results/split_summary.csv

How this step avoids data leakage:
- Duplicates are removed BEFORE splitting, so the same row cannot end up in both train and test.
- The two new features only use each person's own answers, so nothing is learned from other rows.
- Anything that has to learn from the data (scaling, class weights, the threshold) happens later
  and uses the training data only.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (BMI_BINS, BMI_LABELS, COMORBIDITY_COLUMNS, PROCESSED_DIR, RANDOM_STATE,
                    RESULTS_DIR, TARGET, TEST_SIZE, VAL_SIZE)
from data_loader import load_dataset


def remove_duplicates(df):
    """Drop rows that are exact copies of another row."""
    rows_before = len(df)
    share_before = df[TARGET].mean()

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Removed {rows_before - len(df):,} duplicate rows: {rows_before:,} -> {len(df):,}")
    # Most duplicates are healthy people, so removing them raises the diabetes share a little.
    print(f"Diabetes share: {share_before:.2%} -> {df[TARGET].mean():.2%}")
    return df


def add_engineered_features(df):
    """Add BMICategory and ComorbidityCount. The web app uses this same function."""
    df = df.copy()
    # BMI group the way doctors use it: 0 underweight, 1 normal, 2 overweight, 3 obese.
    df["BMICategory"] = pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False).astype(int)
    # Several chronic conditions together are a stronger warning sign than one alone.
    df["ComorbidityCount"] = df[COMORBIDITY_COLUMNS].sum(axis=1)
    return df


def split_data(df):
    """Split into 70% train, 15% validation and 15% test.

    stratify=... keeps the same diabetes share (about 15%) in all three parts.
    """
    # First, cut off the 15% test set.
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)

    # Then split the remaining 85% into train and validation.
    # 15% of all the data is 0.15 / 0.85 = 17.6% of what is left.
    val_share = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=train_val[TARGET], random_state=RANDOM_STATE)
    return train, val, test


def count_shared_rows(part_a, part_b):
    """How many identical rows appear in both parts? It should be 0."""
    shared = part_a.merge(part_b, on=list(part_a.columns), how="inner")
    return len(shared)


def check_no_overlap(train, val, test):
    """Stop with an error if any row appears in two different splits."""
    assert count_shared_rows(train, val) == 0, "train and validation share rows"
    assert count_shared_rows(train, test) == 0, "train and test share rows"
    assert count_shared_rows(val, test) == 0, "validation and test share rows"
    print("Overlap check passed: no row appears in two splits.")


def main():
    df = load_dataset()
    df = remove_duplicates(df)
    df = add_engineered_features(df)
    train, val, test = split_data(df)
    check_no_overlap(train, val, test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        summary_rows.append({
            "split": name,
            "rows": len(part),
            "positives": int(part[TARGET].sum()),
            "positive_share": round(part[TARGET].mean(), 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "split_summary.csv", index=False)
    print("\nSplit summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
