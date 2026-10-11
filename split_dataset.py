#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare train/test and CV splits for multilabel classification.

Study assumption
----------------
The constructed study population assumes:

    number of nonA sites = number of modification-positive sites

Therefore, with --nona_ratio 1.0:

    P(nonA) = 0.5

Definitions
-----------
Modification-positive:
    at least one of label_m6A, label_m1A, label_Am, label_AtoI == 1

Pure m6A:
    label_m6A=1 and all other labels=0

Pure nonA:
    label_nonA=1 and all modification labels=0

Rules
-----
1. Create source_id from site_id/matched_positive_site_id and split the
   original data into development and independent test sets by source_id.
2. Independent test:
   - keep all modification-positive test samples
   - randomly downsample pure nonA so that
       nonA = positive_sites * nona_ratio
   - DO NOT apply max_sample to m6A in the test set
3. Final training:
   - downsample pure m6A to at most max_sample
   - keep all m1A, Am, A-to-I and mixed/multilabel samples
   - downsample pure nonA so that
       nonA = remaining_positive_sites * nona_ratio
4. Cross-validation:
   - validation folds follow the assumed prevalence
   - only CV training folds apply the pure-m6A max_sample cap
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMNS = [
    "label_m6A",
    "label_m1A",
    "label_Am",
    "label_AtoI",
    "label_nonA",
]

POSITIVE_LABEL_COLUMNS = [
    "label_m6A",
    "label_m1A",
    "label_Am",
    "label_AtoI",
]

OUTPUT_COLUMNS = [
    "site_id",
    "source_id",
    "sequence",
    *LABEL_COLUMNS,
]


def parse_csv(value):
    return [x.strip() for x in value.split(",") if x.strip()]

def print_label_counts(df, label_cols, title):
    print(f"\n[INFO] {title}")
    print(df[label_cols].sum().astype(int))
    combinations = df[label_cols].astype(str).agg("".join, axis=1)
    print("[INFO] Label-combination counts (column order shown below):")
    print("[INFO]", ",".join(label_cols))
    print(combinations.value_counts())


def validate_fraction(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, but got {value}.")


def remove_representative_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the 'representative_' prefix from every column name."""
    renamed = {
        column: column.removeprefix("representative_")
        for column in df.columns
    }

    new_names = list(renamed.values())
    duplicated = sorted({
        name for name in new_names
        if new_names.count(name) > 1
    })

    if duplicated:
        raise ValueError(
            "Removing 'representative_' creates duplicate column names: "
            + ", ".join(duplicated)
        )

    return df.rename(columns=renamed)


def add_source_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create source_id from site_id and matched_positive_site_id."""
    if "site_id" not in df.columns:
        raise ValueError("Missing required column: site_id")

    out = df.copy()
    out["source_id"] = out["site_id"].astype("string")

    if "matched_positive_site_id" in out.columns:
        matched = out["matched_positive_site_id"].astype("string").str.strip()
        use_matched = matched.notna() & matched.ne("")
        out.loc[use_matched, "source_id"] = matched.loc[use_matched]

    return out


def validate_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns, labels and sequence values."""
    missing = [column for column in OUTPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    prepared = df.loc[:, OUTPUT_COLUMNS].copy()

    for column in LABEL_COLUMNS:
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin([0, 1])

        if invalid.any():
            examples = prepared.loc[invalid, column].head(5).tolist()
            raise ValueError(
                f"Column '{column}' must contain only 0/1 values. "
                f"Invalid examples: {examples}"
            )

        prepared[column] = numeric.astype(np.int8)

    valid_mask = (
        prepared[
            ["site_id", "source_id", "sequence"]
        ].notna().all(axis=1)
        & prepared["sequence"].astype(str).str.fullmatch(r"[AUGCT]+", na=False)
    )

    removed = int((~valid_mask).sum())
    if removed:
        warnings.warn(f"Removed {removed} invalid rows.")

    prepared = prepared.loc[valid_mask].reset_index(drop=True)
    prepared["sequence"] = prepared["sequence"].astype(str)

    if prepared.empty:
        raise ValueError("No valid rows remain after input validation.")

    if prepared["site_id"].duplicated().any():
        warnings.warn("Duplicate site_id values were found.")

    return prepared


def label_pattern(df: pd.DataFrame) -> pd.Series:
    """Return the complete 5-label pattern, e.g. 10000 or 00001."""
    return df[LABEL_COLUMNS].astype(str).agg("".join, axis=1)


def source_pattern(df: pd.DataFrame) -> pd.Series:
    """Create one label-pattern signature per source_id."""
    work = df.loc[:, ["source_id", *LABEL_COLUMNS]].copy()
    work["_row_pattern"] = label_pattern(work)
    return (
        work.groupby("source_id", sort=False)["_row_pattern"]
        .apply(lambda x: "|".join(sorted(set(x.astype(str)))))
    )


def split_by_pattern(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split source_id groups into development and test sets."""
    rng = np.random.default_rng(seed)

    source_table = source_pattern(df).rename("_pattern").reset_index()

    development_sources = []
    test_sources = []

    for _, group in source_table.groupby("_pattern", sort=False):
        source_ids = rng.permutation(group["source_id"].to_numpy())
        n = len(source_ids)

        if n == 1:
            n_test = 0
        else:
            n_test = int(round(n * test_size))
            n_test = max(1, n_test)
            n_test = min(n - 1, n_test)

        test_sources.extend(source_ids[:n_test].tolist())
        development_sources.extend(source_ids[n_test:].tolist())

    development_df = (
        df.loc[df["source_id"].isin(development_sources)]
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    test_df = (
        df.loc[df["source_id"].isin(test_sources)]
        .sample(frac=1.0, random_state=seed + 1)
        .reset_index(drop=True)
    )

    overlap = set(development_df["source_id"]) & set(test_df["source_id"])
    if overlap:
        raise RuntimeError("source_id leakage detected between development and test.")

    return development_df, test_df


def get_masks(df: pd.DataFrame):
    """Return modification-positive, pure-m6A and pure-nonA masks."""
    positive_mask = df[POSITIVE_LABEL_COLUMNS].any(axis=1)

    pure_m6a_mask = (
        df["label_m6A"].eq(1)
        & df["label_m1A"].eq(0)
        & df["label_Am"].eq(0)
        & df["label_AtoI"].eq(0)
        & df["label_nonA"].eq(0)
    )

    pure_nona_mask = (
        df["label_m6A"].eq(0)
        & df["label_m1A"].eq(0)
        & df["label_Am"].eq(0)
        & df["label_AtoI"].eq(0)
        & df["label_nonA"].eq(1)
    )

    return positive_mask, pure_m6a_mask, pure_nona_mask


def balance_nona_to_positive(
    df: pd.DataFrame,
    nona_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Set pure nonA count relative to all modification-positive sites.

    With nona_ratio=1.0:
        N(nonA) = N(modification-positive)
        P(nonA) = 0.5
    assuming all retained rows belong to one of these two groups.
    """
    if nona_ratio <= 0:
        raise ValueError("--nona_ratio must be > 0.")

    positive_mask, _, pure_nona_mask = get_masks(df)

    positive_df = df.loc[positive_mask]
    nona_df = df.loc[pure_nona_mask]

    other_mask = ~(positive_mask | pure_nona_mask)
    other_df = df.loc[other_mask]

    target_nona = int(round(len(positive_df) * nona_ratio))
    target_nona = min(target_nona, len(nona_df))

    if len(nona_df) > target_nona:
        nona_df = nona_df.sample(
            n=target_nona,
            random_state=seed,
            replace=False,
        )

    result = pd.concat(
        [positive_df, nona_df, other_df],
        ignore_index=True,
    )

    result = result.sample(
        frac=1.0,
        random_state=seed + 1,
    ).reset_index(drop=True)

    if len(other_df):
        warnings.warn(
            f"Retained {len(other_df)} rows that are neither "
            "modification-positive nor pure nonA."
        )

    print(
        f"[INFO] nonA balancing: "
        f"positive={len(positive_df):,}, "
        f"pure nonA={len(nona_df):,}, "
        f"ratio={len(nona_df) / max(len(positive_df), 1):.4f}"
    )

    return result


def prepare_training_dataset(
    df: pd.DataFrame,
    max_sample: int,
    nona_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Cap pure m6A, preserve all other positives, then match nonA to positives."""
    if max_sample <= 0:
        raise ValueError("--max_sample must be > 0.")

    positive_mask, pure_m6a_mask, pure_nona_mask = get_masks(df)

    pure_m6a_df = df.loc[pure_m6a_mask]
    protected_positive_df = df.loc[positive_mask & ~pure_m6a_mask]
    pure_nona_df = df.loc[pure_nona_mask]

    other_mask = ~(positive_mask | pure_nona_mask)
    other_df = df.loc[other_mask]

    original_pure_m6a = len(pure_m6a_df)

    if len(pure_m6a_df) > max_sample:
        pure_m6a_df = pure_m6a_df.sample(
            n=max_sample,
            random_state=seed,
            replace=False,
        )

    positive_df = pd.concat(
        [pure_m6a_df, protected_positive_df],
        ignore_index=True,
    )

    target_nona = int(round(len(positive_df) * nona_ratio))

    if len(pure_nona_df) < target_nona:
        warnings.warn(
            f"Requested {target_nona:,} pure nonA samples, "
            f"but only {len(pure_nona_df):,} are available."
        )
        target_nona = len(pure_nona_df)

    if len(pure_nona_df) > target_nona:
        pure_nona_df = pure_nona_df.sample(
            n=target_nona,
            random_state=seed + 1,
            replace=False,
        )

    result = pd.concat(
        [positive_df, pure_nona_df, other_df],
        ignore_index=True,
    )

    result = result.sample(
        frac=1.0,
        random_state=seed + 2,
    ).reset_index(drop=True)

    print(
        "[INFO] Training sampling: "
        f"pure m6A {original_pure_m6a:,} -> {len(pure_m6a_df):,}; "
        f"protected positive={len(protected_positive_df):,}; "
        f"total positive={len(positive_df):,}; "
        f"pure nonA={len(pure_nona_df):,}"
    )

    return result


def make_cv_assignments(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Assign CV folds by source_id so linked rows stay in the same fold."""
    if n_splits < 2:
        raise ValueError("--n_splits must be at least 2.")

    rng = np.random.default_rng(seed)
    source_table = source_pattern(df).rename("_pattern").reset_index()

    source_to_fold = {}

    for pattern in pd.unique(source_table["_pattern"]):
        source_ids = source_table.loc[
            source_table["_pattern"].eq(pattern), "source_id"
        ].to_numpy()
        source_ids = rng.permutation(source_ids)

        for j, source_id in enumerate(source_ids):
            source_to_fold[source_id] = j % n_splits

    assignments = df["source_id"].map(source_to_fold)

    if assignments.isna().any():
        raise RuntimeError("Some rows were not assigned to a CV fold.")

    return assignments.astype(int).to_numpy()


def save_label_counts(df: pd.DataFrame, output_path: Path) -> None:
    """Save label counts and prevalence."""
    counts = pd.DataFrame({
        "label": LABEL_COLUMNS,
        "positive_count": [
            int(df[column].sum())
            for column in LABEL_COLUMNS
        ],
        "total_rows": len(df),
        "prevalence": [
            float(df[column].mean())
            for column in LABEL_COLUMNS
        ],
    })

    counts.to_csv(output_path, index=False)


def print_summary(name: str, df: pd.DataFrame) -> None:
    """Print dataset summary."""
    positive_mask, pure_m6a_mask, pure_nona_mask = get_masks(df)

    print(f"\n[INFO] {name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Unique source_id: {df['source_id'].nunique():,}")

    for column in LABEL_COLUMNS:
        print(
            f"  {column}: {int(df[column].sum()):,} "
            f"({100.0 * df[column].mean():.3f}%)"
        )

    print(f"  Modification-positive sites: {int(positive_mask.sum()):,}")
    print(f"  Pure m6A : {int(pure_m6a_mask.sum()):,}")
    print(f"  Pure nonA: {int(pure_nona_mask.sum()):,}")

    if len(df):
        print(
            f"  nonA prevalence: "
            f"{100.0 * df['label_nonA'].mean():.3f}%"
        )


def main(args: argparse.Namespace) -> None:
    validate_fraction(args.test_size, "test_size")

    if args.n_splits < 2:
        raise ValueError("--n_splits must be at least 2.")
    if args.nona_ratio <= 0:
        raise ValueError("--nona_ratio must be > 0.")
    if args.max_sample <= 0:
        raise ValueError("--max_sample must be > 0.")

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load original dataset
    # ------------------------------------------------------------
    raw_df = pd.read_csv(args.input_tsv, sep="\t")
    df = remove_representative_prefix(raw_df)
    df = add_source_id(df)
    df = validate_and_prepare_dataframe(df)
    print_summary("Original dataset", df)

    #label_cols"label_m6A,label_m1A,label_Am,label_AtoI,label_nonA"
    #label_cols = parse_csv(args.label_cols)
    print_label_counts(df, LABEL_COLUMNS, "Benchmark dataset")

    # ------------------------------------------------------------
    # 2. Split ORIGINAL data into development and independent test
    # ------------------------------------------------------------
    development_full, test_full = split_by_pattern(
        df=df,
        test_size=args.test_size,
        seed=args.seed,
    )

    # ------------------------------------------------------------
    # 3. Independent test
    #
    # Keep all positive test sites.
    # Do NOT cap m6A.
    # Downsample nonA to positive * nona_ratio.
    # With nona_ratio=1.0, nonA prevalence is approximately 0.5.
    # ------------------------------------------------------------
    test_df = balance_nona_to_positive(
        test_full,
        nona_ratio=args.nona_ratio,
        seed=args.seed + 10,
    )

    # ------------------------------------------------------------
    # 4. Development population for CV
    #
    # Apply only the study's assumed nonA prevalence here.
    # Do NOT cap m6A before creating validation folds.
    # ------------------------------------------------------------
    development_balanced = balance_nona_to_positive(
        development_full,
        nona_ratio=args.nona_ratio,
        seed=args.seed + 20,
    )

    # ------------------------------------------------------------
    # 5. Final training dataset
    #
    # Cap pure m6A, retain all other positives,
    # then make nonA equal to remaining positives * nona_ratio.
    # ------------------------------------------------------------
    train_df = prepare_training_dataset(
        development_full,
        max_sample=args.max_sample,
        nona_ratio=args.nona_ratio,
        seed=args.seed,
    )

    train_df = train_df.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)
    test_df = test_df.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)

    train_path = input_dir / f"train_{args.identity}.csv"
    test_path = input_dir / f"test_{args.identity}.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    save_label_counts(
        train_df,
        input_dir / f"train_label_counts_{args.identity}.csv",
    )
    save_label_counts(
        test_df,
        input_dir / f"test_label_counts_{args.identity}.csv",
    )

    print_summary("Final training dataset", train_df)
    print_summary("Independent test dataset", test_df)

    print(f"\n[INFO] Saved: {train_path}")
    print(f"[INFO] Saved: {test_path}")

    # ------------------------------------------------------------
    # 6. Cross-validation
    #
    # validation:
    #   follows the assumed population, including ~50% nonA
    #
    # training:
    #   pure m6A capped at max_sample
    #   nonA matched to remaining positives
    # ------------------------------------------------------------
    cv_dir = input_dir / "cross_val"
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_assignment = make_cv_assignments(
        development_balanced,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    for fold_zero_based in range(args.n_splits):
        fold_id = fold_zero_based + 1

        fold_valid_df = development_balanced.loc[
            fold_assignment == fold_zero_based
        ].copy()

        fold_train_full_df = development_balanced.loc[
            fold_assignment != fold_zero_based
        ].copy()

        fold_train_df = prepare_training_dataset(
            fold_train_full_df,
            max_sample=args.max_sample,
            nona_ratio=args.nona_ratio,
            seed=args.seed + fold_id * 100,
        )

        fold_train_df = fold_train_df.loc[
            :, OUTPUT_COLUMNS
        ].reset_index(drop=True)

        fold_valid_df = fold_valid_df.loc[
            :, OUTPUT_COLUMNS
        ].reset_index(drop=True)

        fold_dir = cv_dir / str(fold_id)
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_fold_path = fold_dir / f"cv_train_{fold_id}.csv"
        valid_fold_path = fold_dir / f"cv_val_{fold_id}.csv"

        fold_train_df.to_csv(train_fold_path, index=False)
        fold_valid_df.to_csv(valid_fold_path, index=False)

        save_label_counts(
            fold_train_df,
            fold_dir / "train_label_counts.csv",
        )
        save_label_counts(
            fold_valid_df,
            fold_dir / "val_label_counts.csv",
        )

        print_summary(
            f"Fold {fold_id} training",
            fold_train_df,
        )
        print_summary(
            f"Fold {fold_id} validation",
            fold_valid_df,
        )

    print(f"\n[INFO] Done. Output directory: {input_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create multilabel train/test/CV datasets under the study "
            "assumption that nonA prevalence is 0.5 when nona_ratio=1.0."
        )
    )

    parser.add_argument(
        "--input_tsv",
        type=str,
        required=True,
        help="Input dataset TSV.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./",
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.9,
        help="CDHIT-EST identity",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Fraction assigned to the independent test pool.",
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )

    parser.add_argument(
        "--nona_ratio",
        type=float,
        default=1.0,
        help=(
            "N(nonA) / N(modification-positive). "
            "1.0 gives approximately 50%% nonA prevalence."
        ),
    )

    parser.add_argument(
        "--max_sample",
        type=int,
        default=50000,
        help=(
            "Maximum number of pure m6A samples in each training dataset. "
            "This cap is not applied to test/validation datasets."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    main(parser.parse_args())
