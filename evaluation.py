#!/usr/bin/env python3
"""
Faster version of metrics_baseline_21.py.

Main speedups:
  - Vectorized per-label precision/recall/F1/MCC/balanced accuracy/accuracy.
  - Replaced repeated DataFrame filtering with grouped mean/SD aggregation.
  - Reads only true_*, pred_*, prob_* columns from each CSV.
  - Removed unused/duplicate calculations and noisy printing.
  - Fixed the undefined infer_label_names(df) bug when --labels is omitted.
  - Writes paired mean and sample-SD rows across CV folds for val/test results.
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    label_ranking_average_precision_score,
    roc_auc_score,
)


METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "mcc",
    "balanced_accuracy",
    "accuracy",
    "subset accuracy",
    "pr_auc",
    "roc_auc",
    "lrap",
    "absolute_true",
    "absolute_false",
]


def _safe_divide(num: np.ndarray | float, den: np.ndarray | float, default: float = 0.0):
    """Divide with default value when denominator is zero."""
    num_arr = np.asarray(num, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num_arr.shape, den_arr.shape), default, dtype=float)
    return np.divide(num_arr, den_arr, out=out, where=(den_arr != 0))


def table_adjust(df: pd.DataFrame, split: str, machine: str, encode: str) -> pd.DataFrame:
    df = df.copy()
    if "fold" in df.columns:
        df = df.drop(columns="fold")
    df.insert(0, "split", split)
    df.insert(1, "machine", machine)
    df.insert(2, "encode", encode)
    return df


def sorted_cols(columns: Iterable[str], prefix: str) -> list[str]:
    return sorted([c for c in columns if c.startswith(prefix)], key=lambda x: str(x.split("_")[-1]))


def read_roc(path: Path, label_names: list[str] | None = None):
    if not path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")

    # First read only the header, then load only metric-related columns.
    all_columns = pd.read_csv(path, nrows=0).columns.tolist()

    if label_names:
        true_columns = [f"true_{label}" for label in label_names]
        pred_columns = [f"pred_{label}" for label in label_names]
        prob_columns = [f"prob_{label}" for label in label_names]
    else:
        true_columns = sorted_cols(all_columns, "true_")
        pred_columns = sorted_cols(all_columns, "pred_")
        prob_columns = sorted_cols(all_columns, "prob_")

    missing = [c for c in true_columns + pred_columns + prob_columns if c not in all_columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if not true_columns or not pred_columns or not prob_columns:
        raise ValueError(f"No true_*, pred_*, or prob_* columns found in {path}")

    df = pd.read_csv(path, usecols=true_columns + pred_columns + prob_columns)

    y_true = df[true_columns].to_numpy(dtype=np.int8, copy=False)
    y_pred = df[pred_columns].to_numpy(dtype=np.int8, copy=False)
    y_prob = df[prob_columns].to_numpy(dtype=np.float64, copy=False)

    if y_true.shape != y_pred.shape or y_true.shape != y_prob.shape:
        raise ValueError(
            f"Shape mismatch in {path}: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}, y_prob={y_prob.shape}"
        )

    index2label = {idx + 1: col[5:] for idx, col in enumerate(true_columns)}
    return y_true, y_pred, y_prob, index2label


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    value = roc_auc_score(y_true, y_score)
    return 0.5 if np.isnan(value) else float(value)


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0:
        return 0.0
    if negatives == 0:
        return 1.0
    value = average_precision_score(y_true, y_score)
    return 0.0 if np.isnan(value) else float(value)


def safe_lrap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    valid_rows = np.sum(y_true, axis=1) > 0
    if not np.any(valid_rows):
        return 0.0
    value = label_ranking_average_precision_score(y_true[valid_rows], y_prob[valid_rows])
    return 0.0 if np.isnan(value) else float(value)


def confusion_by_label(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.astype(bool, copy=False)
    yp = y_pred.astype(bool, copy=False)
    tp = np.sum(yt & yp, axis=0).astype(float)
    fp = np.sum(~yt & yp, axis=0).astype(float)
    fn = np.sum(yt & ~yp, axis=0).astype(float)
    tn = np.sum(~yt & ~yp, axis=0).astype(float)
    return tp, fp, fn, tn


def per_label_fast_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp, fp, fn, tn = confusion_by_label(y_true, y_pred)
    n = y_true.shape[0]

    precision = _safe_divide(tp, tp + fp, default=0.0)
    recall = _safe_divide(tp, tp + fn, default=0.0)
    f1 = _safe_divide(2 * precision * recall, precision + recall, default=0.0)

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_divide(tp * tn - fp * fn, denom, default=0.0)

    positives = tp + fn
    negatives = tn + fp
    bacc_normal = 0.5 * (_safe_divide(tp, positives) + _safe_divide(tn, negatives))
    one_class = (positives == 0) | (negatives == 0)
    bacc_fallback = (tp + tn) / n
    balanced_accuracy = np.where(one_class, bacc_fallback, bacc_normal)

    accuracy = (tp + tn) / n
    return precision, recall, f1, mcc, balanced_accuracy, accuracy


def binary_fast_metrics(y_true_flat: np.ndarray, y_pred_flat: np.ndarray):
    yt = y_true_flat.astype(bool, copy=False)
    yp = y_pred_flat.astype(bool, copy=False)
    tp = float(np.sum(yt & yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    tn = float(np.sum(~yt & ~yp))

    precision = float(_safe_divide(tp, tp + fp, default=0.0))
    recall = float(_safe_divide(tp, tp + fn, default=0.0))
    f1 = float(_safe_divide(2 * precision * recall, precision + recall, default=0.0))

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float(_safe_divide(tp * tn - fp * fn, denom, default=0.0))

    positives = tp + fn
    negatives = tn + fp
    if positives == 0 or negatives == 0:
        balanced_accuracy = float(np.mean(y_true_flat == y_pred_flat))
    else:
        balanced_accuracy = float(0.5 * (tp / positives + tn / negatives))

    accuracy = float((tp + tn) / (tp + fp + fn + tn))
    return precision, recall, f1, mcc, balanced_accuracy, accuracy


def calc_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split: str,
    fold: int,
) -> pd.DataFrame:
    precision, recall, f1, mcc, bacc, acc = per_label_fast_metrics(y_true, y_pred)
    n_labels = y_true.shape[1]

    # AUC/AP still require ranking/sorting, so keep these per label.
    pr_auc = np.array([safe_ap(y_true[:, k], y_prob[:, k]) for k in range(n_labels)])
    roc_auc = np.array([safe_auc(y_true[:, k], y_prob[:, k]) for k in range(n_labels)])

    return pd.DataFrame(
        {
            "fold": fold,
            "split": split,
            "label": np.arange(1, n_labels + 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "balanced_accuracy": bacc,
            "accuracy": acc,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "lrap": np.nan,
        }
    )


def calc_macro_metrics(per_label_df: pd.DataFrame, split: str, fold: int) -> pd.DataFrame:
    row = {
        "fold": fold,
        "split": split,
        "precision": float(per_label_df["precision"].mean()),
        "recall": float(per_label_df["recall"].mean()),
        "f1": float(per_label_df["f1"].mean()),
        "mcc": float(per_label_df["mcc"].mean()),
        "balanced_accuracy": float(per_label_df["balanced_accuracy"].mean()),
        "accuracy": float(per_label_df["accuracy"].mean()),
        "subset accuracy": np.nan,
        "pr_auc": float(per_label_df["pr_auc"].mean()),
        "roc_auc": float(per_label_df["roc_auc"].mean()),
        "lrap": np.nan,
        "absolute_true": np.nan,
        "absolute_false": np.nan,
    }
    return pd.DataFrame([row])


def calc_micro_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split: str,
    fold: int,
) -> pd.DataFrame:
    yt = y_true.ravel()
    yp = y_pred.ravel()
    ys = y_prob.ravel()
    precision, recall, f1, mcc, bacc, acc = binary_fast_metrics(yt, yp)
    subset_accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
    mismatch_rate = float(np.mean(np.logical_xor(y_true, y_pred)))

    row = {
        "fold": fold,
        "split": split,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": bacc,
        "accuracy": acc,
        "subset accuracy": subset_accuracy,
        "pr_auc": safe_ap(yt, ys),
        "roc_auc": safe_auc(yt, ys),
        "lrap": safe_lrap(y_true, y_prob),
        "absolute_true": subset_accuracy,
        "absolute_false": mismatch_rate,
    }
    return pd.DataFrame([row])


def eval_file(path: Path, split: str, fold: int, label_names: list[str] | None = None):
    y_true, y_pred, y_prob, index2label = read_roc(path, label_names=label_names)
    per_label_df = calc_per_label_metrics(y_true, y_pred, y_prob, split, fold)
    macro_df = calc_macro_metrics(per_label_df, split, fold)
    micro_df = calc_micro_metrics(y_true, y_pred, y_prob, split, fold)
    return macro_df, micro_df, per_label_df, index2label


def mean_sd_numeric_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Return paired mean/sd rows across folds for all numeric metric columns."""
    combined = pd.concat(frames, ignore_index=True)
    numeric = combined.select_dtypes(include=[np.number]).drop(columns="fold", errors="ignore")

    mean_row = numeric.mean(axis=0)
    # ddof=1 gives the sample SD across CV folds (the pandas default).
    sd_row = numeric.std(axis=0, ddof=1)

    summary = pd.DataFrame([mean_row, sd_row])
    summary.insert(0, "mean/sd", ["mean", "sd"])
    return summary


def add_label_after_group_mean_sd(per_label_df: pd.DataFrame) -> pd.DataFrame:
    """Return paired mean/sd rows for each label across folds."""
    metric_columns = [c for c in METRIC_COLUMNS if c in per_label_df.columns]
    grouped = per_label_df.groupby("label", sort=True)[metric_columns]

    mean_df = grouped.mean().reset_index()
    mean_df.insert(1, "mean/sd", "mean")

    sd_df = grouped.std(ddof=1).reset_index()
    sd_df.insert(1, "mean/sd", "sd")

    # Interleave rows as label1-mean, label1-sd, label2-mean, label2-sd, ...
    return (
        pd.concat([mean_df, sd_df], ignore_index=True)
        .sort_values(["label", "mean/sd"], key=lambda s: s.map({"mean": 0, "sd": 1}) if s.name == "mean/sd" else s)
        .reset_index(drop=True)
    )


def write_metric_sheets(
    index2label: dict[int, str],
    machine: str,
    outfile: Path,
    macro_df: pd.DataFrame,
    micro_df: pd.DataFrame,
    per_label_df: pd.DataFrame,
) -> None:
    per_label_df = per_label_df.copy()
    per_label_df["label"] = per_label_df["label"].map(index2label)

    mode = "a" if outfile.exists() else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": mode}
    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(outfile, **writer_kwargs) as writer:
        macro_df.to_excel(writer, sheet_name=f"{machine}_macro", index=False)
        micro_df.to_excel(writer, sheet_name=f"{machine}_micro", index=False)
        per_label_df.to_excel(writer, sheet_name=f"{machine}_per_label", index=False)


def metrics_calculation_save(
    pred_path: Path,
    outfile: Path,
    n_fold: int,
    machine: str,
    encodes: list[str],
    label_names: list[str] | None = None,
) -> None:
    all_macros = []
    all_micros = []
    all_per_labels = []
    index2label = None

    for encode in encodes:
        print(f"{machine}-{encode}")
        for split in ["val", "test"]:
            macros = []
            micros = []
            per_labels = []

            for fold in range(1, n_fold + 1):
                pred_file = pred_path / machine / encode / str(fold) / f"{split}_roc.csv"
                macro_df, micro_df, per_label_df, index2label = eval_file(
                    pred_file, split, fold, label_names=label_names
                )
                macros.append(macro_df)
                micros.append(micro_df)
                per_labels.append(per_label_df)

            macro_df = table_adjust(mean_sd_numeric_frame(macros), split, machine, encode)
            micro_df = table_adjust(mean_sd_numeric_frame(micros), split, machine, encode)
            per_label_df = table_adjust(
                add_label_after_group_mean_sd(pd.concat(per_labels, ignore_index=True)),
                split,
                machine,
                encode,
            )

            all_macros.append(macro_df)
            all_micros.append(micro_df)
            all_per_labels.append(per_label_df)

    if index2label is None:
        raise RuntimeError("No prediction files were processed.")

    write_metric_sheets(
        index2label=index2label,
        machine=machine,
        outfile=outfile,
        macro_df=pd.concat(all_macros, ignore_index=True),
        micro_df=pd.concat(all_micros, ignore_index=True),
        per_label_df=pd.concat(all_per_labels, ignore_index=True),
    )


def split_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine_method", type=str, required=True, help="comma-separated machine methods")
    parser.add_argument("--machine_encode", type=str, required=True, help="comma-separated encodings")
    parser.add_argument("--pred_path", type=str, required=True, help="prediction root directory")
    parser.add_argument("--result_dir", type=str, default=None, help="output directory")
    parser.add_argument("--outfile", type=str, required=True, help="output xlsx filename/path")
    parser.add_argument("--n_fold", type=int, required=True, help="number of folds")
    parser.add_argument("--labels", type=str, default="", help="optional comma-separated label names")
    args = parser.parse_args()

    machine_methods = split_arg(args.machine_method)
    machine_encodes = split_arg(args.machine_encode)
    label_names = split_arg(args.labels) if args.labels else None

    pred_path = Path(args.pred_path)
    result_dir = Path(args.result_dir) if args.result_dir is not None else pred_path.parent
    result_dir.mkdir(parents=True, exist_ok=True)

    outfile = Path(args.outfile)
    if not outfile.is_absolute():
        outfile = result_dir / outfile

    for machine in machine_methods:
        metrics_calculation_save(
            pred_path=pred_path,
            outfile=outfile,
            n_fold=args.n_fold,
            machine=machine,
            encodes=machine_encodes,
            label_names=label_names,
        )

    print(f"\n[DONE] Saved metrics to: {outfile}")


if __name__ == "__main__":
    main()
