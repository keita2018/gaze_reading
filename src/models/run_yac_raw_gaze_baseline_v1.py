#!/usr/bin/env python
"""Build and evaluate a first fixation-free raw-gaze baseline for ZuCo 2.0.

Scope
-----
This is a one-subject pipeline intended as a data/implementation sanity check.
It does not establish cross-subject generalization.

Input
-----
- 14 block-level continuous-gaze Parquet files listed in the sample manifest.
- YAC_all_trials.csv containing 739 sentence-reading trials.
- The 521-trial raw/official mapping CSV.

No fixation, saccade, or blink labels are used to create raw-gaze features.

Evaluation
----------
Seven leave-one-block-pair-out folds:
    fold k test = NR block k + TSR block k
    training    = all remaining NR/TSR blocks

This is safer than a random trial split because trials from the same recording
block never appear in both train and test.

Feature sets
------------
time_quality:
    duration, sample count, valid-data rates, longest invalid-run ratio
position_only:
    robust x/y location and dispersion statistics
dynamics_only:
    sample-to-sample movement and velocity statistics
pupil_only:
    pupil level and sample-to-sample change statistics
raw_no_duration:
    validity + position + dynamics + pupil, excluding duration/sample count
raw_all:
    raw_no_duration + duration/sample count

The script also evaluates official sent_gaze_sacc features on the 521 common
trials using exactly the same seven folds. This within-subject result is a
sanity check and is not the official cross-subject benchmark result.

Outputs
-------
- <SUBJECT>_raw_stat_features.csv
- <SUBJECT>_raw_baseline_fold_metrics.csv
- <SUBJECT>_raw_baseline_summary.csv
- <SUBJECT>_raw_baseline_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_MAP = {
    "NR": 1,
    "TSR": 0,
}


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return safe_float(np.quantile(values, q))


def mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return safe_float(np.mean(values)) if values.size else math.nan


def std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return safe_float(np.std(values, ddof=0)) if values.size else math.nan


def longest_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def consecutive_differences(
    values: np.ndarray,
    timestamps_ms: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return differences and dt only for adjacent valid samples."""
    values = np.asarray(values, dtype=float)
    timestamps_ms = np.asarray(timestamps_ms, dtype=float)
    valid_mask = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(values)
        & np.isfinite(timestamps_ms)
    )

    indices = np.flatnonzero(valid_mask)
    if indices.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    adjacent = np.diff(indices) == 1
    if not np.any(adjacent):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    left = indices[:-1][adjacent]
    right = indices[1:][adjacent]
    differences = values[right] - values[left]
    dt_ms = timestamps_ms[right] - timestamps_ms[left]

    keep = np.isfinite(differences) & np.isfinite(dt_ms) & (dt_ms > 0)
    return differences[keep], dt_ms[keep]


def extract_trial_features(group: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values("sample_index_in_trial")

    t = group["timestamp_ms_abs"].to_numpy(dtype=float)
    x = group["x_px"].to_numpy(dtype=float)
    y = group["y_px"].to_numpy(dtype=float)
    pupil = group["pupil_area"].to_numpy(dtype=float)

    finite_xy = np.isfinite(x) & np.isfinite(y)
    positive_xy = finite_xy & (x > 0) & (y > 0)
    valid_pupil = np.isfinite(pupil) & (pupil > 0)
    valid_basic = positive_xy & valid_pupil

    x_valid = x[positive_xy]
    y_valid = y[positive_xy]
    pupil_valid = pupil[valid_pupil]

    dx, dt_x = consecutive_differences(x, t, positive_xy)
    dy, dt_y = consecutive_differences(y, t, positive_xy)

    # x and y share the same validity mask, so their adjacent pairs should
    # align. Use the shorter length defensively.
    n_xy_diff = min(len(dx), len(dy), len(dt_x), len(dt_y))
    dx = dx[:n_xy_diff]
    dy = dy[:n_xy_diff]
    dt_xy = dt_x[:n_xy_diff]

    if n_xy_diff:
        displacement = np.hypot(dx, dy)
        speed_px_s = displacement / (dt_xy / 1000.0)
    else:
        displacement = np.asarray([], dtype=float)
        speed_px_s = np.asarray([], dtype=float)

    dpupil, _ = consecutive_differences(
        pupil,
        t,
        valid_pupil,
    )

    if len(t) >= 2:
        positive_dt = np.diff(t)
        positive_dt = positive_dt[
            np.isfinite(positive_dt) & (positive_dt > 0)
        ]
        median_dt_ms = (
            safe_float(np.median(positive_dt))
            if positive_dt.size else math.nan
        )
    else:
        median_dt_ms = math.nan

    longest_invalid_samples = longest_true_run(~valid_basic)
    longest_invalid_ms = (
        longest_invalid_samples * median_dt_ms
        if math.isfinite(median_dt_ms)
        else math.nan
    )

    duration_sec = safe_float(group["duration_sec"].iloc[0])
    n_samples = int(len(group))

    features: dict[str, Any] = {
        "trial_id": str(group["trial_id"].iloc[0]),
        "subject_id": str(group["subject_id"].iloc[0]),
        "task": str(group["task"].iloc[0]).upper(),
        "block_id": int(group["block_id"].iloc[0]),
        "trial_index_in_block": int(
            group["trial_index_in_block"].iloc[0]
        ),
        "marker_pair": str(group["marker_pair"].iloc[0]),
        "label_nr": LABEL_MAP[str(group["task"].iloc[0]).upper()],
        "duration_sec": duration_sec,
        "n_samples": n_samples,
        "median_dt_ms": median_dt_ms,
        "valid_xy_rate": safe_float(np.mean(positive_xy)),
        "valid_pupil_rate": safe_float(np.mean(valid_pupil)),
        "valid_basic_rate": safe_float(np.mean(valid_basic)),
        "longest_invalid_run_ms": longest_invalid_ms,
        "longest_invalid_run_ratio": (
            longest_invalid_ms / (duration_sec * 1000.0)
            if math.isfinite(longest_invalid_ms)
            and math.isfinite(duration_sec)
            and duration_sec > 0
            else math.nan
        ),
    }

    for name, values in (("x", x_valid), ("y", y_valid)):
        q05 = quantile(values, 0.05)
        q25 = quantile(values, 0.25)
        q50 = quantile(values, 0.50)
        q75 = quantile(values, 0.75)
        q95 = quantile(values, 0.95)

        features.update(
            {
                f"{name}_mean": mean(values),
                f"{name}_std": std(values),
                f"{name}_q05": q05,
                f"{name}_q25": q25,
                f"{name}_median": q50,
                f"{name}_q75": q75,
                f"{name}_q95": q95,
                f"{name}_iqr": (
                    q75 - q25
                    if math.isfinite(q75) and math.isfinite(q25)
                    else math.nan
                ),
                f"{name}_range90": (
                    q95 - q05
                    if math.isfinite(q95) and math.isfinite(q05)
                    else math.nan
                ),
            }
        )

    if len(x_valid) >= 2 and len(y_valid) == len(x_valid):
        correlation = np.corrcoef(x_valid, y_valid)[0, 1]
        features["xy_correlation"] = safe_float(correlation)
    else:
        features["xy_correlation"] = math.nan

    features.update(
        {
            "dx_mean": mean(dx),
            "dx_std": std(dx),
            "dx_abs_mean": mean(np.abs(dx)),
            "dx_abs_median": quantile(np.abs(dx), 0.50),
            "dx_abs_p90": quantile(np.abs(dx), 0.90),
            "dx_p10": quantile(dx, 0.10),
            "dx_p90": quantile(dx, 0.90),
            "dx_positive_rate": (
                safe_float(np.mean(dx > 0)) if dx.size else math.nan
            ),
            "dx_negative_rate": (
                safe_float(np.mean(dx < 0)) if dx.size else math.nan
            ),
            "dy_mean": mean(dy),
            "dy_std": std(dy),
            "dy_abs_mean": mean(np.abs(dy)),
            "dy_abs_median": quantile(np.abs(dy), 0.50),
            "dy_abs_p90": quantile(np.abs(dy), 0.90),
            "dy_p10": quantile(dy, 0.10),
            "dy_p90": quantile(dy, 0.90),
            "displacement_mean": mean(displacement),
            "displacement_median": quantile(displacement, 0.50),
            "displacement_p90": quantile(displacement, 0.90),
            "speed_mean_px_s": mean(speed_px_s),
            "speed_median_px_s": quantile(speed_px_s, 0.50),
            "speed_p90_px_s": quantile(speed_px_s, 0.90),
            "speed_p95_px_s": quantile(speed_px_s, 0.95),
            "path_length_px": safe_float(np.sum(displacement))
            if displacement.size else math.nan,
            "path_length_px_per_sec": (
                safe_float(np.sum(displacement) / duration_sec)
                if displacement.size
                and math.isfinite(duration_sec)
                and duration_sec > 0
                else math.nan
            ),
        }
    )

    pupil_q05 = quantile(pupil_valid, 0.05)
    pupil_q25 = quantile(pupil_valid, 0.25)
    pupil_q50 = quantile(pupil_valid, 0.50)
    pupil_q75 = quantile(pupil_valid, 0.75)
    pupil_q95 = quantile(pupil_valid, 0.95)

    features.update(
        {
            "pupil_mean": mean(pupil_valid),
            "pupil_std": std(pupil_valid),
            "pupil_q05": pupil_q05,
            "pupil_q25": pupil_q25,
            "pupil_median": pupil_q50,
            "pupil_q75": pupil_q75,
            "pupil_q95": pupil_q95,
            "pupil_iqr": (
                pupil_q75 - pupil_q25
                if math.isfinite(pupil_q75)
                and math.isfinite(pupil_q25)
                else math.nan
            ),
            "pupil_diff_abs_mean": mean(np.abs(dpupil)),
            "pupil_diff_abs_median": quantile(
                np.abs(dpupil), 0.50
            ),
            "pupil_diff_abs_p90": quantile(np.abs(dpupil), 0.90),
        }
    )

    return features


def build_feature_table(
    manifest_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    feature_rows: list[dict[str, Any]] = []

    for row in manifest.itertuples(index=False):
        parquet_path = Path(str(row.sample_parquet))
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)

        print(f"Extracting raw statistics: {parquet_path}")
        block_samples = pd.read_parquet(parquet_path)

        for _, trial_group in block_samples.groupby(
            "trial_id",
            sort=False,
        ):
            feature_rows.append(extract_trial_features(trial_group))

    features = pd.DataFrame(feature_rows)

    if features["trial_id"].duplicated().any():
        duplicate_ids = features.loc[
            features["trial_id"].duplicated(keep=False),
            "trial_id",
        ].tolist()
        raise ValueError(f"Duplicated trial features: {duplicate_ids[:20]}")

    features = features.sort_values(
        ["task", "block_id", "trial_index_in_block"]
    ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding="utf-8-sig")
    return features


def make_feature_sets(
    raw_features: pd.DataFrame,
) -> dict[str, list[str]]:
    time_quality = [
        "duration_sec",
        "n_samples",
        "valid_xy_rate",
        "valid_pupil_rate",
        "valid_basic_rate",
        "longest_invalid_run_ratio",
    ]

    position = [
        column for column in raw_features.columns
        if column.startswith("x_")
        or column.startswith("y_")
        or column == "xy_correlation"
    ]

    # Exclude metadata fields that also begin with x_/y_ by using an explicit
    # numeric whitelist.
    position = [
        column for column in position
        if column not in {"x_px", "y_px"}
    ]

    dynamics_prefixes = (
        "dx_",
        "dy_",
        "displacement_",
        "speed_",
        "path_length_",
    )
    dynamics = [
        column for column in raw_features.columns
        if column.startswith(dynamics_prefixes)
    ]

    pupil = [
        column for column in raw_features.columns
        if column.startswith("pupil_")
    ]

    quality_no_duration = [
        "valid_xy_rate",
        "valid_pupil_rate",
        "valid_basic_rate",
        "longest_invalid_run_ratio",
    ]

    raw_no_duration = list(
        dict.fromkeys(
            quality_no_duration + position + dynamics + pupil
        )
    )
    raw_all = list(
        dict.fromkeys(time_quality + position + dynamics + pupil)
    )

    return {
        "time_quality": time_quality,
        "position_only": position,
        "dynamics_only": dynamics,
        "pupil_only": pupil,
        "raw_no_duration": raw_no_duration,
        "raw_all": raw_all,
    }


def create_model(model_name: str, seed: int) -> Any:
    if model_name == "linear_svm":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_name == "dummy_most_frequent":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DummyClassifier(strategy="most_frequent"),
                ),
            ]
        )

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_one_configuration(
    data: pd.DataFrame,
    dataset_name: str,
    feature_set_name: str,
    feature_columns: list[str],
    model_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for held_out_block in range(1, 8):
        train = data[data["block_id"] != held_out_block].copy()
        test = data[data["block_id"] == held_out_block].copy()

        if set(train["label_nr"].unique()) != {0, 1}:
            raise ValueError(
                f"Training fold {held_out_block} lacks both classes."
            )
        if set(test["label_nr"].unique()) != {0, 1}:
            raise ValueError(
                f"Test fold {held_out_block} lacks both classes."
            )

        x_train = train[feature_columns]
        y_train = train["label_nr"].astype(int).to_numpy()
        x_test = test[feature_columns]
        y_test = test["label_nr"].astype(int).to_numpy()

        model = create_model(model_name, seed)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test).astype(int)

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(x_test), dtype=float)
        elif hasattr(model, "predict_proba"):
            scores = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        else:
            scores = predictions.astype(float)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        try:
            roc_auc = roc_auc_score(y_test, scores)
        except ValueError:
            roc_auc = math.nan

        try:
            pr_auc_nr = average_precision_score(y_test, scores)
        except ValueError:
            pr_auc_nr = math.nan

        fold_rows.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "held_out_block": held_out_block,
                "n_train": len(train),
                "n_test": len(test),
                "n_test_nr": int(np.sum(y_test == 1)),
                "n_test_tsr": int(np.sum(y_test == 0)),
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    y_test, predictions
                ),
                "macro_f1": f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "macro_precision": precision_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "macro_recall": recall_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "roc_auc": roc_auc,
                "pr_auc_nr": pr_auc_nr,
                "tn_tsr_correct": int(tn),
                "fp_tsr_as_nr": int(fp),
                "fn_nr_as_tsr": int(fn),
                "tp_nr_correct": int(tp),
            }
        )

        for row_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "held_out_block": held_out_block,
                    "trial_id": row["trial_id"],
                    "task": row["task"],
                    "true_label_nr": int(y_test[row_index]),
                    "predicted_label_nr": int(predictions[row_index]),
                    "decision_score_nr": safe_float(scores[row_index]),
                }
            )

    return fold_rows, prediction_rows


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "roc_auc",
        "pr_auc_nr",
    ]
    group_columns = ["dataset", "feature_set", "model"]

    summary_rows = []
    for keys, group in folds.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        row["n_folds"] = len(group)
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values(
        ["dataset", "model", "macro_f1_mean"],
        ascending=[True, True, False],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "data/interim/zuco2/YAC/"
            "YAC_sample_manifest.csv"
        ),
    )
    parser.add_argument(
        "--all-trials",
        default=(
            "data/interim/zuco2/YAC/"
            "YAC_all_trials.csv"
        ),
    )
    parser.add_argument(
        "--common-map",
        default=(
            "data/processed/zuco2/YAC/"
            "YAC_sent_gaze_sacc_common_trials.csv"
        ),
    )
    parser.add_argument("--subject", default="YAC")
    parser.add_argument(
        "--output-dir",
        default="outputs/raw_baseline/YAC",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse an existing raw-stat feature CSV.",
    )
    args = parser.parse_args()

    subject = args.subject.upper()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / f"{subject}_raw_stat_features.csv"

    if args.reuse_features and feature_path.is_file():
        print(f"Reusing raw features: {feature_path}")
        raw_features = pd.read_csv(feature_path)
    else:
        raw_features = build_feature_table(
            Path(args.manifest),
            feature_path,
        )

    all_trials = pd.read_csv(args.all_trials)
    expected_trial_ids = set(all_trials["trial_id"])
    actual_trial_ids = set(raw_features["trial_id"])

    if actual_trial_ids != expected_trial_ids:
        missing = sorted(expected_trial_ids - actual_trial_ids)
        extra = sorted(actual_trial_ids - expected_trial_ids)
        raise ValueError(
            f"Raw feature/trial mismatch. Missing={missing[:20]}, "
            f"extra={extra[:20]}"
        )

    feature_sets = make_feature_sets(raw_features)

    feature_set_path = output_dir / f"{subject}_feature_sets.json"
    feature_set_path.write_text(
        json.dumps(feature_sets, indent=2),
        encoding="utf-8",
    )

    common_map = pd.read_csv(args.common_map)
    common_ids = set(common_map["trial_id"])
    common_raw = raw_features[
        raw_features["trial_id"].isin(common_ids)
    ].copy()

    if len(common_raw) != len(common_map):
        raise ValueError(
            f"Common raw count mismatch: raw={len(common_raw)}, "
            f"map={len(common_map)}"
        )

    official_columns = sorted(
        [
            column for column in common_map.columns
            if column.startswith("official_feature_")
            and column[len("official_feature_"):].isdigit()
        ],
        key=lambda column: int(column.split("_")[-1]),
    )
    if not official_columns:
        raise ValueError(
            "No official_feature_<index> columns found in common map."
        )

    official_data = common_map[
        [
            "trial_id",
            "subject_id",
            "task",
            "block_id",
            "trial_index_in_block",
        ]
        + official_columns
    ].copy()
    official_data["task"] = official_data["task"].astype(str).str.upper()
    official_data["label_nr"] = official_data["task"].map(LABEL_MAP)

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    raw_datasets = {
        "full_739": raw_features,
        "common_521": common_raw,
    }

    for dataset_name, dataset in raw_datasets.items():
        for feature_set_name, columns in feature_sets.items():
            rows, predictions = evaluate_one_configuration(
                data=dataset,
                dataset_name=dataset_name,
                feature_set_name=feature_set_name,
                feature_columns=columns,
                model_name="linear_svm",
                seed=args.seed,
            )
            fold_rows.extend(rows)
            prediction_rows.extend(predictions)

        # One dummy run is enough because it ignores feature values.
        dummy_columns = feature_sets["time_quality"]
        rows, predictions = evaluate_one_configuration(
            data=dataset,
            dataset_name=dataset_name,
            feature_set_name="dummy_reference",
            feature_columns=dummy_columns,
            model_name="dummy_most_frequent",
            seed=args.seed,
        )
        fold_rows.extend(rows)
        prediction_rows.extend(predictions)

    rows, predictions = evaluate_one_configuration(
        data=official_data,
        dataset_name="common_521",
        feature_set_name="official_sent_gaze_sacc",
        feature_columns=official_columns,
        model_name="linear_svm",
        seed=args.seed,
    )
    fold_rows.extend(rows)
    prediction_rows.extend(predictions)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_folds(folds)

    folds_path = output_dir / f"{subject}_raw_baseline_fold_metrics.csv"
    summary_path = output_dir / f"{subject}_raw_baseline_summary.csv"
    predictions_path = (
        output_dir / f"{subject}_raw_baseline_predictions.csv"
    )

    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== BASELINE SUMMARY ===")
    display_columns = [
        "dataset",
        "feature_set",
        "model",
        "accuracy_mean",
        "accuracy_sd",
        "macro_f1_mean",
        "macro_f1_sd",
        "roc_auc_mean",
        "pr_auc_nr_mean",
    ]
    print(summary[display_columns].to_string(index=False))
    print("\nWritten:")
    print(f"  {feature_path}")
    print(f"  {feature_set_path}")
    print(f"  {folds_path}")
    print(f"  {summary_path}")
    print(f"  {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
