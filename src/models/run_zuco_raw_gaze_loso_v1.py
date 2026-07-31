#!/usr/bin/env python
"""Extract fixation-free raw-gaze features and run subject-wise LOSO.

This script is the first multi-subject evaluation for the continuous ZuCo 2.0
eye-tracking data prepared by the preceding export pipeline.

Evaluation datasets
-------------------
raw_18:
    All 18 subjects with complete NR/TSR raw ET recordings.

official_overlap_16:
    The 16 raw-ET subjects that also have local official sent_gaze_sacc
    features. This keeps a subject-matched raw-gaze result for later comparison.

Evaluation protocol
-------------------
Leave-one-subject-out (LOSO). All 739 trials from one subject are held out,
and every trial from that subject is excluded from training.

Positive class:
    NR = 1
    TSR = 0

No fixation, saccade, blink-event, word-AOI, or target-position annotations are
used to construct the raw-gaze features.

The script caches one compact feature CSV per subject. If execution is
interrupted, rerunning the same command reuses validated subject feature files.
Use --overwrite-features to force recomputation.

Example
-------
python src/models/run_zuco_raw_gaze_loso.py \
  --manifest data/interim/zuco2/all_subjects/complete_raw_sample_manifest.csv \
  --all-trials data/interim/zuco2/all_subjects/complete_raw_all_trials.csv \
  --output-dir outputs/raw_loso \
  --seed 1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

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
from sklearn.svm import LinearSVC


LABEL_MAP = {"NR": 1, "TSR": 0}
EXPECTED_TRIALS_PER_SUBJECT = 739
EXPECTED_NR_PER_SUBJECT = 349
EXPECTED_TSR_PER_SUBJECT = 390


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def safe_mean(values: np.ndarray) -> float:
    values = finite_values(values)
    return safe_float(np.mean(values)) if values.size else math.nan


def safe_std(values: np.ndarray) -> float:
    values = finite_values(values)
    return safe_float(np.std(values, ddof=0)) if values.size else math.nan


def safe_quantile(values: np.ndarray, q: float) -> float:
    values = finite_values(values)
    return (
        safe_float(np.quantile(values, q))
        if values.size
        else math.nan
    )


def safe_ratio(
    numerator: float,
    denominator: float,
    epsilon: float = 1e-8,
) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or abs(denominator) <= epsilon
    ):
        return math.nan
    return numerator / abs(denominator)


def longest_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def adjacent_differences(
    values: np.ndarray,
    timestamps_ms: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differences for adjacent valid samples only."""
    values = np.asarray(values, dtype=float)
    timestamps_ms = np.asarray(timestamps_ms, dtype=float)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(values)
        & np.isfinite(timestamps_ms)
    )

    indices = np.flatnonzero(valid)
    if indices.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    adjacent = np.diff(indices) == 1
    if not np.any(adjacent):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    left = indices[:-1][adjacent]
    right = indices[1:][adjacent]
    differences = values[right] - values[left]
    dt_ms = timestamps_ms[right] - timestamps_ms[left]

    keep = (
        np.isfinite(differences)
        & np.isfinite(dt_ms)
        & (dt_ms > 0)
    )
    return differences[keep], dt_ms[keep]


def validate_trial_metadata(trials: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "block_id",
        "trial_index_in_block",
        "duration_sec",
        "n_samples",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Missing trial metadata columns: {missing}")

    result = trials.copy()
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()
    result["block_id"] = result["block_id"].astype(int)
    result["trial_index_in_block"] = (
        result["trial_index_in_block"].astype(int)
    )
    result["n_samples"] = result["n_samples"].astype(int)

    if result["trial_id"].duplicated().any():
        duplicate_ids = result.loc[
            result["trial_id"].duplicated(keep=False),
            "trial_id",
        ].tolist()
        raise ValueError(
            f"Duplicate trial metadata IDs: {duplicate_ids[:20]}"
        )

    for subject, group in result.groupby("subject_id"):
        task_counts = group.groupby("task").size().to_dict()
        nr = int(task_counts.get("NR", 0))
        tsr = int(task_counts.get("TSR", 0))
        if (
            len(group) != EXPECTED_TRIALS_PER_SUBJECT
            or nr != EXPECTED_NR_PER_SUBJECT
            or tsr != EXPECTED_TSR_PER_SUBJECT
        ):
            raise ValueError(
                f"{subject} trial counts are invalid: "
                f"total={len(group)}, NR={nr}, TSR={tsr}"
            )

    return result


def extract_trial_features(
    group: pd.DataFrame,
    metadata: pd.Series,
) -> dict[str, Any]:
    group = group.sort_values("sample_index_in_trial")

    t = group["timestamp_ms_abs"].to_numpy(dtype=float)
    x = group["x_px"].to_numpy(dtype=float)
    y = group["y_px"].to_numpy(dtype=float)
    pupil = group["pupil_area"].to_numpy(dtype=float)

    expected_samples = int(metadata["n_samples"])
    if len(group) != expected_samples:
        raise ValueError(
            f"{metadata['trial_id']} sample mismatch: "
            f"Parquet={len(group)}, metadata={expected_samples}"
        )

    finite_xy = np.isfinite(x) & np.isfinite(y)
    valid_xy = finite_xy & (x > 0) & (y > 0)
    valid_pupil = np.isfinite(pupil) & (pupil > 0)
    valid_basic = valid_xy & valid_pupil

    x_valid = x[valid_xy]
    y_valid = y[valid_xy]
    pupil_valid = pupil[valid_pupil]

    dx, dt_x = adjacent_differences(x, t, valid_xy)
    dy, dt_y = adjacent_differences(y, t, valid_xy)
    n_xy = min(len(dx), len(dy), len(dt_x), len(dt_y))
    dx = dx[:n_xy]
    dy = dy[:n_xy]
    dt_xy = dt_x[:n_xy]

    if n_xy:
        displacement = np.hypot(dx, dy)
        speed_px_s = displacement / (dt_xy / 1000.0)
    else:
        displacement = np.asarray([], dtype=float)
        speed_px_s = np.asarray([], dtype=float)

    dpupil, _ = adjacent_differences(
        pupil,
        t,
        valid_pupil,
    )

    if len(t) >= 2:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        median_dt_ms = (
            safe_float(np.median(dt))
            if dt.size
            else math.nan
        )
    else:
        median_dt_ms = math.nan

    duration_sec = safe_float(metadata["duration_sec"])
    longest_invalid_samples = longest_true_run(~valid_basic)
    longest_invalid_ms = (
        longest_invalid_samples * median_dt_ms
        if math.isfinite(median_dt_ms)
        else math.nan
    )

    result: dict[str, Any] = {
        "trial_id": str(metadata["trial_id"]),
        "subject_id": str(metadata["subject_id"]).upper(),
        "task": str(metadata["task"]).upper(),
        "block_id": int(metadata["block_id"]),
        "trial_index_in_block": int(
            metadata["trial_index_in_block"]
        ),
        "marker_pair": str(
            metadata.get("marker_pair", "")
        ),
        "label_nr": LABEL_MAP[str(metadata["task"]).upper()],
        "duration_sec": duration_sec,
        "n_samples": expected_samples,
        "median_dt_ms": median_dt_ms,
        "valid_xy_rate": safe_float(np.mean(valid_xy)),
        "valid_pupil_rate": safe_float(np.mean(valid_pupil)),
        "valid_basic_rate": safe_float(np.mean(valid_basic)),
        "longest_invalid_run_ratio": (
            longest_invalid_ms / (duration_sec * 1000.0)
            if math.isfinite(longest_invalid_ms)
            and math.isfinite(duration_sec)
            and duration_sec > 0
            else math.nan
        ),
    }

    position_quantiles: dict[str, dict[str, float]] = {}
    for name, values in (("x", x_valid), ("y", y_valid)):
        q05 = safe_quantile(values, 0.05)
        q25 = safe_quantile(values, 0.25)
        q50 = safe_quantile(values, 0.50)
        q75 = safe_quantile(values, 0.75)
        q95 = safe_quantile(values, 0.95)
        position_quantiles[name] = {
            "q05": q05,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q95": q95,
        }
        result.update(
            {
                f"{name}_mean": safe_mean(values),
                f"{name}_std": safe_std(values),
                f"{name}_q05": q05,
                f"{name}_q25": q25,
                f"{name}_median": q50,
                f"{name}_q75": q75,
                f"{name}_q95": q95,
                f"{name}_iqr": (
                    q75 - q25
                    if math.isfinite(q75)
                    and math.isfinite(q25)
                    else math.nan
                ),
                f"{name}_range90": (
                    q95 - q05
                    if math.isfinite(q95)
                    and math.isfinite(q05)
                    else math.nan
                ),
            }
        )

    if (
        len(x_valid) >= 2
        and len(x_valid) == len(y_valid)
        and safe_std(x_valid) > 0
        and safe_std(y_valid) > 0
    ):
        result["xy_correlation"] = safe_float(
            np.corrcoef(x_valid, y_valid)[0, 1]
        )
    else:
        result["xy_correlation"] = math.nan

    result.update(
        {
            "dx_mean": safe_mean(dx),
            "dx_std": safe_std(dx),
            "dx_abs_mean": safe_mean(np.abs(dx)),
            "dx_abs_median": safe_quantile(np.abs(dx), 0.50),
            "dx_abs_p90": safe_quantile(np.abs(dx), 0.90),
            "dx_p10": safe_quantile(dx, 0.10),
            "dx_p90": safe_quantile(dx, 0.90),
            "dx_positive_rate": (
                safe_float(np.mean(dx > 0))
                if dx.size
                else math.nan
            ),
            "dx_negative_rate": (
                safe_float(np.mean(dx < 0))
                if dx.size
                else math.nan
            ),
            "dy_mean": safe_mean(dy),
            "dy_std": safe_std(dy),
            "dy_abs_mean": safe_mean(np.abs(dy)),
            "dy_abs_median": safe_quantile(np.abs(dy), 0.50),
            "dy_abs_p90": safe_quantile(np.abs(dy), 0.90),
            "dy_p10": safe_quantile(dy, 0.10),
            "dy_p90": safe_quantile(dy, 0.90),
            "displacement_mean": safe_mean(displacement),
            "displacement_median": safe_quantile(
                displacement,
                0.50,
            ),
            "displacement_p90": safe_quantile(
                displacement,
                0.90,
            ),
            "speed_mean_px_s": safe_mean(speed_px_s),
            "speed_median_px_s": safe_quantile(
                speed_px_s,
                0.50,
            ),
            "speed_p90_px_s": safe_quantile(
                speed_px_s,
                0.90,
            ),
            "speed_p95_px_s": safe_quantile(
                speed_px_s,
                0.95,
            ),
        }
    )

    pupil_q05 = safe_quantile(pupil_valid, 0.05)
    pupil_q25 = safe_quantile(pupil_valid, 0.25)
    pupil_q50 = safe_quantile(pupil_valid, 0.50)
    pupil_q75 = safe_quantile(pupil_valid, 0.75)
    pupil_q95 = safe_quantile(pupil_valid, 0.95)
    pupil_std = safe_std(pupil_valid)
    pupil_iqr = (
        pupil_q75 - pupil_q25
        if math.isfinite(pupil_q75)
        and math.isfinite(pupil_q25)
        else math.nan
    )
    pupil_range90 = (
        pupil_q95 - pupil_q05
        if math.isfinite(pupil_q95)
        and math.isfinite(pupil_q05)
        else math.nan
    )
    pupil_diff_abs_mean = safe_mean(np.abs(dpupil))
    pupil_diff_abs_median = safe_quantile(
        np.abs(dpupil),
        0.50,
    )
    pupil_diff_abs_p90 = safe_quantile(
        np.abs(dpupil),
        0.90,
    )

    result.update(
        {
            "pupil_mean": safe_mean(pupil_valid),
            "pupil_std": pupil_std,
            "pupil_q05": pupil_q05,
            "pupil_q25": pupil_q25,
            "pupil_median": pupil_q50,
            "pupil_q75": pupil_q75,
            "pupil_q95": pupil_q95,
            "pupil_iqr": pupil_iqr,
            "pupil_range90": pupil_range90,
            "pupil_diff_abs_mean": pupil_diff_abs_mean,
            "pupil_diff_abs_median": pupil_diff_abs_median,
            "pupil_diff_abs_p90": pupil_diff_abs_p90,
            "pupil_std_rel": safe_ratio(
                pupil_std,
                pupil_q50,
            ),
            "pupil_iqr_rel": safe_ratio(
                pupil_iqr,
                pupil_q50,
            ),
            "pupil_range90_rel": safe_ratio(
                pupil_range90,
                pupil_q50,
            ),
            "pupil_diff_abs_mean_rel": safe_ratio(
                pupil_diff_abs_mean,
                pupil_q50,
            ),
            "pupil_diff_abs_median_rel": safe_ratio(
                pupil_diff_abs_median,
                pupil_q50,
            ),
            "pupil_diff_abs_p90_rel": safe_ratio(
                pupil_diff_abs_p90,
                pupil_q50,
            ),
        }
    )

    return result


def validate_subject_feature_table(
    features: pd.DataFrame,
    expected_trials: pd.DataFrame,
    subject: str,
) -> None:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "label_nr",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(
            f"{subject} cached feature columns missing: {missing}"
        )

    if len(features) != EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(
            f"{subject} cached features contain {len(features)} rows."
        )
    if features["trial_id"].duplicated().any():
        raise ValueError(
            f"{subject} cached features contain duplicate trial IDs."
        )

    feature_ids = set(features["trial_id"])
    expected_ids = set(expected_trials["trial_id"])
    if feature_ids != expected_ids:
        missing_ids = sorted(expected_ids - feature_ids)
        extra_ids = sorted(feature_ids - expected_ids)
        raise ValueError(
            f"{subject} cached trial mismatch: "
            f"missing={missing_ids[:10]}, extra={extra_ids[:10]}"
        )

    task_counts = features.groupby("task").size().to_dict()
    if (
        int(task_counts.get("NR", 0)) != EXPECTED_NR_PER_SUBJECT
        or int(task_counts.get("TSR", 0))
        != EXPECTED_TSR_PER_SUBJECT
    ):
        raise ValueError(
            f"{subject} cached task counts are invalid: "
            f"{task_counts}"
        )


def extract_subject_features(
    subject: str,
    subject_manifest: pd.DataFrame,
    subject_trials: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    metadata_lookup = subject_trials.set_index(
        "trial_id",
        verify_integrity=True,
    )
    rows: list[dict[str, Any]] = []

    manifest = subject_manifest.sort_values(
        ["task", "block_id"]
    )
    if len(manifest) != 14:
        raise ValueError(
            f"{subject} manifest has {len(manifest)} rows, expected 14."
        )

    for block_index, manifest_row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        parquet_path = Path(str(manifest_row.sample_parquet))
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)

        print(
            f"[{subject} {block_index}/14] "
            f"Extracting: {parquet_path}"
        )
        samples = pd.read_parquet(parquet_path)

        for trial_id, trial_group in samples.groupby(
            "trial_id",
            sort=False,
        ):
            if trial_id not in metadata_lookup.index:
                raise ValueError(
                    f"Parquet trial missing from metadata: {trial_id}"
                )
            rows.append(
                extract_trial_features(
                    trial_group,
                    metadata_lookup.loc[trial_id],
                )
            )

        del samples

    features = pd.DataFrame(rows).sort_values(
        ["task", "block_id", "trial_index_in_block"]
    ).reset_index(drop=True)

    validate_subject_feature_table(
        features,
        subject_trials,
        subject,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return features


def build_feature_sets() -> dict[str, list[str]]:
    time_only = ["duration_sec", "n_samples"]

    quality_only = [
        "valid_xy_rate",
        "valid_pupil_rate",
        "valid_basic_rate",
        "longest_invalid_run_ratio",
    ]

    position_location = [
        "x_mean",
        "x_q05",
        "x_q25",
        "x_median",
        "x_q75",
        "x_q95",
        "y_mean",
        "y_q05",
        "y_q25",
        "y_median",
        "y_q75",
        "y_q95",
    ]

    position_dispersion = [
        "x_std",
        "x_iqr",
        "x_range90",
        "y_std",
        "y_iqr",
        "y_range90",
        "xy_correlation",
    ]

    dynamics_local = [
        "dx_mean",
        "dx_std",
        "dx_abs_mean",
        "dx_abs_median",
        "dx_abs_p90",
        "dx_p10",
        "dx_p90",
        "dx_positive_rate",
        "dx_negative_rate",
        "dy_mean",
        "dy_std",
        "dy_abs_mean",
        "dy_abs_median",
        "dy_abs_p90",
        "dy_p10",
        "dy_p90",
        "displacement_mean",
        "displacement_median",
        "displacement_p90",
        "speed_mean_px_s",
        "speed_median_px_s",
        "speed_p90_px_s",
        "speed_p95_px_s",
    ]

    pupil_level = [
        "pupil_mean",
        "pupil_q05",
        "pupil_q25",
        "pupil_median",
        "pupil_q75",
        "pupil_q95",
    ]

    pupil_variation_absolute = [
        "pupil_std",
        "pupil_iqr",
        "pupil_range90",
        "pupil_diff_abs_mean",
        "pupil_diff_abs_median",
        "pupil_diff_abs_p90",
    ]

    pupil_relative = [
        "pupil_std_rel",
        "pupil_iqr_rel",
        "pupil_range90_rel",
        "pupil_diff_abs_mean_rel",
        "pupil_diff_abs_median_rel",
        "pupil_diff_abs_p90_rel",
    ]

    gaze_relative = list(
        dict.fromkeys(
            quality_only
            + position_dispersion
            + dynamics_local
        )
    )

    strict_relative = list(
        dict.fromkeys(gaze_relative + pupil_relative)
    )

    strict_absolute = list(
        dict.fromkeys(
            quality_only
            + position_location
            + position_dispersion
            + dynamics_local
            + pupil_level
            + pupil_variation_absolute
        )
    )

    return {
        "dummy_reference": quality_only,
        "time_only": time_only,
        "quality_only": quality_only,
        "position_location": position_location,
        "position_dispersion": position_dispersion,
        "dynamics_local": dynamics_local,
        "gaze_relative": gaze_relative,
        "pupil_level": pupil_level,
        "pupil_relative": pupil_relative,
        "strict_relative": strict_relative,
        "strict_absolute": strict_absolute,
    }


def make_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "linear_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        C=1.0,
                        dual=False,
                        max_iter=20000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_name == "dummy_most_frequent":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DummyClassifier(
                        strategy="most_frequent",
                    ),
                ),
            ]
        )

    raise ValueError(model_name)


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = math.nan

    try:
        pr_auc_nr = float(
            average_precision_score(y_true, scores)
        )
    except ValueError:
        pr_auc_nr = math.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc,
        "pr_auc_nr": pr_auc_nr,
        "tn_tsr_correct": int(tn),
        "fp_tsr_as_nr": int(fp),
        "fn_nr_as_tsr": int(fn),
        "tp_nr_correct": int(tp),
    }


def evaluate_loso(
    data: pd.DataFrame,
    dataset_name: str,
    feature_set_name: str,
    feature_columns: list[str],
    model_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    subjects = sorted(data["subject_id"].unique())
    if len(subjects) < 2:
        raise ValueError(
            f"{dataset_name} needs at least two subjects."
        )

    for fold_index, held_out_subject in enumerate(
        subjects,
        start=1,
    ):
        train = data[
            data["subject_id"] != held_out_subject
        ].copy()
        test = data[
            data["subject_id"] == held_out_subject
        ].copy()

        y_train = train["label_nr"].astype(int).to_numpy()
        y_test = test["label_nr"].astype(int).to_numpy()

        if set(np.unique(y_train)) != {0, 1}:
            raise ValueError(
                f"Training fold lacks both classes: "
                f"{held_out_subject}"
            )
        if set(np.unique(y_test)) != {0, 1}:
            raise ValueError(
                f"Test fold lacks both classes: "
                f"{held_out_subject}"
            )

        model = make_model(model_name, seed)
        model.fit(train[feature_columns], y_train)
        y_pred = model.predict(
            test[feature_columns]
        ).astype(int)

        if hasattr(model, "decision_function"):
            scores = np.asarray(
                model.decision_function(
                    test[feature_columns]
                ),
                dtype=float,
            )
        elif hasattr(model, "predict_proba"):
            scores = np.asarray(
                model.predict_proba(
                    test[feature_columns]
                )[:, 1],
                dtype=float,
            )
        else:
            scores = y_pred.astype(float)

        metrics = score_predictions(
            y_test,
            y_pred,
            scores,
        )

        fold_rows.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "fold_index": fold_index,
                "held_out_subject": held_out_subject,
                "n_train_subjects": len(subjects) - 1,
                "n_train": len(train),
                "n_test": len(test),
                "n_test_nr": int(np.sum(y_test == 1)),
                "n_test_tsr": int(np.sum(y_test == 0)),
                **metrics,
            }
        )

        for row_index, (_, row) in enumerate(
            test.iterrows()
        ):
            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "fold_index": fold_index,
                    "held_out_subject": held_out_subject,
                    "trial_id": row["trial_id"],
                    "task": row["task"],
                    "true_label_nr": int(y_test[row_index]),
                    "predicted_label_nr": int(y_pred[row_index]),
                    "score_nr": safe_float(scores[row_index]),
                }
            )

        print(
            f"{dataset_name} | {feature_set_name} | "
            f"test={held_out_subject} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    return fold_rows, prediction_rows


def summarize_results(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "roc_auc",
        "pr_auc_nr",
    ]

    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "feature_set", "model"]

    for keys, fold_group in folds.groupby(
        group_columns,
        sort=False,
    ):
        dataset, feature_set, model = keys
        pred_group = predictions[
            (predictions["dataset"] == dataset)
            & (
                predictions["feature_set"]
                == feature_set
            )
            & (predictions["model"] == model)
        ]

        y_true = (
            pred_group["true_label_nr"]
            .astype(int)
            .to_numpy()
        )
        y_pred = (
            pred_group["predicted_label_nr"]
            .astype(int)
            .to_numpy()
        )
        scores = (
            pred_group["score_nr"]
            .astype(float)
            .to_numpy()
        )
        pooled = score_predictions(
            y_true,
            y_pred,
            scores,
        )

        row: dict[str, Any] = {
            "dataset": dataset,
            "feature_set": feature_set,
            "model": model,
            "n_folds": len(fold_group),
            "n_oof_predictions": len(pred_group),
        }

        for metric in metric_names:
            values = pd.to_numeric(
                fold_group[metric],
                errors="coerce",
            )
            row[f"{metric}_subject_mean"] = float(
                values.mean()
            )
            row[f"{metric}_subject_sd"] = float(
                values.std(ddof=1)
            )
            row[f"{metric}_pooled_oof"] = float(
                pooled[metric]
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["dataset", "macro_f1_subject_mean"],
        ascending=[True, False],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "data/interim/zuco2/all_subjects/"
            "complete_raw_sample_manifest.csv"
        ),
    )
    parser.add_argument(
        "--all-trials",
        default=(
            "data/interim/zuco2/all_subjects/"
            "complete_raw_all_trials.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/raw_loso",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    trials = validate_trial_metadata(
        pd.read_csv(args.all_trials)
    )

    manifest["subject_id"] = (
        manifest["subject_id"]
        .astype(str)
        .str.upper()
    )
    manifest["task"] = (
        manifest["task"].astype(str).str.upper()
    )
    manifest["block_id"] = (
        manifest["block_id"].astype(int)
    )

    subjects = sorted(trials["subject_id"].unique())
    if len(subjects) != 18:
        raise ValueError(
            f"Expected 18 subjects, found {len(subjects)}: "
            f"{subjects}"
        )

    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "subject_features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    subject_feature_frames: list[pd.DataFrame] = []

    for subject_index, subject in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 72)
        print(
            f"Subject {subject_index}/{len(subjects)}: "
            f"{subject}"
        )

        subject_trials = trials[
            trials["subject_id"] == subject
        ].copy()
        subject_manifest = manifest[
            manifest["subject_id"] == subject
        ].copy()
        feature_path = (
            feature_dir
            / f"{subject}_raw_relative_features.csv"
        )

        if feature_path.is_file() and not args.overwrite_features:
            print(f"Reusing cached features: {feature_path}")
            features = pd.read_csv(feature_path)
            validate_subject_feature_table(
                features,
                subject_trials,
                subject,
            )
        else:
            features = extract_subject_features(
                subject=subject,
                subject_manifest=subject_manifest,
                subject_trials=subject_trials,
                output_path=feature_path,
            )

        subject_feature_frames.append(features)

    all_features = pd.concat(
        subject_feature_frames,
        ignore_index=True,
    )

    if len(all_features) != len(trials):
        raise ValueError(
            f"Combined feature count={len(all_features)}, "
            f"expected={len(trials)}"
        )
    if all_features["trial_id"].duplicated().any():
        raise ValueError(
            "Combined features contain duplicate trial IDs."
        )
    if set(all_features["trial_id"]) != set(
        trials["trial_id"]
    ):
        raise ValueError(
            "Combined feature trial IDs do not match metadata."
        )

    official_flags = (
        trials[
            [
                "trial_id",
                "has_local_official_sent_gaze_sacc",
            ]
        ]
        if "has_local_official_sent_gaze_sacc"
        in trials.columns
        else None
    )
    if official_flags is None:
        raise ValueError(
            "all-trials CSV lacks "
            "has_local_official_sent_gaze_sacc."
        )

    all_features = all_features.merge(
        official_flags,
        on="trial_id",
        how="left",
        validate="one_to_one",
    )
    all_features[
        "has_local_official_sent_gaze_sacc"
    ] = pd.to_numeric(
        all_features[
            "has_local_official_sent_gaze_sacc"
        ],
        errors="coerce",
    ).fillna(0).astype(int)

    combined_features_path = (
        output_dir
        / "complete_raw_18_features.csv"
    )
    all_features.to_csv(
        combined_features_path,
        index=False,
        encoding="utf-8-sig",
    )

    feature_sets = build_feature_sets()
    missing_features = {
        name: sorted(
            set(columns) - set(all_features.columns)
        )
        for name, columns in feature_sets.items()
        if set(columns) - set(all_features.columns)
    }
    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    feature_sets_path = output_dir / "feature_sets.json"
    feature_sets_path.write_text(
        json.dumps(
            feature_sets,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    datasets = {
        "raw_18": all_features,
        "official_overlap_16": all_features[
            all_features[
                "has_local_official_sent_gaze_sacc"
            ].eq(1)
        ].copy(),
    }

    if (
        datasets["raw_18"]["subject_id"].nunique()
        != 18
    ):
        raise ValueError("raw_18 does not contain 18 subjects.")
    if (
        datasets["official_overlap_16"][
            "subject_id"
        ].nunique()
        != 16
    ):
        raise ValueError(
            "official_overlap_16 does not contain 16 subjects."
        )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for dataset_name, dataset in datasets.items():
        print("\n" + "#" * 72)
        print(
            f"LOSO dataset: {dataset_name}, "
            f"subjects={dataset['subject_id'].nunique()}, "
            f"trials={len(dataset)}"
        )

        for feature_set_name, columns in feature_sets.items():
            model_name = (
                "dummy_most_frequent"
                if feature_set_name == "dummy_reference"
                else "linear_svm"
            )
            rows, predictions = evaluate_loso(
                data=dataset,
                dataset_name=dataset_name,
                feature_set_name=feature_set_name,
                feature_columns=columns,
                model_name=model_name,
                seed=args.seed,
            )
            fold_rows.extend(rows)
            prediction_rows.extend(predictions)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_results(
        folds,
        predictions,
    )

    folds_path = output_dir / "raw_loso_fold_metrics.csv"
    predictions_path = (
        output_dir / "raw_loso_predictions.csv"
    )
    summary_path = output_dir / "raw_loso_summary.csv"

    folds.to_csv(
        folds_path,
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "dataset",
        "feature_set",
        "model",
        "n_folds",
        "accuracy_subject_mean",
        "macro_f1_subject_mean",
        "macro_f1_subject_sd",
        "roc_auc_subject_mean",
        "pr_auc_nr_subject_mean",
        "macro_f1_pooled_oof",
    ]

    print("\n=== RAW GAZE LOSO SUMMARY ===")
    print(
        summary[display_columns].to_string(index=False)
    )
    print("\nWritten:")
    print(f"  {combined_features_path}")
    print(f"  {feature_sets_path}")
    print(f"  {folds_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
