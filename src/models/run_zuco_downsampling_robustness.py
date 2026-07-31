#!/usr/bin/env python
"""Evaluate raw-gaze robustness to lower sampling rates.

This script reuses the feature extraction and LOSO functions from
run_zuco_raw_gaze_loso.py. Continuous 500 Hz trials are deterministically
sampled on a regular time grid using the nearest observed raw sample.

Rates evaluated by default:
    500, 250, 125, 60, 30, 15 Hz

Protocol:
- Rate-matched training and testing.
- 18-fold leave-one-subject-out (LOSO).
- NR=1, TSR=0.
- The same feature definitions and LinearSVC pipeline are used at every rate.
- No fixation, saccade, blink-event, word-AOI, or target-position annotation
  is used.

The 500 Hz condition reuses the already extracted feature table by default.
Lower-rate features are cached separately for every subject and rate, so an
interrupted run can be resumed without recomputing completed caches.

This is a simple-decimation experiment. It intentionally does not low-pass
filter before sampling, because the goal is to approximate the observations
available from a lower-rate tracker rather than reconstruct an ideal
band-limited signal.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_zuco_raw_gaze_loso import (
    build_feature_sets,
    evaluate_loso,
    extract_trial_features,
    summarize_results,
    validate_subject_feature_table,
    validate_trial_metadata,
)


EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS_PER_SUBJECT = 739
BASE_RATE_HZ = 500


def parse_rates(text: str) -> list[int]:
    rates: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        rate = int(item)
        if rate <= 0 or rate > BASE_RATE_HZ:
            raise ValueError(
                f"Sampling rate must be within 1..{BASE_RATE_HZ}: {rate}"
            )
        rates.append(rate)

    rates = sorted(set(rates), reverse=True)
    if BASE_RATE_HZ not in rates:
        rates.insert(0, BASE_RATE_HZ)
    return rates


def nearest_regular_grid_indices(
    timestamps_ms: np.ndarray,
    target_rate_hz: int,
) -> np.ndarray:
    """Select nearest source samples to a regular target-rate grid."""
    timestamps = np.asarray(timestamps_ms, dtype=float)

    if timestamps.ndim != 1 or timestamps.size == 0:
        return np.asarray([], dtype=np.int64)

    finite = np.isfinite(timestamps)
    if not np.all(finite):
        valid_indices = np.flatnonzero(finite)
        if valid_indices.size == 0:
            return np.asarray([], dtype=np.int64)
        timestamps = timestamps[valid_indices]
    else:
        valid_indices = None

    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Timestamps are not monotonically increasing.")

    if target_rate_hz >= BASE_RATE_HZ:
        selected = np.arange(timestamps.size, dtype=np.int64)
    elif timestamps.size == 1:
        selected = np.asarray([0], dtype=np.int64)
    else:
        step_ms = 1000.0 / float(target_rate_hz)
        start = float(timestamps[0])
        end = float(timestamps[-1])

        n_grid = int(math.floor((end - start) / step_ms)) + 1
        grid = start + np.arange(n_grid, dtype=float) * step_ms

        right = np.searchsorted(timestamps, grid, side="left")
        right = np.clip(right, 0, timestamps.size - 1)
        left = np.clip(right - 1, 0, timestamps.size - 1)

        left_distance = np.abs(grid - timestamps[left])
        right_distance = np.abs(timestamps[right] - grid)
        choose_right = right_distance < left_distance
        selected = np.where(choose_right, right, left)
        selected = np.unique(selected.astype(np.int64))

    if valid_indices is not None:
        selected = valid_indices[selected]

    return selected


def downsample_trial(
    trial_group: pd.DataFrame,
    target_rate_hz: int,
) -> pd.DataFrame:
    group = trial_group.sort_values("sample_index_in_trial")
    if target_rate_hz == BASE_RATE_HZ:
        return group

    indices = nearest_regular_grid_indices(
        group["timestamp_ms_abs"].to_numpy(dtype=float),
        target_rate_hz,
    )
    if indices.size == 0:
        raise ValueError("Downsampling produced an empty trial.")

    return group.iloc[indices].copy()


def validate_rate_feature_table(
    features: pd.DataFrame,
    expected_trials: pd.DataFrame,
    subject: str,
    rate_hz: int,
) -> None:
    validate_subject_feature_table(
        features=features,
        expected_trials=expected_trials,
        subject=subject,
    )

    if "sampling_rate_hz" not in features.columns:
        raise ValueError(
            f"{subject} {rate_hz} Hz cache lacks sampling_rate_hz."
        )
    values = pd.to_numeric(
        features["sampling_rate_hz"],
        errors="coerce",
    )
    if not values.eq(rate_hz).all():
        raise ValueError(
            f"{subject} cache contains rates other than {rate_hz} Hz."
        )


def extract_missing_rates_for_subject(
    subject: str,
    subject_manifest: pd.DataFrame,
    subject_trials: pd.DataFrame,
    missing_rates: list[int],
    cache_paths: dict[int, Path],
) -> dict[int, pd.DataFrame]:
    metadata_lookup = subject_trials.set_index(
        "trial_id",
        verify_integrity=True,
    )
    rows_by_rate: dict[int, list[dict[str, Any]]] = {
        rate: [] for rate in missing_rates
    }

    manifest = subject_manifest.sort_values(["task", "block_id"])
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
            f"[{subject} {block_index}/14] Reading once for rates "
            f"{missing_rates}: {parquet_path}"
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

            original_metadata = metadata_lookup.loc[trial_id].copy()
            original_metadata["trial_id"] = trial_id

            for rate_hz in missing_rates:
                sampled = downsample_trial(trial_group, rate_hz)

                metadata = original_metadata.copy()
                metadata["n_samples"] = len(sampled)

                row = extract_trial_features(sampled, metadata)
                row["sampling_rate_hz"] = rate_hz
                row["effective_rate_hz"] = (
                    (len(sampled) - 1)
                    / float(row["duration_sec"])
                    if len(sampled) >= 2
                    and math.isfinite(float(row["duration_sec"]))
                    and float(row["duration_sec"]) > 0
                    else math.nan
                )
                rows_by_rate[rate_hz].append(row)

        del samples

    results: dict[int, pd.DataFrame] = {}
    for rate_hz in missing_rates:
        features = (
            pd.DataFrame(rows_by_rate[rate_hz])
            .sort_values(
                ["task", "block_id", "trial_index_in_block"]
            )
            .reset_index(drop=True)
        )
        validate_rate_feature_table(
            features=features,
            expected_trials=subject_trials,
            subject=subject,
            rate_hz=rate_hz,
        )

        output_path = cache_paths[rate_hz]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        results[rate_hz] = features

    return results


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0

    for rank, original_index in enumerate(order):
        raw = p_values[original_index]
        value = min(1.0, (n - rank) * raw)
        running = max(running, value)
        adjusted[original_index] = running

    return adjusted.tolist()


def build_rate_retention(summary: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy_subject_mean",
        "macro_f1_subject_mean",
        "roc_auc_subject_mean",
        "pr_auc_nr_subject_mean",
        "macro_f1_pooled_oof",
    ]

    reference = summary[
        summary["sampling_rate_hz"].eq(BASE_RATE_HZ)
    ].set_index(["feature_set", "model"])

    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        key = (row.feature_set, row.model)
        if key not in reference.index:
            raise ValueError(f"Missing 500 Hz reference for {key}.")

        ref = reference.loc[key]
        output: dict[str, Any] = {
            "sampling_rate_hz": int(row.sampling_rate_hz),
            "feature_set": row.feature_set,
            "model": row.model,
        }
        for metric in metric_names:
            current = float(getattr(row, metric))
            baseline = float(ref[metric])
            output[metric] = current
            output[f"{metric}_500hz"] = baseline
            output[f"{metric}_difference_vs_500hz"] = current - baseline
            output[f"{metric}_retention_ratio"] = (
                current / baseline
                if math.isfinite(baseline) and baseline != 0
                else math.nan
            )
        rows.append(output)

    return pd.DataFrame(rows).sort_values(
        ["feature_set", "sampling_rate_hz"],
        ascending=[True, False],
    )


def build_paired_rate_tests(
    folds: pd.DataFrame,
) -> pd.DataFrame:
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    rows: list[dict[str, Any]] = []

    for feature_set in sorted(folds["feature_set"].unique()):
        for metric in metrics:
            baseline = (
                folds[
                    folds["feature_set"].eq(feature_set)
                    & folds["sampling_rate_hz"].eq(BASE_RATE_HZ)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )

            row_indices: list[int] = []
            raw_p_values: list[float] = []

            lower_rates = sorted(
                [
                    int(rate)
                    for rate in folds["sampling_rate_hz"].unique()
                    if int(rate) != BASE_RATE_HZ
                ],
                reverse=True,
            )

            for rate_hz in lower_rates:
                current = (
                    folds[
                        folds["feature_set"].eq(feature_set)
                        & folds["sampling_rate_hz"].eq(rate_hz)
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )
                paired = pd.concat(
                    [
                        current.rename("lower_rate"),
                        baseline.rename("rate_500hz"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()

                differences = (
                    paired["lower_rate"] - paired["rate_500hz"]
                )

                if len(paired) == 0:
                    statistic = math.nan
                    p_value = math.nan
                elif np.allclose(differences.to_numpy(), 0.0):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    result = wilcoxon(
                        paired["lower_rate"],
                        paired["rate_500hz"],
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)

                rows.append(
                    {
                        "feature_set": feature_set,
                        "metric": metric,
                        "sampling_rate_hz": rate_hz,
                        "reference_rate_hz": BASE_RATE_HZ,
                        "n_subjects": len(paired),
                        "lower_rate_mean": float(
                            paired["lower_rate"].mean()
                        ),
                        "rate_500hz_mean": float(
                            paired["rate_500hz"].mean()
                        ),
                        "mean_difference": float(
                            differences.mean()
                        ),
                        "median_difference": float(
                            differences.median()
                        ),
                        "wins": int(np.sum(differences > 0)),
                        "ties": int(
                            np.sum(np.isclose(differences, 0))
                        ),
                        "losses": int(np.sum(differences < 0)),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "p_holm_within_feature_metric": math.nan,
                    }
                )
                row_indices.append(len(rows) - 1)
                raw_p_values.append(p_value)

            finite_positions = [
                index
                for index, value in enumerate(raw_p_values)
                if math.isfinite(value)
            ]
            if finite_positions:
                adjusted = holm_adjust(
                    [
                        raw_p_values[index]
                        for index in finite_positions
                    ]
                )
                for local_position, adjusted_value in zip(
                    finite_positions,
                    adjusted,
                ):
                    rows[
                        row_indices[local_position]
                    ][
                        "p_holm_within_feature_metric"
                    ] = adjusted_value

    return pd.DataFrame(rows)


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
        "--base-features",
        default=(
            "outputs/raw_loso/"
            "complete_raw_18_features.csv"
        ),
    )
    parser.add_argument(
        "--rates",
        default="500,250,125,60,30,15",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/downsampling_robustness",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
    )
    parser.add_argument(
        "--recompute-500hz",
        action="store_true",
    )
    args = parser.parse_args()

    rates = parse_rates(args.rates)
    manifest = pd.read_csv(args.manifest)
    trials = validate_trial_metadata(pd.read_csv(args.all_trials))

    manifest["subject_id"] = (
        manifest["subject_id"].astype(str).str.upper()
    )
    manifest["task"] = manifest["task"].astype(str).str.upper()
    manifest["block_id"] = manifest["block_id"].astype(int)

    subjects = sorted(trials["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} subjects, found "
            f"{len(subjects)}: {subjects}"
        )

    output_dir = Path(args.output_dir)
    cache_root = output_dir / "subject_rate_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_features: pd.DataFrame | None = None
    if (
        BASE_RATE_HZ in rates
        and not args.recompute_500hz
    ):
        base_path = Path(args.base_features)
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        base_features = pd.read_csv(base_path)
        base_features["subject_id"] = (
            base_features["subject_id"]
            .astype(str)
            .str.upper()
        )
        base_features["sampling_rate_hz"] = BASE_RATE_HZ
        if "effective_rate_hz" not in base_features.columns:
            base_features["effective_rate_hz"] = (
                (base_features["n_samples"] - 1)
                / base_features["duration_sec"]
            )

    frames_by_rate: dict[int, list[pd.DataFrame]] = {
        rate: [] for rate in rates
    }

    for subject_index, subject in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"Subject {subject_index}/{len(subjects)}: {subject}"
        )

        subject_trials = trials[
            trials["subject_id"].eq(subject)
        ].copy()
        subject_manifest = manifest[
            manifest["subject_id"].eq(subject)
        ].copy()

        cache_paths = {
            rate: (
                cache_root
                / f"{rate:04d}hz"
                / f"{subject}_{rate:04d}hz_features.csv"
            )
            for rate in rates
        }

        missing_rates: list[int] = []

        for rate_hz in rates:
            if (
                rate_hz == BASE_RATE_HZ
                and base_features is not None
            ):
                subject_features = base_features[
                    base_features["subject_id"].eq(subject)
                ].copy()
                validate_rate_feature_table(
                    features=subject_features,
                    expected_trials=subject_trials,
                    subject=subject,
                    rate_hz=rate_hz,
                )
                frames_by_rate[rate_hz].append(subject_features)
                print(
                    f"Reusing existing 500 Hz features for {subject}."
                )
                continue

            cache_path = cache_paths[rate_hz]
            if (
                cache_path.is_file()
                and not args.overwrite_features
            ):
                cached = pd.read_csv(cache_path)
                validate_rate_feature_table(
                    features=cached,
                    expected_trials=subject_trials,
                    subject=subject,
                    rate_hz=rate_hz,
                )
                frames_by_rate[rate_hz].append(cached)
                print(f"Reusing cache: {cache_path}")
            else:
                missing_rates.append(rate_hz)

        if missing_rates:
            extracted = extract_missing_rates_for_subject(
                subject=subject,
                subject_manifest=subject_manifest,
                subject_trials=subject_trials,
                missing_rates=missing_rates,
                cache_paths=cache_paths,
            )
            for rate_hz in missing_rates:
                frames_by_rate[rate_hz].append(
                    extracted[rate_hz]
                )

    combined_by_rate: dict[int, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []

    for rate_hz in rates:
        combined = pd.concat(
            frames_by_rate[rate_hz],
            ignore_index=True,
        )
        if len(combined) != len(trials):
            raise ValueError(
                f"{rate_hz} Hz feature count={len(combined)}, "
                f"expected={len(trials)}"
            )
        if combined["trial_id"].duplicated().any():
            raise ValueError(
                f"{rate_hz} Hz contains duplicate trial IDs."
            )
        if set(combined["trial_id"]) != set(trials["trial_id"]):
            raise ValueError(
                f"{rate_hz} Hz trial IDs do not match metadata."
            )

        combined_by_rate[rate_hz] = combined
        combined_path = (
            output_dir
            / f"complete_raw_18_{rate_hz:04d}hz_features.csv"
        )
        combined.to_csv(
            combined_path,
            index=False,
            encoding="utf-8-sig",
        )

        for subject, group in combined.groupby("subject_id"):
            audit_rows.append(
                {
                    "sampling_rate_hz": rate_hz,
                    "subject_id": subject,
                    "n_trials": len(group),
                    "median_n_samples": float(
                        group["n_samples"].median()
                    ),
                    "median_effective_rate_hz": float(
                        group["effective_rate_hz"].median()
                    ),
                    "median_valid_basic_rate": float(
                        group["valid_basic_rate"].median()
                    ),
                }
            )

    feature_sets = build_feature_sets()
    missing_columns = {
        rate_hz: {
            name: sorted(set(columns) - set(data.columns))
            for name, columns in feature_sets.items()
            if set(columns) - set(data.columns)
        }
        for rate_hz, data in combined_by_rate.items()
    }
    missing_columns = {
        rate: value
        for rate, value in missing_columns.items()
        if value
    }
    if missing_columns:
        raise ValueError(
            f"Missing feature columns by rate: {missing_columns}"
        )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for rate_hz in rates:
        dataset_name = f"rate_{rate_hz:04d}hz"
        data = combined_by_rate[rate_hz]

        print("\n" + "#" * 78)
        print(
            f"LOSO at {rate_hz} Hz: "
            f"subjects={data['subject_id'].nunique()}, "
            f"trials={len(data)}"
        )

        for feature_set_name, columns in feature_sets.items():
            model_name = (
                "dummy_most_frequent"
                if feature_set_name == "dummy_reference"
                else "linear_svm"
            )
            rows, predictions = evaluate_loso(
                data=data,
                dataset_name=dataset_name,
                feature_set_name=feature_set_name,
                feature_columns=columns,
                model_name=model_name,
                seed=args.seed,
            )
            for row in rows:
                row["sampling_rate_hz"] = rate_hz
            for row in predictions:
                row["sampling_rate_hz"] = rate_hz
            fold_rows.extend(rows)
            prediction_rows.extend(predictions)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_results(folds, predictions)

    summary["sampling_rate_hz"] = (
        summary["dataset"]
        .str.extract(r"rate_(\d+)hz")[0]
        .astype(int)
    )
    summary = summary.sort_values(
        ["feature_set", "sampling_rate_hz"],
        ascending=[True, False],
    )

    retention = build_rate_retention(summary)
    paired_tests = build_paired_rate_tests(folds)
    audit = pd.DataFrame(audit_rows).sort_values(
        ["sampling_rate_hz", "subject_id"],
        ascending=[False, True],
    )

    audit_path = output_dir / "downsampling_rate_audit.csv"
    folds_path = output_dir / "downsampling_loso_fold_metrics.csv"
    predictions_path = (
        output_dir / "downsampling_loso_predictions.csv"
    )
    summary_path = output_dir / "downsampling_loso_summary.csv"
    retention_path = (
        output_dir / "downsampling_retention_vs_500hz.csv"
    )
    paired_path = (
        output_dir / "downsampling_paired_tests_vs_500hz.csv"
    )

    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
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
    retention.to_csv(
        retention_path,
        index=False,
        encoding="utf-8-sig",
    )
    paired_tests.to_csv(
        paired_path,
        index=False,
        encoding="utf-8-sig",
    )

    important_sets = {
        "dummy_reference",
        "time_only",
        "quality_only",
        "gaze_relative",
        "pupil_relative",
        "strict_relative",
        "strict_absolute",
    }
    display = summary[
        summary["feature_set"].isin(important_sets)
    ][
        [
            "sampling_rate_hz",
            "feature_set",
            "n_folds",
            "accuracy_subject_mean",
            "macro_f1_subject_mean",
            "macro_f1_subject_sd",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
            "macro_f1_pooled_oof",
        ]
    ].sort_values(
        ["sampling_rate_hz", "macro_f1_subject_mean"],
        ascending=[False, False],
    )

    print("\n=== DOWNSAMPLING ROBUSTNESS SUMMARY ===")
    print(display.to_string(index=False))
    print("\nWritten:")
    print(f"  {audit_path}")
    print(f"  {folds_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    print(f"  {retention_path}")
    print(f"  {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
