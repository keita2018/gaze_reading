#!/usr/bin/env python
"""Compare clean-trained Soft-AOI and Hard-AOI under 40% missingness.

Protocol
--------
- Clean 30 Hz training data.
- Test data with deterministic additional missingness.
- Modes: random and one-second burst.
- Three corruption repeats by default.
- Dual holdout: unseen subject + unseen sentence fold.
- Fixed LinearSVC C=0.01.
- Same 25 comparable features for Soft-AOI and Hard-AOI.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from extract_zuco_soft_aoi_features import (  # noqa: E402
    downsample_trial as soft_downsample_trial,
)
from run_zuco_hard_aoi_all_subjects import (  # noqa: E402
    HARD_COMMON25,
    HARD_DISTRIBUTION_COMMON,
    HARD_QUALITY,
    extract_hard_trial_features,
    load_trial_map,
)
from run_zuco_missingness_30hz import (  # noqa: E402
    stable_seed as missingness_seed,
)
from run_zuco_soft_aoi_ablation import (  # noqa: E402
    holm_adjust,
    load_sentence_folds,
)
from run_zuco_soft_aoi_confound_controls import (  # noqa: E402
    METRICS,
    decision_scores,
    load_merged_data,
    make_svm,
    score_predictions,
)
from run_zuco_soft_aoi_coordinate_noise_30hz import (  # noqa: E402
    load_word_template,
)
from run_zuco_soft_aoi_missingness_30hz import (  # noqa: E402
    MODES,
    degrade_samples,
    soft_cache_path,
    validate_soft_cache,
)
from run_zuco_soft_vs_hard_aoi_clean_30hz import (  # noqa: E402
    HARD_NAME,
    SOFT_COMMON_DISTRIBUTION,
    SOFT_NAME,
    SOFT_QUALITY,
    add_hard_features,
    audit_shared_quality,
    build_feature_sets,
    normalize_hard_table,
)


RATE_HZ = 30
EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_PER_SUBJECT = 739


def parse_percentages(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values or any(value <= 0 or value >= 100 for value in values):
        raise ValueError("--percentages must be within (0, 100).")
    return values


def hard_cache_path(
    root: Path,
    subject: str,
    mode: str,
    percentage: float,
    repeat: int,
) -> Path:
    return (
        root
        / mode
        / f"missing_{percentage:05.1f}pct"
        / f"repeat_{repeat:02d}"
        / (
            f"{subject}_hard_aoi_{mode}_{percentage:05.1f}pct"
            f"_repeat_{repeat:02d}.csv"
        )
    )


def validate_hard_cache(
    frame: pd.DataFrame,
    subject: str,
    expected_ids: set[str],
    mode: str,
    percentage: float,
    repeat: int,
) -> pd.DataFrame:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "task_sentence_index",
        "missingness_mode",
        "additional_missing_percent",
        "corruption_repeat",
        *HARD_COMMON25,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{subject} Hard-AOI missingness cache lacks: {missing}"
        )

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = result["subject_id"].astype(str).str.upper()
    result["task"] = result["task"].astype(str).str.upper()
    result["missingness_mode"] = (
        result["missingness_mode"].astype(str).str.lower()
    )

    if len(result) != EXPECTED_PER_SUBJECT:
        raise ValueError(
            f"{subject} Hard cache rows={len(result)}, "
            f"expected={EXPECTED_PER_SUBJECT}."
        )
    if result["trial_id"].duplicated().any():
        raise ValueError(f"{subject} Hard cache has duplicate trial IDs.")
    if set(result["trial_id"]) != expected_ids:
        raise ValueError(f"{subject} Hard cache trial IDs do not match.")
    if set(result["subject_id"]) != {subject}:
        raise ValueError(f"{subject} Hard cache subject is invalid.")
    if set(result["missingness_mode"]) != {mode}:
        raise ValueError(f"{subject} Hard cache mode is invalid.")
    if not np.allclose(
        pd.to_numeric(
            result["additional_missing_percent"],
            errors="coerce",
        ),
        percentage,
    ):
        raise ValueError(f"{subject} Hard cache percentage is invalid.")
    if not pd.to_numeric(
        result["corruption_repeat"],
        errors="coerce",
    ).eq(repeat).all():
        raise ValueError(f"{subject} Hard cache repeat is invalid.")
    if np.isinf(result[HARD_COMMON25].to_numpy(dtype=float)).any():
        raise ValueError(f"{subject} Hard cache contains infinity.")
    return result


def load_soft_conditions(
    root: Path,
    processed_root: Path,
    subjects: list[str],
    percentages: list[float],
    repeats: int,
) -> dict[tuple[str, float, int], pd.DataFrame]:
    output_lists: dict[
        tuple[str, float, int], list[pd.DataFrame]
    ] = {
        (mode, percentage, repeat): []
        for mode in MODES
        for percentage in percentages
        for repeat in range(1, repeats + 1)
    }
    missing_paths: list[Path] = []

    for subject in subjects:
        trial_map = load_trial_map(
            processed_root
            / subject
            / f"{subject}_all_trials_with_sentence_index.csv",
            subject,
        )
        expected_ids = set(trial_map["trial_id"].astype(str))

        for condition in output_lists:
            mode, percentage, repeat = condition
            path = soft_cache_path(
                root,
                subject,
                mode,
                percentage,
                repeat,
            )
            if not path.is_file():
                missing_paths.append(path)
                continue
            output_lists[condition].append(
                validate_soft_cache(
                    pd.read_csv(path),
                    subject,
                    expected_ids,
                    mode,
                    percentage,
                    repeat,
                )
            )

    if missing_paths:
        examples = "\n".join(
            f"  {path}" for path in missing_paths[:12]
        )
        raise FileNotFoundError(
            "Soft-AOI missingness caches are missing. "
            "Run run_zuco_soft_aoi_missingness_30hz.py first. "
            f"Missing={len(missing_paths)}; examples:\n{examples}"
        )

    output: dict[tuple[str, float, int], pd.DataFrame] = {}
    for condition, frames in output_lists.items():
        frame = pd.concat(frames, ignore_index=True)
        if len(frame) != EXPECTED_TRIALS:
            raise ValueError(
                f"Soft condition {condition} rows={len(frame)}, "
                f"expected={EXPECTED_TRIALS}."
            )
        output[condition] = frame
    return output


def extract_or_load_hard_conditions(
    manifest: pd.DataFrame,
    subjects: list[str],
    processed_root: Path,
    cache_root: Path,
    aoi_lookup: dict[tuple[str, int], pd.DataFrame],
    percentages: list[float],
    repeats: int,
    burst_seconds: float,
    seed: int,
    sigma_word_height_ratio: float,
    active_threshold: float,
    overwrite: bool,
) -> dict[tuple[str, float, int], pd.DataFrame]:
    combined: dict[
        tuple[str, float, int], list[pd.DataFrame]
    ] = {
        (mode, percentage, repeat): []
        for mode in MODES
        for percentage in percentages
        for repeat in range(1, repeats + 1)
    }

    for subject_index, subject in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"Hard-AOI missingness subject "
            f"{subject_index:02d}/{len(subjects)}: {subject}"
        )

        trial_map = load_trial_map(
            processed_root
            / subject
            / f"{subject}_all_trials_with_sentence_index.csv",
            subject,
        )
        expected_ids = set(trial_map["trial_id"].astype(str))
        trial_lookup = trial_map.set_index(
            "trial_id",
            drop=False,
            verify_integrity=True,
        )
        paths = {
            condition: hard_cache_path(
                cache_root,
                subject,
                condition[0],
                condition[1],
                condition[2],
            )
            for condition in combined
        }

        if not overwrite and all(path.is_file() for path in paths.values()):
            for condition, path in paths.items():
                mode, percentage, repeat = condition
                combined[condition].append(
                    validate_hard_cache(
                        pd.read_csv(path),
                        subject,
                        expected_ids,
                        mode,
                        percentage,
                        repeat,
                    )
                )
            print("  Loaded all existing Hard-AOI missingness caches.")
            continue

        needed = {
            condition
            for condition, path in paths.items()
            if overwrite or not path.is_file()
        }
        rows_by_condition: dict[
            tuple[str, float, int], list[dict[str, Any]]
        ] = {condition: [] for condition in needed}

        subject_manifest = manifest[
            manifest["subject_id"].eq(subject)
        ].sort_values(["task", "block_id"])
        if len(subject_manifest) != 14:
            raise ValueError(
                f"{subject} manifest rows={len(subject_manifest)}, expected=14."
            )

        if needed:
            for block_index, manifest_row in enumerate(
                subject_manifest.itertuples(index=False),
                start=1,
            ):
                sample_path = Path(str(manifest_row.sample_parquet))
                if not sample_path.is_file():
                    raise FileNotFoundError(sample_path)
                print(
                    f"  [{block_index:02d}/14] {sample_path.name}; "
                    f"conditions={len(needed)}"
                )

                samples = pd.read_parquet(sample_path)
                required = {
                    "trial_id",
                    "timestamp_ms_abs",
                    "x_px",
                    "y_px",
                    "pupil_area",
                    "valid_basic",
                }
                missing = sorted(required - set(samples.columns))
                if missing:
                    raise ValueError(
                        f"{sample_path} lacks columns: {missing}"
                    )

                samples["trial_id"] = samples["trial_id"].astype(str)
                for trial_id, trial_samples in samples.groupby(
                    "trial_id",
                    sort=False,
                ):
                    trial_id = str(trial_id)
                    if trial_id not in trial_lookup.index:
                        raise ValueError(
                            f"Trial missing from map: {trial_id}"
                        )
                    trial_row = trial_lookup.loc[trial_id].copy()
                    aoi_key = (
                        str(trial_row["task"]).upper(),
                        int(trial_row["task_sentence_index"]),
                    )
                    if aoi_key not in aoi_lookup:
                        raise ValueError(
                            f"AOI template missing for key={aoi_key}"
                        )

                    sampled = soft_downsample_trial(
                        trial_samples,
                        RATE_HZ,
                    )
                    valid_before = int(
                        pd.to_numeric(
                            sampled["valid_basic"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .astype(bool)
                        .sum()
                    )

                    for mode, percentage, repeat in needed:
                        rng = np.random.default_rng(
                            missingness_seed(
                                base_seed=seed,
                                trial_id=trial_id,
                                mode=mode,
                                percentage=percentage,
                                repeat=repeat,
                            )
                        )
                        degraded, removed = degrade_samples(
                            sampled=sampled,
                            mode=mode,
                            percentage=percentage,
                            burst_seconds=burst_seconds,
                            rng=rng,
                        )
                        row = extract_hard_trial_features(
                            samples=degraded,
                            trial_row=trial_row,
                            aoi=aoi_lookup[aoi_key],
                            rate_hz=500,
                            sigma_word_height_ratio=(
                                sigma_word_height_ratio
                            ),
                            active_threshold=active_threshold,
                        )
                        row["hard_aoi_rate_hz"] = RATE_HZ
                        row["missingness_mode"] = mode
                        row["additional_missing_percent"] = float(
                            percentage
                        )
                        row["corruption_repeat"] = int(repeat)
                        row["realized_additional_missing_percent"] = (
                            100.0 * removed / valid_before
                            if valid_before > 0
                            else math.nan
                        )
                        rows_by_condition[
                            (mode, percentage, repeat)
                        ].append(row)
                del samples

        for condition, path in paths.items():
            mode, percentage, repeat = condition
            if condition in rows_by_condition:
                frame = pd.DataFrame(
                    rows_by_condition[condition]
                ).sort_values("trial_id").reset_index(drop=True)
                frame = validate_hard_cache(
                    frame,
                    subject,
                    expected_ids,
                    mode,
                    percentage,
                    repeat,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(
                    path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"  Written: {path}")
            else:
                frame = validate_hard_cache(
                    pd.read_csv(path),
                    subject,
                    expected_ids,
                    mode,
                    percentage,
                    repeat,
                )
            combined[condition].append(frame)

    output: dict[tuple[str, float, int], pd.DataFrame] = {}
    for condition, frames in combined.items():
        frame = pd.concat(frames, ignore_index=True)
        if len(frame) != EXPECTED_TRIALS:
            raise ValueError(
                f"Hard condition {condition} rows={len(frame)}, "
                f"expected={EXPECTED_TRIALS}."
            )
        output[condition] = frame
    return output


def make_condition_table(
    metadata: pd.DataFrame,
    soft: pd.DataFrame,
    hard: pd.DataFrame,
) -> pd.DataFrame:
    soft_columns = SOFT_COMMON_DISTRIBUTION + SOFT_QUALITY
    soft_small = soft[
        ["trial_id", "subject_id", "task"] + soft_columns
    ].rename(
        columns={
            name: f"soft__{name}"
            for name in soft_columns
        }
    )
    hard_small = hard[
        ["trial_id", "subject_id", "task"]
        + HARD_DISTRIBUTION_COMMON
        + HARD_QUALITY
    ].rename(
        columns={
            name: f"hard__{name}"
            for name in (
                HARD_DISTRIBUTION_COMMON + HARD_QUALITY
            )
        }
    )

    merged = metadata.merge(
        soft_small,
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    ).merge(
        hard_small,
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_TRIALS:
        raise ValueError(
            f"Condition merge rows={len(merged)}, "
            f"expected={EXPECTED_TRIALS}."
        )
    audit_shared_quality(merged)
    return merged


def evaluate(
    clean: pd.DataFrame,
    conditions: dict[tuple[str, float, int], pd.DataFrame],
    feature_sets: dict[str, list[str]],
    c_value: float,
    seed: int,
) -> pd.DataFrame:
    subjects = sorted(clean["subject_id"].unique())
    folds = sorted(clean["sentence_fold"].unique())
    prediction_rows: list[dict[str, Any]] = []

    for subject_index, held_out in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"Missingness evaluation subject "
            f"{subject_index:02d}/{len(subjects)}: {held_out}"
        )
        for fold_index in folds:
            train = clean[
                (clean["subject_id"] != held_out)
                & (clean["sentence_fold"] != fold_index)
            ].copy()
            y_train = train["label_nr"].to_numpy(dtype=int)

            models = {}
            for feature_set, columns in feature_sets.items():
                model = make_svm(c_value, seed)
                model.fit(train[columns], y_train)
                models[feature_set] = model

            for (mode, percentage, repeat), condition in conditions.items():
                test = condition[
                    (condition["subject_id"] == held_out)
                    & (condition["sentence_fold"] == fold_index)
                ].copy().reset_index(drop=True)
                if test.empty:
                    raise ValueError(
                        f"Empty test: {held_out}, fold={fold_index}, "
                        f"{mode}, {percentage}, repeat={repeat}."
                    )
                if set(train["sentence_key"]) & set(test["sentence_key"]):
                    raise ValueError("Sentence leakage detected.")

                y_test = test["label_nr"].to_numpy(dtype=int)
                for feature_set, columns in feature_sets.items():
                    model = models[feature_set]
                    y_pred = model.predict(test[columns]).astype(int)
                    scores = decision_scores(model, test[columns])

                    for row_index, row in test.iterrows():
                        prediction_rows.append(
                            {
                                "missingness_mode": mode,
                                "additional_missing_percent": float(
                                    percentage
                                ),
                                "corruption_repeat": int(repeat),
                                "feature_set": feature_set,
                                "held_out_subject": held_out,
                                "sentence_fold": int(fold_index),
                                "trial_id": str(row["trial_id"]),
                                "subject_id": str(row["subject_id"]),
                                "task": str(row["task"]),
                                "label_nr": int(row["label_nr"]),
                                "prediction_nr": int(y_pred[row_index]),
                                "decision_score_nr": float(
                                    scores[row_index]
                                ),
                            }
                        )
    return pd.DataFrame(prediction_rows)


def aggregate_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    keys = [
        "missingness_mode",
        "additional_missing_percent",
        "corruption_repeat",
        "feature_set",
        "held_out_subject",
    ]
    for values, group in predictions.groupby(keys, sort=True):
        mode, percentage, repeat, feature_set, subject = values
        repeat_rows.append(
            {
                "missingness_mode": mode,
                "additional_missing_percent": float(percentage),
                "corruption_repeat": int(repeat),
                "feature_set": feature_set,
                "held_out_subject": subject,
                **score_predictions(
                    group["label_nr"].to_numpy(dtype=int),
                    group["prediction_nr"].to_numpy(dtype=int),
                    group["decision_score_nr"].to_numpy(dtype=float),
                ),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)

    subject_metrics = (
        repeat_metrics.groupby(
            [
                "missingness_mode",
                "additional_missing_percent",
                "feature_set",
                "held_out_subject",
            ],
            as_index=False,
        )[METRICS]
        .mean()
    )

    summary_rows: list[dict[str, Any]] = []
    for values, group in subject_metrics.groupby(
        [
            "missingness_mode",
            "additional_missing_percent",
            "feature_set",
        ],
        sort=True,
    ):
        mode, percentage, feature_set = values
        row: dict[str, Any] = {
            "missingness_mode": mode,
            "additional_missing_percent": float(percentage),
            "feature_set": feature_set,
            "n_subjects": int(group["held_out_subject"].nunique()),
            "n_input_features": 25,
        }
        for metric in METRICS:
            row[f"{metric}_subject_mean"] = float(
                group[metric].mean()
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return repeat_metrics, subject_metrics, summary


def build_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (mode, percentage), condition in subject_metrics.groupby(
        ["missingness_mode", "additional_missing_percent"],
        sort=True,
    ):
        condition_rows: list[dict[str, Any]] = []
        for metric in METRICS:
            soft = (
                condition[
                    condition["feature_set"].eq(SOFT_NAME)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            hard = (
                condition[
                    condition["feature_set"].eq(HARD_NAME)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            paired = pd.concat(
                [
                    hard.rename("hard"),
                    soft.rename("soft"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            diff = paired["soft"] - paired["hard"]

            if np.allclose(diff.to_numpy(dtype=float), 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    paired["soft"],
                    paired["hard"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            condition_rows.append(
                {
                    "missingness_mode": mode,
                    "additional_missing_percent": float(percentage),
                    "metric": metric,
                    "hard_mean": float(paired["hard"].mean()),
                    "soft_mean": float(paired["soft"].mean()),
                    "mean_difference_soft_minus_hard": float(
                        diff.mean()
                    ),
                    "soft_wins": int(np.sum(diff > 0)),
                    "ties": int(np.sum(np.isclose(diff, 0.0))),
                    "soft_losses": int(np.sum(diff < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_across_metrics": math.nan,
                }
            )

        adjusted = holm_adjust(
            [float(row["p_value"]) for row in condition_rows]
        )
        for row, adjusted_value in zip(
            condition_rows,
            adjusted,
        ):
            row["p_holm_across_metrics"] = adjusted_value
        rows.extend(condition_rows)

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
        "--processed-root",
        default="data/processed/zuco2",
    )
    parser.add_argument(
        "--clean-raw-features",
        default=(
            "outputs/downsampling_robustness/"
            "complete_raw_18_0030hz_features.csv"
        ),
    )
    parser.add_argument(
        "--raw-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--clean-soft-features",
        default=(
            "data/processed/zuco2/soft_aoi_all_subjects/"
            "zuco2_soft_aoi_features_30hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--soft-feature-manifest",
        default=(
            "data/processed/zuco2/YAC/soft_aoi/"
            "YAC_soft_aoi_feature_manifest_30hz.json"
        ),
    )
    parser.add_argument(
        "--clean-hard-features",
        default=(
            "data/processed/zuco2/hard_aoi_all_subjects/"
            "zuco2_hard_aoi_features_30hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--word-template",
        default=(
            "data/processed/zuco2/soft_aoi_template/"
            "zuco_soft_aoi_word_template.csv"
        ),
    )
    parser.add_argument(
        "--sentence-folds-file",
        default=(
            "outputs/soft_aoi_confound_controls_30hz/"
            "dual_holdout_sentence_folds.csv"
        ),
    )
    parser.add_argument(
        "--soft-cache-root",
        default=(
            "outputs/soft_aoi_missingness_30hz/"
            "subject_soft_aoi_missingness_features"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/soft_vs_hard_aoi_missingness_30hz",
    )
    parser.add_argument("--percentages", default="40")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--burst-seconds", type=float, default=1.0)
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sigma-word-height-ratio",
        type=float,
        default=2.0 / 3.0,
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--overwrite-hard-features",
        action="store_true",
    )
    args = parser.parse_args()

    percentages = parse_percentages(args.percentages)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.burst_seconds <= 0:
        raise ValueError("--burst-seconds must be positive.")

    manifest = pd.read_csv(args.manifest)
    required_manifest = {
        "subject_id",
        "task",
        "block_id",
        "sample_parquet",
    }
    missing_manifest = sorted(
        required_manifest - set(manifest.columns)
    )
    if missing_manifest:
        raise ValueError(
            f"Manifest lacks columns: {missing_manifest}"
        )
    manifest["subject_id"] = (
        manifest["subject_id"].astype(str).str.upper()
    )
    manifest["task"] = manifest["task"].astype(str).str.upper()
    subjects = sorted(manifest["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Manifest subjects={len(subjects)}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    processed_root = Path(args.processed_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean, _, _, _ = load_merged_data(
        raw_path=Path(args.clean_raw_features),
        raw_sets_path=Path(args.raw_feature_sets),
        soft_path=Path(args.clean_soft_features),
        soft_manifest_path=Path(args.soft_feature_manifest),
        word_template_path=Path(args.word_template),
    )
    clean = add_hard_features(
        clean,
        normalize_hard_table(Path(args.clean_hard_features)),
    )
    audit_shared_quality(clean)

    sentence_folds = load_sentence_folds(
        Path(args.sentence_folds_file)
    )
    clean = clean.merge(
        sentence_folds[["sentence_key", "sentence_fold"]],
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if clean["sentence_fold"].isna().any():
        raise ValueError("Clean trials lack sentence folds.")
    clean["sentence_fold"] = clean["sentence_fold"].astype(int)

    aoi_lookup = load_word_template(Path(args.word_template))
    soft_conditions = load_soft_conditions(
        root=Path(args.soft_cache_root),
        processed_root=processed_root,
        subjects=subjects,
        percentages=percentages,
        repeats=args.repeats,
    )
    hard_conditions = extract_or_load_hard_conditions(
        manifest=manifest,
        subjects=subjects,
        processed_root=processed_root,
        cache_root=(
            output_dir / "subject_hard_aoi_missingness_features"
        ),
        aoi_lookup=aoi_lookup,
        percentages=percentages,
        repeats=args.repeats,
        burst_seconds=args.burst_seconds,
        seed=args.seed,
        sigma_word_height_ratio=args.sigma_word_height_ratio,
        active_threshold=args.active_threshold,
        overwrite=args.overwrite_hard_features,
    )

    metadata = clean[
        [
            "trial_id",
            "subject_id",
            "task",
            "task_sentence_index",
            "sentence_key",
            "label_nr",
            "sentence_fold",
        ]
    ].copy()
    conditions = {
        condition: make_condition_table(
            metadata,
            soft_conditions[condition],
            hard_conditions[condition],
        )
        for condition in soft_conditions
    }
    feature_sets = build_feature_sets()

    print("\n=== SOFT VS HARD MISSINGNESS INPUT ===")
    print(f"subjects: {len(subjects)}")
    print(f"trials: {len(clean)}")
    print(f"modes: {MODES}")
    print(f"percentages: {percentages}")
    print(f"repeats: {args.repeats}")
    print("shared quality audit: PASS for clean and all conditions")

    predictions = evaluate(
        clean=clean,
        conditions=conditions,
        feature_sets=feature_sets,
        c_value=args.c_value,
        seed=args.seed,
    )
    repeat_metrics, subject_metrics, summary = aggregate_metrics(
        predictions
    )
    tests = build_tests(subject_metrics)

    paths = {
        "predictions": output_dir / "soft_vs_hard_missingness_predictions.csv",
        "repeat_metrics": (
            output_dir
            / "soft_vs_hard_missingness_subject_repeat_metrics.csv"
        ),
        "subject_metrics": (
            output_dir
            / "soft_vs_hard_missingness_subject_metrics.csv"
        ),
        "summary": output_dir / "soft_vs_hard_missingness_summary.csv",
        "tests": output_dir / "soft_vs_hard_missingness_tests.csv",
        "manifest": output_dir / "soft_vs_hard_missingness_manifest.json",
    }
    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    repeat_metrics.to_csv(
        paths["repeat_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    subject_metrics.to_csv(
        paths["subject_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        paths["summary"],
        index=False,
        encoding="utf-8-sig",
    )
    tests.to_csv(
        paths["tests"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "protocol": (
                    "clean30_train_missing30_test_dual_holdout"
                ),
                "modes": MODES,
                "percentages": percentages,
                "repeats": int(args.repeats),
                "burst_seconds": float(args.burst_seconds),
                "c_value": float(args.c_value),
                "seed": int(args.seed),
                "feature_sets": feature_sets,
                "hard_assignment": "argmax_q_with_qmax_mass",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== SOFT-AOI VS HARD-AOI MISSINGNESS SUMMARY ===")
    print(
        summary[
            [
                "missingness_mode",
                "additional_missing_percent",
                "feature_set",
                "n_input_features",
                "macro_f1_subject_mean",
                "balanced_accuracy_subject_mean",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ].to_string(index=False)
    )

    print("\n=== SOFT-AOI VS HARD-AOI MISSINGNESS TESTS ===")
    print(
        tests[
            tests["metric"].isin(
                ["macro_f1", "roc_auc", "pr_auc_nr"]
            )
        ][
            [
                "missingness_mode",
                "additional_missing_percent",
                "metric",
                "hard_mean",
                "soft_mean",
                "mean_difference_soft_minus_hard",
                "soft_wins",
                "ties",
                "soft_losses",
                "p_value",
                "p_holm_across_metrics",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
