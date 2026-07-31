#!/usr/bin/env python
"""Evaluate Soft-AOI robustness to random and burst missingness at 30 Hz.

This experiment reuses the deterministic missingness conditions from
run_zuco_missingness_30hz.py and evaluates three representations:

- rawstat_all (18 features)
- soft_full_all (41 features)
- soft_compact_distribution_membership (27 features)

Protocol
--------
- Train on clean 30 Hz trials.
- Test on clean or additionally corrupted 30 Hz trials.
- Exclude the held-out subject from training.
- Exclude every test sentence from training.
- Fix LinearSVC C=0.01.
- Average three deterministic corruption repeats per subject and condition.

Missingness modes
-----------------
random
    Remove valid samples independently.

burst
    Remove the same target percentage in contiguous windows, one second by
    default.

Existing RawStat missingness caches are reused. Soft-AOI missingness features
are cached per subject, mode, percentage, and repeat so extraction can resume.
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
    CONTENT_FEATURES,
    MEMBERSHIP_QUALITY_FEATURES,
    downsample_trial as soft_downsample_trial,
    extract_trial_features as extract_soft_trial_features,
)
from run_zuco_missingness_30hz import (  # noqa: E402
    add_burst_missingness,
    add_random_missingness,
    current_valid_basic_mask,
    stable_seed as missingness_seed,
)
from run_zuco_soft_aoi_ablation import (  # noqa: E402
    DISTRIBUTION_BASE,
    load_sentence_folds,
)
from run_zuco_soft_aoi_confound_controls import (  # noqa: E402
    METRICS,
    decision_scores,
    derive_raw_feature_sets,
    make_svm,
    score_predictions,
)
from run_zuco_soft_aoi_coordinate_noise_30hz import (  # noqa: E402
    build_feature_sets,
    load_soft_manifest,
    load_word_template,
    merge_condition,
    normalize_basic_table,
    prepare_subject_map,
)


RATE_HZ = 30
EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_PER_SUBJECT = 739
EXPECTED_NR = 6282
EXPECTED_TSR = 7020

COMPACT_NAME = "soft_compact_distribution_membership"
FEATURE_SET_ORDER = [
    "rawstat_all",
    "soft_full_all",
    COMPACT_NAME,
]
MODES = ["random", "burst"]


def parse_percentages(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values:
        raise ValueError("--percentages is empty.")
    if any(value <= 0 or value >= 100 for value in values):
        raise ValueError(
            "Every percentage must be within (0, 100)."
        )
    return values


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return result.tolist()

    order = finite[np.argsort(values[finite])]
    running = 0.0
    total = len(order)
    for rank, original_index in enumerate(order):
        adjusted = min(
            1.0,
            (total - rank) * float(values[original_index]),
        )
        running = max(running, adjusted)
        result[original_index] = running
    return result.tolist()


def raw_cache_path(
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
            f"{subject}_{mode}_{percentage:05.1f}pct"
            f"_repeat_{repeat:02d}.csv"
        )
    )


def soft_cache_path(
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
            f"{subject}_soft_aoi_{mode}_{percentage:05.1f}pct"
            f"_repeat_{repeat:02d}.csv"
        )
    )


def validate_raw_caches(
    root: Path,
    subjects: list[str],
    percentages: list[float],
    repeats: int,
) -> None:
    missing: list[Path] = []
    for subject in subjects:
        for mode in MODES:
            for percentage in percentages:
                for repeat in range(1, repeats + 1):
                    path = raw_cache_path(
                        root,
                        subject,
                        mode,
                        percentage,
                        repeat,
                    )
                    if not path.is_file():
                        missing.append(path)

    if missing:
        examples = "\n".join(
            f"  {path}" for path in missing[:15]
        )
        raise FileNotFoundError(
            "Existing RawStat missingness caches are missing. "
            "Run run_zuco_missingness_30hz.py first. "
            f"Missing={len(missing)}; examples:\n{examples}"
        )


def update_validity_columns(samples: pd.DataFrame) -> pd.DataFrame:
    """Synchronize validity flags after x/y/pupil samples are zeroed."""
    result = samples.copy()
    valid_basic = current_valid_basic_mask(result)

    if "valid_basic" in result.columns:
        result["valid_basic"] = valid_basic
    if "valid_xy" in result.columns:
        x = result["x_px"].to_numpy(dtype=float)
        y = result["y_px"].to_numpy(dtype=float)
        result["valid_xy"] = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
            & (y > 0)
        )
    if "valid_pupil" in result.columns:
        pupil = result["pupil_area"].to_numpy(dtype=float)
        result["valid_pupil"] = (
            np.isfinite(pupil) & (pupil > 0)
        )
    return result


def degrade_samples(
    sampled: pd.DataFrame,
    mode: str,
    percentage: float,
    burst_seconds: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    if mode == "random":
        degraded, removed = add_random_missingness(
            samples=sampled,
            percentage=percentage,
            rng=rng,
        )
    elif mode == "burst":
        degraded, removed = add_burst_missingness(
            samples=sampled,
            percentage=percentage,
            burst_seconds=burst_seconds,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown missingness mode: {mode}")

    return update_validity_columns(degraded), int(removed)


def validate_soft_cache(
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
        "realized_additional_missing_percent",
        *CONTENT_FEATURES,
        *MEMBERSHIP_QUALITY_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{subject} Soft-AOI missingness cache lacks: {missing}"
        )

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()
    result["missingness_mode"] = (
        result["missingness_mode"].astype(str).str.lower()
    )

    if len(result) != EXPECTED_PER_SUBJECT:
        raise ValueError(
            f"{subject} cache rows={len(result)}, "
            f"expected={EXPECTED_PER_SUBJECT}."
        )
    if set(result["trial_id"]) != expected_ids:
        raise ValueError(
            f"{subject} cache trial IDs do not match."
        )
    if result["trial_id"].duplicated().any():
        raise ValueError(
            f"{subject} cache contains duplicate trial IDs."
        )
    if set(result["subject_id"]) != {subject}:
        raise ValueError(
            f"{subject} cache contains another subject."
        )
    if set(result["missingness_mode"]) != {mode}:
        raise ValueError(
            f"{subject} cache has wrong mode."
        )
    if not np.allclose(
        pd.to_numeric(
            result["additional_missing_percent"],
            errors="coerce",
        ),
        percentage,
    ):
        raise ValueError(
            f"{subject} cache has wrong percentage."
        )
    if not pd.to_numeric(
        result["corruption_repeat"],
        errors="coerce",
    ).eq(repeat).all():
        raise ValueError(
            f"{subject} cache has wrong repeat."
        )

    feature_columns = (
        CONTENT_FEATURES + MEMBERSHIP_QUALITY_FEATURES
    )
    if np.isinf(
        result[feature_columns].to_numpy(dtype=float)
    ).any():
        raise ValueError(
            f"{subject} cache contains infinity."
        )
    return result


def extract_subject_soft_conditions(
    subject: str,
    subject_manifest: pd.DataFrame,
    subject_map: pd.DataFrame,
    aoi_lookup: dict[tuple[str, int], pd.DataFrame],
    conditions: list[tuple[str, float, int]],
    cache_paths: dict[tuple[str, float, int], Path],
    seed: int,
    burst_seconds: float,
    sigma_word_height_ratio: float,
    active_threshold: float,
) -> dict[tuple[str, float, int], pd.DataFrame]:
    trial_lookup = subject_map.set_index(
        "trial_id",
        drop=False,
        verify_integrity=True,
    )
    rows_by_condition: dict[
        tuple[str, float, int],
        list[dict[str, Any]],
    ] = {
        condition: [] for condition in conditions
    }

    block_manifest = subject_manifest.sort_values(
        ["task", "block_id"]
    )
    if len(block_manifest) != 14:
        raise ValueError(
            f"{subject} manifest rows={len(block_manifest)}, "
            "expected=14."
        )

    for block_index, manifest_row in enumerate(
        block_manifest.itertuples(index=False),
        start=1,
    ):
        sample_path = Path(str(manifest_row.sample_parquet))
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)

        print(
            f"[{subject} {block_index}/14] "
            f"Soft-AOI extraction for {len(conditions)} "
            f"missingness conditions: {sample_path}"
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
                f"{sample_path} lacks sample columns: {missing}"
            )

        for trial_id_value, trial_samples in samples.groupby(
            "trial_id",
            sort=False,
        ):
            trial_id = str(trial_id_value)
            if trial_id not in trial_lookup.index:
                raise ValueError(
                    f"Trial missing from subject map: {trial_id}"
                )

            trial_row = trial_lookup.loc[trial_id].copy()
            key = (
                str(trial_row["task"]).upper(),
                int(trial_row["task_sentence_index"]),
            )
            if key not in aoi_lookup:
                raise ValueError(
                    f"AOI template missing for key={key}"
                )

            sampled = soft_downsample_trial(
                trial_samples,
                RATE_HZ,
            )
            original_valid = int(
                np.sum(current_valid_basic_mask(sampled))
            )

            for mode, percentage, repeat in conditions:
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

                # Input is already at 30 Hz. Passing 500 prevents a
                # second downsampling pass; the stored rate is corrected.
                row = extract_soft_trial_features(
                    samples=degraded,
                    trial_row=trial_row,
                    aoi=aoi_lookup[key],
                    rate_hz=500,
                    sigma_word_height_ratio=(
                        sigma_word_height_ratio
                    ),
                    active_threshold=active_threshold,
                )
                row["soft_aoi_rate_hz"] = RATE_HZ
                row["missingness_mode"] = mode
                row["additional_missing_percent"] = float(
                    percentage
                )
                row["corruption_repeat"] = int(repeat)
                row["additional_missing_samples"] = int(
                    removed
                )
                row["original_valid_basic_samples"] = int(
                    original_valid
                )
                row[
                    "realized_additional_missing_percent"
                ] = (
                    100.0 * removed / original_valid
                    if original_valid > 0
                    else math.nan
                )
                rows_by_condition[
                    (mode, percentage, repeat)
                ].append(row)

        del samples

    expected_ids = set(subject_map["trial_id"].astype(str))
    results: dict[
        tuple[str, float, int],
        pd.DataFrame,
    ] = {}

    for condition in conditions:
        mode, percentage, repeat = condition
        frame = (
            pd.DataFrame(rows_by_condition[condition])
            .sort_values(
                [
                    "task",
                    "block_id",
                    "trial_index_in_block",
                ]
            )
            .reset_index(drop=True)
        )
        frame = validate_soft_cache(
            frame=frame,
            subject=subject,
            expected_ids=expected_ids,
            mode=mode,
            percentage=percentage,
            repeat=repeat,
        )
        path = cache_paths[condition]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        results[condition] = frame

    return results


def extract_or_load_soft_conditions(
    manifest: pd.DataFrame,
    subjects: list[str],
    processed_root: Path,
    cache_root: Path,
    aoi_lookup: dict[tuple[str, int], pd.DataFrame],
    percentages: list[float],
    repeats: int,
    seed: int,
    burst_seconds: float,
    sigma_word_height_ratio: float,
    active_threshold: float,
    overwrite: bool,
) -> dict[tuple[str, float, int], pd.DataFrame]:
    conditions = [
        (mode, percentage, repeat)
        for mode in MODES
        for percentage in percentages
        for repeat in range(1, repeats + 1)
    ]
    frames_by_condition: dict[
        tuple[str, float, int],
        list[pd.DataFrame],
    ] = {
        condition: [] for condition in conditions
    }

    for subject_index, subject in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Soft-AOI missingness subject "
            f"{subject_index}/{len(subjects)}: {subject}"
        )
        subject_map = prepare_subject_map(
            processed_root
            / subject
            / f"{subject}_all_trials_with_sentence_index.csv",
            subject,
        )
        expected_ids = set(subject_map["trial_id"])
        subject_manifest = manifest[
            manifest["subject_id"].eq(subject)
        ].copy()

        cache_paths = {
            condition: soft_cache_path(
                cache_root,
                subject,
                condition[0],
                condition[1],
                condition[2],
            )
            for condition in conditions
        }

        missing_conditions: list[
            tuple[str, float, int]
        ] = []
        for condition in conditions:
            mode, percentage, repeat = condition
            path = cache_paths[condition]
            if path.is_file() and not overwrite:
                cached = validate_soft_cache(
                    frame=pd.read_csv(path),
                    subject=subject,
                    expected_ids=expected_ids,
                    mode=mode,
                    percentage=percentage,
                    repeat=repeat,
                )
                frames_by_condition[condition].append(cached)
                print(f"Reusing Soft-AOI cache: {path}")
            else:
                missing_conditions.append(condition)

        if missing_conditions:
            extracted = extract_subject_soft_conditions(
                subject=subject,
                subject_manifest=subject_manifest,
                subject_map=subject_map,
                aoi_lookup=aoi_lookup,
                conditions=missing_conditions,
                cache_paths=cache_paths,
                seed=seed,
                burst_seconds=burst_seconds,
                sigma_word_height_ratio=(
                    sigma_word_height_ratio
                ),
                active_threshold=active_threshold,
            )
            for condition in missing_conditions:
                frames_by_condition[condition].append(
                    extracted[condition]
                )

    result: dict[
        tuple[str, float, int],
        pd.DataFrame,
    ] = {}
    for condition, frames in frames_by_condition.items():
        result[condition] = normalize_basic_table(
            pd.concat(frames, ignore_index=True),
            (
                f"Soft-AOI missingness "
                f"{condition[0]} {condition[1]}% "
                f"repeat={condition[2]}"
            ),
        )
    return result


def load_raw_conditions(
    cache_root: Path,
    subjects: list[str],
    percentages: list[float],
    repeats: int,
) -> dict[tuple[str, float, int], pd.DataFrame]:
    result: dict[
        tuple[str, float, int],
        pd.DataFrame,
    ] = {}
    for mode in MODES:
        for percentage in percentages:
            for repeat in range(1, repeats + 1):
                frames = [
                    pd.read_csv(
                        raw_cache_path(
                            cache_root,
                            subject,
                            mode,
                            percentage,
                            repeat,
                        )
                    )
                    for subject in subjects
                ]
                result[(mode, percentage, repeat)] = (
                    normalize_basic_table(
                        pd.concat(
                            frames,
                            ignore_index=True,
                        ),
                        (
                            f"RawStat missingness "
                            f"{mode} {percentage}% "
                            f"repeat={repeat}"
                        ),
                    )
                )
    return result


def evaluate(
    clean: pd.DataFrame,
    conditions: dict[
        tuple[str, float, int],
        pd.DataFrame,
    ],
    feature_sets: dict[str, list[str]],
    percentages: list[float],
    repeats: int,
    c_value: float,
    seed: int,
) -> pd.DataFrame:
    subjects = sorted(clean["subject_id"].unique())
    sentence_folds = sorted(
        clean["sentence_fold"].unique()
    )
    rows: list[dict[str, Any]] = []

    eval_conditions: list[
        tuple[str, float, int]
    ] = [("clean", 0.0, 0)]
    eval_conditions.extend(
        (mode, percentage, repeat)
        for mode in MODES
        for percentage in percentages
        for repeat in range(1, repeats + 1)
    )

    for condition_index, condition in enumerate(
        eval_conditions,
        start=1,
    ):
        mode, percentage, repeat = condition
        test_table = (
            clean
            if mode == "clean"
            else conditions[condition]
        )
        print("\n" + "#" * 78)
        print(
            f"Evaluation condition "
            f"{condition_index}/{len(eval_conditions)}: "
            f"mode={mode}, missing={percentage:g}%, "
            f"repeat={repeat}"
        )

        for subject_index, held_out in enumerate(
            subjects,
            start=1,
        ):
            print(
                f"  Subject {subject_index:02d}/"
                f"{len(subjects)}: {held_out}"
            )
            for sentence_fold in sentence_folds:
                train = clean[
                    (clean["subject_id"] != held_out)
                    & (
                        clean["sentence_fold"]
                        != sentence_fold
                    )
                ]
                test = test_table[
                    (test_table["subject_id"] == held_out)
                    & (
                        test_table["sentence_fold"]
                        == sentence_fold
                    )
                ]

                if test.empty:
                    raise ValueError(
                        f"Empty test split: {held_out}, "
                        f"fold={sentence_fold}, "
                        f"mode={mode}, missing={percentage}."
                    )
                if held_out in set(train["subject_id"]):
                    raise ValueError(
                        "Subject leakage detected."
                    )
                if set(train["sentence_key"]) & set(
                    test["sentence_key"]
                ):
                    raise ValueError(
                        "Sentence leakage detected."
                    )

                y_train = train[
                    "label_nr"
                ].to_numpy(dtype=int)
                y_test = test[
                    "label_nr"
                ].to_numpy(dtype=int)

                for feature_set in FEATURE_SET_ORDER:
                    columns = feature_sets[feature_set]
                    model = make_svm(c_value, seed)
                    model.fit(train[columns], y_train)
                    y_pred = model.predict(
                        test[columns]
                    ).astype(int)
                    scores = decision_scores(
                        model,
                        test[columns],
                    )

                    ordered = test.reset_index(drop=True)
                    for index, row in ordered.iterrows():
                        rows.append(
                            {
                                "missingness_mode": mode,
                                "additional_missing_percent": (
                                    float(percentage)
                                ),
                                "corruption_repeat": int(repeat),
                                "feature_set": feature_set,
                                "held_out_subject": held_out,
                                "sentence_fold": int(
                                    sentence_fold
                                ),
                                "n_input_features": int(
                                    len(columns)
                                ),
                                "trial_id": row["trial_id"],
                                "subject_id": row["subject_id"],
                                "task": row["task"],
                                "sentence_key": row[
                                    "sentence_key"
                                ],
                                "label_nr": int(
                                    row["label_nr"]
                                ),
                                "prediction_nr": int(
                                    y_pred[index]
                                ),
                                "decision_score_nr": float(
                                    scores[index]
                                ),
                            }
                        )

    predictions = pd.DataFrame(rows)
    expected_conditions = (
        1
        + len(MODES)
        * len(percentages)
        * repeats
    )
    expected_rows = (
        expected_conditions
        * len(FEATURE_SET_ORDER)
        * EXPECTED_TRIALS
    )
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Prediction rows={len(predictions)}, "
            f"expected={expected_rows}."
        )
    duplicate_columns = [
        "missingness_mode",
        "additional_missing_percent",
        "corruption_repeat",
        "feature_set",
        "trial_id",
    ]
    if predictions[
        duplicate_columns
    ].duplicated().any():
        raise ValueError(
            "Predictions contain duplicate condition rows."
        )
    return predictions


def subject_repeat_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "missingness_mode",
        "additional_missing_percent",
        "corruption_repeat",
        "feature_set",
        "held_out_subject",
    ]
    for values, group in predictions.groupby(
        keys,
        sort=False,
    ):
        (
            mode,
            percentage,
            repeat,
            feature_set,
            subject,
        ) = values
        metrics = score_predictions(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction_nr"].to_numpy(dtype=int),
            group[
                "decision_score_nr"
            ].to_numpy(dtype=float),
        )
        rows.append(
            {
                "missingness_mode": mode,
                "additional_missing_percent": float(
                    percentage
                ),
                "corruption_repeat": int(repeat),
                "feature_set": feature_set,
                "held_out_subject": subject,
                "n_input_features": int(
                    group["n_input_features"].iloc[0]
                ),
                "n_predictions": int(len(group)),
                **metrics,
            }
        )

    result = pd.DataFrame(rows)
    if not result["n_predictions"].eq(
        EXPECTED_PER_SUBJECT
    ).all():
        raise ValueError(
            "Not every subject/repeat metric uses 739 trials."
        )
    return result


def average_repeats(
    subject_repeat: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "missingness_mode",
        "additional_missing_percent",
        "feature_set",
        "held_out_subject",
        "n_input_features",
    ]
    result = (
        subject_repeat.groupby(
            group_columns,
            as_index=False,
            sort=False,
        )[list(METRICS)]
        .mean()
    )
    counts = result.groupby(
        [
            "missingness_mode",
            "additional_missing_percent",
            "feature_set",
        ]
    )["held_out_subject"].nunique()
    if not counts.eq(EXPECTED_SUBJECTS).all():
        raise ValueError(
            f"Subject counts are invalid: {counts.to_dict()}"
        )
    return result


def summarize(
    subject_condition: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "missingness_mode",
        "additional_missing_percent",
        "feature_set",
    ]
    for values, group in subject_condition.groupby(
        keys,
        sort=False,
    ):
        mode, percentage, feature_set = values
        row: dict[str, Any] = {
            "missingness_mode": mode,
            "additional_missing_percent": float(
                percentage
            ),
            "feature_set": feature_set,
            "n_subjects": int(len(group)),
            "n_input_features": int(
                group["n_input_features"].iloc[0]
            ),
        }
        for metric in METRICS:
            numeric = pd.to_numeric(
                group[metric],
                errors="coerce",
            )
            row[f"{metric}_subject_mean"] = float(
                numeric.mean()
            )
            row[f"{metric}_subject_std"] = float(
                numeric.std(ddof=1)
            )
        rows.append(row)

    mode_order = {
        "clean": 0,
        "random": 1,
        "burst": 2,
    }
    result = pd.DataFrame(rows)
    result["_mode_order"] = result[
        "missingness_mode"
    ].map(mode_order)
    result = result.sort_values(
        [
            "_mode_order",
            "additional_missing_percent",
            "macro_f1_subject_mean",
        ],
        ascending=[True, True, False],
    ).drop(columns="_mode_order")
    return result


def wilcoxon_pair(
    reference: pd.Series,
    candidate: pd.Series,
) -> dict[str, Any]:
    paired = pd.concat(
        [
            reference.rename("reference"),
            candidate.rename("candidate"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    differences = (
        paired["candidate"] - paired["reference"]
    )

    if len(paired) == 0:
        statistic = math.nan
        p_value = math.nan
    elif np.allclose(
        differences.to_numpy(dtype=float),
        0.0,
    ):
        statistic = 0.0
        p_value = 1.0
    else:
        test = wilcoxon(
            paired["candidate"],
            paired["reference"],
            alternative="two-sided",
            zero_method="wilcox",
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)

    return {
        "n_subjects": int(len(paired)),
        "reference_mean": float(
            paired["reference"].mean()
        ),
        "candidate_mean": float(
            paired["candidate"].mean()
        ),
        "mean_difference_candidate_minus_reference": float(
            differences.mean()
        ),
        "wins": int(np.sum(differences > 0)),
        "ties": int(
            np.sum(np.isclose(differences, 0.0))
        ),
        "losses": int(np.sum(differences < 0)),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def build_tests(
    subject_condition: pd.DataFrame,
    percentages: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ]

    # Missing condition versus clean.
    for metric in metric_names:
        for feature_set in FEATURE_SET_ORDER:
            clean = (
                subject_condition[
                    subject_condition[
                        "missingness_mode"
                    ].eq("clean")
                    & subject_condition[
                        "feature_set"
                    ].eq(feature_set)
                ]
                .set_index(
                    "held_out_subject"
                )[metric]
                .sort_index()
            )
            for mode in MODES:
                for percentage in percentages:
                    missing = (
                        subject_condition[
                            subject_condition[
                                "missingness_mode"
                            ].eq(mode)
                            & subject_condition[
                                "additional_missing_percent"
                            ].eq(percentage)
                            & subject_condition[
                                "feature_set"
                            ].eq(feature_set)
                        ]
                        .set_index(
                            "held_out_subject"
                        )[metric]
                        .sort_index()
                    )
                    rows.append(
                        {
                            "test_family": (
                                "missingness_vs_clean"
                            ),
                            "metric": metric,
                            "missingness_mode": mode,
                            "additional_missing_percent": (
                                float(percentage)
                            ),
                            "comparison": (
                                f"{feature_set}_{mode}_"
                                f"{percentage:g}_vs_clean"
                            ),
                            "reference_feature_set": (
                                feature_set
                            ),
                            "candidate_feature_set": (
                                feature_set
                            ),
                            **wilcoxon_pair(
                                clean,
                                missing,
                            ),
                            "p_holm_within_family_metric": (
                                math.nan
                            ),
                        }
                    )

    # Representation comparisons at each condition.
    comparisons = [
        (
            "rawstat_all",
            "soft_full_all",
            "soft_full_vs_rawstat",
        ),
        (
            "rawstat_all",
            COMPACT_NAME,
            "soft_compact_vs_rawstat",
        ),
        (
            "soft_full_all",
            COMPACT_NAME,
            "soft_compact_vs_full",
        ),
    ]
    conditions = [("clean", 0.0)]
    conditions.extend(
        (mode, percentage)
        for mode in MODES
        for percentage in percentages
    )

    for metric in metric_names:
        for mode, percentage in conditions:
            condition = subject_condition[
                subject_condition[
                    "missingness_mode"
                ].eq(mode)
                & subject_condition[
                    "additional_missing_percent"
                ].eq(percentage)
            ]
            for (
                reference_name,
                candidate_name,
                comparison,
            ) in comparisons:
                reference = (
                    condition[
                        condition["feature_set"].eq(
                            reference_name
                        )
                    ]
                    .set_index(
                        "held_out_subject"
                    )[metric]
                    .sort_index()
                )
                candidate = (
                    condition[
                        condition["feature_set"].eq(
                            candidate_name
                        )
                    ]
                    .set_index(
                        "held_out_subject"
                    )[metric]
                    .sort_index()
                )
                rows.append(
                    {
                        "test_family": (
                            "feature_comparison"
                        ),
                        "metric": metric,
                        "missingness_mode": mode,
                        "additional_missing_percent": (
                            float(percentage)
                        ),
                        "comparison": comparison,
                        "reference_feature_set": (
                            reference_name
                        ),
                        "candidate_feature_set": (
                            candidate_name
                        ),
                        **wilcoxon_pair(
                            reference,
                            candidate,
                        ),
                        "p_holm_within_family_metric": (
                            math.nan
                        ),
                    }
                )

    tests = pd.DataFrame(rows)
    for (family, metric), indices in tests.groupby(
        ["test_family", "metric"]
    ).groups.items():
        positions = list(indices)
        adjusted = holm_adjust(
            tests.loc[
                positions,
                "p_value",
            ].tolist()
        )
        tests.loc[
            positions,
            "p_holm_within_family_metric",
        ] = adjusted
    return tests


def quality_summary(
    clean_soft: pd.DataFrame,
    missing_soft: dict[
        tuple[str, float, int],
        pd.DataFrame,
    ],
    percentages: list[float],
    repeats: int,
) -> pd.DataFrame:
    quality_columns = [
        "soft_aoi_valid_rate",
        "soft_aoi_active_rate",
        "soft_aoi_word_box_hit_rate",
        "soft_aoi_sentence_envelope_hit_rate",
        "soft_aoi_qmax_mean",
        "soft_aoi_nearest_distance_median_word_heights",
    ]

    frames: list[pd.DataFrame] = []
    clean = clean_soft[
        [
            "trial_id",
            "subject_id",
            "task",
        ]
        + quality_columns
    ].copy()
    clean["missingness_mode"] = "clean"
    clean["additional_missing_percent"] = 0.0
    clean["corruption_repeat"] = 0
    clean[
        "realized_additional_missing_percent"
    ] = 0.0
    frames.append(clean)

    for mode in MODES:
        for percentage in percentages:
            for repeat in range(1, repeats + 1):
                source = missing_soft[
                    (mode, percentage, repeat)
                ]
                frame = source[
                    [
                        "trial_id",
                        "subject_id",
                        "task",
                        "realized_additional_missing_percent",
                    ]
                    + quality_columns
                ].copy()
                frame["missingness_mode"] = mode
                frame[
                    "additional_missing_percent"
                ] = float(percentage)
                frame["corruption_repeat"] = int(
                    repeat
                )
                frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )
    aggregation_columns = [
        "realized_additional_missing_percent",
        *quality_columns,
    ]
    subject_repeat = (
        combined.groupby(
            [
                "missingness_mode",
                "additional_missing_percent",
                "corruption_repeat",
                "subject_id",
            ],
            as_index=False,
        )[aggregation_columns]
        .median()
    )
    subject_condition = (
        subject_repeat.groupby(
            [
                "missingness_mode",
                "additional_missing_percent",
                "subject_id",
            ],
            as_index=False,
        )[aggregation_columns]
        .mean()
    )

    rows: list[dict[str, Any]] = []
    keys = [
        "missingness_mode",
        "additional_missing_percent",
    ]
    for values, group in subject_condition.groupby(
        keys,
        sort=False,
    ):
        mode, percentage = values
        row: dict[str, Any] = {
            "missingness_mode": mode,
            "additional_missing_percent": float(
                percentage
            ),
            "n_subjects": int(
                group["subject_id"].nunique()
            ),
        }
        for column in aggregation_columns:
            row[
                f"{column}_subject_median_mean"
            ] = float(group[column].mean())
        rows.append(row)
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
        "--raw-missingness-cache-root",
        default=(
            "outputs/missingness_30hz/"
            "subject_missingness_features"
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
        "--output-dir",
        default="outputs/soft_aoi_missingness_30hz",
    )
    parser.add_argument(
        "--percentages",
        default="5,10,20,40",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--burst-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument("--c-value", type=float, default=0.01)
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overwrite-soft-features",
        action="store_true",
    )
    args = parser.parse_args()

    percentages = parse_percentages(args.percentages)
    if args.repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )
    if args.burst_seconds <= 0:
        raise ValueError(
            "--burst-seconds must be positive."
        )
    if args.c_value <= 0:
        raise ValueError(
            "--c-value must be positive."
        )
    if args.sigma_word_height_ratio <= 0:
        raise ValueError(
            "--sigma-word-height-ratio must be positive."
        )
    if not (0 < args.active_threshold <= 1):
        raise ValueError(
            "--active-threshold must be within (0, 1]."
        )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
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
        manifest["subject_id"]
        .astype(str)
        .str.upper()
    )
    manifest["task"] = (
        manifest["task"].astype(str).str.upper()
    )
    subjects = sorted(
        manifest["subject_id"].unique()
    )
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Manifest subjects={len(subjects)}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    validate_raw_caches(
        root=Path(
            args.raw_missingness_cache_root
        ),
        subjects=subjects,
        percentages=percentages,
        repeats=args.repeats,
    )

    raw_sets_path = Path(args.raw_feature_sets)
    if not raw_sets_path.is_file():
        raise FileNotFoundError(raw_sets_path)
    raw_sets = json.loads(
        raw_sets_path.read_text(
            encoding="utf-8"
        )
    )
    _, raw_all, raw_source = (
        derive_raw_feature_sets(raw_sets)
    )

    load_soft_manifest(
        Path(args.soft_feature_manifest)
    )
    aoi_lookup = load_word_template(
        Path(args.word_template)
    )
    sentence_folds = load_sentence_folds(
        Path(args.sentence_folds_file)
    )

    clean_raw = normalize_basic_table(
        pd.read_csv(args.clean_raw_features),
        "Clean RawStat",
    )
    clean_soft = normalize_basic_table(
        pd.read_csv(args.clean_soft_features),
        "Clean Soft-AOI",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    soft_cache_root = (
        output_dir
        / "subject_soft_aoi_missingness_features"
    )

    print("=== SOFT-AOI MISSINGNESS INPUT ===")
    print(f"subjects: {len(subjects)}")
    print(f"trials: {EXPECTED_TRIALS}")
    print(f"modes: {MODES}")
    print(f"percentages: {percentages}")
    print(f"repeats: {args.repeats}")
    print(f"burst seconds: {args.burst_seconds}")
    print(f"fixed C: {args.c_value}")
    print(
        f"RawStat feature source: {raw_source}"
    )
    print(
        f"Soft-AOI cache root: {soft_cache_root}"
    )

    missing_soft = extract_or_load_soft_conditions(
        manifest=manifest,
        subjects=subjects,
        processed_root=Path(
            args.processed_root
        ),
        cache_root=soft_cache_root,
        aoi_lookup=aoi_lookup,
        percentages=percentages,
        repeats=args.repeats,
        seed=args.seed,
        burst_seconds=args.burst_seconds,
        sigma_word_height_ratio=(
            args.sigma_word_height_ratio
        ),
        active_threshold=args.active_threshold,
        overwrite=args.overwrite_soft_features,
    )
    missing_raw = load_raw_conditions(
        cache_root=Path(
            args.raw_missingness_cache_root
        ),
        subjects=subjects,
        percentages=percentages,
        repeats=args.repeats,
    )

    soft_all = (
        CONTENT_FEATURES
        + MEMBERSHIP_QUALITY_FEATURES
    )
    clean = merge_condition(
        raw=clean_raw,
        soft=clean_soft,
        raw_columns=raw_all,
        soft_columns=soft_all,
        sentence_folds=sentence_folds,
        name="clean",
    )
    condition_tables = {
        condition: merge_condition(
            raw=missing_raw[condition],
            soft=missing_soft[condition],
            raw_columns=raw_all,
            soft_columns=soft_all,
            sentence_folds=sentence_folds,
            name=(
                f"{condition[0]} "
                f"{condition[1]}% "
                f"repeat={condition[2]}"
            ),
        )
        for condition in missing_soft
    }
    feature_sets = build_feature_sets(
        raw_all
    )

    predictions = evaluate(
        clean=clean,
        conditions=condition_tables,
        feature_sets=feature_sets,
        percentages=percentages,
        repeats=args.repeats,
        c_value=args.c_value,
        seed=args.seed,
    )
    repeat_metrics = subject_repeat_metrics(
        predictions
    )
    condition_metrics = average_repeats(
        repeat_metrics
    )
    summary = summarize(
        condition_metrics
    )
    tests = build_tests(
        condition_metrics,
        percentages,
    )
    soft_quality = quality_summary(
        clean_soft=clean_soft,
        missing_soft=missing_soft,
        percentages=percentages,
        repeats=args.repeats,
    )

    paths = {
        "predictions": (
            output_dir
            / "soft_aoi_missingness_predictions.csv"
        ),
        "subject_repeat": (
            output_dir
            / "soft_aoi_missingness_subject_repeat_metrics.csv"
        ),
        "subject_condition": (
            output_dir
            / "soft_aoi_missingness_subject_condition_metrics.csv"
        ),
        "summary": (
            output_dir
            / "soft_aoi_missingness_summary.csv"
        ),
        "tests": (
            output_dir
            / "soft_aoi_missingness_pairwise_tests.csv"
        ),
        "quality": (
            output_dir
            / "soft_aoi_missingness_quality_summary.csv"
        ),
        "feature_sets": (
            output_dir
            / "soft_aoi_missingness_feature_sets.json"
        ),
        "manifest": (
            output_dir
            / "soft_aoi_missingness_manifest.json"
        ),
    }

    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    repeat_metrics.to_csv(
        paths["subject_repeat"],
        index=False,
        encoding="utf-8-sig",
    )
    condition_metrics.to_csv(
        paths["subject_condition"],
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
    soft_quality.to_csv(
        paths["quality"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["feature_sets"].write_text(
        json.dumps(
            feature_sets,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "protocol": (
                    "train_clean_test_missing_"
                    "dual_unseen_subject_unseen_sentence"
                ),
                "sampling_rate_hz": RATE_HZ,
                "modes": MODES,
                "additional_missing_percentages": (
                    percentages
                ),
                "repeats": int(args.repeats),
                "burst_seconds": float(
                    args.burst_seconds
                ),
                "c_value": float(args.c_value),
                "seed": int(args.seed),
                "sigma_word_height_ratio": float(
                    args.sigma_word_height_ratio
                ),
                "active_threshold": float(
                    args.active_threshold
                ),
                "n_subjects": EXPECTED_SUBJECTS,
                "n_trials": EXPECTED_TRIALS,
                "feature_counts": {
                    name: len(columns)
                    for name, columns in (
                        feature_sets.items()
                    )
                },
                "raw_feature_source": raw_source,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    display = summary[
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
    ]
    print(
        "\n=== SOFT-AOI MISSINGNESS SUMMARY ==="
    )
    print(display.to_string(index=False))

    key_tests = tests[
        tests["metric"].eq("macro_f1")
        & (
            tests["test_family"].eq(
                "missingness_vs_clean"
            )
            | tests["comparison"].isin(
                [
                    "soft_full_vs_rawstat",
                    "soft_compact_vs_rawstat",
                    "soft_compact_vs_full",
                ]
            )
        )
    ][
        [
            "test_family",
            "missingness_mode",
            "additional_missing_percent",
            "comparison",
            "reference_feature_set",
            "candidate_feature_set",
            "reference_mean",
            "candidate_mean",
            "mean_difference_candidate_minus_reference",
            "wins",
            "ties",
            "losses",
            "p_value",
            "p_holm_within_family_metric",
        ]
    ]
    print(
        "\n=== KEY MACRO-F1 MISSINGNESS TESTS ==="
    )
    print(key_tests.to_string(index=False))

    print(
        "\n=== SOFT-AOI QUALITY BY MISSINGNESS ==="
    )
    print(soft_quality.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
