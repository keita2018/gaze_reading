#!/usr/bin/env python
"""Evaluate Soft-AOI under simultaneous coordinate noise and missingness.

The experiment mirrors the existing RawStat combined-degradation conditions:

moderate_random
    Gaussian noise sigma=25 px + random missingness=20%.

moderate_burst
    Gaussian noise sigma=25 px + burst missingness=20%.

high_random
    Gaussian noise sigma=50 px + random missingness=40%.

high_burst
    Gaussian noise sigma=50 px + burst missingness=40%.

extreme_random
    Gaussian noise sigma=100 px + random missingness=40%.

extreme_burst
    Gaussian noise sigma=100 px + burst missingness=40%.

Protocol
--------
- Train on clean 30 Hz trials.
- Test on clean or simultaneously degraded 30 Hz trials.
- Exclude the held-out subject from training.
- Exclude every test sentence from training.
- Use fixed LinearSVC C=0.01.
- Average deterministic corruption repeats per subject and condition.

Existing RawStat combined-degradation caches are reused. Soft-AOI combined
features are cached per subject, condition, and repeat so extraction can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
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
from run_zuco_coordinate_noise_30hz import (  # noqa: E402
    add_coordinate_noise,
)
from run_zuco_missingness_30hz import (  # noqa: E402
    add_burst_missingness,
    add_random_missingness,
    current_valid_basic_mask,
)
from run_zuco_soft_aoi_ablation import (  # noqa: E402
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
from run_zuco_soft_aoi_missingness_30hz import (  # noqa: E402
    update_validity_columns,
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


@dataclass(frozen=True)
class Condition:
    name: str
    sigma_px: float
    missingness_mode: str
    missing_percent: float


CONDITIONS = [
    Condition("moderate_random", 25.0, "random", 20.0),
    Condition("moderate_burst", 25.0, "burst", 20.0),
    Condition("high_random", 50.0, "random", 40.0),
    Condition("high_burst", 50.0, "burst", 40.0),
    Condition("extreme_random", 100.0, "random", 40.0),
    Condition("extreme_burst", 100.0, "burst", 40.0),
]


def stable_seed(
    base_seed: int,
    trial_id: str,
    condition_name: str,
    repeat: int,
) -> int:
    payload = (
        f"{base_seed}|{trial_id}|{condition_name}|{repeat}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


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
    condition_name: str,
    repeat: int,
) -> Path:
    return (
        root
        / condition_name
        / f"repeat_{repeat:02d}"
        / (
            f"{subject}_{condition_name}"
            f"_repeat_{repeat:02d}.csv"
        )
    )


def soft_cache_path(
    root: Path,
    subject: str,
    condition_name: str,
    repeat: int,
) -> Path:
    return (
        root
        / condition_name
        / f"repeat_{repeat:02d}"
        / (
            f"{subject}_soft_aoi_{condition_name}"
            f"_repeat_{repeat:02d}.csv"
        )
    )


def validate_raw_caches(
    root: Path,
    subjects: list[str],
    repeats: int,
) -> None:
    missing: list[Path] = []
    for subject in subjects:
        for condition in CONDITIONS:
            for repeat in range(1, repeats + 1):
                path = raw_cache_path(
                    root,
                    subject,
                    condition.name,
                    repeat,
                )
                if not path.is_file():
                    missing.append(path)

    if missing:
        examples = "\n".join(
            f"  {path}" for path in missing[:15]
        )
        raise FileNotFoundError(
            "Existing RawStat combined-degradation caches are missing. "
            "Run run_zuco_combined_degradation_30hz.py first. "
            f"Missing={len(missing)}; examples:\n{examples}"
        )


def degrade_samples(
    sampled: pd.DataFrame,
    condition: Condition,
    repeat: int,
    seed: int,
    trial_id: str,
    burst_seconds: float,
) -> tuple[pd.DataFrame, int, int]:
    rng = np.random.default_rng(
        stable_seed(
            base_seed=seed,
            trial_id=trial_id,
            condition_name=condition.name,
            repeat=repeat,
        )
    )

    noisy = add_coordinate_noise(
        sampled=sampled,
        sigma_px=condition.sigma_px,
        rng=rng,
    )
    valid_before_missing = int(
        np.sum(current_valid_basic_mask(noisy))
    )

    if condition.missingness_mode == "random":
        degraded, removed = add_random_missingness(
            samples=noisy,
            percentage=condition.missing_percent,
            rng=rng,
        )
    elif condition.missingness_mode == "burst":
        degraded, removed = add_burst_missingness(
            samples=noisy,
            percentage=condition.missing_percent,
            burst_seconds=burst_seconds,
            rng=rng,
        )
    else:
        raise ValueError(
            f"Unknown missingness mode: "
            f"{condition.missingness_mode}"
        )

    return (
        update_validity_columns(degraded),
        valid_before_missing,
        int(removed),
    )


def validate_soft_cache(
    frame: pd.DataFrame,
    subject: str,
    expected_ids: set[str],
    condition: Condition,
    repeat: int,
) -> pd.DataFrame:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "task_sentence_index",
        "combined_condition",
        "noise_sigma_px",
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
            f"{subject} Soft-AOI combined cache lacks: {missing}"
        )

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()
    result["combined_condition"] = (
        result["combined_condition"].astype(str)
    )
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
    if set(result["combined_condition"]) != {
        condition.name
    }:
        raise ValueError(
            f"{subject} cache has wrong condition."
        )
    if not np.allclose(
        pd.to_numeric(
            result["noise_sigma_px"],
            errors="coerce",
        ),
        condition.sigma_px,
    ):
        raise ValueError(
            f"{subject} cache has wrong noise sigma."
        )
    if set(result["missingness_mode"]) != {
        condition.missingness_mode
    }:
        raise ValueError(
            f"{subject} cache has wrong missingness mode."
        )
    if not np.allclose(
        pd.to_numeric(
            result["additional_missing_percent"],
            errors="coerce",
        ),
        condition.missing_percent,
    ):
        raise ValueError(
            f"{subject} cache has wrong missingness percent."
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
    condition_repeats: list[tuple[Condition, int]],
    cache_paths: dict[tuple[str, int], Path],
    seed: int,
    burst_seconds: float,
    sigma_word_height_ratio: float,
    active_threshold: float,
) -> dict[tuple[str, int], pd.DataFrame]:
    trial_lookup = subject_map.set_index(
        "trial_id",
        drop=False,
        verify_integrity=True,
    )
    rows_by_condition: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = {
        (condition.name, repeat): []
        for condition, repeat in condition_repeats
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
            f"Soft-AOI extraction for "
            f"{len(condition_repeats)} combined conditions: "
            f"{sample_path}"
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

            for condition, repeat in condition_repeats:
                (
                    degraded,
                    valid_before_missing,
                    removed,
                ) = degrade_samples(
                    sampled=sampled,
                    condition=condition,
                    repeat=repeat,
                    seed=seed,
                    trial_id=trial_id,
                    burst_seconds=burst_seconds,
                )

                # Input is already 30 Hz. Passing 500 prevents a second
                # downsampling pass; stored rate is corrected afterward.
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
                row["combined_condition"] = condition.name
                row["noise_sigma_px"] = float(
                    condition.sigma_px
                )
                row["missingness_mode"] = (
                    condition.missingness_mode
                )
                row["additional_missing_percent"] = float(
                    condition.missing_percent
                )
                row["corruption_repeat"] = int(repeat)
                row["valid_before_missing"] = int(
                    valid_before_missing
                )
                row["additional_missing_samples"] = int(
                    removed
                )
                row[
                    "realized_additional_missing_percent"
                ] = (
                    100.0 * removed / valid_before_missing
                    if valid_before_missing > 0
                    else math.nan
                )
                rows_by_condition[
                    (condition.name, repeat)
                ].append(row)

        del samples

    expected_ids = set(subject_map["trial_id"].astype(str))
    results: dict[tuple[str, int], pd.DataFrame] = {}

    for condition, repeat in condition_repeats:
        key = (condition.name, repeat)
        frame = (
            pd.DataFrame(rows_by_condition[key])
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
            condition=condition,
            repeat=repeat,
        )
        path = cache_paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        results[key] = frame

    return results


def extract_or_load_soft_conditions(
    manifest: pd.DataFrame,
    subjects: list[str],
    processed_root: Path,
    cache_root: Path,
    aoi_lookup: dict[tuple[str, int], pd.DataFrame],
    repeats: int,
    seed: int,
    burst_seconds: float,
    sigma_word_height_ratio: float,
    active_threshold: float,
    overwrite: bool,
) -> dict[tuple[str, int], pd.DataFrame]:
    condition_repeats = [
        (condition, repeat)
        for condition in CONDITIONS
        for repeat in range(1, repeats + 1)
    ]
    condition_lookup = {
        condition.name: condition for condition in CONDITIONS
    }
    frames_by_condition: dict[
        tuple[str, int],
        list[pd.DataFrame],
    ] = {
        (condition.name, repeat): []
        for condition, repeat in condition_repeats
    }

    for subject_index, subject in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Soft-AOI combined degradation subject "
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
            (condition.name, repeat): soft_cache_path(
                cache_root,
                subject,
                condition.name,
                repeat,
            )
            for condition, repeat in condition_repeats
        }

        missing: list[tuple[Condition, int]] = []
        for condition, repeat in condition_repeats:
            key = (condition.name, repeat)
            path = cache_paths[key]
            if path.is_file() and not overwrite:
                cached = validate_soft_cache(
                    frame=pd.read_csv(path),
                    subject=subject,
                    expected_ids=expected_ids,
                    condition=condition,
                    repeat=repeat,
                )
                frames_by_condition[key].append(cached)
                print(f"Reusing Soft-AOI cache: {path}")
            else:
                missing.append((condition, repeat))

        if missing:
            extracted = extract_subject_soft_conditions(
                subject=subject,
                subject_manifest=subject_manifest,
                subject_map=subject_map,
                aoi_lookup=aoi_lookup,
                condition_repeats=missing,
                cache_paths=cache_paths,
                seed=seed,
                burst_seconds=burst_seconds,
                sigma_word_height_ratio=(
                    sigma_word_height_ratio
                ),
                active_threshold=active_threshold,
            )
            for condition, repeat in missing:
                key = (condition.name, repeat)
                frames_by_condition[key].append(
                    extracted[key]
                )

    result: dict[tuple[str, int], pd.DataFrame] = {}
    for key, frames in frames_by_condition.items():
        condition_name, repeat = key
        condition = condition_lookup[condition_name]
        combined = normalize_basic_table(
            pd.concat(frames, ignore_index=True),
            (
                f"Soft-AOI combined {condition_name} "
                f"repeat={repeat}"
            ),
        )
        # Revalidate each subject after concatenation metadata normalization.
        for subject in subjects:
            validate_soft_cache(
                frame=combined[
                    combined["subject_id"].eq(subject)
                ],
                subject=subject,
                expected_ids=set(
                    combined.loc[
                        combined["subject_id"].eq(subject),
                        "trial_id",
                    ]
                ),
                condition=condition,
                repeat=repeat,
            )
        result[key] = combined
    return result


def load_raw_conditions(
    cache_root: Path,
    subjects: list[str],
    repeats: int,
) -> dict[tuple[str, int], pd.DataFrame]:
    result: dict[tuple[str, int], pd.DataFrame] = {}
    for condition in CONDITIONS:
        for repeat in range(1, repeats + 1):
            frames = [
                pd.read_csv(
                    raw_cache_path(
                        cache_root,
                        subject,
                        condition.name,
                        repeat,
                    )
                )
                for subject in subjects
            ]
            result[(condition.name, repeat)] = (
                normalize_basic_table(
                    pd.concat(frames, ignore_index=True),
                    (
                        f"RawStat combined {condition.name} "
                        f"repeat={repeat}"
                    ),
                )
            )
    return result


def evaluate(
    clean: pd.DataFrame,
    conditions: dict[tuple[str, int], pd.DataFrame],
    feature_sets: dict[str, list[str]],
    repeats: int,
    c_value: float,
    seed: int,
) -> pd.DataFrame:
    subjects = sorted(clean["subject_id"].unique())
    sentence_folds = sorted(
        clean["sentence_fold"].unique()
    )
    rows: list[dict[str, Any]] = []

    eval_conditions: list[tuple[str, int]] = [
        ("clean", 0)
    ]
    eval_conditions.extend(
        (condition.name, repeat)
        for condition in CONDITIONS
        for repeat in range(1, repeats + 1)
    )

    for condition_index, key in enumerate(
        eval_conditions,
        start=1,
    ):
        condition_name, repeat = key
        test_table = (
            clean
            if condition_name == "clean"
            else conditions[key]
        )
        print("\n" + "#" * 78)
        print(
            f"Evaluation condition "
            f"{condition_index}/{len(eval_conditions)}: "
            f"{condition_name}, repeat={repeat}"
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
                        f"condition={condition_name}."
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
                                "combined_condition": (
                                    condition_name
                                ),
                                "corruption_repeat": int(
                                    repeat
                                ),
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
    expected_conditions = 1 + len(CONDITIONS) * repeats
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
        "combined_condition",
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
        "combined_condition",
        "corruption_repeat",
        "feature_set",
        "held_out_subject",
    ]
    for values, group in predictions.groupby(
        keys,
        sort=False,
    ):
        (
            condition_name,
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
                "combined_condition": condition_name,
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
    repeat_metrics: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "combined_condition",
        "feature_set",
        "held_out_subject",
        "n_input_features",
    ]
    result = (
        repeat_metrics.groupby(
            group_columns,
            as_index=False,
            sort=False,
        )[list(METRICS)]
        .mean()
    )
    counts = result.groupby(
        ["combined_condition", "feature_set"]
    )["held_out_subject"].nunique()
    if not counts.eq(EXPECTED_SUBJECTS).all():
        raise ValueError(
            f"Subject counts are invalid: {counts.to_dict()}"
        )
    return result


def summarize(
    condition_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    condition_lookup = {
        condition.name: condition for condition in CONDITIONS
    }

    for values, group in condition_metrics.groupby(
        ["combined_condition", "feature_set"],
        sort=False,
    ):
        condition_name, feature_set = values
        condition = condition_lookup.get(condition_name)
        row: dict[str, Any] = {
            "combined_condition": condition_name,
            "noise_sigma_px": (
                0.0 if condition is None else condition.sigma_px
            ),
            "missingness_mode": (
                "clean"
                if condition is None
                else condition.missingness_mode
            ),
            "additional_missing_percent": (
                0.0
                if condition is None
                else condition.missing_percent
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

    order = {
        "clean": 0,
        "moderate_random": 1,
        "moderate_burst": 2,
        "high_random": 3,
        "high_burst": 4,
        "extreme_random": 5,
        "extreme_burst": 6,
    }
    result = pd.DataFrame(rows)
    result["_order"] = result[
        "combined_condition"
    ].map(order)
    return result.sort_values(
        [
            "_order",
            "macro_f1_subject_mean",
        ],
        ascending=[True, False],
    ).drop(columns="_order")


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
    condition_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ]

    # Each degraded condition versus clean.
    for metric in metrics:
        for feature_set in FEATURE_SET_ORDER:
            clean = (
                condition_metrics[
                    condition_metrics[
                        "combined_condition"
                    ].eq("clean")
                    & condition_metrics[
                        "feature_set"
                    ].eq(feature_set)
                ]
                .set_index(
                    "held_out_subject"
                )[metric]
                .sort_index()
            )
            for condition in CONDITIONS:
                degraded = (
                    condition_metrics[
                        condition_metrics[
                            "combined_condition"
                        ].eq(condition.name)
                        & condition_metrics[
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
                            "combined_vs_clean"
                        ),
                        "metric": metric,
                        "combined_condition": (
                            condition.name
                        ),
                        "comparison": (
                            f"{feature_set}_"
                            f"{condition.name}_vs_clean"
                        ),
                        "reference_feature_set": (
                            feature_set
                        ),
                        "candidate_feature_set": (
                            feature_set
                        ),
                        **wilcoxon_pair(
                            clean,
                            degraded,
                        ),
                        "p_holm_within_family_metric": (
                            math.nan
                        ),
                    }
                )

    # Representation comparisons.
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
    condition_names = ["clean"] + [
        condition.name for condition in CONDITIONS
    ]

    for metric in metrics:
        for condition_name in condition_names:
            condition = condition_metrics[
                condition_metrics[
                    "combined_condition"
                ].eq(condition_name)
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
                        "combined_condition": (
                            condition_name
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
    condition_soft: dict[tuple[str, int], pd.DataFrame],
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
    clean["combined_condition"] = "clean"
    clean["corruption_repeat"] = 0
    clean["noise_sigma_px"] = 0.0
    clean["missingness_mode"] = "clean"
    clean["additional_missing_percent"] = 0.0
    clean[
        "realized_additional_missing_percent"
    ] = 0.0
    frames.append(clean)

    condition_lookup = {
        condition.name: condition for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        for repeat in range(1, repeats + 1):
            source = condition_soft[
                (condition.name, repeat)
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
            frame["combined_condition"] = (
                condition.name
            )
            frame["corruption_repeat"] = int(repeat)
            frame["noise_sigma_px"] = float(
                condition.sigma_px
            )
            frame["missingness_mode"] = (
                condition.missingness_mode
            )
            frame[
                "additional_missing_percent"
            ] = float(condition.missing_percent)
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
                "combined_condition",
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
                "combined_condition",
                "subject_id",
            ],
            as_index=False,
        )[aggregation_columns]
        .mean()
    )

    rows: list[dict[str, Any]] = []
    for condition_name, group in subject_condition.groupby(
        "combined_condition",
        sort=False,
    ):
        condition = condition_lookup.get(condition_name)
        row: dict[str, Any] = {
            "combined_condition": condition_name,
            "noise_sigma_px": (
                0.0 if condition is None else condition.sigma_px
            ),
            "missingness_mode": (
                "clean"
                if condition is None
                else condition.missingness_mode
            ),
            "additional_missing_percent": (
                0.0
                if condition is None
                else condition.missing_percent
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
        "--raw-combined-cache-root",
        default=(
            "outputs/combined_degradation_30hz/"
            "subject_combined_features"
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
        default=(
            "outputs/"
            "soft_aoi_combined_degradation_30hz"
        ),
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
        root=Path(args.raw_combined_cache_root),
        subjects=subjects,
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
        / "subject_soft_aoi_combined_features"
    )

    print(
        "=== SOFT-AOI COMBINED DEGRADATION INPUT ==="
    )
    print(f"subjects: {len(subjects)}")
    print(f"trials: {EXPECTED_TRIALS}")
    print(
        "conditions: "
        + ", ".join(
            condition.name for condition in CONDITIONS
        )
    )
    print(f"repeats: {args.repeats}")
    print(f"burst seconds: {args.burst_seconds}")
    print(f"fixed C: {args.c_value}")
    print(
        f"RawStat feature source: {raw_source}"
    )
    print(
        f"Soft-AOI cache root: {soft_cache_root}"
    )

    condition_soft = extract_or_load_soft_conditions(
        manifest=manifest,
        subjects=subjects,
        processed_root=Path(
            args.processed_root
        ),
        cache_root=soft_cache_root,
        aoi_lookup=aoi_lookup,
        repeats=args.repeats,
        seed=args.seed,
        burst_seconds=args.burst_seconds,
        sigma_word_height_ratio=(
            args.sigma_word_height_ratio
        ),
        active_threshold=args.active_threshold,
        overwrite=args.overwrite_soft_features,
    )
    condition_raw = load_raw_conditions(
        cache_root=Path(
            args.raw_combined_cache_root
        ),
        subjects=subjects,
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
        key: merge_condition(
            raw=condition_raw[key],
            soft=condition_soft[key],
            raw_columns=raw_all,
            soft_columns=soft_all,
            sentence_folds=sentence_folds,
            name=(
                f"{key[0]} repeat={key[1]}"
            ),
        )
        for key in condition_soft
    }
    feature_sets = build_feature_sets(
        raw_all
    )

    predictions = evaluate(
        clean=clean,
        conditions=condition_tables,
        feature_sets=feature_sets,
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
        condition_metrics
    )
    soft_quality = quality_summary(
        clean_soft=clean_soft,
        condition_soft=condition_soft,
        repeats=args.repeats,
    )

    paths = {
        "predictions": (
            output_dir
            / "soft_aoi_combined_predictions.csv"
        ),
        "subject_repeat": (
            output_dir
            / "soft_aoi_combined_subject_repeat_metrics.csv"
        ),
        "subject_condition": (
            output_dir
            / "soft_aoi_combined_subject_condition_metrics.csv"
        ),
        "summary": (
            output_dir
            / "soft_aoi_combined_summary.csv"
        ),
        "tests": (
            output_dir
            / "soft_aoi_combined_pairwise_tests.csv"
        ),
        "quality": (
            output_dir
            / "soft_aoi_combined_quality_summary.csv"
        ),
        "feature_sets": (
            output_dir
            / "soft_aoi_combined_feature_sets.json"
        ),
        "manifest": (
            output_dir
            / "soft_aoi_combined_manifest.json"
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
                    "train_clean_test_combined_degradation_"
                    "dual_unseen_subject_unseen_sentence"
                ),
                "sampling_rate_hz": RATE_HZ,
                "conditions": [
                    {
                        "name": condition.name,
                        "noise_sigma_px": (
                            condition.sigma_px
                        ),
                        "missingness_mode": (
                            condition.missingness_mode
                        ),
                        "additional_missing_percent": (
                            condition.missing_percent
                        ),
                    }
                    for condition in CONDITIONS
                ],
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
            "combined_condition",
            "noise_sigma_px",
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
        "\n=== SOFT-AOI COMBINED DEGRADATION SUMMARY ==="
    )
    print(display.to_string(index=False))

    key_tests = tests[
        tests["metric"].eq("macro_f1")
        & (
            tests["test_family"].eq(
                "combined_vs_clean"
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
            "combined_condition",
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
        "\n=== KEY MACRO-F1 COMBINED TESTS ==="
    )
    print(key_tests.to_string(index=False))

    print(
        "\n=== SOFT-AOI QUALITY BY COMBINED DEGRADATION ==="
    )
    print(soft_quality.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
