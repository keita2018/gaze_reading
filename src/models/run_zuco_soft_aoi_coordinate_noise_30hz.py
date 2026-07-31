#!/usr/bin/env python
"""Evaluate Soft-AOI robustness to Gaussian coordinate noise at 30 Hz.

The script has two responsibilities:

1. Recompute Soft-AOI features after adding deterministic Gaussian noise to
   the same 30 Hz samples used by the existing RawStat coordinate-noise
   experiment.
2. Evaluate clean-trained models on noisy held-out trials under the strict
   unseen-subject + unseen-sentence protocol.

The existing RawStat noise caches are reused. No RawStat features are
recomputed. Soft-AOI noise features are cached per subject, sigma, and repeat,
so interrupted extraction can be resumed.

Feature sets
------------
rawstat_all
    Existing 18-dimensional rate-invariant RawStat vector.

soft_full_all
    All 41 Soft-AOI features.

soft_compact_distribution_membership
    The 27-dimensional rate-robust candidate:
    15 distribution features + 12 membership/quality features.

Evaluation
----------
- train clean 30 Hz, test noisy 30 Hz;
- held-out subject is absent from training;
- every test sentence is absent from training;
- LinearSVC C is fixed at 0.01;
- three deterministic corruption repeats by default.
"""

from __future__ import annotations

import argparse
import hashlib
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


def coordinate_noise_seed(
    base_seed: int,
    trial_id: str,
    sigma_px: float,
    repeat: int,
) -> int:
    payload = (
        f"{base_seed}|{trial_id}|{sigma_px:.8f}|{repeat}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def add_coordinate_noise(
    sampled: pd.DataFrame,
    sigma_px: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = sampled.copy()
    if sigma_px <= 0:
        return result

    x = result["x_px"].to_numpy(dtype=float)
    y = result["y_px"].to_numpy(dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )
    if np.any(valid):
        x[valid] = np.maximum(
            x[valid]
            + rng.normal(0.0, sigma_px, int(np.sum(valid))),
            1e-6,
        )
        y[valid] = np.maximum(
            y[valid]
            + rng.normal(0.0, sigma_px, int(np.sum(valid))),
            1e-6,
        )
    result["x_px"] = x
    result["y_px"] = y
    return result


def parse_sigmas(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values:
        raise ValueError("--sigmas is empty.")
    if any(value <= 0 for value in values):
        raise ValueError("Every noise sigma must be positive.")
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


def normalize_basic_table(
    frame: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    required = {"trial_id", "subject_id", "task"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks columns: {missing}")

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()

    if len(result) != EXPECTED_TRIALS:
        raise ValueError(
            f"{name} rows={len(result)}, expected={EXPECTED_TRIALS}."
        )
    if result["trial_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate trial IDs.")
    if result["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"{name} subjects={result['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )
    counts = result.groupby("task").size().to_dict()
    if (
        int(counts.get("NR", 0)) != EXPECTED_NR
        or int(counts.get("TSR", 0)) != EXPECTED_TSR
    ):
        raise ValueError(f"{name} task counts are invalid: {counts}")
    return result


def load_soft_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = list(manifest.get("content_feature_columns", []))
    quality = list(
        manifest.get("membership_quality_feature_columns", [])
    )
    if content != CONTENT_FEATURES:
        raise ValueError(
            "Soft manifest content features differ from extractor."
        )
    if quality != MEMBERSHIP_QUALITY_FEATURES:
        raise ValueError(
            "Soft manifest quality features differ from extractor."
        )
    return manifest


def load_word_template(
    path: Path,
) -> dict[tuple[str, int], pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "task",
        "task_sentence_index",
        "word_index",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "height",
        "word_progress",
        "line_index",
        "line_progress",
        "line_index_norm",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Word template lacks columns: {missing}")
    frame["task"] = frame["task"].astype(str).str.upper()
    frame["task_sentence_index"] = pd.to_numeric(
        frame["task_sentence_index"], errors="raise"
    ).astype(int)

    lookup = {
        (str(task), int(index)): group.sort_values("word_index")
        for (task, index), group in frame.groupby(
            ["task", "task_sentence_index"],
            sort=False,
        )
    }
    if len(lookup) != 739:
        raise ValueError(
            f"Word-template sentence count={len(lookup)}, expected=739."
        )
    return lookup


def raw_noise_cache_path(
    root: Path,
    subject: str,
    sigma_px: float,
    repeat: int,
) -> Path:
    return (
        root
        / f"sigma_{sigma_px:06.1f}px"
        / f"repeat_{repeat:02d}"
        / (
            f"{subject}_sigma_{sigma_px:06.1f}"
            f"_repeat_{repeat:02d}.csv"
        )
    )


def soft_noise_cache_path(
    root: Path,
    subject: str,
    sigma_px: float,
    repeat: int,
) -> Path:
    return (
        root
        / f"sigma_{sigma_px:06.1f}px"
        / f"repeat_{repeat:02d}"
        / (
            f"{subject}_soft_aoi_sigma_{sigma_px:06.1f}"
            f"_repeat_{repeat:02d}.csv"
        )
    )


def validate_raw_noise_caches(
    root: Path,
    subjects: list[str],
    sigmas: list[float],
    repeats: int,
) -> None:
    missing: list[Path] = []
    for subject in subjects:
        for sigma_px in sigmas:
            for repeat in range(1, repeats + 1):
                path = raw_noise_cache_path(
                    root, subject, sigma_px, repeat
                )
                if not path.is_file():
                    missing.append(path)
    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:15])
        raise FileNotFoundError(
            "Existing RawStat coordinate-noise caches are missing. "
            "Run run_zuco_coordinate_noise_30hz.py first. "
            f"Missing={len(missing)}; examples:\n{examples}"
        )


def validate_subject_soft_cache(
    frame: pd.DataFrame,
    subject: str,
    expected_ids: set[str],
    sigma_px: float,
    repeat: int,
) -> pd.DataFrame:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "task_sentence_index",
        "noise_sigma_px",
        "corruption_repeat",
        *CONTENT_FEATURES,
        *MEMBERSHIP_QUALITY_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{subject} Soft-AOI noise cache lacks columns: {missing}"
        )

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()

    if len(result) != EXPECTED_PER_SUBJECT:
        raise ValueError(
            f"{subject} cache rows={len(result)}, "
            f"expected={EXPECTED_PER_SUBJECT}."
        )
    if set(result["trial_id"]) != expected_ids:
        raise ValueError(f"{subject} cache trial IDs do not match.")
    if result["trial_id"].duplicated().any():
        raise ValueError(f"{subject} cache contains duplicate IDs.")
    if set(result["subject_id"]) != {subject}:
        raise ValueError(f"{subject} cache subject column is invalid.")
    if not np.allclose(
        pd.to_numeric(
            result["noise_sigma_px"], errors="coerce"
        ),
        sigma_px,
    ):
        raise ValueError(f"{subject} cache has the wrong sigma.")
    if not pd.to_numeric(
        result["corruption_repeat"], errors="coerce"
    ).eq(repeat).all():
        raise ValueError(f"{subject} cache has the wrong repeat.")

    numeric_columns = CONTENT_FEATURES + MEMBERSHIP_QUALITY_FEATURES
    if np.isinf(
        result[numeric_columns].to_numpy(dtype=float)
    ).any():
        raise ValueError(f"{subject} cache contains infinity.")
    return result


def extract_subject_soft_noise(
    subject: str,
    subject_manifest: pd.DataFrame,
    subject_map: pd.DataFrame,
    aoi_lookup: dict[tuple[str, int], pd.DataFrame],
    conditions: list[tuple[float, int]],
    cache_paths: dict[tuple[float, int], Path],
    seed: int,
    sigma_word_height_ratio: float,
    active_threshold: float,
) -> dict[tuple[float, int], pd.DataFrame]:
    # Keep trial_id both as the lookup index and as a normal column.
    # extract_trial_features expects trial_row["trial_id"] to exist.
    trial_lookup = subject_map.set_index(
        "trial_id",
        drop=False,
        verify_integrity=True,
    )
    rows_by_condition: dict[
        tuple[float, int], list[dict[str, Any]]
    ] = {condition: [] for condition in conditions}

    block_manifest = subject_manifest.sort_values(
        ["task", "block_id"]
    )
    if len(block_manifest) != 14:
        raise ValueError(
            f"{subject} manifest rows={len(block_manifest)}, expected=14."
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
            f"Soft-AOI extraction for {len(conditions)} noise conditions: "
            f"{sample_path}"
        )
        samples = pd.read_parquet(sample_path)
        required = {
            "trial_id",
            "sample_index_in_trial",
            "timestamp_ms_abs",
            "x_px",
            "y_px",
            "valid_basic",
        }
        missing = sorted(required - set(samples.columns))
        if missing:
            raise ValueError(
                f"{sample_path} lacks sample columns: {missing}"
            )

        for trial_id_value, trial_samples in samples.groupby(
            "trial_id", sort=False
        ):
            trial_id = str(trial_id_value)
            if trial_id not in trial_lookup.index:
                raise ValueError(
                    f"Trial missing from subject map: {trial_id}"
                )
            trial_row = trial_lookup.loc[trial_id]
            key = (
                str(trial_row["task"]).upper(),
                int(trial_row["task_sentence_index"]),
            )
            if key not in aoi_lookup:
                raise ValueError(f"AOI template missing for key={key}")

            sampled = soft_downsample_trial(
                trial_samples, RATE_HZ
            )

            for sigma_px, repeat in conditions:
                rng = np.random.default_rng(
                    coordinate_noise_seed(
                        base_seed=seed,
                        trial_id=trial_id,
                        sigma_px=sigma_px,
                        repeat=repeat,
                    )
                )
                degraded = add_coordinate_noise(
                    sampled=sampled,
                    sigma_px=sigma_px,
                    rng=rng,
                )
                # The input is already sampled at 30 Hz. Passing 500 prevents
                # a second downsampling pass; the stored rate is corrected.
                row = extract_soft_trial_features(
                    samples=degraded,
                    trial_row=trial_row,
                    aoi=aoi_lookup[key],
                    rate_hz=500,
                    sigma_word_height_ratio=sigma_word_height_ratio,
                    active_threshold=active_threshold,
                )
                row["soft_aoi_rate_hz"] = RATE_HZ
                row["noise_sigma_px"] = float(sigma_px)
                row["corruption_repeat"] = int(repeat)
                rows_by_condition[(sigma_px, repeat)].append(row)

        del samples

    expected_ids = set(subject_map["trial_id"].astype(str))
    results: dict[tuple[float, int], pd.DataFrame] = {}

    for sigma_px, repeat in conditions:
        condition = (sigma_px, repeat)
        frame = (
            pd.DataFrame(rows_by_condition[condition])
            .sort_values(
                ["task", "block_id", "trial_index_in_block"]
            )
            .reset_index(drop=True)
        )
        frame = validate_subject_soft_cache(
            frame=frame,
            subject=subject,
            expected_ids=expected_ids,
            sigma_px=sigma_px,
            repeat=repeat,
        )
        path = cache_paths[condition]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        results[condition] = frame

    return results


def prepare_subject_map(
    path: Path,
    subject: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "trial_id",
        "subject_id",
        "task",
        "block_id",
        "trial_index_in_block",
        "task_sentence_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")

    frame["trial_id"] = frame["trial_id"].astype(str)
    frame["subject_id"] = (
        frame["subject_id"].astype(str).str.upper()
    )
    frame["task"] = frame["task"].astype(str).str.upper()
    frame["task_sentence_index"] = pd.to_numeric(
        frame["task_sentence_index"], errors="raise"
    ).astype(int)

    if len(frame) != EXPECTED_PER_SUBJECT:
        raise ValueError(
            f"{subject} map rows={len(frame)}, expected={EXPECTED_PER_SUBJECT}."
        )
    if set(frame["subject_id"]) != {subject}:
        raise ValueError(f"{subject} map contains another subject.")
    if frame["trial_id"].duplicated().any():
        raise ValueError(f"{subject} map contains duplicate IDs.")
    return frame


def extract_or_load_soft_noise(
    manifest: pd.DataFrame,
    subjects: list[str],
    processed_root: Path,
    cache_root: Path,
    aoi_lookup: dict[tuple[str, int], pd.DataFrame],
    sigmas: list[float],
    repeats: int,
    seed: int,
    sigma_word_height_ratio: float,
    active_threshold: float,
    overwrite: bool,
) -> dict[tuple[float, int], pd.DataFrame]:
    conditions = [
        (sigma_px, repeat)
        for sigma_px in sigmas
        for repeat in range(1, repeats + 1)
    ]
    frames_by_condition: dict[
        tuple[float, int], list[pd.DataFrame]
    ] = {condition: [] for condition in conditions}

    for subject_index, subject in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"Soft-AOI noise subject {subject_index}/{len(subjects)}: "
            f"{subject}"
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
            condition: soft_noise_cache_path(
                cache_root,
                subject,
                condition[0],
                condition[1],
            )
            for condition in conditions
        }

        missing_conditions: list[tuple[float, int]] = []
        for sigma_px, repeat in conditions:
            path = cache_paths[(sigma_px, repeat)]
            if path.is_file() and not overwrite:
                cached = pd.read_csv(path)
                cached = validate_subject_soft_cache(
                    frame=cached,
                    subject=subject,
                    expected_ids=expected_ids,
                    sigma_px=sigma_px,
                    repeat=repeat,
                )
                frames_by_condition[(sigma_px, repeat)].append(
                    cached
                )
                print(f"Reusing Soft-AOI cache: {path}")
            else:
                missing_conditions.append((sigma_px, repeat))

        if missing_conditions:
            extracted = extract_subject_soft_noise(
                subject=subject,
                subject_manifest=subject_manifest,
                subject_map=subject_map,
                aoi_lookup=aoi_lookup,
                conditions=missing_conditions,
                cache_paths=cache_paths,
                seed=seed,
                sigma_word_height_ratio=sigma_word_height_ratio,
                active_threshold=active_threshold,
            )
            for condition in missing_conditions:
                frames_by_condition[condition].append(
                    extracted[condition]
                )

    combined: dict[tuple[float, int], pd.DataFrame] = {}
    for condition in conditions:
        frame = normalize_basic_table(
            pd.concat(
                frames_by_condition[condition],
                ignore_index=True,
            ),
            (
                f"Soft-AOI noise sigma={condition[0]}, "
                f"repeat={condition[1]}"
            ),
        )
        combined[condition] = frame
    return combined


def load_raw_noise_conditions(
    cache_root: Path,
    subjects: list[str],
    sigmas: list[float],
    repeats: int,
) -> dict[tuple[float, int], pd.DataFrame]:
    result: dict[tuple[float, int], pd.DataFrame] = {}
    for sigma_px in sigmas:
        for repeat in range(1, repeats + 1):
            frames = [
                pd.read_csv(
                    raw_noise_cache_path(
                        cache_root,
                        subject,
                        sigma_px,
                        repeat,
                    )
                )
                for subject in subjects
            ]
            result[(sigma_px, repeat)] = normalize_basic_table(
                pd.concat(frames, ignore_index=True),
                (
                    f"RawStat noise sigma={sigma_px}, "
                    f"repeat={repeat}"
                ),
            )
    return result


def merge_condition(
    raw: pd.DataFrame,
    soft: pd.DataFrame,
    raw_columns: list[str],
    soft_columns: list[str],
    sentence_folds: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    missing_raw = sorted(set(raw_columns) - set(raw.columns))
    missing_soft = sorted(set(soft_columns) - set(soft.columns))
    if missing_raw:
        raise ValueError(f"{name} RawStat lacks: {missing_raw}")
    if missing_soft:
        raise ValueError(f"{name} Soft-AOI lacks: {missing_soft}")

    raw = raw.copy()
    soft = soft.copy()
    for column in raw_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    for column in soft_columns:
        soft[column] = pd.to_numeric(soft[column], errors="coerce")

    if np.isinf(raw[raw_columns].to_numpy(dtype=float)).any():
        raise ValueError(f"{name} RawStat contains infinity.")
    if np.isinf(soft[soft_columns].to_numpy(dtype=float)).any():
        raise ValueError(f"{name} Soft-AOI contains infinity.")

    raw_rename = {
        column: f"raw__{column}" for column in raw_columns
    }
    soft_rename = {
        column: f"soft__{column}" for column in soft_columns
    }

    raw_small = raw[
        ["trial_id", "subject_id", "task"] + raw_columns
    ].rename(columns=raw_rename)
    soft_small = soft[
        [
            "trial_id",
            "subject_id",
            "task",
            "task_sentence_index",
        ]
        + soft_columns
    ].rename(columns=soft_rename)

    merged = raw_small.merge(
        soft_small,
        on=["trial_id", "subject_id", "task"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts = merged["_merge"].value_counts().to_dict()
    if int(counts.get("both", 0)) != EXPECTED_TRIALS:
        raise ValueError(f"{name} Raw/Soft merge failed: {counts}")
    merged = merged.drop(columns="_merge")
    merged["task_sentence_index"] = pd.to_numeric(
        merged["task_sentence_index"], errors="raise"
    ).astype(int)
    merged["sentence_key"] = merged.apply(
        lambda row: (
            f"{str(row['task']).upper()}:"
            f"{int(row['task_sentence_index']):03d}"
        ),
        axis=1,
    )
    merged = merged.merge(
        sentence_folds[["sentence_key", "sentence_fold"]],
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if merged["sentence_fold"].isna().any():
        raise ValueError(f"{name} has trials without sentence folds.")
    merged["sentence_fold"] = merged["sentence_fold"].astype(int)
    merged["label_nr"] = (
        merged["task"].map({"NR": 1, "TSR": 0}).astype(int)
    )
    return merged.sort_values("trial_id").reset_index(drop=True)


def build_feature_sets(
    raw_columns: list[str],
) -> dict[str, list[str]]:
    soft_all = CONTENT_FEATURES + MEMBERSHIP_QUALITY_FEATURES
    compact = DISTRIBUTION_BASE + MEMBERSHIP_QUALITY_FEATURES

    if len(raw_columns) != 18:
        raise ValueError(
            f"RawStat feature count={len(raw_columns)}, expected=18."
        )
    if len(soft_all) != 41:
        raise ValueError(
            f"Full Soft-AOI count={len(soft_all)}, expected=41."
        )
    if len(compact) != 27:
        raise ValueError(
            f"Compact Soft-AOI count={len(compact)}, expected=27."
        )

    return {
        "rawstat_all": [
            f"raw__{column}" for column in raw_columns
        ],
        "soft_full_all": [
            f"soft__{column}" for column in soft_all
        ],
        COMPACT_NAME: [
            f"soft__{column}" for column in compact
        ],
    }


def evaluate_clean_train_noisy_test(
    clean: pd.DataFrame,
    condition_tables: dict[tuple[float, int], pd.DataFrame],
    feature_sets: dict[str, list[str]],
    sigmas: list[float],
    repeats: int,
    c_value: float,
    seed: int,
) -> pd.DataFrame:
    subjects = sorted(clean["subject_id"].unique())
    sentence_fold_values = sorted(clean["sentence_fold"].unique())
    rows: list[dict[str, Any]] = []

    conditions: list[tuple[float, int]] = [(0.0, 0)]
    conditions.extend(
        (sigma_px, repeat)
        for sigma_px in sigmas
        for repeat in range(1, repeats + 1)
    )

    for condition_index, (sigma_px, repeat) in enumerate(
        conditions,
        start=1,
    ):
        test_table = (
            clean
            if sigma_px == 0
            else condition_tables[(sigma_px, repeat)]
        )
        print("\n" + "#" * 78)
        print(
            f"Evaluation condition {condition_index}/{len(conditions)}: "
            f"sigma={sigma_px:g}px, repeat={repeat}"
        )

        for subject_index, held_out in enumerate(subjects, start=1):
            print(
                f"  Subject {subject_index:02d}/{len(subjects)}: "
                f"{held_out}"
            )
            for sentence_fold in sentence_fold_values:
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
                        f"fold={sentence_fold}, sigma={sigma_px}."
                    )
                if set(train["sentence_key"]) & set(
                    test["sentence_key"]
                ):
                    raise ValueError("Sentence leakage detected.")
                if held_out in set(train["subject_id"]):
                    raise ValueError("Subject leakage detected.")

                y_train = train["label_nr"].to_numpy(dtype=int)
                y_test = test["label_nr"].to_numpy(dtype=int)

                for feature_set in FEATURE_SET_ORDER:
                    columns = feature_sets[feature_set]
                    model = make_svm(c_value, seed)
                    model.fit(train[columns], y_train)
                    y_pred = model.predict(test[columns]).astype(int)
                    scores = decision_scores(model, test[columns])
                    metrics = score_predictions(
                        y_test, y_pred, scores
                    )

                    ordered = test.reset_index(drop=True)
                    for index, row in ordered.iterrows():
                        rows.append(
                            {
                                "noise_sigma_px": float(sigma_px),
                                "corruption_repeat": int(repeat),
                                "feature_set": feature_set,
                                "held_out_subject": held_out,
                                "sentence_fold": int(sentence_fold),
                                "n_input_features": int(len(columns)),
                                "trial_id": row["trial_id"],
                                "subject_id": row["subject_id"],
                                "task": row["task"],
                                "sentence_key": row["sentence_key"],
                                "label_nr": int(row["label_nr"]),
                                "prediction_nr": int(y_pred[index]),
                                "decision_score_nr": float(scores[index]),
                            }
                        )

    predictions = pd.DataFrame(rows)
    expected_conditions = 1 + len(sigmas) * repeats
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
        "noise_sigma_px",
        "corruption_repeat",
        "feature_set",
        "trial_id",
    ]
    if predictions[duplicate_columns].duplicated().any():
        raise ValueError("Predictions contain duplicate condition rows.")
    return predictions


def build_subject_repeat_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "noise_sigma_px",
        "corruption_repeat",
        "feature_set",
        "held_out_subject",
    ]
    for values, group in predictions.groupby(keys, sort=False):
        sigma_px, repeat, feature_set, subject = values
        metrics = score_predictions(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction_nr"].to_numpy(dtype=int),
            group["decision_score_nr"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "noise_sigma_px": float(sigma_px),
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
    if not result["n_predictions"].eq(EXPECTED_PER_SUBJECT).all():
        raise ValueError(
            "Not every subject/repeat metric uses 739 trials."
        )
    return result


def average_repeats(
    subject_repeat: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = list(METRICS)
    group_columns = [
        "noise_sigma_px",
        "feature_set",
        "held_out_subject",
        "n_input_features",
    ]
    result = (
        subject_repeat.groupby(
            group_columns,
            as_index=False,
            sort=False,
        )[metric_columns]
        .mean()
    )
    counts = result.groupby(
        ["noise_sigma_px", "feature_set"]
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
    keys = ["noise_sigma_px", "feature_set"]
    for values, group in subject_condition.groupby(keys, sort=False):
        sigma_px, feature_set = values
        row: dict[str, Any] = {
            "noise_sigma_px": float(sigma_px),
            "feature_set": feature_set,
            "n_subjects": int(len(group)),
            "n_input_features": int(
                group["n_input_features"].iloc[0]
            ),
        }
        for metric in METRICS:
            numeric = pd.to_numeric(
                group[metric], errors="coerce"
            )
            row[f"{metric}_subject_mean"] = float(numeric.mean())
            row[f"{metric}_subject_std"] = float(
                numeric.std(ddof=1)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["noise_sigma_px", "macro_f1_subject_mean"],
        ascending=[True, False],
    )


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
    differences = paired["candidate"] - paired["reference"]

    if len(paired) == 0:
        statistic = math.nan
        p_value = math.nan
    elif np.allclose(differences.to_numpy(dtype=float), 0.0):
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
        "reference_mean": float(paired["reference"].mean()),
        "candidate_mean": float(paired["candidate"].mean()),
        "mean_difference_candidate_minus_reference": float(
            differences.mean()
        ),
        "wins": int(np.sum(differences > 0)),
        "ties": int(np.sum(np.isclose(differences, 0.0))),
        "losses": int(np.sum(differences < 0)),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def build_tests(
    subject_condition: pd.DataFrame,
    sigmas: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Degradation versus clean within each feature set.
    for metric in [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ]:
        for feature_set in FEATURE_SET_ORDER:
            clean = (
                subject_condition[
                    subject_condition["noise_sigma_px"].eq(0)
                    & subject_condition["feature_set"].eq(
                        feature_set
                    )
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            for sigma_px in sigmas:
                noisy = (
                    subject_condition[
                        subject_condition[
                            "noise_sigma_px"
                        ].eq(sigma_px)
                        & subject_condition["feature_set"].eq(
                            feature_set
                        )
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )
                rows.append(
                    {
                        "test_family": "noise_vs_clean",
                        "metric": metric,
                        "noise_sigma_px": float(sigma_px),
                        "comparison": (
                            f"{feature_set}_sigma_{sigma_px:g}_vs_clean"
                        ),
                        "reference_feature_set": feature_set,
                        "candidate_feature_set": feature_set,
                        **wilcoxon_pair(clean, noisy),
                        "p_holm_within_family_metric": math.nan,
                    }
                )

    # Feature comparisons at each noise level including clean.
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
    for metric in [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ]:
        for sigma_px in [0.0] + list(sigmas):
            condition = subject_condition[
                subject_condition["noise_sigma_px"].eq(
                    sigma_px
                )
            ]
            for reference_name, candidate_name, comparison in comparisons:
                reference = (
                    condition[
                        condition["feature_set"].eq(
                            reference_name
                        )
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )
                candidate = (
                    condition[
                        condition["feature_set"].eq(
                            candidate_name
                        )
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )
                rows.append(
                    {
                        "test_family": "feature_comparison",
                        "metric": metric,
                        "noise_sigma_px": float(sigma_px),
                        "comparison": comparison,
                        "reference_feature_set": reference_name,
                        "candidate_feature_set": candidate_name,
                        **wilcoxon_pair(reference, candidate),
                        "p_holm_within_family_metric": math.nan,
                    }
                )

    tests = pd.DataFrame(rows)
    for (family, metric), indices in tests.groupby(
        ["test_family", "metric"]
    ).groups.items():
        positions = list(indices)
        adjusted = holm_adjust(
            tests.loc[positions, "p_value"].tolist()
        )
        tests.loc[
            positions,
            "p_holm_within_family_metric",
        ] = adjusted
    return tests


def summarize_soft_quality(
    clean_soft: pd.DataFrame,
    noisy_soft: dict[tuple[float, int], pd.DataFrame],
    sigmas: list[float],
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
        ["trial_id", "subject_id", "task"] + quality_columns
    ].copy()
    clean["noise_sigma_px"] = 0.0
    clean["corruption_repeat"] = 0
    frames.append(clean)

    for sigma_px in sigmas:
        for repeat in range(1, repeats + 1):
            frame = noisy_soft[(sigma_px, repeat)][
                ["trial_id", "subject_id", "task"] + quality_columns
            ].copy()
            frame["noise_sigma_px"] = float(sigma_px)
            frame["corruption_repeat"] = int(repeat)
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    subject_repeat = (
        combined.groupby(
            [
                "noise_sigma_px",
                "corruption_repeat",
                "subject_id",
            ],
            as_index=False,
        )[quality_columns]
        .median()
    )
    subject_condition = (
        subject_repeat.groupby(
            ["noise_sigma_px", "subject_id"],
            as_index=False,
        )[quality_columns]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for sigma_px, group in subject_condition.groupby(
        "noise_sigma_px", sort=True
    ):
        row: dict[str, Any] = {
            "noise_sigma_px": float(sigma_px),
            "n_subjects": int(group["subject_id"].nunique()),
        }
        for column in quality_columns:
            row[f"{column}_subject_median_mean"] = float(
                group[column].mean()
            )
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
        "--raw-noise-cache-root",
        default=(
            "outputs/coordinate_noise_30hz/"
            "subject_noise_features"
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
        default="outputs/soft_aoi_coordinate_noise_30hz",
    )
    parser.add_argument(
        "--sigmas",
        default="10,25,50,100",
    )
    parser.add_argument("--repeats", type=int, default=3)
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

    sigmas = parse_sigmas(args.sigmas)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.c_value <= 0:
        raise ValueError("--c-value must be positive.")
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
        manifest["subject_id"].astype(str).str.upper()
    )
    manifest["task"] = manifest["task"].astype(str).str.upper()
    subjects = sorted(manifest["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Manifest subjects={len(subjects)}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    validate_raw_noise_caches(
        root=Path(args.raw_noise_cache_root),
        subjects=subjects,
        sigmas=sigmas,
        repeats=args.repeats,
    )

    raw_sets_path = Path(args.raw_feature_sets)
    if not raw_sets_path.is_file():
        raise FileNotFoundError(raw_sets_path)
    raw_sets = json.loads(
        raw_sets_path.read_text(encoding="utf-8")
    )
    _, raw_all, raw_source = derive_raw_feature_sets(raw_sets)

    load_soft_manifest(Path(args.soft_feature_manifest))
    aoi_lookup = load_word_template(Path(args.word_template))
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
    output_dir.mkdir(parents=True, exist_ok=True)
    soft_cache_root = (
        output_dir / "subject_soft_aoi_noise_features"
    )

    print("=== SOFT-AOI COORDINATE-NOISE INPUT ===")
    print(f"subjects: {len(subjects)}")
    print(f"trials: {EXPECTED_TRIALS}")
    print(f"sigmas: {sigmas}")
    print(f"repeats: {args.repeats}")
    print(f"fixed C: {args.c_value}")
    print(f"RawStat feature source: {raw_source}")
    print(f"Soft-AOI cache root: {soft_cache_root}")

    noisy_soft = extract_or_load_soft_noise(
        manifest=manifest,
        subjects=subjects,
        processed_root=Path(args.processed_root),
        cache_root=soft_cache_root,
        aoi_lookup=aoi_lookup,
        sigmas=sigmas,
        repeats=args.repeats,
        seed=args.seed,
        sigma_word_height_ratio=(
            args.sigma_word_height_ratio
        ),
        active_threshold=args.active_threshold,
        overwrite=args.overwrite_soft_features,
    )
    noisy_raw = load_raw_noise_conditions(
        cache_root=Path(args.raw_noise_cache_root),
        subjects=subjects,
        sigmas=sigmas,
        repeats=args.repeats,
    )

    soft_all = CONTENT_FEATURES + MEMBERSHIP_QUALITY_FEATURES
    clean = merge_condition(
        raw=clean_raw,
        soft=clean_soft,
        raw_columns=raw_all,
        soft_columns=soft_all,
        sentence_folds=sentence_folds,
        name="clean",
    )
    conditions = {
        condition: merge_condition(
            raw=noisy_raw[condition],
            soft=noisy_soft[condition],
            raw_columns=raw_all,
            soft_columns=soft_all,
            sentence_folds=sentence_folds,
            name=(
                f"sigma={condition[0]}, "
                f"repeat={condition[1]}"
            ),
        )
        for condition in noisy_soft
    }
    feature_sets = build_feature_sets(raw_all)

    predictions = evaluate_clean_train_noisy_test(
        clean=clean,
        condition_tables=conditions,
        feature_sets=feature_sets,
        sigmas=sigmas,
        repeats=args.repeats,
        c_value=args.c_value,
        seed=args.seed,
    )
    subject_repeat = build_subject_repeat_metrics(predictions)
    subject_condition = average_repeats(subject_repeat)
    summary = summarize(subject_condition)
    tests = build_tests(subject_condition, sigmas)
    quality_summary = summarize_soft_quality(
        clean_soft=clean_soft,
        noisy_soft=noisy_soft,
        sigmas=sigmas,
        repeats=args.repeats,
    )

    paths = {
        "predictions": (
            output_dir
            / "soft_aoi_coordinate_noise_predictions.csv"
        ),
        "subject_repeat": (
            output_dir
            / "soft_aoi_coordinate_noise_subject_repeat_metrics.csv"
        ),
        "subject_condition": (
            output_dir
            / "soft_aoi_coordinate_noise_subject_condition_metrics.csv"
        ),
        "summary": (
            output_dir
            / "soft_aoi_coordinate_noise_summary.csv"
        ),
        "tests": (
            output_dir
            / "soft_aoi_coordinate_noise_pairwise_tests.csv"
        ),
        "quality": (
            output_dir
            / "soft_aoi_coordinate_noise_quality_summary.csv"
        ),
        "feature_sets": (
            output_dir
            / "soft_aoi_coordinate_noise_feature_sets.json"
        ),
        "manifest": (
            output_dir
            / "soft_aoi_coordinate_noise_manifest.json"
        ),
    }
    predictions.to_csv(
        paths["predictions"], index=False, encoding="utf-8-sig"
    )
    subject_repeat.to_csv(
        paths["subject_repeat"], index=False, encoding="utf-8-sig"
    )
    subject_condition.to_csv(
        paths["subject_condition"],
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        paths["summary"], index=False, encoding="utf-8-sig"
    )
    tests.to_csv(
        paths["tests"], index=False, encoding="utf-8-sig"
    )
    quality_summary.to_csv(
        paths["quality"], index=False, encoding="utf-8-sig"
    )
    paths["feature_sets"].write_text(
        json.dumps(
            feature_sets, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "protocol": (
                    "train_clean_test_noisy_dual_unseen_"
                    "subject_unseen_sentence"
                ),
                "sampling_rate_hz": RATE_HZ,
                "noise_sigmas_px": sigmas,
                "repeats": int(args.repeats),
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
                    for name, columns in feature_sets.items()
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
            "noise_sigma_px",
            "feature_set",
            "n_input_features",
            "macro_f1_subject_mean",
            "balanced_accuracy_subject_mean",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
        ]
    ]
    print("\n=== SOFT-AOI COORDINATE-NOISE SUMMARY ===")
    print(display.to_string(index=False))

    key_tests = tests[
        tests["metric"].eq("macro_f1")
        & (
            tests["test_family"].eq("noise_vs_clean")
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
            "noise_sigma_px",
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
    print("\n=== KEY MACRO-F1 COORDINATE-NOISE TESTS ===")
    print(key_tests.to_string(index=False))

    print("\n=== SOFT-AOI MEMBERSHIP QUALITY BY NOISE ===")
    print(quality_summary.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
