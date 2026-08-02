#!/usr/bin/env python
"""Fixed-I-VT event baseline for ZuCo reading-goal classification.

This script re-detects fixation-like events from continuous gaze after each
measurement degradation, constructs the nine official-formula sentence
features, and evaluates a clean-trained classifier with the same unseen-subject
plus unseen-sentence protocol used by the Soft-AOI study.

Important interpretation
------------------------
The resulting features are NOT identical to the official EyeLink event
features. They are a transparent re-detected event baseline:

- continuous gaze is first resampled to 30 Hz (or 10 Hz for ``rate10``);
- fixation-like segments are detected with a fixed I-VT threshold;
- segments shorter than 100 ms are discarded;
- inter-fixation transitions approximate saccades;
- the published nine-dimensional sentence formulas are then applied.

Supported conditions
--------------------
clean
    Clean 30 Hz gaze.
rate10
    Clean gaze resampled to 10 Hz; model is still trained on clean 30 Hz.
noise50
    50 px Gaussian coordinate noise, three deterministic repeats by default.
random40
    Additional random 40% missingness.
burst40
    Additional one-second burst missingness totaling 40%.
high_random
    50 px noise followed by random 40% missingness.
high_burst
    50 px noise followed by burst 40% missingness.

The corruption functions and deterministic seeds are imported from the existing
Soft-AOI robustness scripts so that the event baseline sees the same degraded
gaze realizations.
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
from scipy.stats import pearsonr, spearmanr, wilcoxon

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from extract_zuco_soft_aoi_features import (  # noqa: E402
    downsample_trial,
)
from run_zuco_hard_aoi_all_subjects import (  # noqa: E402
    load_trial_map,
)
from run_zuco_missingness_30hz import (  # noqa: E402
    stable_seed as missingness_seed,
)
from run_zuco_soft_aoi_ablation import (  # noqa: E402
    holm_adjust,
    load_sentence_folds,
)
from run_zuco_soft_aoi_combined_degradation_30hz import (  # noqa: E402
    CONDITIONS as COMBINED_CONDITIONS,
    degrade_samples as degrade_combined_samples,
)
from run_zuco_soft_aoi_confound_controls import (  # noqa: E402
    METRICS,
    decision_scores,
    make_svm,
    score_predictions,
)
from run_zuco_soft_aoi_coordinate_noise_30hz import (  # noqa: E402
    add_coordinate_noise,
    coordinate_noise_seed,
    load_word_template,
)
from run_zuco_soft_aoi_missingness_30hz import (  # noqa: E402
    degrade_samples as degrade_missing_samples,
)


EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_PER_SUBJECT = 739
EXPECTED_COMMON_SUBJECTS = 16
EXPECTED_COMMON_TRIALS = 11324

ZUCO_WIDTH_PX = 800.0
ZUCO_HEIGHT_PX = 600.0
ZUCO_ACTIVE_WIDTH_MM = 400.0
ZUCO_ACTIVE_HEIGHT_MM = 298.9
ZUCO_VIEWING_DISTANCE_MM = 680.0

FORMULA_NAMES = [
    "omission_rate",
    "fixation_number",
    "reading_speed",
    "mean_sacc_dur",
    "max_sacc_velocity",
    "mean_sacc_velocity",
    "max_sacc_dur",
    "mean_sacc_amp",
    "max_sacc_amp",
]
IVT_COLUMNS = [f"ivt_{name}" for name in FORMULA_NAMES]

SUPPORTED_CONDITIONS = (
    "clean",
    "rate10",
    "noise50",
    "random40",
    "burst40",
    "high_random",
    "high_burst",
)


def parse_conditions(text: str) -> list[str]:
    values = [
        item.strip().lower()
        for item in text.split(",")
        if item.strip()
    ]
    if not values:
        raise ValueError("--conditions must not be empty.")
    unknown = sorted(set(values) - set(SUPPORTED_CONDITIONS))
    if unknown:
        raise ValueError(
            f"Unsupported conditions: {unknown}; "
            f"supported={SUPPORTED_CONDITIONS}"
        )
    # Preserve user order while removing duplicates.
    return list(dict.fromkeys(values))


def condition_repeat_count(
    condition: str,
    repeats: int,
) -> int:
    return 1 if condition in {"clean", "rate10"} else repeats


def condition_rate_hz(condition: str) -> int:
    return 10 if condition == "rate10" else 30


def combined_condition_lookup() -> dict[str, Any]:
    result = {
        condition.name: condition
        for condition in COMBINED_CONDITIONS
        if condition.name in {"high_random", "high_burst"}
    }
    if set(result) != {"high_random", "high_burst"}:
        raise ValueError(
            "Existing combined-degradation module does not contain "
            "high_random and high_burst."
        )
    return result


def zuco_degrees_per_pixel() -> tuple[float, float]:
    horizontal_fov_deg = math.degrees(
        2.0
        * math.atan(
            (ZUCO_ACTIVE_WIDTH_MM / 2.0)
            / ZUCO_VIEWING_DISTANCE_MM
        )
    )
    vertical_fov_deg = math.degrees(
        2.0
        * math.atan(
            (ZUCO_ACTIVE_HEIGHT_MM / 2.0)
            / ZUCO_VIEWING_DISTANCE_MM
        )
    )
    return (
        horizontal_fov_deg / ZUCO_WIDTH_PX,
        vertical_fov_deg / ZUCO_HEIGHT_PX,
    )


def angular_distance_deg(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    deg_per_px_x, deg_per_px_y = zuco_degrees_per_pixel()
    return np.hypot(
        (x1 - x0) * deg_per_px_x,
        (y1 - y0) * deg_per_px_y,
    )


def contiguous_valid_segments(
    frame: pd.DataFrame,
    expected_rate_hz: int,
) -> list[np.ndarray]:
    """Return indices of valid, temporally contiguous sample segments."""
    required = {
        "timestamp_ms_abs",
        "x_px",
        "y_px",
        "valid_basic",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sample frame lacks columns: {missing}")

    t = pd.to_numeric(
        frame["timestamp_ms_abs"],
        errors="coerce",
    ).to_numpy(dtype=float)
    x = pd.to_numeric(
        frame["x_px"],
        errors="coerce",
    ).to_numpy(dtype=float)
    y = pd.to_numeric(
        frame["y_px"],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid_basic = (
        pd.to_numeric(
            frame["valid_basic"],
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
        .to_numpy()
    )
    valid = (
        valid_basic
        & np.isfinite(t)
        & np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return []

    expected_dt = 1000.0 / float(expected_rate_hz)
    max_gap_ms = 1.75 * expected_dt
    breaks = np.ones(len(valid_indices), dtype=bool)
    if len(valid_indices) >= 2:
        previous = valid_indices[:-1]
        current = valid_indices[1:]
        index_adjacent = current == previous + 1
        time_adjacent = (
            (t[current] - t[previous]) > 0
        ) & (
            (t[current] - t[previous]) <= max_gap_ms
        )
        breaks[1:] = ~(index_adjacent & time_adjacent)

    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], len(valid_indices)]
    return [
        valid_indices[start:end]
        for start, end in zip(starts, ends)
        if end > start
    ]


def low_velocity_edge_runs(
    low_velocity: np.ndarray,
) -> list[tuple[int, int]]:
    """Return inclusive edge-index runs where every edge is below threshold."""
    if len(low_velocity) == 0:
        return []
    true_indices = np.flatnonzero(low_velocity)
    if len(true_indices) == 0:
        return []

    split_positions = np.flatnonzero(
        np.diff(true_indices) > 1
    ) + 1
    groups = np.split(true_indices, split_positions)
    return [
        (int(group[0]), int(group[-1]))
        for group in groups
        if len(group)
    ]


def detect_ivt_fixations(
    sampled: pd.DataFrame,
    rate_hz: int,
    velocity_threshold_deg_s: float,
    min_fixation_ms: float,
) -> pd.DataFrame:
    """Detect fixation-like segments with an edge-connected I-VT rule."""
    if velocity_threshold_deg_s <= 0:
        raise ValueError(
            "velocity_threshold_deg_s must be positive."
        )
    if min_fixation_ms <= 0:
        raise ValueError("min_fixation_ms must be positive.")

    frame = sampled.sort_values(
        "timestamp_ms_abs"
    ).reset_index(drop=True)
    t_all = pd.to_numeric(
        frame["timestamp_ms_abs"],
        errors="coerce",
    ).to_numpy(dtype=float)
    x_all = pd.to_numeric(
        frame["x_px"],
        errors="coerce",
    ).to_numpy(dtype=float)
    y_all = pd.to_numeric(
        frame["y_px"],
        errors="coerce",
    ).to_numpy(dtype=float)

    expected_dt = 1000.0 / float(rate_hz)
    rows: list[dict[str, Any]] = []

    for segment_id, indices in enumerate(
        contiguous_valid_segments(frame, rate_hz)
    ):
        if len(indices) < 2:
            continue

        t = t_all[indices]
        x = x_all[indices]
        y = y_all[indices]
        dt_s = np.diff(t) / 1000.0
        amplitude_deg = angular_distance_deg(
            x[:-1],
            y[:-1],
            x[1:],
            y[1:],
        )
        velocity = np.full(len(dt_s), np.inf, dtype=float)
        valid_dt = np.isfinite(dt_s) & (dt_s > 0)
        velocity[valid_dt] = (
            amplitude_deg[valid_dt] / dt_s[valid_dt]
        )
        low_velocity = (
            np.isfinite(velocity)
            & (velocity <= velocity_threshold_deg_s)
        )

        for first_edge, last_edge in low_velocity_edge_runs(
            low_velocity
        ):
            # A run of edges e0..e1 contains sample points e0..e1+1.
            point_start = first_edge
            point_end = last_edge + 1
            t_start = float(t[point_start])
            t_last = float(t[point_end])

            local_dt = np.diff(
                t[point_start : point_end + 1]
            )
            local_dt = local_dt[
                np.isfinite(local_dt) & (local_dt > 0)
            ]
            sample_interval_ms = (
                float(np.median(local_dt))
                if len(local_dt)
                else expected_dt
            )
            end_ms = t_last + sample_interval_ms
            duration_ms = end_ms - t_start

            if duration_ms + 1e-9 < min_fixation_ms:
                continue

            x_points = x[point_start : point_end + 1]
            y_points = y[point_start : point_end + 1]
            rows.append(
                {
                    "fixation_index": len(rows),
                    "segment_id": int(segment_id),
                    "start_ms": t_start,
                    "end_ms": float(end_ms),
                    "duration_ms": float(duration_ms),
                    "x_px": float(np.mean(x_points)),
                    "y_px": float(np.mean(y_points)),
                    "n_samples": int(len(x_points)),
                    "mean_internal_velocity_deg_s": float(
                        np.mean(
                            velocity[
                                first_edge : last_edge + 1
                            ]
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "fixation_index",
            "segment_id",
            "start_ms",
            "end_ms",
            "duration_ms",
            "x_px",
            "y_px",
            "n_samples",
            "mean_internal_velocity_deg_s",
        ],
    )


def rectangle_membership(
    x: np.ndarray,
    y: np.ndarray,
    aoi: pd.DataFrame,
) -> np.ndarray:
    return (
        (
            x[:, None]
            >= aoi["x_min"].to_numpy(dtype=float)[None, :]
        )
        & (
            x[:, None]
            <= aoi["x_max"].to_numpy(dtype=float)[None, :]
        )
        & (
            y[:, None]
            >= aoi["y_min"].to_numpy(dtype=float)[None, :]
        )
        & (
            y[:, None]
            <= aoi["y_max"].to_numpy(dtype=float)[None, :]
        )
    )


def event_formula_features(
    fixations: pd.DataFrame,
    aoi: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Apply the exact nine benchmark-compatible sentence formulas.

    Order and formulas follow the frozen official sent_gaze_sacc model:

    0 omission_rate
    1 fixation_number
    2 reading_speed = duration_inside_ms / 100 / n_words
    3 mean_sacc_dur
    4 max_sacc_velocity
    5 mean_sacc_velocity
    6 max_sacc_dur
    7 mean_sacc_amp
    8 max_sacc_amp

    Missing saccade aggregates are zero-filled, matching the benchmark
    extractor used by the frozen model.
    """
    n_words = int(len(aoi))
    if n_words <= 0:
        raise ValueError(
            "Event formula requires at least one word AOI."
        )

    if fixations.empty:
        values = {
            "ivt_omission_rate": 1.0,
            "ivt_fixation_number": 0.0,
            "ivt_reading_speed": 0.0,
            "ivt_mean_sacc_dur": 0.0,
            "ivt_max_sacc_velocity": 0.0,
            "ivt_mean_sacc_velocity": 0.0,
            "ivt_max_sacc_dur": 0.0,
            "ivt_mean_sacc_amp": 0.0,
            "ivt_max_sacc_amp": 0.0,
        }
        return values, {
            "ivt_n_fixations": 0,
            "ivt_n_fixations_inside_words": 0,
            "ivt_n_unique_fixated_words": 0,
            "ivt_n_saccades": 0,
        }

    frame = fixations.sort_values(
        ["start_ms", "fixation_index"]
    ).reset_index(drop=True)
    x = frame["x_px"].to_numpy(dtype=float)
    y = frame["y_px"].to_numpy(dtype=float)

    membership = rectangle_membership(x, y, aoi)
    inside_any = membership.any(axis=1)
    word_assignment = np.full(len(frame), -1, dtype=int)

    if inside_any.any():
        areas = (
            (aoi["x_max"] - aoi["x_min"])
            * (aoi["y_max"] - aoi["y_min"])
        ).to_numpy(dtype=float)
        for index in np.flatnonzero(inside_any):
            candidates = np.flatnonzero(membership[index])
            word_assignment[index] = int(
                candidates[np.argmin(areas[candidates])]
            )

    in_word = word_assignment >= 0
    unique_words = np.unique(word_assignment[in_word])

    omission_rate = 1.0 - (
        len(unique_words) / float(n_words)
    )
    fixation_number = (
        float(np.sum(in_word)) / float(n_words)
    )
    duration_inside_ms = float(
        frame.loc[in_word, "duration_ms"].sum()
    )
    reading_speed = (
        duration_inside_ms / 100.0 / float(n_words)
    )

    starts = frame["start_ms"].to_numpy(dtype=float)
    ends = frame["end_ms"].to_numpy(dtype=float)
    transition_duration_ms = starts[1:] - ends[:-1]
    amplitude_deg = angular_distance_deg(
        x[:-1],
        y[:-1],
        x[1:],
        y[1:],
    )
    valid_saccade = (
        np.isfinite(transition_duration_ms)
        & (transition_duration_ms > 0)
        & np.isfinite(amplitude_deg)
    )
    saccade_duration = transition_duration_ms[valid_saccade]
    saccade_amplitude = amplitude_deg[valid_saccade]
    saccade_velocity = (
        saccade_amplitude
        / (saccade_duration / 1000.0)
        if len(saccade_duration)
        else np.asarray([], dtype=float)
    )

    if len(saccade_duration):
        mean_sacc_dur = float(np.mean(saccade_duration))
        max_sacc_dur = float(np.max(saccade_duration))
        mean_sacc_velocity = float(np.mean(saccade_velocity))
        max_sacc_velocity = float(np.max(saccade_velocity))
        mean_sacc_amp = float(np.mean(saccade_amplitude))
        max_sacc_amp = float(np.max(saccade_amplitude))
    else:
        mean_sacc_dur = 0.0
        max_sacc_dur = 0.0
        mean_sacc_velocity = 0.0
        max_sacc_velocity = 0.0
        mean_sacc_amp = 0.0
        max_sacc_amp = 0.0

    values = {
        "ivt_omission_rate": float(omission_rate),
        "ivt_fixation_number": float(fixation_number),
        "ivt_reading_speed": float(reading_speed),
        "ivt_mean_sacc_dur": mean_sacc_dur,
        "ivt_max_sacc_velocity": max_sacc_velocity,
        "ivt_mean_sacc_velocity": mean_sacc_velocity,
        "ivt_max_sacc_dur": max_sacc_dur,
        "ivt_mean_sacc_amp": mean_sacc_amp,
        "ivt_max_sacc_amp": max_sacc_amp,
    }
    audit = {
        "ivt_n_fixations": int(len(frame)),
        "ivt_n_fixations_inside_words": int(np.sum(in_word)),
        "ivt_n_unique_fixated_words": int(len(unique_words)),
        "ivt_n_saccades": int(len(saccade_duration)),
    }
    return values, audit

def load_common_official_maps(
    common_map_path: Path,
) -> tuple[pd.DataFrame, list[str], bool]:
    """Load the already consolidated Common16 official trial map.

    The project stores the matched 16-subject, 11,324-trial table at
    ``outputs/common16_comparison/common16_trial_map.csv``.  Subject-level
    ``*_sent_gaze_sacc_common_trials.csv`` files are not guaranteed to exist
    for all 16 subjects, so they must not be discovered with a directory glob.
    """
    if not common_map_path.is_file():
        raise FileNotFoundError(common_map_path)

    common = pd.read_csv(common_map_path)
    required = {"trial_id", "subject_id", "task"}
    missing = sorted(required - set(common.columns))
    if missing:
        raise ValueError(
            f"Common16 map lacks columns: {missing}"
        )

    common["trial_id"] = common["trial_id"].astype(str)
    common["subject_id"] = (
        common["subject_id"].astype(str).str.upper()
    )
    common["task"] = (
        common["task"].astype(str).str.upper()
    )
    official_columns = sorted(
        [
            column
            for column in common.columns
            if column.startswith("official_feature_")
            and column.rsplit("_", 1)[-1].isdigit()
        ],
        key=lambda column: int(
            column.rsplit("_", 1)[-1]
        ),
    )
    if len(official_columns) != 9:
        raise ValueError(
            f"Official feature count={len(official_columns)}, "
            "expected=9."
        )
    if common["subject_id"].nunique() != EXPECTED_COMMON_SUBJECTS:
        raise ValueError(
            f"Common subjects={common['subject_id'].nunique()}, "
            f"expected={EXPECTED_COMMON_SUBJECTS}."
        )
    if len(common) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Common trials={len(common)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if common["trial_id"].duplicated().any():
        raise ValueError(
            "Common official maps contain duplicate trial IDs."
        )

    return common, official_columns


def apply_condition(
    sampled: pd.DataFrame,
    condition: str,
    repeat: int,
    trial_id: str,
    seed: int,
    burst_seconds: float,
    combined_lookup: dict[str, Any],
) -> pd.DataFrame:
    if condition in {"clean", "rate10"}:
        return sampled.copy()

    if condition == "noise50":
        rng = np.random.default_rng(
            coordinate_noise_seed(
                base_seed=seed,
                trial_id=trial_id,
                sigma_px=50.0,
                repeat=repeat,
            )
        )
        return add_coordinate_noise(
            sampled=sampled,
            sigma_px=50.0,
            rng=rng,
        )

    if condition in {"random40", "burst40"}:
        mode = (
            "random"
            if condition == "random40"
            else "burst"
        )
        rng = np.random.default_rng(
            missingness_seed(
                base_seed=seed,
                trial_id=trial_id,
                mode=mode,
                percentage=40.0,
                repeat=repeat,
            )
        )
        degraded, _ = degrade_missing_samples(
            sampled=sampled,
            mode=mode,
            percentage=40.0,
            burst_seconds=burst_seconds,
            rng=rng,
        )
        return degraded

    if condition in {"high_random", "high_burst"}:
        degraded, _, _ = degrade_combined_samples(
            sampled=sampled,
            condition=combined_lookup[condition],
            repeat=repeat,
            seed=seed,
            trial_id=trial_id,
            burst_seconds=burst_seconds,
        )
        return degraded

    raise ValueError(f"Unhandled condition: {condition}")


def cache_path(
    root: Path,
    subject: str,
    condition: str,
    repeat: int,
    velocity_threshold: float,
    min_fixation_ms: float,
) -> Path:
    threshold_text = (
        f"{velocity_threshold:g}".replace(".", "p")
    )
    duration_text = (
        f"{min_fixation_ms:g}".replace(".", "p")
    )
    return (
        root
        / f"ivt_v{threshold_text}_d{duration_text}"
        / condition
        / f"repeat_{repeat:02d}"
        / f"{subject}_ivt_{condition}_repeat_{repeat:02d}.csv"
    )


def validate_cache(
    frame: pd.DataFrame,
    subject: str,
    expected_ids: set[str],
    condition: str,
    repeat: int,
) -> pd.DataFrame:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "task_sentence_index",
        "condition",
        "corruption_repeat",
        *IVT_COLUMNS,
        "ivt_n_fixations",
        "ivt_n_fixations_inside_words",
        "ivt_n_unique_fixated_words",
        "ivt_n_saccades",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{subject} I-VT cache lacks columns: {missing}"
        )

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = (
        result["task"].astype(str).str.upper()
    )
    result["condition"] = (
        result["condition"].astype(str).str.lower()
    )

    if len(result) != EXPECTED_PER_SUBJECT:
        raise ValueError(
            f"{subject} cache rows={len(result)}, "
            f"expected={EXPECTED_PER_SUBJECT}."
        )
    if result["trial_id"].duplicated().any():
        raise ValueError(
            f"{subject} cache has duplicate trial IDs."
        )
    if set(result["trial_id"]) != expected_ids:
        raise ValueError(
            f"{subject} cache trial IDs do not match."
        )
    if set(result["subject_id"]) != {subject}:
        raise ValueError(
            f"{subject} cache contains another subject."
        )
    if set(result["condition"]) != {condition}:
        raise ValueError(
            f"{subject} cache condition is invalid."
        )
    if not pd.to_numeric(
        result["corruption_repeat"],
        errors="coerce",
    ).eq(repeat).all():
        raise ValueError(
            f"{subject} cache repeat is invalid."
        )
    if np.isinf(
        result[IVT_COLUMNS].to_numpy(dtype=float)
    ).any():
        raise ValueError(
            f"{subject} cache contains infinity."
        )
    return result


def extract_conditions(
    manifest: pd.DataFrame,
    subjects: list[str],
    processed_root: Path,
    aoi_lookup: dict[tuple[str, int], pd.DataFrame],
    output_root: Path,
    conditions: list[str],
    repeats: int,
    seed: int,
    burst_seconds: float,
    velocity_threshold: float,
    min_fixation_ms: float,
    overwrite: bool,
) -> dict[tuple[str, int], pd.DataFrame]:
    combined_lookup = combined_condition_lookup()
    output_lists: dict[
        tuple[str, int], list[pd.DataFrame]
    ] = {
        (condition, repeat): []
        for condition in conditions
        for repeat in range(
            1,
            condition_repeat_count(condition, repeats) + 1,
        )
    }

    for subject_index, subject in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"I-VT extraction subject "
            f"{subject_index:02d}/{len(subjects)}: {subject}"
        )

        trial_map = load_trial_map(
            processed_root
            / subject
            / f"{subject}_all_trials_with_sentence_index.csv",
            subject,
        )
        expected_ids = set(
            trial_map["trial_id"].astype(str)
        )
        trial_lookup = trial_map.set_index(
            "trial_id",
            drop=False,
            verify_integrity=True,
        )

        paths = {
            key: cache_path(
                output_root,
                subject,
                key[0],
                key[1],
                velocity_threshold,
                min_fixation_ms,
            )
            for key in output_lists
        }
        if not overwrite and all(
            path.is_file()
            for path in paths.values()
        ):
            for key, path in paths.items():
                output_lists[key].append(
                    validate_cache(
                        pd.read_csv(path),
                        subject,
                        expected_ids,
                        key[0],
                        key[1],
                    )
                )
            print("  Loaded all existing I-VT caches.")
            continue

        needed = {
            key
            for key, path in paths.items()
            if overwrite or not path.is_file()
        }
        rows_by_key: dict[
            tuple[str, int], list[dict[str, Any]]
        ] = {
            key: []
            for key in needed
        }

        subject_manifest = manifest[
            manifest["subject_id"].eq(subject)
        ].sort_values(["task", "block_id"])
        if len(subject_manifest) != 14:
            raise ValueError(
                f"{subject} manifest rows="
                f"{len(subject_manifest)}, expected=14."
            )

        if needed:
            for block_index, manifest_row in enumerate(
                subject_manifest.itertuples(index=False),
                start=1,
            ):
                sample_path = Path(
                    str(manifest_row.sample_parquet)
                )
                if not sample_path.is_file():
                    raise FileNotFoundError(sample_path)
                print(
                    f"  [{block_index:02d}/14] "
                    f"{sample_path.name}; "
                    f"conditions={len(needed)}"
                )

                samples = pd.read_parquet(sample_path)
                required_samples = {
                    "trial_id",
                    "timestamp_ms_abs",
                    "x_px",
                    "y_px",
                    "valid_basic",
                }
                missing_samples = sorted(
                    required_samples - set(samples.columns)
                )
                if missing_samples:
                    raise ValueError(
                        f"{sample_path} lacks columns: "
                        f"{missing_samples}"
                    )
                samples["trial_id"] = (
                    samples["trial_id"].astype(str)
                )

                for trial_id, trial_samples in samples.groupby(
                    "trial_id",
                    sort=False,
                ):
                    trial_id = str(trial_id)
                    if trial_id not in trial_lookup.index:
                        raise ValueError(
                            f"Trial missing from map: {trial_id}"
                        )
                    trial_row = trial_lookup.loc[
                        trial_id
                    ].copy()
                    aoi_key = (
                        str(trial_row["task"]).upper(),
                        int(
                            trial_row[
                                "task_sentence_index"
                            ]
                        ),
                    )
                    if aoi_key not in aoi_lookup:
                        raise ValueError(
                            f"AOI template missing: {aoi_key}"
                        )
                    aoi = aoi_lookup[aoi_key]

                    sampled_by_rate: dict[int, pd.DataFrame] = {}
                    for condition, repeat in needed:
                        rate_hz = condition_rate_hz(
                            condition
                        )
                        if rate_hz not in sampled_by_rate:
                            sampled_by_rate[rate_hz] = (
                                downsample_trial(
                                    trial_samples,
                                    rate_hz,
                                )
                            )
                        sampled = sampled_by_rate[rate_hz]
                        degraded = apply_condition(
                            sampled=sampled,
                            condition=condition,
                            repeat=repeat,
                            trial_id=trial_id,
                            seed=seed,
                            burst_seconds=burst_seconds,
                            combined_lookup=combined_lookup,
                        )
                        fixations = detect_ivt_fixations(
                            sampled=degraded,
                            rate_hz=rate_hz,
                            velocity_threshold_deg_s=(
                                velocity_threshold
                            ),
                            min_fixation_ms=(
                                min_fixation_ms
                            ),
                        )
                        values, audit = (
                            event_formula_features(
                                fixations=fixations,
                                aoi=aoi,
                            )
                        )
                        rows_by_key[
                            (condition, repeat)
                        ].append(
                            {
                                "subject_id": subject,
                                "task": str(
                                    trial_row["task"]
                                ).upper(),
                                "block_id": int(
                                    trial_row["block_id"]
                                ),
                                "trial_index_in_block": int(
                                    trial_row[
                                        "trial_index_in_block"
                                    ]
                                ),
                                "trial_id": trial_id,
                                "task_sentence_index": int(
                                    trial_row[
                                        "task_sentence_index"
                                    ]
                                ),
                                "condition": condition,
                                "corruption_repeat": int(
                                    repeat
                                ),
                                "sampling_rate_hz": int(
                                    rate_hz
                                ),
                                "ivt_velocity_threshold_deg_s": float(
                                    velocity_threshold
                                ),
                                "ivt_min_fixation_ms": float(
                                    min_fixation_ms
                                ),
                                **values,
                                **audit,
                            }
                        )
                del samples

        for key, path in paths.items():
            condition, repeat = key
            if key in rows_by_key:
                frame = pd.DataFrame(
                    rows_by_key[key]
                ).sort_values(
                    "trial_id"
                ).reset_index(drop=True)
                frame = validate_cache(
                    frame,
                    subject,
                    expected_ids,
                    condition,
                    repeat,
                )
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                frame.to_csv(
                    path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"  Written: {path}")
            else:
                frame = validate_cache(
                    pd.read_csv(path),
                    subject,
                    expected_ids,
                    condition,
                    repeat,
                )
            output_lists[key].append(frame)

    output: dict[tuple[str, int], pd.DataFrame] = {}
    for key, frames in output_lists.items():
        combined = pd.concat(
            frames,
            ignore_index=True,
        )
        if len(combined) != EXPECTED_TRIALS:
            raise ValueError(
                f"I-VT condition {key} rows="
                f"{len(combined)}, "
                f"expected={EXPECTED_TRIALS}."
            )
        if (
            combined["subject_id"].nunique()
            != EXPECTED_SUBJECTS
        ):
            raise ValueError(
                f"I-VT condition {key} subject "
                "count is invalid."
            )
        output[key] = combined
    return output


def build_metadata(
    processed_root: Path,
    subjects: list[str],
    sentence_folds: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for subject in subjects:
        frame = load_trial_map(
            processed_root
            / subject
            / f"{subject}_all_trials_with_sentence_index.csv",
            subject,
        )
        frame = frame.copy()
        frame["subject_id"] = subject
        frame["task"] = (
            frame["task"].astype(str).str.upper()
        )
        frame["trial_id"] = (
            frame["trial_id"].astype(str)
        )
        frame["label_nr"] = frame["task"].map(
            {"NR": 1, "TSR": 0}
        )
        frame["sentence_key"] = frame.apply(
            lambda row: (
                f"{row['task']}:"
                f"{int(row['task_sentence_index']):03d}"
            ),
            axis=1,
        )
        frames.append(frame)

    metadata = pd.concat(
        frames,
        ignore_index=True,
    )
    if len(metadata) != EXPECTED_TRIALS:
        raise ValueError(
            f"Metadata rows={len(metadata)}, "
            f"expected={EXPECTED_TRIALS}."
        )
    metadata = metadata.merge(
        sentence_folds[
            ["sentence_key", "sentence_fold"]
        ],
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if metadata["sentence_fold"].isna().any():
        raise ValueError(
            "Some trials lack sentence-fold assignments."
        )
    metadata["sentence_fold"] = (
        metadata["sentence_fold"].astype(int)
    )
    metadata["label_nr"] = (
        metadata["label_nr"].astype(int)
    )
    return metadata[
        [
            "trial_id",
            "subject_id",
            "task",
            "task_sentence_index",
            "sentence_key",
            "sentence_fold",
            "label_nr",
        ]
    ].copy()


def condition_table(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "trial_id",
        "subject_id",
        "task",
        *IVT_COLUMNS,
        "ivt_n_fixations",
        "ivt_n_fixations_inside_words",
        "ivt_n_unique_fixated_words",
        "ivt_n_saccades",
    ]
    result = metadata.merge(
        features[columns],
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    )
    if len(result) != EXPECTED_TRIALS:
        raise ValueError(
            f"Condition table rows={len(result)}, "
            f"expected={EXPECTED_TRIALS}."
        )
    return result


def evaluate_clean_train_conditions(
    clean: pd.DataFrame,
    conditions: dict[tuple[str, int], pd.DataFrame],
    c_value: float,
    seed: int,
) -> pd.DataFrame:
    subjects = sorted(
        clean["subject_id"].unique()
    )
    folds = sorted(
        clean["sentence_fold"].unique()
    )
    rows: list[dict[str, Any]] = []

    for subject_index, held_out in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"I-VT evaluation subject "
            f"{subject_index:02d}/{len(subjects)}: "
            f"{held_out}"
        )
        for fold_index in folds:
            train = clean[
                (clean["subject_id"] != held_out)
                & (
                    clean["sentence_fold"]
                    != fold_index
                )
            ].copy()
            y_train = train[
                "label_nr"
            ].to_numpy(dtype=int)
            model = make_svm(c_value, seed)
            model.fit(
                train[IVT_COLUMNS],
                y_train,
            )

            for (
                condition_name,
                repeat,
            ), test_table in conditions.items():
                test = test_table[
                    (
                        test_table["subject_id"]
                        == held_out
                    )
                    & (
                        test_table["sentence_fold"]
                        == fold_index
                    )
                ].copy().reset_index(drop=True)
                if test.empty:
                    raise ValueError(
                        f"Empty test: {held_out}, "
                        f"fold={fold_index}, "
                        f"condition={condition_name}, "
                        f"repeat={repeat}."
                    )
                if set(
                    train["sentence_key"]
                ) & set(test["sentence_key"]):
                    raise ValueError(
                        "Sentence leakage detected."
                    )

                y_pred = model.predict(
                    test[IVT_COLUMNS]
                ).astype(int)
                scores = decision_scores(
                    model,
                    test[IVT_COLUMNS],
                )

                for row_index, row in test.iterrows():
                    rows.append(
                        {
                            "condition": condition_name,
                            "corruption_repeat": int(
                                repeat
                            ),
                            "held_out_subject": (
                                held_out
                            ),
                            "sentence_fold": int(
                                fold_index
                            ),
                            "trial_id": str(
                                row["trial_id"]
                            ),
                            "subject_id": str(
                                row["subject_id"]
                            ),
                            "task": str(row["task"]),
                            "label_nr": int(
                                row["label_nr"]
                            ),
                            "prediction_nr": int(
                                y_pred[row_index]
                            ),
                            "decision_score_nr": float(
                                scores[row_index]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    for values, group in predictions.groupby(
        [
            "condition",
            "corruption_repeat",
            "held_out_subject",
        ],
        sort=True,
    ):
        condition, repeat, subject = values
        repeat_rows.append(
            {
                "condition": condition,
                "corruption_repeat": int(
                    repeat
                ),
                "held_out_subject": subject,
                "n_predictions": int(
                    len(group)
                ),
                **score_predictions(
                    group["label_nr"].to_numpy(
                        dtype=int
                    ),
                    group[
                        "prediction_nr"
                    ].to_numpy(dtype=int),
                    group[
                        "decision_score_nr"
                    ].to_numpy(dtype=float),
                ),
            }
        )
    repeat_metrics = pd.DataFrame(
        repeat_rows
    )

    subject_metrics = (
        repeat_metrics.groupby(
            [
                "condition",
                "held_out_subject",
            ],
            as_index=False,
        )[METRICS]
        .mean()
    )

    summary_rows: list[dict[str, Any]] = []
    for condition, group in subject_metrics.groupby(
        "condition",
        sort=True,
    ):
        row: dict[str, Any] = {
            "condition": condition,
            "n_subjects": int(
                group[
                    "held_out_subject"
                ].nunique()
            ),
            "n_input_features": 9,
        }
        for metric in METRICS:
            row[
                f"{metric}_subject_mean"
            ] = float(group[metric].mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return (
        repeat_metrics,
        subject_metrics,
        summary,
    )


def clean_vs_degraded_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if "clean" not in set(
        subject_metrics["condition"]
    ):
        return pd.DataFrame()

    clean = (
        subject_metrics[
            subject_metrics["condition"].eq(
                "clean"
            )
        ]
        .set_index("held_out_subject")
        .sort_index()
    )
    rows: list[dict[str, Any]] = []

    for condition in sorted(
        set(subject_metrics["condition"])
        - {"clean"}
    ):
        degraded = (
            subject_metrics[
                subject_metrics[
                    "condition"
                ].eq(condition)
            ]
            .set_index("held_out_subject")
            .sort_index()
        )
        condition_rows = []
        for metric in METRICS:
            paired = pd.concat(
                [
                    clean[metric].rename(
                        "clean"
                    ),
                    degraded[metric].rename(
                        "degraded"
                    ),
                ],
                axis=1,
                join="inner",
            ).dropna()
            difference = (
                paired["degraded"]
                - paired["clean"]
            )
            if np.allclose(
                difference.to_numpy(
                    dtype=float
                ),
                0.0,
            ):
                statistic, p_value = (
                    0.0,
                    1.0,
                )
            else:
                test = wilcoxon(
                    paired["degraded"],
                    paired["clean"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(
                    test.statistic
                )
                p_value = float(
                    test.pvalue
                )
            condition_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "clean_mean": float(
                        paired["clean"].mean()
                    ),
                    "degraded_mean": float(
                        paired[
                            "degraded"
                        ].mean()
                    ),
                    "mean_difference_degraded_minus_clean": float(
                        difference.mean()
                    ),
                    "degraded_wins": int(
                        np.sum(difference > 0)
                    ),
                    "ties": int(
                        np.sum(
                            np.isclose(
                                difference,
                                0.0,
                            )
                        )
                    ),
                    "degraded_losses": int(
                        np.sum(difference < 0)
                    ),
                    "wilcoxon_statistic": (
                        statistic
                    ),
                    "p_value": p_value,
                    "p_holm_across_metrics": (
                        math.nan
                    ),
                }
            )
        adjusted = holm_adjust(
            [
                float(row["p_value"])
                for row in condition_rows
            ]
        )
        for row, adjusted_value in zip(
            condition_rows,
            adjusted,
        ):
            row[
                "p_holm_across_metrics"
            ] = adjusted_value
        rows.extend(condition_rows)
    return pd.DataFrame(rows)


def safe_correlation(
    x: pd.Series,
    y: pd.Series,
    method: str,
) -> tuple[float, float, int]:
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()
    if len(frame) < 3:
        return math.nan, math.nan, len(frame)
    if (
        frame["x"].nunique() < 2
        or frame["y"].nunique() < 2
    ):
        return math.nan, math.nan, len(frame)

    if method == "pearson":
        result = pearsonr(
            frame["x"],
            frame["y"],
        )
    elif method == "spearman":
        result = spearmanr(
            frame["x"],
            frame["y"],
        )
    else:
        raise ValueError(method)
    return (
        float(result.statistic),
        float(result.pvalue),
        len(frame),
    )


def official_audit(
    clean_features: pd.DataFrame,
    common: pd.DataFrame,
    official_columns: list[str],
) -> pd.DataFrame:
    merged = common[
        [
            "trial_id",
            "subject_id",
            "task",
            *official_columns,
        ]
    ].merge(
        clean_features[
            [
                "trial_id",
                "subject_id",
                "task",
                *IVT_COLUMNS,
            ]
        ],
        on=[
            "trial_id",
            "subject_id",
            "task",
        ],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Official audit rows="
            f"{len(merged)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )

    rows = []
    for index, name in enumerate(
        FORMULA_NAMES
    ):
        official_column = (
            official_columns[index]
        )
        ivt_column = f"ivt_{name}"
        pearson_r, pearson_p, n_pearson = (
            safe_correlation(
                merged[official_column],
                merged[ivt_column],
                "pearson",
            )
        )
        spearman_r, spearman_p, n_spearman = (
            safe_correlation(
                merged[official_column],
                merged[ivt_column],
                "spearman",
            )
        )
        official_values = pd.to_numeric(
            merged[official_column],
            errors="coerce",
        )
        ivt_values = pd.to_numeric(
            merged[ivt_column],
            errors="coerce",
        )
        rows.append(
            {
                "formula_index": index,
                "formula_name": name,
                "official_column": (
                    official_column
                ),
                "ivt_column": ivt_column,
                "n_common_pearson": n_pearson,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "n_common_spearman": (
                    n_spearman
                ),
                "spearman_rho": (
                    spearman_r
                ),
                "spearman_p": (
                    spearman_p
                ),
                "official_median": float(
                    official_values.median()
                ),
                "ivt_median": float(
                    ivt_values.median()
                ),
                "official_mean": float(
                    official_values.mean()
                ),
                "ivt_mean": float(
                    ivt_values.mean()
                ),
                "ivt_missing_rate": float(
                    ivt_values.isna().mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def event_count_summary(
    features: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    rows = []
    for (condition, repeat), frame in (
        features.items()
    ):
        rows.append(
            {
                "condition": condition,
                "corruption_repeat": repeat,
                "median_fixations_per_trial": float(
                    frame[
                        "ivt_n_fixations"
                    ].median()
                ),
                "mean_fixations_per_trial": float(
                    frame[
                        "ivt_n_fixations"
                    ].mean()
                ),
                "zero_fixation_trial_rate": float(
                    frame[
                        "ivt_n_fixations"
                    ].eq(0).mean()
                ),
                "median_saccades_per_trial": float(
                    frame[
                        "ivt_n_saccades"
                    ].median()
                ),
                "mean_valid_formula_count": float(
                    frame[
                        IVT_COLUMNS
                    ].notna().sum(
                        axis=1
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "data/interim/zuco2/"
            "all_subjects/"
            "complete_raw_sample_manifest.csv"
        ),
    )
    parser.add_argument(
        "--processed-root",
        default=(
            "data/processed/zuco2"
        ),
    )
    parser.add_argument(
        "--common-map",
        default=(
            "outputs/common16_comparison/"
            "common16_trial_map.csv"
        ),
    )
    parser.add_argument(
        "--word-template",
        default=(
            "data/processed/zuco2/"
            "soft_aoi_template/"
            "zuco_soft_aoi_word_template.csv"
        ),
    )
    parser.add_argument(
        "--sentence-folds-file",
        default=(
            "outputs/"
            "soft_aoi_confound_controls_30hz/"
            "dual_holdout_sentence_folds.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "ivt_event_baseline_fixed_benchmark_exact"
        ),
    )
    parser.add_argument(
        "--conditions",
        default="clean",
        help=(
            "Comma-separated subset of: "
            + ",".join(
                SUPPORTED_CONDITIONS
            )
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--velocity-threshold-deg-s",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--min-fixation-ms",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--burst-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
    )
    args = parser.parse_args()

    conditions = parse_conditions(
        args.conditions
    )
    if "clean" not in conditions:
        # The model is always trained on clean features.
        conditions = ["clean"] + conditions
    if args.repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )
    if args.c_value <= 0:
        raise ValueError(
            "--c-value must be positive."
        )
    if args.burst_seconds <= 0:
        raise ValueError(
            "--burst-seconds must be positive."
        )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )
    manifest = pd.read_csv(
        manifest_path
    )
    required_manifest = {
        "subject_id",
        "task",
        "block_id",
        "sample_parquet",
    }
    missing_manifest = sorted(
        required_manifest
        - set(manifest.columns)
    )
    if missing_manifest:
        raise ValueError(
            f"Manifest lacks columns: "
            f"{missing_manifest}"
        )
    manifest["subject_id"] = (
        manifest["subject_id"]
        .astype(str)
        .str.upper()
    )
    manifest["task"] = (
        manifest["task"]
        .astype(str)
        .str.upper()
    )
    subjects = sorted(
        manifest[
            "subject_id"
        ].unique()
    )
    if (
        len(subjects)
        != EXPECTED_SUBJECTS
    ):
        raise ValueError(
            f"Manifest subjects="
            f"{len(subjects)}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    processed_root = Path(
        args.processed_root
    )
    (
        common,
        official_columns,
    ) = load_common_official_maps(
        Path(args.common_map)
    )
    aoi_lookup = load_word_template(
        Path(args.word_template)
    )
    sentence_folds = (
        load_sentence_folds(
            Path(
                args.sentence_folds_file
            )
        )
    )
    metadata = build_metadata(
        processed_root,
        subjects,
        sentence_folds,
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    feature_root = (
        output_dir
        / "subject_ivt_features"
    )

    extracted = extract_conditions(
        manifest=manifest,
        subjects=subjects,
        processed_root=processed_root,
        aoi_lookup=aoi_lookup,
        output_root=feature_root,
        conditions=conditions,
        repeats=args.repeats,
        seed=args.seed,
        burst_seconds=(
            args.burst_seconds
        ),
        velocity_threshold=(
            args.velocity_threshold_deg_s
        ),
        min_fixation_ms=(
            args.min_fixation_ms
        ),
        overwrite=(
            args.overwrite_features
        ),
    )

    condition_tables = {
        key: condition_table(
            metadata,
            frame,
        )
        for key, frame in extracted.items()
    }
    clean_table = condition_tables[
        ("clean", 1)
    ]

    print(
        "\n=== FIXED I-VT EVENT BASELINE INPUT ==="
    )
    print(f"subjects: {len(subjects)}")
    print(f"trials: {len(metadata)}")
    print(f"conditions: {conditions}")
    print(f"repeats: {args.repeats}")
    print(
        "velocity threshold: "
        f"{args.velocity_threshold_deg_s} deg/s"
    )
    print(
        "minimum fixation: "
        f"{args.min_fixation_ms} ms"
    )
    print(
        "reading-speed formula: "
        "duration_inside_ms / 100 / n_words"
    )

    predictions = (
        evaluate_clean_train_conditions(
            clean=clean_table,
            conditions=condition_tables,
            c_value=args.c_value,
            seed=args.seed,
        )
    )
    (
        repeat_metrics,
        subject_metrics,
        summary,
    ) = aggregate_metrics(
        predictions
    )
    tests = clean_vs_degraded_tests(
        subject_metrics
    )
    audit = official_audit(
        clean_features=extracted[
            ("clean", 1)
        ],
        common=common,
        official_columns=official_columns,
    )
    count_summary = (
        event_count_summary(extracted)
    )

    paths = {
        "predictions": (
            output_dir
            / "ivt_fixed_predictions.csv"
        ),
        "repeat_metrics": (
            output_dir
            / "ivt_fixed_subject_repeat_metrics.csv"
        ),
        "subject_metrics": (
            output_dir
            / "ivt_fixed_subject_metrics.csv"
        ),
        "summary": (
            output_dir
            / "ivt_fixed_summary.csv"
        ),
        "tests": (
            output_dir
            / "ivt_fixed_clean_vs_degraded_tests.csv"
        ),
        "official_audit": (
            output_dir
            / "ivt_fixed_vs_official_common16_audit.csv"
        ),
        "event_counts": (
            output_dir
            / "ivt_fixed_event_count_summary.csv"
        ),
        "manifest": (
            output_dir
            / "ivt_fixed_manifest.json"
        ),
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
    audit.to_csv(
        paths["official_audit"],
        index=False,
        encoding="utf-8-sig",
    )
    count_summary.to_csv(
        paths["event_counts"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "baseline": (
                    "fixed_ivt_re_detected_event_formula"
                ),
                "interpretation": (
                    "Approximate re-detected event baseline; "
                    "not identical to EyeLink official events."
                ),
                "conditions": conditions,
                "repeats": int(
                    args.repeats
                ),
                "velocity_threshold_deg_s": float(
                    args.velocity_threshold_deg_s
                ),
                "min_fixation_ms": float(
                    args.min_fixation_ms
                ),
                "c_value": float(
                    args.c_value
                ),
                "seed": int(args.seed),
                "common_map": str(
                    Path(args.common_map)
                ),
                "formula_names": (
                    FORMULA_NAMES
                ),
                "reading_speed_formula": (
                    "duration_inside_ms / 100 / n_words"
                ),
                "official_benchmark_order": FORMULA_NAMES,
                "screen_geometry": {
                    "width_px": (
                        ZUCO_WIDTH_PX
                    ),
                    "height_px": (
                        ZUCO_HEIGHT_PX
                    ),
                    "active_width_mm": (
                        ZUCO_ACTIVE_WIDTH_MM
                    ),
                    "active_height_mm": (
                        ZUCO_ACTIVE_HEIGHT_MM
                    ),
                    "viewing_distance_mm": (
                        ZUCO_VIEWING_DISTANCE_MM
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== FIXED I-VT EVENT BASELINE SUMMARY ==="
    )
    print(
        summary[
            [
                "condition",
                "n_input_features",
                "macro_f1_subject_mean",
                "balanced_accuracy_subject_mean",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== FIXED I-VT EVENT COUNT SUMMARY ==="
    )
    print(
        count_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== FIXED I-VT VS OFFICIAL COMMON16 AUDIT ==="
    )
    print(
        audit[
            [
                "formula_index",
                "formula_name",
                "pearson_r",
                "spearman_rho",
                "official_median",
                "ivt_median",
                "ivt_missing_rate",
            ]
        ].to_string(index=False)
    )

    if not tests.empty:
        print(
            "\n=== FIXED I-VT CLEAN VS DEGRADED TESTS ==="
        )
        print(
            tests[
                tests["metric"].isin(
                    [
                        "macro_f1",
                        "roc_auc",
                        "pr_auc_nr",
                    ]
                )
            ][
                [
                    "condition",
                    "metric",
                    "clean_mean",
                    "degraded_mean",
                    "mean_difference_degraded_minus_clean",
                    "degraded_wins",
                    "ties",
                    "degraded_losses",
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
