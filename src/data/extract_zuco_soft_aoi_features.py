#!/usr/bin/env python
"""Extract fixation-free, trial-level Soft-AOI features from ZuCo raw gaze.

The script maps 30 Hz (or another requested rate) raw gaze samples to the word
rectangles in ``zuco_soft_aoi_word_template.csv``. It does not use fixation,
saccade, blink-event, or official word-level gaze annotations.

Soft membership
---------------
For gaze sample g_t and word rectangle A_j, d(g_t, A_j) is zero inside the
rectangle and otherwise the Euclidean distance to its nearest point. Membership
is

    q_j(t) = exp(-d(g_t, A_j)^2 / (2 sigma^2)).

By default sigma is defined relative to each sentence's median word height:

    sigma = (2 / 3) * median_word_height.

For the current ZuCo template this is approximately 10 px because the median
word height is approximately 15 px. A sample is considered Soft-AOI-active when
max_j q_j(t) >= 0.1. This threshold corresponds to a maximum rectangle distance
of approximately 2.146 * sigma.

Outputs
-------
- <SUBJECT>_soft_aoi_features_<RATE>hz.csv
- <SUBJECT>_soft_aoi_task_summary_<RATE>hz.csv
- <SUBJECT>_soft_aoi_feature_manifest_<RATE>hz.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_RATE_HZ = 500
EXPECTED_TASK_COUNTS = {"NR": 349, "TSR": 390}


def nearest_regular_grid_indices(
    timestamps_ms: np.ndarray,
    target_rate_hz: int,
) -> np.ndarray:
    """Select source samples nearest to a regular time grid."""
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
        choose_left = np.abs(grid - timestamps[left]) <= np.abs(
            timestamps[right] - grid
        )
        selected = np.where(choose_left, left, right).astype(np.int64)
        selected = np.unique(selected)

    if valid_indices is not None:
        selected = valid_indices[selected]
    return selected


def downsample_trial(
    trial_group: pd.DataFrame,
    target_rate_hz: int,
) -> pd.DataFrame:
    group = trial_group.sort_values("sample_index_in_trial")
    if target_rate_hz >= BASE_RATE_HZ:
        return group.copy()
    indices = nearest_regular_grid_indices(
        group["timestamp_ms_abs"].to_numpy(dtype=float),
        target_rate_hz,
    )
    if indices.size == 0:
        raise ValueError("Downsampling produced an empty trial.")
    return group.iloc[indices].copy()


def to_bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"}).to_numpy(dtype=bool)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return math.nan
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    if not math.isfinite(mean):
        return math.nan
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    variance = np.average((values[mask] - mean) ** 2, weights=weights[mask])
    return float(math.sqrt(max(float(variance), 0.0)))


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return math.nan
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(quantile) * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(values) - 1)
    return float(values[index])


def weighted_slope(
    time_sec: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    time_sec = np.asarray(time_sec, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (
        np.isfinite(time_sec)
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if np.sum(mask) < 2:
        return math.nan
    x = time_sec[mask]
    y = values[mask]
    w = weights[mask]
    x_mean = float(np.average(x, weights=w))
    y_mean = float(np.average(y, weights=w))
    denominator = float(np.sum(w * (x - x_mean) ** 2))
    if denominator <= 0:
        return math.nan
    numerator = float(np.sum(w * (x - x_mean) * (y - y_mean)))
    return numerator / denominator


def safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.nan
    return float(numerator / denominator)


def normalized_entropy(probabilities: np.ndarray) -> float:
    raw = np.asarray(probabilities, dtype=float)
    n_categories = int(raw.size)
    p = raw[np.isfinite(raw) & (raw > 0)]
    if n_categories <= 1 or p.size == 0:
        return 0.0
    p = p / p.sum()
    entropy = -float(np.sum(p * np.log(p)))
    return entropy / math.log(n_categories)


def rectangle_distances(
    x: np.ndarray,
    y: np.ndarray,
    aoi: pd.DataFrame,
) -> np.ndarray:
    px = np.asarray(x, dtype=float)[:, None]
    py = np.asarray(y, dtype=float)[:, None]
    xmin = aoi["x_min"].to_numpy(dtype=float)[None, :]
    xmax = aoi["x_max"].to_numpy(dtype=float)[None, :]
    ymin = aoi["y_min"].to_numpy(dtype=float)[None, :]
    ymax = aoi["y_max"].to_numpy(dtype=float)[None, :]
    dx = np.maximum(np.maximum(xmin - px, px - xmax), 0.0)
    dy = np.maximum(np.maximum(ymin - py, py - ymax), 0.0)
    return np.sqrt(dx * dx + dy * dy)


def extract_trial_features(
    samples: pd.DataFrame,
    trial_row: pd.Series,
    aoi: pd.DataFrame,
    rate_hz: int,
    sigma_word_height_ratio: float,
    active_threshold: float,
) -> dict[str, Any]:
    downsampled = downsample_trial(samples, rate_hz)
    n_samples = len(downsampled)

    valid = (
        to_bool_array(downsampled["valid_basic"])
        & np.isfinite(downsampled["x_px"].to_numpy(dtype=float))
        & np.isfinite(downsampled["y_px"].to_numpy(dtype=float))
        & (downsampled["x_px"].to_numpy(dtype=float) > 0)
        & (downsampled["y_px"].to_numpy(dtype=float) > 0)
    )
    valid_positions = np.flatnonzero(valid)
    valid_samples = downsampled.iloc[valid_positions].copy()
    n_valid = len(valid_samples)

    n_words = len(aoi)
    n_lines = int(aoi["line_index"].max()) + 1
    median_word_height = float(aoi["height"].median())
    if not math.isfinite(median_word_height) or median_word_height <= 0:
        raise ValueError(
            f"Invalid median word height for {trial_row['trial_id']}: "
            f"{median_word_height}"
        )
    sigma_px = sigma_word_height_ratio * median_word_height
    if sigma_px <= 0:
        raise ValueError("Derived sigma must be positive.")

    timestamps_all = downsampled["timestamp_ms_abs"].to_numpy(dtype=float)
    duration_sec = (
        float((timestamps_all[-1] - timestamps_all[0]) / 1000.0)
        if len(timestamps_all) >= 2
        else 0.0
    )

    base: dict[str, Any] = {
        "subject_id": str(trial_row["subject_id"]).upper(),
        "task": str(trial_row["task"]).upper(),
        "block_id": int(trial_row["block_id"]),
        "trial_index_in_block": int(trial_row["trial_index_in_block"]),
        "trial_id": str(trial_row["trial_id"]),
        "task_sentence_index": int(trial_row["task_sentence_index"]),
        "bounds_source_subject": str(
            aoi["bounds_source_subject"].iloc[0]
            if "bounds_source_subject" in aoi.columns
            else ""
        ),
        "soft_aoi_rate_hz": int(rate_hz),
        "soft_aoi_sigma_word_height_ratio": float(
            sigma_word_height_ratio
        ),
        "soft_aoi_sigma_px": float(sigma_px),
        "soft_aoi_active_threshold": float(active_threshold),
        "soft_aoi_n_words": int(n_words),
        "soft_aoi_n_lines": int(n_lines),
        "soft_aoi_n_samples": int(n_samples),
        "soft_aoi_n_valid_samples": int(n_valid),
        "soft_aoi_duration_sec": float(duration_sec),
        "soft_aoi_valid_rate": safe_fraction(n_valid, n_samples),
    }

    feature_names = CONTENT_FEATURES + MEMBERSHIP_QUALITY_FEATURES
    if n_valid == 0:
        for name in feature_names:
            if name not in base:
                base[name] = math.nan
        return base

    x = valid_samples["x_px"].to_numpy(dtype=float)
    y = valid_samples["y_px"].to_numpy(dtype=float)
    distances = rectangle_distances(x, y, aoi)
    nearest = distances.min(axis=1)
    q = np.exp(-(distances**2) / (2.0 * sigma_px**2))
    q_max = q.max(axis=1)
    dominant = q.argmax(axis=1)
    active = q_max >= active_threshold
    n_active = int(np.sum(active))

    sentence_x_min = float(aoi["x_min"].min())
    sentence_x_max = float(aoi["x_max"].max())
    sentence_y_min = float(aoi["y_min"].min())
    sentence_y_max = float(aoi["y_max"].max())
    envelope_hit = (
        (x >= sentence_x_min)
        & (x <= sentence_x_max)
        & (y >= sentence_y_min)
        & (y <= sentence_y_max)
    )

    base.update(
        {
            "soft_aoi_active_rate": safe_fraction(n_active, n_valid),
            "soft_aoi_word_box_hit_rate": float(np.mean(nearest == 0)),
            "soft_aoi_sentence_envelope_hit_rate": float(
                np.mean(envelope_hit)
            ),
            "soft_aoi_qmax_mean": float(np.mean(q_max)),
            "soft_aoi_qmax_median": float(np.median(q_max)),
            "soft_aoi_qmax_q10": float(np.quantile(q_max, 0.10)),
            "soft_aoi_qmax_ge_0_5_rate": float(np.mean(q_max >= 0.5)),
            "soft_aoi_nearest_distance_mean_word_heights": float(
                np.mean(nearest / median_word_height)
            ),
            "soft_aoi_nearest_distance_median_word_heights": float(
                np.median(nearest / median_word_height)
            ),
            "soft_aoi_nearest_distance_q90_word_heights": float(
                np.quantile(nearest / median_word_height, 0.90)
            ),
            "soft_aoi_nearest_distance_q95_word_heights": float(
                np.quantile(nearest / median_word_height, 0.95)
            ),
        }
    )

    if n_active == 0:
        for name in CONTENT_FEATURES:
            base[name] = math.nan
        return base

    q_active = q[active]
    qmax_active = q_max[active]
    q_sum = q_active.sum(axis=1, keepdims=True)
    posterior = q_active / np.maximum(q_sum, np.finfo(float).tiny)

    word_progress_values = aoi["word_progress"].to_numpy(dtype=float)
    line_progress_values = aoi["line_progress"].to_numpy(dtype=float)
    line_norm_values = aoi["line_index_norm"].to_numpy(dtype=float)
    line_index_values = aoi["line_index"].to_numpy(dtype=int)

    expected_progress = posterior @ word_progress_values
    expected_line_progress = posterior @ line_progress_values
    expected_line_norm = posterior @ line_norm_values

    sample_entropy = np.zeros(n_active, dtype=float)
    if n_words > 1:
        sample_entropy = -np.sum(
            posterior * np.log(np.maximum(posterior, np.finfo(float).tiny)),
            axis=1,
        ) / math.log(n_words)

    aggregate_q = q_active.sum(axis=0)
    occupancy = aggregate_q / aggregate_q.sum()
    occupancy_sorted = np.sort(occupancy)[::-1]
    occupancy_entropy = normalized_entropy(occupancy)
    effective_fraction = (
        math.exp(occupancy_entropy * math.log(n_words)) / n_words
        if n_words > 1
        else 1.0
    )

    active_dominant = dominant[active]
    dominant_lines = line_index_values[active_dominant]
    dominant_line_progress = line_progress_values[active_dominant]

    time_valid = (
        valid_samples["timestamp_ms_abs"].to_numpy(dtype=float)
        - timestamps_all[0]
    ) / 1000.0
    time_active = time_valid[active]
    midpoint = duration_sec / 2.0
    first = time_active <= midpoint
    second = time_active > midpoint

    active_positions_in_downsampled = valid_positions[active]
    adjacent = np.diff(active_positions_in_downsampled) == 1
    word_diff = np.diff(active_dominant)
    line_diff = np.diff(dominant_lines)
    previous_line_progress = dominant_line_progress[:-1]
    current_line_progress = dominant_line_progress[1:]

    word_diff_adj = word_diff[adjacent]
    line_diff_adj = line_diff[adjacent]
    previous_line_progress_adj = previous_line_progress[adjacent]
    current_line_progress_adj = current_line_progress[adjacent]
    n_pairs = len(word_diff_adj)
    changed = word_diff_adj != 0
    n_changed = int(np.sum(changed))
    forward = word_diff_adj > 0
    backward = word_diff_adj < 0
    stationary = word_diff_adj == 0
    line_forward = line_diff_adj > 0
    line_backward = line_diff_adj < 0
    line_return = (
        (line_diff_adj == 1)
        & (previous_line_progress_adj >= 0.60)
        & (current_line_progress_adj <= 0.40)
    )

    base.update(
        {
            "soft_aoi_sample_entropy_mean": weighted_mean(
                sample_entropy, qmax_active
            ),
            "soft_aoi_sample_entropy_median": weighted_quantile(
                sample_entropy, qmax_active, 0.50
            ),
            "soft_aoi_occupancy_entropy_norm": float(occupancy_entropy),
            "soft_aoi_occupancy_effective_word_fraction": float(
                effective_fraction
            ),
            "soft_aoi_occupancy_top1_share": float(occupancy_sorted[0]),
            "soft_aoi_occupancy_top3_share": float(
                occupancy_sorted[: min(3, n_words)].sum()
            ),
            "soft_aoi_dominant_word_coverage": float(
                len(np.unique(active_dominant)) / n_words
            ),
            "soft_aoi_dominant_line_coverage": float(
                len(np.unique(dominant_lines)) / n_lines
            ),
            "soft_aoi_progress_mean": weighted_mean(
                expected_progress, qmax_active
            ),
            "soft_aoi_progress_std": weighted_std(
                expected_progress, qmax_active
            ),
            "soft_aoi_progress_q25": weighted_quantile(
                expected_progress, qmax_active, 0.25
            ),
            "soft_aoi_progress_median": weighted_quantile(
                expected_progress, qmax_active, 0.50
            ),
            "soft_aoi_progress_q75": weighted_quantile(
                expected_progress, qmax_active, 0.75
            ),
            "soft_aoi_progress_slope_per_sec": weighted_slope(
                time_active, expected_progress, qmax_active
            ),
            "soft_aoi_line_progress_mean": weighted_mean(
                expected_line_progress, qmax_active
            ),
            "soft_aoi_line_position_mean": weighted_mean(
                expected_line_norm, qmax_active
            ),
            "soft_aoi_progress_first_half": weighted_mean(
                expected_progress[first], qmax_active[first]
            ),
            "soft_aoi_progress_second_half": weighted_mean(
                expected_progress[second], qmax_active[second]
            ),
            "soft_aoi_transition_rate_per_sec": safe_fraction(
                n_changed, duration_sec
            ),
            "soft_aoi_forward_transition_fraction": safe_fraction(
                int(np.sum(forward)), n_pairs
            ),
            "soft_aoi_backward_transition_fraction": safe_fraction(
                int(np.sum(backward)), n_pairs
            ),
            "soft_aoi_stationary_transition_fraction": safe_fraction(
                int(np.sum(stationary)), n_pairs
            ),
            "soft_aoi_mean_abs_word_jump_norm": (
                float(np.mean(np.abs(word_diff_adj))) / max(n_words - 1, 1)
                if n_pairs
                else math.nan
            ),
            "soft_aoi_backward_jump_mean_norm": (
                float(np.mean(np.abs(word_diff_adj[backward])))
                / max(n_words - 1, 1)
                if np.any(backward)
                else 0.0
            ),
            "soft_aoi_line_forward_fraction": safe_fraction(
                int(np.sum(line_forward)), n_pairs
            ),
            "soft_aoi_line_backward_fraction": safe_fraction(
                int(np.sum(line_backward)), n_pairs
            ),
            "soft_aoi_line_return_fraction": safe_fraction(
                int(np.sum(line_return)), n_pairs
            ),
            "soft_aoi_regression_events_per_sec": safe_fraction(
                int(np.sum(backward)), duration_sec
            ),
        }
    )
    first_value = base["soft_aoi_progress_first_half"]
    second_value = base["soft_aoi_progress_second_half"]
    base["soft_aoi_progress_half_delta"] = (
        float(second_value - first_value)
        if math.isfinite(first_value) and math.isfinite(second_value)
        else math.nan
    )
    return base


MEMBERSHIP_QUALITY_FEATURES = [
    "soft_aoi_valid_rate",
    "soft_aoi_active_rate",
    "soft_aoi_word_box_hit_rate",
    "soft_aoi_sentence_envelope_hit_rate",
    "soft_aoi_qmax_mean",
    "soft_aoi_qmax_median",
    "soft_aoi_qmax_q10",
    "soft_aoi_qmax_ge_0_5_rate",
    "soft_aoi_nearest_distance_mean_word_heights",
    "soft_aoi_nearest_distance_median_word_heights",
    "soft_aoi_nearest_distance_q90_word_heights",
    "soft_aoi_nearest_distance_q95_word_heights",
]

CONTENT_FEATURES = [
    "soft_aoi_sample_entropy_mean",
    "soft_aoi_sample_entropy_median",
    "soft_aoi_occupancy_entropy_norm",
    "soft_aoi_occupancy_effective_word_fraction",
    "soft_aoi_occupancy_top1_share",
    "soft_aoi_occupancy_top3_share",
    "soft_aoi_dominant_word_coverage",
    "soft_aoi_dominant_line_coverage",
    "soft_aoi_progress_mean",
    "soft_aoi_progress_std",
    "soft_aoi_progress_q25",
    "soft_aoi_progress_median",
    "soft_aoi_progress_q75",
    "soft_aoi_progress_slope_per_sec",
    "soft_aoi_line_progress_mean",
    "soft_aoi_line_position_mean",
    "soft_aoi_progress_first_half",
    "soft_aoi_progress_second_half",
    "soft_aoi_progress_half_delta",
    "soft_aoi_transition_rate_per_sec",
    "soft_aoi_forward_transition_fraction",
    "soft_aoi_backward_transition_fraction",
    "soft_aoi_stationary_transition_fraction",
    "soft_aoi_mean_abs_word_jump_norm",
    "soft_aoi_backward_jump_mean_norm",
    "soft_aoi_line_forward_fraction",
    "soft_aoi_line_backward_fraction",
    "soft_aoi_line_return_fraction",
    "soft_aoi_regression_events_per_sec",
]


FEATURE_DESCRIPTIONS = {
    "soft_aoi_valid_rate": "Fraction of downsampled samples with valid x, y, and pupil.",
    "soft_aoi_active_rate": "Fraction of valid samples whose maximum word membership is at least the active threshold.",
    "soft_aoi_word_box_hit_rate": "Fraction of valid samples lying inside at least one word rectangle.",
    "soft_aoi_sentence_envelope_hit_rate": "Fraction of valid samples inside the sentence's axis-aligned envelope.",
    "soft_aoi_qmax_mean": "Mean maximum word Soft-AOI membership.",
    "soft_aoi_qmax_median": "Median maximum word Soft-AOI membership.",
    "soft_aoi_qmax_q10": "10th percentile of maximum word membership.",
    "soft_aoi_qmax_ge_0_5_rate": "Fraction of valid samples with maximum membership at least 0.5.",
    "soft_aoi_nearest_distance_mean_word_heights": "Mean nearest-word distance divided by sentence median word height.",
    "soft_aoi_nearest_distance_median_word_heights": "Median nearest-word distance divided by sentence median word height.",
    "soft_aoi_nearest_distance_q90_word_heights": "90th percentile normalized nearest-word distance.",
    "soft_aoi_nearest_distance_q95_word_heights": "95th percentile normalized nearest-word distance.",
    "soft_aoi_sample_entropy_mean": "Membership-weighted mean normalized entropy of the sample-level word posterior.",
    "soft_aoi_sample_entropy_median": "Membership-weighted median normalized sample-level posterior entropy.",
    "soft_aoi_occupancy_entropy_norm": "Normalized entropy of aggregate word Soft-AOI mass.",
    "soft_aoi_occupancy_effective_word_fraction": "Entropy-derived effective number of occupied words divided by sentence word count.",
    "soft_aoi_occupancy_top1_share": "Largest word share of aggregate Soft-AOI mass.",
    "soft_aoi_occupancy_top3_share": "Three largest words' share of aggregate Soft-AOI mass.",
    "soft_aoi_dominant_word_coverage": "Fraction of sentence words selected as dominant AOI at least once.",
    "soft_aoi_dominant_line_coverage": "Fraction of sentence lines selected as dominant AOI at least once.",
    "soft_aoi_progress_mean": "Membership-weighted mean normalized sentence word progress.",
    "soft_aoi_progress_std": "Membership-weighted standard deviation of normalized sentence progress.",
    "soft_aoi_progress_q25": "Membership-weighted 25th percentile of sentence progress.",
    "soft_aoi_progress_median": "Membership-weighted median sentence progress.",
    "soft_aoi_progress_q75": "Membership-weighted 75th percentile of sentence progress.",
    "soft_aoi_progress_slope_per_sec": "Membership-weighted linear slope of sentence progress per second.",
    "soft_aoi_line_progress_mean": "Membership-weighted mean within-line progress.",
    "soft_aoi_line_position_mean": "Membership-weighted mean normalized line index.",
    "soft_aoi_progress_first_half": "Membership-weighted progress mean in the first trial half.",
    "soft_aoi_progress_second_half": "Membership-weighted progress mean in the second trial half.",
    "soft_aoi_progress_half_delta": "Second-half progress minus first-half progress.",
    "soft_aoi_transition_rate_per_sec": "Dominant-word changes per trial second among consecutive active samples.",
    "soft_aoi_forward_transition_fraction": "Fraction of consecutive active pairs moving to a later word.",
    "soft_aoi_backward_transition_fraction": "Fraction of consecutive active pairs moving to an earlier word.",
    "soft_aoi_stationary_transition_fraction": "Fraction of consecutive active pairs retaining the same dominant word.",
    "soft_aoi_mean_abs_word_jump_norm": "Mean absolute dominant-word jump divided by sentence length minus one.",
    "soft_aoi_backward_jump_mean_norm": "Mean backward jump magnitude normalized by sentence length minus one.",
    "soft_aoi_line_forward_fraction": "Fraction of consecutive active pairs moving to a later line.",
    "soft_aoi_line_backward_fraction": "Fraction of consecutive active pairs moving to an earlier line.",
    "soft_aoi_line_return_fraction": "Fraction of pairs matching a line-end to next-line-start return pattern.",
    "soft_aoi_regression_events_per_sec": "Backward dominant-word transitions per trial second.",
}


def validate_inputs(
    trial_map: pd.DataFrame,
    word_template: pd.DataFrame,
    subject: str,
    expected_trials: int,
) -> None:
    required_trial = {
        "subject_id",
        "task",
        "block_id",
        "trial_index_in_block",
        "trial_id",
        "task_sentence_index",
    }
    missing = sorted(required_trial - set(trial_map.columns))
    if missing:
        raise ValueError(f"Missing trial-map columns: {missing}")

    required_aoi = {
        "task",
        "task_sentence_index",
        "word_index",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "height",
        "word_progress",
        "line_index",
        "line_progress",
        "line_index_norm",
    }
    missing = sorted(required_aoi - set(word_template.columns))
    if missing:
        raise ValueError(f"Missing word-template columns: {missing}")

    subjects = set(trial_map["subject_id"].astype(str).str.upper())
    if subjects != {subject}:
        raise ValueError(
            f"Trial map subjects={sorted(subjects)}, expected only {subject}."
        )
    if len(trial_map) != expected_trials:
        raise ValueError(
            f"Trial map rows={len(trial_map)}, expected={expected_trials}."
        )
    if trial_map["trial_id"].duplicated().any():
        raise ValueError("Trial map contains duplicate trial IDs.")

    counts = trial_map.groupby("task").size().to_dict()
    for task, expected in EXPECTED_TASK_COUNTS.items():
        actual = int(counts.get(task, 0))
        if actual != expected:
            raise ValueError(
                f"Task trial count mismatch for {task}: "
                f"actual={actual}, expected={expected}."
            )

    sentence_keys = trial_map[["task", "task_sentence_index"]]
    template_keys = word_template[["task", "task_sentence_index"]].drop_duplicates()
    merged = sentence_keys.merge(
        template_keys,
        on=["task", "task_sentence_index"],
        how="left",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Some trial-map sentences lack AOI templates.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="YAC")
    parser.add_argument(
        "--interim-dir",
        default="data/interim/zuco2/YAC",
    )
    parser.add_argument(
        "--trial-map",
        default=(
            "data/processed/zuco2/YAC/"
            "YAC_all_trials_with_sentence_index.csv"
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
        "--output-dir",
        default="data/processed/zuco2/YAC/soft_aoi",
    )
    parser.add_argument("--rate-hz", type=int, default=30)
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
    parser.add_argument("--expected-trials", type=int, default=739)
    args = parser.parse_args()

    subject = args.subject.upper()
    interim_dir = Path(args.interim_dir)
    trial_map_path = Path(args.trial_map)
    word_template_path = Path(args.word_template)
    output_dir = Path(args.output_dir)

    if args.rate_hz <= 0 or args.rate_hz > BASE_RATE_HZ:
        raise ValueError(f"--rate-hz must be within 1..{BASE_RATE_HZ}.")
    if args.sigma_word_height_ratio <= 0:
        raise ValueError("--sigma-word-height-ratio must be positive.")
    if not (0 < args.active_threshold <= 1):
        raise ValueError("--active-threshold must be within (0, 1].")
    if not interim_dir.is_dir():
        raise FileNotFoundError(interim_dir)
    if not trial_map_path.is_file():
        raise FileNotFoundError(trial_map_path)
    if not word_template_path.is_file():
        raise FileNotFoundError(word_template_path)

    trial_map = pd.read_csv(trial_map_path)
    trial_map["subject_id"] = trial_map["subject_id"].astype(str).str.upper()
    trial_map["task"] = trial_map["task"].astype(str).str.upper()
    word_template = pd.read_csv(word_template_path)
    word_template["task"] = word_template["task"].astype(str).str.upper()

    validate_inputs(
        trial_map=trial_map,
        word_template=word_template,
        subject=subject,
        expected_trials=args.expected_trials,
    )

    trial_lookup = trial_map.set_index("trial_id", drop=False)
    aoi_lookup = {
        (str(task), int(sentence_index)): group.sort_values("word_index")
        for (task, sentence_index), group in word_template.groupby(
            ["task", "task_sentence_index"], sort=False
        )
    }

    sample_files = sorted(interim_dir.glob(f"{subject}_*_samples.parquet"))
    if len(sample_files) != 14:
        raise ValueError(
            f"Found {len(sample_files)} sample Parquet files; expected 14."
        )

    rows: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    for sample_path in sample_files:
        print(f"Processing {sample_path}")
        samples = pd.read_parquet(sample_path)
        required_sample = {
            "trial_id",
            "sample_index_in_trial",
            "timestamp_ms_abs",
            "x_px",
            "y_px",
            "valid_basic",
        }
        missing = sorted(required_sample - set(samples.columns))
        if missing:
            raise ValueError(
                f"Missing sample columns in {sample_path}: {missing}"
            )

        for trial_id, trial_samples in samples.groupby("trial_id", sort=False):
            trial_id = str(trial_id)
            if trial_id in processed_ids:
                raise ValueError(f"Duplicate trial across Parquet files: {trial_id}")
            if trial_id not in trial_lookup.index:
                raise ValueError(f"Trial missing from trial map: {trial_id}")
            trial_row = trial_lookup.loc[trial_id]
            key = (
                str(trial_row["task"]).upper(),
                int(trial_row["task_sentence_index"]),
            )
            if key not in aoi_lookup:
                raise ValueError(f"AOI template missing for key={key}")
            rows.append(
                extract_trial_features(
                    samples=trial_samples,
                    trial_row=trial_row,
                    aoi=aoi_lookup[key],
                    rate_hz=args.rate_hz,
                    sigma_word_height_ratio=args.sigma_word_height_ratio,
                    active_threshold=args.active_threshold,
                )
            )
            processed_ids.add(trial_id)

    missing_ids = sorted(set(trial_lookup.index.astype(str)) - processed_ids)
    if missing_ids:
        raise ValueError(f"Unprocessed trial IDs: {missing_ids[:30]}")

    features = pd.DataFrame(rows).sort_values(
        ["task", "task_sentence_index"]
    ).reset_index(drop=True)
    if len(features) != args.expected_trials:
        raise ValueError(
            f"Feature rows={len(features)}, expected={args.expected_trials}."
        )
    if features["trial_id"].duplicated().any():
        raise ValueError("Output contains duplicate trial IDs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / (
        f"{subject}_soft_aoi_features_{args.rate_hz}hz.csv"
    )
    summary_path = output_dir / (
        f"{subject}_soft_aoi_task_summary_{args.rate_hz}hz.csv"
    )
    manifest_path = output_dir / (
        f"{subject}_soft_aoi_feature_manifest_{args.rate_hz}hz.json"
    )

    features.to_csv(feature_path, index=False, encoding="utf-8-sig")

    summary_columns = [
        "soft_aoi_valid_rate",
        "soft_aoi_active_rate",
        "soft_aoi_word_box_hit_rate",
        "soft_aoi_qmax_mean",
        "soft_aoi_occupancy_entropy_norm",
        "soft_aoi_dominant_word_coverage",
        "soft_aoi_progress_slope_per_sec",
        "soft_aoi_backward_transition_fraction",
        "soft_aoi_regression_events_per_sec",
    ]
    summary = features.groupby("task")[summary_columns].agg(
        ["count", "mean", "median", "std"]
    )
    summary.to_csv(summary_path, encoding="utf-8-sig")

    manifest = {
        "subject": subject,
        "sampling_rate_hz": args.rate_hz,
        "sigma_definition": "sentence_median_word_height_times_ratio",
        "sigma_word_height_ratio": args.sigma_word_height_ratio,
        "active_threshold": args.active_threshold,
        "distance_definition": (
            "Euclidean distance from gaze point to nearest point of each "
            "word rectangle; zero inside rectangle."
        ),
        "membership_definition": "exp(-distance^2/(2*sigma^2))",
        "content_feature_columns": CONTENT_FEATURES,
        "membership_quality_feature_columns": MEMBERSHIP_QUALITY_FEATURES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "excluded_from_feature_sets_but_retained_as_metadata": [
            "task",
            "task_sentence_index",
            "block_id",
            "trial_index_in_block",
            "bounds_source_subject",
            "soft_aoi_n_words",
            "soft_aoi_n_lines",
            "soft_aoi_sigma_px",
            "soft_aoi_duration_sec",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== SOFT-AOI FEATURE SUMMARY ===")
    print(f"subject: {subject}")
    print(f"trials: {len(features)}")
    print(f"sampling rate: {args.rate_hz} Hz")
    print(
        "median sigma px: "
        f"{features['soft_aoi_sigma_px'].median():.4f}"
    )
    print(f"content features: {len(CONTENT_FEATURES)}")
    print(
        "membership/quality features: "
        f"{len(MEMBERSHIP_QUALITY_FEATURES)}"
    )
    print("\n=== TASK MEDIANS ===")
    print(
        features.groupby("task").agg(
            trials=("trial_id", "size"),
            valid_rate=("soft_aoi_valid_rate", "median"),
            active_rate=("soft_aoi_active_rate", "median"),
            box_hit_rate=("soft_aoi_word_box_hit_rate", "median"),
            qmax_mean=("soft_aoi_qmax_mean", "median"),
            occupancy_entropy=(
                "soft_aoi_occupancy_entropy_norm",
                "median",
            ),
            word_coverage=(
                "soft_aoi_dominant_word_coverage",
                "median",
            ),
            progress_slope=(
                "soft_aoi_progress_slope_per_sec",
                "median",
            ),
            backward_fraction=(
                "soft_aoi_backward_transition_fraction",
                "median",
            ),
        ).to_string()
    )
    print("\nWritten:")
    print(f"  {feature_path}")
    print(f"  {summary_path}")
    print(f"  {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
