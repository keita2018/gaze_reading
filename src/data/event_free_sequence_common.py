#!/usr/bin/env python
"""Shared utilities for event-free gaze-sequence modeling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


CHANNEL_NAMES = [
    "x_text_norm",
    "y_text_norm",
    "soft_word_progress",
    "soft_line_index",
    "soft_line_progress",
    "qmax",
    "membership_entropy_norm",
    "nearest_distance_word_heights",
    "word_box_hit",
    "valid",
    "active",
    "delta_x_text",
    "delta_y_text",
    "delta_progress",
    "elapsed_norm",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_seed(
    base_seed: int,
    trial_id: str,
    condition: str,
    repeat: int,
) -> int:
    payload = f"{base_seed}|{trial_id}|{condition}|{repeat}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def normalize_id(value: Any) -> str:
    return str(value).strip().lower()


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
        .fillna(False)
    )


def canonicalize_aoi(aoi: pd.DataFrame) -> pd.DataFrame:
    """Convert OneStop or WebQAmGaze AOIs to the common rectangle schema."""
    frame = aoi.copy()

    rename = {}
    aliases = {
        "x_min": ("x_min", "ia_left", "left", "xmin"),
        "x_max": ("x_max", "ia_right", "right", "xmax"),
        "y_min": ("y_min", "ia_top", "top", "ymin"),
        "y_max": ("y_max", "ia_bottom", "bottom", "ymax"),
    }
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for canonical, candidates in aliases.items():
        if canonical in frame.columns:
            continue
        for candidate in candidates:
            if candidate.lower() in lower:
                rename[lower[candidate.lower()]] = canonical
                break
    frame = frame.rename(columns=rename)

    required = {"x_min", "x_max", "y_min", "y_max"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AOI table lacks rectangle columns: {missing}")

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        np.isfinite(frame["x_min"])
        & np.isfinite(frame["x_max"])
        & np.isfinite(frame["y_min"])
        & np.isfinite(frame["y_max"])
        & frame["x_max"].gt(frame["x_min"])
        & frame["y_max"].gt(frame["y_min"])
    ].copy()
    if frame.empty:
        raise ValueError("AOI table contains no valid rectangles.")

    frame["x_center"] = (frame["x_min"] + frame["x_max"]) / 2.0
    frame["y_center"] = (frame["y_min"] + frame["y_max"]) / 2.0
    frame["width"] = frame["x_max"] - frame["x_min"]
    frame["height"] = frame["y_max"] - frame["y_min"]

    if "line_index" not in frame.columns:
        median_height = float(frame["height"].median())
        tolerance = max(1.0, 0.60 * median_height)
        ordered = frame.sort_values(["y_center", "x_center"]).index
        line_centers: list[float] = []
        assignment: dict[Any, int] = {}
        for index in ordered:
            y_value = float(frame.at[index, "y_center"])
            if not line_centers:
                line_centers.append(y_value)
                assignment[index] = 0
                continue
            distances = np.abs(np.asarray(line_centers, dtype=float) - y_value)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= tolerance:
                assignment[index] = nearest
                line_centers[nearest] = (line_centers[nearest] + y_value) / 2.0
            else:
                line_centers.append(y_value)
                assignment[index] = len(line_centers) - 1
        order = np.argsort(np.asarray(line_centers, dtype=float))
        remap = {int(old): int(new) for new, old in enumerate(order)}
        frame["line_index"] = [remap[assignment[index]] for index in frame.index]
    else:
        frame["line_index"] = pd.to_numeric(
            frame["line_index"], errors="raise"
        ).astype(int)

    if "word_index" not in frame.columns:
        order = frame.sort_values(["line_index", "x_center", "y_center"]).index
        mapping = {index: position for position, index in enumerate(order)}
        frame["word_index"] = [mapping[index] for index in frame.index]
    else:
        frame["word_index"] = pd.to_numeric(
            frame["word_index"], errors="raise"
        ).astype(int)

    frame = frame.sort_values("word_index").reset_index(drop=True)
    n_words = len(frame)
    n_lines = int(frame["line_index"].max()) + 1

    if "word_progress" not in frame.columns:
        frame["word_progress"] = (
            frame["word_index"] / float(max(n_words - 1, 1))
        )
    if "line_index_norm" not in frame.columns:
        frame["line_index_norm"] = (
            frame["line_index"] / float(max(n_lines - 1, 1))
        )
    if "line_progress" not in frame.columns:
        frame["line_progress"] = 0.0
        for _, group in frame.groupby("line_index", sort=True):
            values = (
                np.arange(len(group), dtype=float) / float(len(group) - 1)
                if len(group) > 1
                else np.zeros(1, dtype=float)
            )
            frame.loc[group.index, "line_progress"] = values

    return frame


def canonicalize_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Convert a dense 30 Hz sample table to the common schema."""
    frame = samples.copy()
    rename = {}

    if "sample_index_in_trial" not in frame.columns:
        for candidate in ("sample_index_30hz", "sample_index", "index_30hz"):
            if candidate in frame.columns:
                rename[candidate] = "sample_index_in_trial"
                break
    if "timestamp_ms_abs" not in frame.columns:
        for candidate in (
            "target_timestamp_ms",
            "t_ms",
            "timestamp_ms",
            "time_ms",
        ):
            if candidate in frame.columns:
                rename[candidate] = "timestamp_ms_abs"
                break
    if "valid_basic" not in frame.columns:
        for candidate in ("valid", "valid_xy", "is_valid"):
            if candidate in frame.columns:
                rename[candidate] = "valid_basic"
                break
    frame = frame.rename(columns=rename)

    required = {
        "sample_index_in_trial",
        "x_px",
        "y_px",
        "valid_basic",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sample table lacks columns: {missing}")

    frame["sample_index_in_trial"] = pd.to_numeric(
        frame["sample_index_in_trial"], errors="raise"
    ).astype(int)
    frame["x_px"] = pd.to_numeric(frame["x_px"], errors="coerce")
    frame["y_px"] = pd.to_numeric(frame["y_px"], errors="coerce")
    frame["valid_basic"] = to_bool_series(frame["valid_basic"])

    if "timestamp_ms_abs" not in frame.columns:
        frame["timestamp_ms_abs"] = (
            frame["sample_index_in_trial"].astype(float) * (1000.0 / 30.0)
        )
    else:
        frame["timestamp_ms_abs"] = pd.to_numeric(
            frame["timestamp_ms_abs"], errors="coerce"
        )

    frame = (
        frame.sort_values(["sample_index_in_trial", "timestamp_ms_abs"])
        .drop_duplicates("sample_index_in_trial", keep="first")
        .reset_index(drop=True)
    )
    return frame


def _rectangle_distances(
    x: np.ndarray,
    y: np.ndarray,
    aoi: pd.DataFrame,
) -> np.ndarray:
    x_min = aoi["x_min"].to_numpy(dtype=float)
    x_max = aoi["x_max"].to_numpy(dtype=float)
    y_min = aoi["y_min"].to_numpy(dtype=float)
    y_max = aoi["y_max"].to_numpy(dtype=float)

    dx = np.maximum(
        np.maximum(x_min[None, :] - x[:, None], 0.0),
        x[:, None] - x_max[None, :],
    )
    dy = np.maximum(
        np.maximum(y_min[None, :] - y[:, None], 0.0),
        y[:, None] - y_max[None, :],
    )
    return np.hypot(dx, dy)


def compute_event_free_tokens(
    samples: pd.DataFrame,
    aoi: pd.DataFrame,
    sigma_word_height_ratio: float = 2.0 / 3.0,
    active_threshold: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build event-free sample tokens without fixation/saccade detection."""
    sample_frame = canonicalize_samples(samples)
    aoi_frame = canonicalize_aoi(aoi)

    x = sample_frame["x_px"].to_numpy(dtype=float)
    y = sample_frame["y_px"].to_numpy(dtype=float)
    valid = (
        sample_frame["valid_basic"].to_numpy(dtype=bool)
        & np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )

    x_min = float(aoi_frame["x_min"].min())
    x_max = float(aoi_frame["x_max"].max())
    y_min = float(aoi_frame["y_min"].min())
    y_max = float(aoi_frame["y_max"].max())
    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)
    median_word_height = float(aoi_frame["height"].median())
    sigma = max(median_word_height * sigma_word_height_ratio, 1e-6)

    x_safe = np.where(valid, x, x_min)
    y_safe = np.where(valid, y, y_min)
    distances = _rectangle_distances(x_safe, y_safe, aoi_frame)
    q = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
    q[~valid, :] = 0.0
    qmax = q.max(axis=1)
    active = valid & (qmax >= active_threshold)

    weights = np.zeros_like(q, dtype=float)
    row_sum = q.sum(axis=1)
    usable = active & (row_sum > 0)
    weights[usable] = q[usable] / row_sum[usable, None]

    word_progress = aoi_frame["word_progress"].to_numpy(dtype=float)
    line_index_norm = aoi_frame["line_index_norm"].to_numpy(dtype=float)
    line_progress = aoi_frame["line_progress"].to_numpy(dtype=float)

    soft_progress = weights @ word_progress
    soft_line_index = weights @ line_index_norm
    soft_line_progress = weights @ line_progress

    n_words = len(aoi_frame)
    if n_words > 1:
        entropy = -np.sum(
            np.where(weights > 0, weights * np.log(np.maximum(weights, 1e-12)), 0.0),
            axis=1,
        ) / math.log(n_words)
    else:
        entropy = np.zeros(len(sample_frame), dtype=float)

    nearest = distances.min(axis=1) / max(median_word_height, 1e-6)
    nearest[~valid] = 0.0
    nearest = np.clip(nearest, 0.0, 10.0) / 10.0
    box_hit = valid & (distances.min(axis=1) <= 1e-9)

    x_norm = np.clip((x_safe - x_min) / width, -1.0, 2.0)
    y_norm = np.clip((y_safe - y_min) / height, -1.0, 2.0)
    x_norm[~valid] = 0.0
    y_norm[~valid] = 0.0

    timestamps = sample_frame["timestamp_ms_abs"].to_numpy(dtype=float)
    finite_time = np.isfinite(timestamps)
    if finite_time.any():
        start = float(np.nanmin(timestamps[finite_time]))
        end = float(np.nanmax(timestamps[finite_time]))
        elapsed_norm = (
            np.clip((timestamps - start) / max(end - start, 1e-6), 0.0, 1.0)
        )
        elapsed_norm[~finite_time] = 0.0
    else:
        elapsed_norm = np.linspace(0.0, 1.0, len(sample_frame), dtype=float)

    delta_x = np.zeros(len(sample_frame), dtype=float)
    delta_y = np.zeros(len(sample_frame), dtype=float)
    delta_progress = np.zeros(len(sample_frame), dtype=float)
    if len(sample_frame) > 1:
        consecutive_valid = valid[1:] & valid[:-1]
        consecutive_active = active[1:] & active[:-1]
        delta_x[1:] = np.where(
            consecutive_valid,
            np.clip(x_norm[1:] - x_norm[:-1], -2.0, 2.0),
            0.0,
        )
        delta_y[1:] = np.where(
            consecutive_valid,
            np.clip(y_norm[1:] - y_norm[:-1], -2.0, 2.0),
            0.0,
        )
        delta_progress[1:] = np.where(
            consecutive_active,
            np.clip(soft_progress[1:] - soft_progress[:-1], -1.0, 1.0),
            0.0,
        )

    tokens = np.column_stack(
        [
            x_norm,
            y_norm,
            soft_progress,
            soft_line_index,
            soft_line_progress,
            qmax,
            entropy,
            nearest,
            box_hit.astype(float),
            valid.astype(float),
            active.astype(float),
            delta_x,
            delta_y,
            delta_progress,
            elapsed_norm,
        ]
    ).astype(np.float32)

    if tokens.shape[1] != len(CHANNEL_NAMES):
        raise RuntimeError("Token channel count mismatch.")
    if not np.isfinite(tokens).all():
        raise RuntimeError("Token array contains NaN or infinity.")

    audit = {
        "n_samples": int(len(sample_frame)),
        "n_valid": int(valid.sum()),
        "valid_rate": float(valid.mean()) if len(valid) else 0.0,
        "n_active": int(active.sum()),
        "active_rate": float(active.mean()) if len(active) else 0.0,
        "word_box_hit_rate": float(box_hit.mean()) if len(box_hit) else 0.0,
        "qmax_mean": float(qmax[valid].mean()) if valid.any() else 0.0,
        "n_words": int(n_words),
        "n_lines": int(aoi_frame["line_index"].nunique()),
        "median_word_height_px": median_word_height,
        "sigma_px": sigma,
        "aoi_x_min": x_min,
        "aoi_x_max": x_max,
        "aoi_y_min": y_min,
        "aoi_y_max": y_max,
    }
    return tokens, audit


def resize_sequence(
    tokens: np.ndarray,
    max_length: int,
) -> tuple[np.ndarray, int, str]:
    """Pad short sequences and uniformly subsample long sequences."""
    if max_length < 8:
        raise ValueError("max_length must be at least 8.")
    original_length = int(len(tokens))
    if original_length <= 0:
        raise ValueError("Cannot resize an empty sequence.")

    if original_length > max_length:
        indices = np.linspace(
            0,
            original_length - 1,
            max_length,
        ).round().astype(np.int64)
        resized = tokens[indices]
        length = max_length
        mode = "uniform_subsample"
    else:
        resized = np.zeros(
            (max_length, tokens.shape[1]),
            dtype=np.float32,
        )
        resized[:original_length] = tokens
        length = original_length
        mode = "pad"
    return resized, length, mode


def pr_auc_trapezoid(
    y_true: np.ndarray,
    score: np.ndarray,
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, score)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def optimize_macro_f1_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
) -> tuple[float, float]:
    candidates = np.unique(
        np.concatenate(
            [
                np.quantile(score, np.linspace(0.01, 0.99, 399)),
                np.asarray([0.0, float(np.median(score))]),
            ]
        )
    )
    best_threshold = 0.0
    best_value = -1.0
    for threshold in candidates:
        prediction = (score >= threshold).astype(int)
        value = f1_score(
            y_true,
            prediction,
            average="macro",
            zero_division=0,
        )
        if value > best_value + 1e-12:
            best_value = float(value)
            best_threshold = float(threshold)
        elif abs(value - best_value) <= 1e-12:
            if abs(threshold) < abs(best_threshold):
                best_threshold = float(threshold)
    return best_threshold, best_value


def calculate_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = (score >= threshold).astype(int)
    return {
        "n_trials": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(
            f1_score(y_true, prediction, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc_nr": pr_auc_trapezoid(y_true, score),
        "average_precision_nr": float(
            average_precision_score(y_true, score)
        ),
        "nr_prediction_rate": float(prediction.mean()),
    }


def write_hash_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "file_hash_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(
        root / "file_hash_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
