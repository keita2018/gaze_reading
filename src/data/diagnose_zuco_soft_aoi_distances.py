#!/usr/bin/env python
"""Audit ZuCo Soft-AOI distances for one subject at a target sampling rate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_TASK_COUNTS = {"NR": 349, "TSR": 390}


def parse_sigmas(text: str) -> list[float]:
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x <= 0 for x in values):
        raise ValueError("--sigmas must contain positive numbers")
    return values


def nearest_regular_grid_indices(timestamps_ms: np.ndarray, target_rate_hz: int) -> np.ndarray:
    timestamps = np.asarray(timestamps_ms, dtype=float)
    if timestamps.ndim != 1 or timestamps.size == 0:
        return np.asarray([], dtype=np.int64)
    finite = np.isfinite(timestamps)
    valid_indices = np.flatnonzero(finite)
    if valid_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    ts = timestamps[valid_indices]
    if np.any(np.diff(ts) < 0):
        raise ValueError("Timestamps are not monotonically increasing")
    if ts.size == 1:
        return valid_indices[:1]
    step_ms = 1000.0 / float(target_rate_hz)
    grid = ts[0] + np.arange(int(math.floor((ts[-1] - ts[0]) / step_ms)) + 1) * step_ms
    right = np.searchsorted(ts, grid, side="left")
    right = np.clip(right, 0, len(ts) - 1)
    left = np.clip(right - 1, 0, len(ts) - 1)
    choose_left = np.abs(grid - ts[left]) <= np.abs(ts[right] - grid)
    selected = np.where(choose_left, left, right)
    selected = np.unique(selected.astype(np.int64))
    return valid_indices[selected]


def q(values: np.ndarray, percentile: float) -> float:
    return float(np.quantile(values, percentile)) if values.size else math.nan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="YAC")
    parser.add_argument("--interim-dir", default="data/interim/zuco2/YAC")
    parser.add_argument("--trial-map", default="data/processed/zuco2/YAC/YAC_all_trials_with_sentence_index.csv")
    parser.add_argument("--word-template", default="data/processed/zuco2/soft_aoi_template/zuco_soft_aoi_word_template.csv")
    parser.add_argument("--output-dir", default="outputs/soft_aoi_diagnostics/YAC")
    parser.add_argument("--rate-hz", type=int, default=30)
    parser.add_argument("--sigmas", default="5,10,15,30")
    args = parser.parse_args()

    subject = args.subject.upper()
    interim_dir = Path(args.interim_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigmas = parse_sigmas(args.sigmas)

    trial_map = pd.read_csv(args.trial_map)
    trial_map["task"] = trial_map["task"].astype(str).str.upper()
    trial_map["trial_id"] = trial_map["trial_id"].astype(str)
    if trial_map["trial_id"].duplicated().any():
        raise ValueError("Duplicate trial_id values in trial map")
    expected_total = sum(EXPECTED_TASK_COUNTS.values())
    if len(trial_map) != expected_total:
        raise ValueError(f"trial map rows={len(trial_map)}, expected={expected_total}")

    words = pd.read_csv(args.word_template)
    words["task"] = words["task"].astype(str).str.upper()
    words_by_key = {
        (task, int(sentence_index)): group.sort_values("word_index").copy()
        for (task, sentence_index), group in words.groupby(["task", "task_sentence_index"], sort=False)
    }
    trial_lookup = trial_map.set_index("trial_id")

    parquet_files = sorted(interim_dir.glob(f"{subject}_*_samples.parquet"))
    if len(parquet_files) != 14:
        raise FileNotFoundError(f"Expected 14 sample Parquet files, found {len(parquet_files)} under {interim_dir}")

    rows: list[dict[str, object]] = []
    pooled: dict[str, list[np.ndarray]] = {"ALL": [], "NR": [], "TSR": []}
    seen_trials: set[str] = set()

    for parquet_path in parquet_files:
        print(f"Processing {parquet_path}")
        samples = pd.read_parquet(parquet_path)
        required = {"trial_id", "sample_index_in_trial", "timestamp_ms_abs", "x_px", "y_px", "valid_basic"}
        missing = sorted(required - set(samples.columns))
        if missing:
            raise ValueError(f"Missing sample columns in {parquet_path}: {missing}")

        for trial_id, group in samples.groupby("trial_id", sort=False):
            trial_id = str(trial_id)
            if trial_id not in trial_lookup.index:
                raise KeyError(f"Trial not found in map: {trial_id}")
            if trial_id in seen_trials:
                raise ValueError(f"Duplicate trial across Parquet files: {trial_id}")
            seen_trials.add(trial_id)

            meta = trial_lookup.loc[trial_id]
            task = str(meta["task"]).upper()
            sentence_index = int(meta["task_sentence_index"])
            aoi = words_by_key.get((task, sentence_index))
            if aoi is None or aoi.empty:
                raise KeyError(f"No AOI for task={task}, sentence={sentence_index}")

            group = group.sort_values("sample_index_in_trial")
            indices = nearest_regular_grid_indices(group["timestamp_ms_abs"].to_numpy(dtype=float), args.rate_hz)
            if indices.size == 0:
                raise ValueError(f"Downsampling produced no samples: {trial_id}")
            down = group.iloc[indices].copy()

            valid = down["valid_basic"].astype(bool).to_numpy()
            x = down["x_px"].to_numpy(dtype=float)
            y = down["y_px"].to_numpy(dtype=float)
            valid &= np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]

            if x.size:
                px = x[:, None]
                py = y[:, None]
                xmin = aoi["x_min"].to_numpy(dtype=float)[None, :]
                xmax = aoi["x_max"].to_numpy(dtype=float)[None, :]
                ymin = aoi["y_min"].to_numpy(dtype=float)[None, :]
                ymax = aoi["y_max"].to_numpy(dtype=float)[None, :]
                dx = np.maximum(np.maximum(xmin - px, px - xmax), 0.0)
                dy = np.maximum(np.maximum(ymin - py, py - ymax), 0.0)
                nearest = np.sqrt(dx * dx + dy * dy).min(axis=1)
                envelope = (
                    (x >= float(aoi["x_min"].min()))
                    & (x <= float(aoi["x_max"].max()))
                    & (y >= float(aoi["y_min"].min()))
                    & (y <= float(aoi["y_max"].max()))
                )
            else:
                nearest = np.asarray([], dtype=float)
                envelope = np.asarray([], dtype=bool)

            pooled["ALL"].append(nearest)
            pooled[task].append(nearest)
            row: dict[str, object] = {
                "subject_id": subject,
                "task": task,
                "trial_id": trial_id,
                "task_sentence_index": sentence_index,
                "sampling_rate_hz": args.rate_hz,
                "n_samples": len(down),
                "n_valid": int(x.size),
                "valid_rate": float(x.size / len(down)),
                "word_box_hit_rate": float(np.mean(nearest == 0.0)) if nearest.size else math.nan,
                "sentence_envelope_hit_rate": float(np.mean(envelope)) if envelope.size else math.nan,
                "nearest_distance_mean_px": float(np.mean(nearest)) if nearest.size else math.nan,
                "nearest_distance_q50_px": q(nearest, 0.50),
                "nearest_distance_q75_px": q(nearest, 0.75),
                "nearest_distance_q90_px": q(nearest, 0.90),
                "nearest_distance_q95_px": q(nearest, 0.95),
                "nearest_distance_q99_px": q(nearest, 0.99),
                "nearest_distance_max_px": float(np.max(nearest)) if nearest.size else math.nan,
            }
            for sigma in sigmas:
                name = f"mean_max_soft_sigma_{sigma:g}px"
                row[name] = float(np.mean(np.exp(-(nearest ** 2) / (2.0 * sigma ** 2)))) if nearest.size else math.nan
            rows.append(row)

    if seen_trials != set(trial_lookup.index.astype(str)):
        missing_trials = sorted(set(trial_lookup.index.astype(str)) - seen_trials)
        extra_trials = sorted(seen_trials - set(trial_lookup.index.astype(str)))
        raise ValueError(f"Trial coverage mismatch; missing={missing_trials[:20]}, extra={extra_trials[:20]}")

    trial_summary = pd.DataFrame(rows).sort_values(["task", "task_sentence_index"]).reset_index(drop=True)
    pooled_rows: list[dict[str, object]] = []
    for task in ("ALL", "NR", "TSR"):
        arrays = [a for a in pooled[task] if a.size]
        distances = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
        pooled_row: dict[str, object] = {
            "task": task,
            "n_valid_samples": int(distances.size),
            "word_box_hit_rate": float(np.mean(distances == 0.0)) if distances.size else math.nan,
            "nearest_distance_mean_px": float(np.mean(distances)) if distances.size else math.nan,
            "nearest_distance_q50_px": q(distances, 0.50),
            "nearest_distance_q75_px": q(distances, 0.75),
            "nearest_distance_q90_px": q(distances, 0.90),
            "nearest_distance_q95_px": q(distances, 0.95),
            "nearest_distance_q99_px": q(distances, 0.99),
            "nearest_distance_max_px": float(np.max(distances)) if distances.size else math.nan,
        }
        for sigma in sigmas:
            pooled_row[f"mean_max_soft_sigma_{sigma:g}px"] = float(np.mean(np.exp(-(distances ** 2) / (2.0 * sigma ** 2)))) if distances.size else math.nan
        pooled_rows.append(pooled_row)
    pooled_summary = pd.DataFrame(pooled_rows)

    trial_path = output_dir / f"{subject}_soft_aoi_distance_trial_summary_{args.rate_hz}hz.csv"
    pooled_path = output_dir / f"{subject}_soft_aoi_distance_pooled_summary_{args.rate_hz}hz.csv"
    trial_summary.to_csv(trial_path, index=False, encoding="utf-8-sig")
    pooled_summary.to_csv(pooled_path, index=False, encoding="utf-8-sig")

    print("\n=== TRIAL COVERAGE ===")
    print(trial_summary.groupby("task").agg(trials=("trial_id", "size"), median_valid_rate=("valid_rate", "median"), median_box_hit_rate=("word_box_hit_rate", "median"), median_q95_distance_px=("nearest_distance_q95_px", "median")).to_string())
    print("\n=== POOLED DISTANCE SUMMARY ===")
    print(pooled_summary.to_string(index=False))
    print("\nWritten:")
    print(f"  {trial_path}")
    print(f"  {pooled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
