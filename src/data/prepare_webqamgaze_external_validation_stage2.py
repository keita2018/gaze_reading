#!/usr/bin/env python
"""Build the formal WebQAmGaze English external-validation data foundation.

This script converts the official participant-level pandas-Series JSON files into:

1. a participant quality/filter audit;
2. a formal NR/IS trial manifest;
3. native irregular WebGazer samples;
4. 30 Hz gap-aware resampled samples;
5. official preprocessed fixation events.

No model is fitted or applied. WebQAmGaze labels are never used for training,
feature selection, preprocessing estimation, threshold selection, or calibration.

Fixed external label mapping
----------------------------
- information seeking (IS) -> 0, corresponding to ZuCo TSR
- normal reading (NR)      -> 1, corresponding to ZuCo NR

Predeclared participant-quality filter
--------------------------------------
The primary-candidate flag requires:
- fixation_error is False;
- target_error is False;
- experiment_reload is False;
- questions_duplicated is False;
- approved_flag >= --min-approved-flag;
- valid positive screen_x and screen_y;
- parsable raw and filtered gaze dictionaries.

Every participant and trial remains in the audit outputs. Exclusion reasons are
stored explicitly. The filter is based only on metadata/quality flags, not on
classification performance or NR/IS separability.

Expected WebQAmGaze participant schema
--------------------------------------
- webgazer_raw_data:
    JSON string mapping trial_name -> [[x, y, t_ms], ...]
- webgazer_filtered_data:
    JSON string mapping trial_name -> [[x, y, t_ms, duration_ms], ...]
- set_trials:
    ordered list of ten trial names
- trial_i_name / trial_i_condition / trial_i_time:
    trial metadata, with nr or is condition

Outputs
-------
participant_quality_audit.csv
trial_manifest_all.csv
trial_manifest_primary_candidates.csv
native_samples.parquet
samples_30hz.parquet
filtered_fixation_events.parquet
official_filter_code_excerpt.txt
stage2_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "pyarrow is required. Install it in the active venv with: "
        "python -m pip install pyarrow"
    ) from exc


TRIAL_FIELD_PATTERN = re.compile(r"^trial_(\d+)_name$")
SET_PATTERN = re.compile(r"(mturk_EN_v\d+(?:_[A-Za-z0-9_]+)?)", re.IGNORECASE)

RAW_COLUMNS = [
    "external_trial_id",
    "participant_id",
    "set_name",
    "trial_index",
    "trial_name",
    "task",
    "label_nr",
    "sample_index",
    "x_px",
    "y_px",
    "t_ms",
    "x_norm",
    "y_norm",
    "finite_xy",
    "finite_t",
    "in_screen",
    "valid_xy",
    "source_file",
]

RESAMPLED_COLUMNS = [
    "external_trial_id",
    "participant_id",
    "set_name",
    "trial_index",
    "trial_name",
    "task",
    "label_nr",
    "sample_index_30hz",
    "t_ms",
    "t_sec",
    "x_px",
    "y_px",
    "x_norm",
    "y_norm",
    "valid_xy",
    "interpolation_segment_id",
    "source_file",
]

FIXATION_COLUMNS = [
    "external_trial_id",
    "participant_id",
    "set_name",
    "trial_index",
    "trial_name",
    "task",
    "label_nr",
    "fixation_index",
    "x_px",
    "y_px",
    "t_ms",
    "duration_ms",
    "x_norm",
    "y_norm",
    "finite_xy",
    "finite_t",
    "finite_duration",
    "in_screen",
    "valid_fixation",
    "source_file",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def json_load_string(value: Any, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        result = value
    elif isinstance(value, str):
        result = json.loads(value)
    else:
        raise TypeError(
            f"{field_name} must be a JSON string or dict, got {type(value).__name__}."
        )
    if not isinstance(result, dict):
        raise TypeError(f"{field_name} decoded to {type(result).__name__}, expected dict.")
    return result


def read_participant_series(path: Path) -> pd.Series:
    try:
        result = pd.read_json(path, typ="series")
    except Exception:
        with path.open("r", encoding="utf-8") as handle:
            result = pd.Series(json.load(handle))
    if not isinstance(result, pd.Series):
        raise TypeError(f"{path} did not load as pandas Series.")
    return result


def scalar_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def make_external_trial_id(
    participant_id: str,
    set_name: str,
    trial_index: int,
    trial_name: str,
) -> str:
    return f"{participant_id}::{set_name}::{trial_index:02d}::{trial_name}"


def condition_to_label(condition: Any) -> tuple[str, int | None]:
    task = safe_string(condition).strip().upper()
    if task == "NR":
        return "NR", 1
    if task == "IS":
        return "IS", 0
    return task, None


def trial_indexes(series: pd.Series) -> list[int]:
    indexes: list[int] = []
    for name in map(str, series.index):
        match = TRIAL_FIELD_PATTERN.match(name)
        if match:
            indexes.append(int(match.group(1)))
    return sorted(set(indexes))


def coerce_points(
    payload: Any,
    expected_min_length: int,
) -> tuple[np.ndarray, int, int]:
    if not isinstance(payload, list):
        return np.empty((0, expected_min_length), dtype=float), 0, 1

    rows: list[list[float]] = []
    malformed = 0
    extra_values = 0
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) < expected_min_length:
            malformed += 1
            continue
        converted = [finite_number(item[index]) for index in range(expected_min_length)]
        rows.append(converted)
        extra_values += max(0, len(item) - expected_min_length)

    if not rows:
        return np.empty((0, expected_min_length), dtype=float), malformed, extra_values
    return np.asarray(rows, dtype=float), malformed, extra_values


def frame_native_samples(
    *,
    points: np.ndarray,
    external_trial_id: str,
    participant_id: str,
    set_name: str,
    trial_index: int,
    trial_name: str,
    task: str,
    label_nr: int,
    screen_x: float,
    screen_y: float,
    source_file: str,
) -> pd.DataFrame:
    if len(points) == 0:
        return pd.DataFrame(columns=RAW_COLUMNS)

    x = points[:, 0]
    y = points[:, 1]
    t = points[:, 2]
    finite_xy = np.isfinite(x) & np.isfinite(y)
    finite_t = np.isfinite(t)
    valid_screen = (
        math.isfinite(screen_x)
        and math.isfinite(screen_y)
        and screen_x > 0
        and screen_y > 0
    )
    if valid_screen:
        in_screen = finite_xy & (x >= 0) & (x <= screen_x) & (y >= 0) & (y <= screen_y)
        x_norm = x / screen_x
        y_norm = y / screen_y
    else:
        in_screen = np.zeros(len(points), dtype=bool)
        x_norm = np.full(len(points), np.nan)
        y_norm = np.full(len(points), np.nan)

    result = pd.DataFrame(
        {
            "external_trial_id": external_trial_id,
            "participant_id": participant_id,
            "set_name": set_name,
            "trial_index": int(trial_index),
            "trial_name": trial_name,
            "task": task,
            "label_nr": int(label_nr),
            "sample_index": np.arange(len(points), dtype=np.int64),
            "x_px": x,
            "y_px": y,
            "t_ms": t,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "finite_xy": finite_xy,
            "finite_t": finite_t,
            "in_screen": in_screen,
            "valid_xy": finite_xy & finite_t & in_screen,
            "source_file": source_file,
        }
    )
    return result[RAW_COLUMNS]


def frame_fixations(
    *,
    points: np.ndarray,
    external_trial_id: str,
    participant_id: str,
    set_name: str,
    trial_index: int,
    trial_name: str,
    task: str,
    label_nr: int,
    screen_x: float,
    screen_y: float,
    source_file: str,
) -> pd.DataFrame:
    if len(points) == 0:
        return pd.DataFrame(columns=FIXATION_COLUMNS)

    x = points[:, 0]
    y = points[:, 1]
    t = points[:, 2]
    duration = points[:, 3]
    finite_xy = np.isfinite(x) & np.isfinite(y)
    finite_t = np.isfinite(t)
    finite_duration = np.isfinite(duration) & (duration >= 0)
    valid_screen = (
        math.isfinite(screen_x)
        and math.isfinite(screen_y)
        and screen_x > 0
        and screen_y > 0
    )
    if valid_screen:
        in_screen = finite_xy & (x >= 0) & (x <= screen_x) & (y >= 0) & (y <= screen_y)
        x_norm = x / screen_x
        y_norm = y / screen_y
    else:
        in_screen = np.zeros(len(points), dtype=bool)
        x_norm = np.full(len(points), np.nan)
        y_norm = np.full(len(points), np.nan)

    result = pd.DataFrame(
        {
            "external_trial_id": external_trial_id,
            "participant_id": participant_id,
            "set_name": set_name,
            "trial_index": int(trial_index),
            "trial_name": trial_name,
            "task": task,
            "label_nr": int(label_nr),
            "fixation_index": np.arange(len(points), dtype=np.int64),
            "x_px": x,
            "y_px": y,
            "t_ms": t,
            "duration_ms": duration,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "finite_xy": finite_xy,
            "finite_t": finite_t,
            "finite_duration": finite_duration,
            "in_screen": in_screen,
            "valid_fixation": finite_xy & finite_t & finite_duration & in_screen,
            "source_file": source_file,
        }
    )
    return result[FIXATION_COLUMNS]


def gap_aware_resample(
    native: pd.DataFrame,
    *,
    hz: float,
    max_gap_ms: float,
) -> pd.DataFrame:
    """Linearly interpolate only inside contiguous valid segments.

    A segment is broken whenever the gap between consecutive valid native samples
    exceeds max_gap_ms. No interpolation is performed across invalid/out-of-screen
    samples or long temporal gaps.
    """
    if native.empty:
        return pd.DataFrame(columns=RESAMPLED_COLUMNS)

    valid = native[native["valid_xy"]].copy()
    valid = valid[
        np.isfinite(valid["t_ms"])
        & np.isfinite(valid["x_px"])
        & np.isfinite(valid["y_px"])
    ]
    if len(valid) < 2:
        return pd.DataFrame(columns=RESAMPLED_COLUMNS)

    valid = (
        valid.sort_values(["t_ms", "sample_index"])
        .groupby("t_ms", as_index=False)
        .agg(
            {
                "x_px": "mean",
                "y_px": "mean",
                "x_norm": "mean",
                "y_norm": "mean",
            }
        )
    )
    if len(valid) < 2:
        return pd.DataFrame(columns=RESAMPLED_COLUMNS)

    t = valid["t_ms"].to_numpy(dtype=float)
    gaps = np.diff(t)
    segment_ids = np.concatenate(
        [[0], np.cumsum((gaps > max_gap_ms) | (gaps <= 0))]
    ).astype(int)
    valid["segment_id"] = segment_ids

    step_ms = 1000.0 / float(hz)
    records: list[pd.DataFrame] = []

    meta = native.iloc[0]
    for segment_id, segment in valid.groupby("segment_id", sort=True):
        segment = segment.sort_values("t_ms")
        if len(segment) < 2:
            continue
        start = float(segment["t_ms"].iloc[0])
        end = float(segment["t_ms"].iloc[-1])
        if end <= start:
            continue

        first_grid = math.ceil(start / step_ms) * step_ms
        last_grid = math.floor(end / step_ms) * step_ms
        if last_grid < first_grid:
            continue
        grid = np.arange(first_grid, last_grid + step_ms * 0.25, step_ms)
        source_t = segment["t_ms"].to_numpy(dtype=float)

        frame = pd.DataFrame(
            {
                "external_trial_id": meta["external_trial_id"],
                "participant_id": meta["participant_id"],
                "set_name": meta["set_name"],
                "trial_index": int(meta["trial_index"]),
                "trial_name": meta["trial_name"],
                "task": meta["task"],
                "label_nr": int(meta["label_nr"]),
                "sample_index_30hz": np.arange(len(grid), dtype=np.int64),
                "t_ms": grid,
                "t_sec": grid / 1000.0,
                "x_px": np.interp(
                    grid, source_t, segment["x_px"].to_numpy(dtype=float)
                ),
                "y_px": np.interp(
                    grid, source_t, segment["y_px"].to_numpy(dtype=float)
                ),
                "x_norm": np.interp(
                    grid, source_t, segment["x_norm"].to_numpy(dtype=float)
                ),
                "y_norm": np.interp(
                    grid, source_t, segment["y_norm"].to_numpy(dtype=float)
                ),
                "valid_xy": True,
                "interpolation_segment_id": int(segment_id),
                "source_file": meta["source_file"],
            }
        )
        records.append(frame)

    if not records:
        return pd.DataFrame(columns=RESAMPLED_COLUMNS)

    result = pd.concat(records, ignore_index=True)
    result["sample_index_30hz"] = np.arange(len(result), dtype=np.int64)
    return result[RESAMPLED_COLUMNS]


def sample_metrics(
    points: np.ndarray,
    screen_x: float,
    screen_y: float,
    trial_time_ms: float,
) -> dict[str, Any]:
    if len(points) == 0:
        return {
            "n_raw_samples": 0,
            "n_finite_xy_samples": 0,
            "n_finite_t_samples": 0,
            "n_in_screen_samples": 0,
            "n_valid_xy_samples": 0,
            "valid_xy_rate": 0.0,
            "sample_t_min_ms": np.nan,
            "sample_t_max_ms": np.nan,
            "sample_duration_ms": np.nan,
            "median_dt_ms": np.nan,
            "native_rate_hz_from_median_dt": np.nan,
            "nonpositive_dt_count": 0,
            "duplicate_timestamp_count": 0,
            "sample_to_trial_time_ratio": np.nan,
        }

    x, y, t = points[:, 0], points[:, 1], points[:, 2]
    finite_xy = np.isfinite(x) & np.isfinite(y)
    finite_t = np.isfinite(t)
    screen_ok = (
        math.isfinite(screen_x)
        and math.isfinite(screen_y)
        and screen_x > 0
        and screen_y > 0
    )
    in_screen = (
        finite_xy & (x >= 0) & (x <= screen_x) & (y >= 0) & (y <= screen_y)
        if screen_ok
        else np.zeros(len(points), dtype=bool)
    )
    valid = finite_xy & finite_t & in_screen

    finite_times = np.sort(t[finite_t])
    if len(finite_times):
        t_min = float(finite_times[0])
        t_max = float(finite_times[-1])
        duration = t_max - t_min
    else:
        t_min = t_max = duration = np.nan

    unique_times = np.unique(finite_times)
    dt = np.diff(unique_times)
    positive_dt = dt[dt > 0]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else np.nan
    rate = 1000.0 / median_dt if math.isfinite(median_dt) and median_dt > 0 else np.nan

    ratio = (
        duration / trial_time_ms
        if math.isfinite(duration)
        and math.isfinite(trial_time_ms)
        and trial_time_ms > 0
        else np.nan
    )
    return {
        "n_raw_samples": int(len(points)),
        "n_finite_xy_samples": int(finite_xy.sum()),
        "n_finite_t_samples": int(finite_t.sum()),
        "n_in_screen_samples": int(in_screen.sum()),
        "n_valid_xy_samples": int(valid.sum()),
        "valid_xy_rate": float(valid.mean()) if len(points) else 0.0,
        "sample_t_min_ms": t_min,
        "sample_t_max_ms": t_max,
        "sample_duration_ms": duration,
        "median_dt_ms": median_dt,
        "native_rate_hz_from_median_dt": rate,
        "nonpositive_dt_count": int((np.diff(finite_times) <= 0).sum())
        if len(finite_times) >= 2 else 0,
        "duplicate_timestamp_count": int(len(finite_times) - len(unique_times)),
        "sample_to_trial_time_ratio": ratio,
    }


def fixation_metrics(points: np.ndarray) -> dict[str, Any]:
    if len(points) == 0:
        return {
            "n_filtered_fixations": 0,
            "n_valid_duration_fixations": 0,
            "filtered_fixation_duration_sum_ms": 0.0,
        }
    duration = points[:, 3]
    valid_duration = np.isfinite(duration) & (duration >= 0)
    return {
        "n_filtered_fixations": int(len(points)),
        "n_valid_duration_fixations": int(valid_duration.sum()),
        "filtered_fixation_duration_sum_ms": float(
            np.nansum(np.where(valid_duration, duration, np.nan))
        ),
    }


class ParquetAppender:
    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = path
        self.columns = list(columns)
        self.writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame = frame[self.columns].copy()
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        else:
            if table.schema != self.writer.schema:
                table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows_written += len(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def notebook_filter_excerpt(root: Path) -> str:
    notebook_path = root / "scripts" / "Loading Eye-Tracking Data + Word Boundaries.ipynb"
    if not notebook_path.is_file():
        return "Notebook not found.\n"
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Could not parse notebook: {type(exc).__name__}: {exc}\n"

    lines: list[str] = []
    filter_pattern = re.compile(
        r"fixation_error|target_error|experiment_reload|questions_duplicated|"
        r"approved_flag|sample_rate|filter|no_fixation|no_target",
        re.IGNORECASE,
    )
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", [])
        if isinstance(source, str):
            source = source.splitlines()
        source_lines = [str(line).rstrip("\r\n") for line in source]
        if any(filter_pattern.search(line) for line in source_lines):
            lines.append(f"### cell {cell_index}")
            lines.extend(source_lines)
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage2"),
    )
    parser.add_argument("--min-approved-flag", type=float, default=1.0)
    parser.add_argument("--resample-hz", type=float, default=30.0)
    parser.add_argument(
        "--max-interpolation-gap-ms",
        type=float,
        default=150.0,
    )
    parser.add_argument(
        "--minimum-valid-native-samples",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    if args.resample_hz <= 0:
        raise ValueError("--resample-hz must be positive.")
    if args.max_interpolation_gap_ms <= 0:
        raise ValueError("--max-interpolation-gap-ms must be positive.")
    if args.minimum_valid_native_samples < 2:
        raise ValueError("--minimum-valid-native-samples must be at least 2.")

    root = args.root.resolve()
    preprocessed = root / "scripts" / "pre_processed_data"
    if not preprocessed.is_dir():
        raise FileNotFoundError(preprocessed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_paths = sorted(preprocessed.glob("*_EN_*.json"))
    if not participant_paths:
        raise FileNotFoundError("No English participant JSON files found.")

    paths = {
        "participant_audit": output_dir / "participant_quality_audit.csv",
        "trial_all": output_dir / "trial_manifest_all.csv",
        "trial_primary": output_dir / "trial_manifest_primary_candidates.csv",
        "native_samples": output_dir / "native_samples.parquet",
        "samples_30hz": output_dir / "samples_30hz.parquet",
        "fixations": output_dir / "filtered_fixation_events.parquet",
        "filter_excerpt": output_dir / "official_filter_code_excerpt.txt",
        "manifest": output_dir / "stage2_manifest.json",
    }

    native_writer = ParquetAppender(paths["native_samples"], RAW_COLUMNS)
    resampled_writer = ParquetAppender(paths["samples_30hz"], RESAMPLED_COLUMNS)
    fixation_writer = ParquetAppender(paths["fixations"], FIXATION_COLUMNS)

    participant_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    participant_filter_counts: Counter[str] = Counter()
    trial_status_counts: Counter[str] = Counter()
    task_counts_all: Counter[str] = Counter()
    task_counts_primary: Counter[str] = Counter()

    try:
        for participant_path in participant_paths:
            source_file = participant_path.relative_to(root).as_posix()
            participant_id_from_name = participant_path.stem.split("_mturk_EN_", 1)[0]

            try:
                series = read_participant_series(participant_path)
                participant_parse_error = ""
            except Exception as exc:
                participant_rows.append(
                    {
                        "participant_id": participant_id_from_name,
                        "source_file": source_file,
                        "read_status": "error",
                        "read_error": f"{type(exc).__name__}: {exc}",
                        "primary_participant_filter_pass": False,
                        "participant_exclusion_reasons": "participant_json_read_error",
                    }
                )
                continue

            participant_id = safe_string(series.get("worker_id")) or participant_id_from_name
            set_name = safe_string(series.get("set_name"))
            screen_x = finite_number(series.get("screen_x"))
            screen_y = finite_number(series.get("screen_y"))
            approved_flag = finite_number(series.get("approved_flag"))
            fixation_error = scalar_bool(series.get("fixation_error"), default=None)
            target_error = scalar_bool(series.get("target_error"), default=None)
            experiment_reload = scalar_bool(series.get("experiment_reload"), default=None)
            questions_duplicated = scalar_bool(
                series.get("questions_duplicated"), default=None
            )
            sample_rate_reported = finite_number(series.get("webgazer_sample_rate"))
            set_trials = series.get("set_trials")
            if not isinstance(set_trials, list):
                set_trials = []

            raw_parse_error = ""
            filtered_parse_error = ""
            try:
                raw_by_trial = json_load_string(
                    series.get("webgazer_raw_data"),
                    "webgazer_raw_data",
                )
            except Exception as exc:
                raw_by_trial = {}
                raw_parse_error = f"{type(exc).__name__}: {exc}"

            try:
                filtered_by_trial = json_load_string(
                    series.get("webgazer_filtered_data"),
                    "webgazer_filtered_data",
                )
            except Exception as exc:
                filtered_by_trial = {}
                filtered_parse_error = f"{type(exc).__name__}: {exc}"

            screen_valid = (
                math.isfinite(screen_x)
                and math.isfinite(screen_y)
                and screen_x > 0
                and screen_y > 0
            )
            approved_pass = (
                math.isfinite(approved_flag)
                and approved_flag >= args.min_approved_flag
            )
            filter_no_fixation_error = fixation_error is False
            filter_no_target_error = target_error is False
            filter_no_reload = experiment_reload is False
            filter_no_duplicate_questions = questions_duplicated is False
            filter_raw_parsed = not raw_parse_error and bool(raw_by_trial)
            filter_filtered_parsed = (
                not filtered_parse_error and bool(filtered_by_trial)
            )

            participant_filter_flags = {
                "no_fixation_error": filter_no_fixation_error,
                "no_target_error": filter_no_target_error,
                "no_experiment_reload": filter_no_reload,
                "no_duplicated_questions": filter_no_duplicate_questions,
                "approved_flag_threshold": approved_pass,
                "valid_screen_resolution": screen_valid,
                "raw_gaze_dictionary_parsed": filter_raw_parsed,
                "filtered_gaze_dictionary_parsed": filter_filtered_parsed,
            }
            participant_exclusion_reasons = [
                name for name, passed in participant_filter_flags.items() if not passed
            ]
            participant_pass = not participant_exclusion_reasons

            for name, passed in participant_filter_flags.items():
                participant_filter_counts[
                    f"{name}__{'pass' if passed else 'fail'}"
                ] += 1
            participant_filter_counts[
                "primary_participant_filter_pass"
                if participant_pass
                else "primary_participant_filter_fail"
            ] += 1

            indexes = trial_indexes(series)
            participant_rows.append(
                {
                    "participant_id": participant_id,
                    "participant_id_filename": participant_id_from_name,
                    "set_name": set_name,
                    "source_file": source_file,
                    "read_status": "ok",
                    "read_error": participant_parse_error,
                    "participant_type": safe_string(series.get("participant_type")),
                    "platform_type": safe_string(series.get("platform_type")),
                    "worker_age": series.get("worker_age"),
                    "worker_lang": safe_string(series.get("worker_lang")),
                    "worker_fluency": series.get("worker_fluency"),
                    "screen_x": screen_x,
                    "screen_y": screen_y,
                    "webgazer_sample_rate_reported_hz": sample_rate_reported,
                    "approved_flag": approved_flag,
                    "total_correct_answers": series.get("total_correct_answers"),
                    "fixation_error": fixation_error,
                    "target_error": target_error,
                    "experiment_reload": experiment_reload,
                    "questions_duplicated": questions_duplicated,
                    "n_set_trials": len(set_trials),
                    "n_trial_indexes_detected": len(indexes),
                    "n_raw_trial_keys": len(raw_by_trial),
                    "n_filtered_trial_keys": len(filtered_by_trial),
                    "raw_parse_error": raw_parse_error,
                    "filtered_parse_error": filtered_parse_error,
                    "filter_no_fixation_error": filter_no_fixation_error,
                    "filter_no_target_error": filter_no_target_error,
                    "filter_no_experiment_reload": filter_no_reload,
                    "filter_no_duplicated_questions": filter_no_duplicate_questions,
                    "filter_approved_flag_threshold": approved_pass,
                    "filter_valid_screen_resolution": screen_valid,
                    "filter_raw_gaze_dictionary_parsed": filter_raw_parsed,
                    "filter_filtered_gaze_dictionary_parsed": filter_filtered_parsed,
                    "primary_participant_filter_pass": participant_pass,
                    "participant_exclusion_reasons": "|".join(
                        participant_exclusion_reasons
                    ),
                }
            )

            for trial_index in indexes:
                trial_name = safe_string(series.get(f"trial_{trial_index}_name"))
                condition_raw = series.get(f"trial_{trial_index}_condition")
                task, label_nr = condition_to_label(condition_raw)
                trial_time_ms = finite_number(series.get(f"trial_{trial_index}_time"))
                external_trial_id = make_external_trial_id(
                    participant_id,
                    set_name,
                    trial_index,
                    trial_name,
                )

                expected_from_set_trials = (
                    safe_string(set_trials[trial_index])
                    if trial_index < len(set_trials)
                    else ""
                )
                set_trial_match = bool(
                    expected_from_set_trials
                    and trial_name
                    and expected_from_set_trials == trial_name
                )
                expected_condition_by_position = "NR" if trial_index < 5 else "IS"
                position_condition_match = task == expected_condition_by_position
                valid_label = label_nr in {0, 1}
                raw_key_exists = trial_name in raw_by_trial
                filtered_key_exists = trial_name in filtered_by_trial

                raw_points, raw_malformed, raw_extra = coerce_points(
                    raw_by_trial.get(trial_name, []),
                    expected_min_length=3,
                )
                fixation_points, fixation_malformed, fixation_extra = coerce_points(
                    filtered_by_trial.get(trial_name, []),
                    expected_min_length=4,
                )

                metrics = sample_metrics(
                    raw_points,
                    screen_x,
                    screen_y,
                    trial_time_ms,
                )
                fix_metrics = fixation_metrics(fixation_points)

                native_frame = pd.DataFrame(columns=RAW_COLUMNS)
                fixation_frame = pd.DataFrame(columns=FIXATION_COLUMNS)
                resampled_frame = pd.DataFrame(columns=RESAMPLED_COLUMNS)

                if valid_label:
                    native_frame = frame_native_samples(
                        points=raw_points,
                        external_trial_id=external_trial_id,
                        participant_id=participant_id,
                        set_name=set_name,
                        trial_index=trial_index,
                        trial_name=trial_name,
                        task=task,
                        label_nr=int(label_nr),
                        screen_x=screen_x,
                        screen_y=screen_y,
                        source_file=source_file,
                    )
                    fixation_frame = frame_fixations(
                        points=fixation_points,
                        external_trial_id=external_trial_id,
                        participant_id=participant_id,
                        set_name=set_name,
                        trial_index=trial_index,
                        trial_name=trial_name,
                        task=task,
                        label_nr=int(label_nr),
                        screen_x=screen_x,
                        screen_y=screen_y,
                        source_file=source_file,
                    )
                    resampled_frame = gap_aware_resample(
                        native_frame,
                        hz=args.resample_hz,
                        max_gap_ms=args.max_interpolation_gap_ms,
                    )

                native_writer.append(native_frame)
                fixation_writer.append(fixation_frame)
                resampled_writer.append(resampled_frame)

                structural_flags = {
                    "valid_task_label": valid_label,
                    "trial_name_nonempty": bool(trial_name),
                    "set_trials_name_match": set_trial_match,
                    "condition_position_match": position_condition_match,
                    "raw_trial_key_exists": raw_key_exists,
                    "filtered_trial_key_exists": filtered_key_exists,
                    "trial_time_positive": (
                        math.isfinite(trial_time_ms) and trial_time_ms > 0
                    ),
                    "minimum_valid_native_samples": (
                        metrics["n_valid_xy_samples"]
                        >= args.minimum_valid_native_samples
                    ),
                    "resampled_samples_present": len(resampled_frame) >= 2,
                }
                trial_exclusion_reasons = [
                    name for name, passed in structural_flags.items() if not passed
                ]
                structural_pass = not trial_exclusion_reasons
                primary_candidate = participant_pass and structural_pass

                status = (
                    "primary_candidate"
                    if primary_candidate
                    else (
                        "participant_filter_fail"
                        if not participant_pass
                        else "trial_structure_fail"
                    )
                )
                trial_status_counts[status] += 1
                if task:
                    task_counts_all[task] += 1
                    if primary_candidate:
                        task_counts_primary[task] += 1

                trial_rows.append(
                    {
                        "external_trial_id": external_trial_id,
                        "participant_id": participant_id,
                        "set_name": set_name,
                        "source_file": source_file,
                        "trial_index": trial_index,
                        "trial_name": trial_name,
                        "set_trials_expected_name": expected_from_set_trials,
                        "task": task,
                        "label_nr": label_nr,
                        "fixed_label_mapping": (
                            "NR=1;IS=0"
                        ),
                        "trial_time_ms": trial_time_ms,
                        "question_name": safe_string(
                            series.get(f"question_{trial_index}_name")
                        ),
                        "question_correct_flag": series.get(
                            f"question_{trial_index}_correct_flag"
                        ),
                        "pre_question_name": safe_string(
                            series.get(f"pre_question_{trial_index}_name")
                        ),
                        "screen_x": screen_x,
                        "screen_y": screen_y,
                        "webgazer_sample_rate_reported_hz": sample_rate_reported,
                        "participant_filter_pass": participant_pass,
                        "participant_exclusion_reasons": "|".join(
                            participant_exclusion_reasons
                        ),
                        "set_trials_name_match": set_trial_match,
                        "condition_position_match": position_condition_match,
                        "raw_trial_key_exists": raw_key_exists,
                        "filtered_trial_key_exists": filtered_key_exists,
                        "raw_malformed_record_count": raw_malformed,
                        "raw_extra_value_count": raw_extra,
                        "fixation_malformed_record_count": fixation_malformed,
                        "fixation_extra_value_count": fixation_extra,
                        **metrics,
                        **fix_metrics,
                        "n_resampled_30hz_samples": int(len(resampled_frame)),
                        "n_resampled_segments": int(
                            resampled_frame["interpolation_segment_id"].nunique()
                        )
                        if not resampled_frame.empty
                        else 0,
                        "trial_structural_pass": structural_pass,
                        "trial_exclusion_reasons": "|".join(
                            trial_exclusion_reasons
                        ),
                        "primary_external_candidate": primary_candidate,
                        "candidate_status": status,
                    }
                )
    finally:
        native_writer.close()
        resampled_writer.close()
        fixation_writer.close()

    participants = pd.DataFrame(participant_rows)
    trials = pd.DataFrame(trial_rows)
    participants.to_csv(
        paths["participant_audit"],
        index=False,
        encoding="utf-8-sig",
    )
    trials.to_csv(
        paths["trial_all"],
        index=False,
        encoding="utf-8-sig",
    )
    primary_trials = trials[
        trials["primary_external_candidate"].fillna(False)
    ].copy()
    primary_trials.to_csv(
        paths["trial_primary"],
        index=False,
        encoding="utf-8-sig",
    )

    filter_excerpt = notebook_filter_excerpt(root)
    paths["filter_excerpt"].write_text(
        filter_excerpt,
        encoding="utf-8",
    )

    primary_participants = set(primary_trials["participant_id"].astype(str))
    manifest = {
        "stage": "webqamgaze_external_validation_stage2_formal_extraction",
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "repository_root": str(root),
        "repository_git_commit": git_commit(root),
        "language": "EN",
        "fixed_external_label_mapping": {
            "IS_information_seeking": 0,
            "NR_normal_reading": 1,
        },
        "participant_filter": {
            "fixation_error": False,
            "target_error": False,
            "experiment_reload": False,
            "questions_duplicated": False,
            "minimum_approved_flag": float(args.min_approved_flag),
            "valid_positive_screen_resolution": True,
            "raw_gaze_dictionary_parsed": True,
            "filtered_gaze_dictionary_parsed": True,
        },
        "trial_structural_filter": {
            "valid_task_label": True,
            "trial_name_nonempty": True,
            "set_trials_name_match": True,
            "condition_position_match": True,
            "raw_trial_key_exists": True,
            "filtered_trial_key_exists": True,
            "trial_time_positive": True,
            "minimum_valid_native_samples": int(
                args.minimum_valid_native_samples
            ),
            "resampled_samples_present": True,
        },
        "resampling": {
            "target_hz": float(args.resample_hz),
            "method": (
                "linear interpolation within contiguous valid in-screen segments"
            ),
            "maximum_interpolation_gap_ms": float(
                args.max_interpolation_gap_ms
            ),
            "no_interpolation_across_invalid_or_long_gap_segments": True,
        },
        "counts": {
            "participant_files": len(participant_paths),
            "participant_rows": len(participants),
            "primary_participants": len(primary_participants),
            "all_trials": len(trials),
            "primary_candidate_trials": len(primary_trials),
            "native_sample_rows": native_writer.rows_written,
            "resampled_sample_rows": resampled_writer.rows_written,
            "filtered_fixation_rows": fixation_writer.rows_written,
            "participant_filter_counts": dict(
                sorted(participant_filter_counts.items())
            ),
            "trial_status_counts": dict(sorted(trial_status_counts.items())),
            "task_counts_all": dict(sorted(task_counts_all.items())),
            "task_counts_primary": dict(sorted(task_counts_primary.items())),
        },
        "constraints": [
            "No WebQAmGaze data are used to fit or refit a classifier.",
            "No external labels are used for feature selection, preprocessing estimation, C selection, or threshold calibration.",
            "All participants and trials remain available in audit outputs.",
            "Primary-candidate filtering uses only predeclared quality and structural criteria.",
            "Frozen ZuCo model artifacts remain unchanged.",
            "This stage does not yet construct Soft-AOI or official nine-dimensional feature vectors.",
        ],
        "outputs": {
            key: str(value) for key, value in paths.items() if key != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE EXTERNAL VALIDATION STAGE 2 ===")
    print(f"Root: {root}")
    print(f"Git commit: {manifest['repository_git_commit']}")
    print(f"Participant files: {len(participant_paths)}")
    print(
        "Primary-filter participants with eligible trials: "
        f"{len(primary_participants)}"
    )
    print(f"All trials: {len(trials)}")
    print(f"Primary candidate trials: {len(primary_trials)}")
    print("Task counts, all trials:")
    for task, count in sorted(task_counts_all.items()):
        print(f"  {task}: {count}")
    print("Task counts, primary candidates:")
    for task, count in sorted(task_counts_primary.items()):
        print(f"  {task}: {count}")
    print("Trial status counts:")
    for status, count in sorted(trial_status_counts.items()):
        print(f"  {status}: {count}")
    print(f"Native sample rows: {native_writer.rows_written}")
    print(f"30 Hz sample rows: {resampled_writer.rows_written}")
    print(f"Filtered fixation rows: {fixation_writer.rows_written}")
    print("Participant filter failures:")
    failure_names = [
        key for key in sorted(participant_filter_counts)
        if key.endswith("__fail")
    ]
    for key in failure_names:
        print(f"  {key.removesuffix('__fail')}: {participant_filter_counts[key]}")
    print("Written:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
