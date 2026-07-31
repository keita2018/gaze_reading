#!/usr/bin/env python
"""Export sentence-level continuous gaze samples from a ZuCo ET session.

The script pairs event code 10 (candidate sentence start) with the next
event code 11 (candidate sentence end), slices the 500 Hz continuous sample
array, and writes:

1. A sample-level Parquet file.
2. A trial-level CSV summary.
3. A CSV containing all non-10/11 events for later interpretation.

No fixation/saccade labels are used in the exported sample input.

Example:
    python src/data/export_zuco_session_trials.py \
        data/raw/zuco2/YAC/session_et/raw/YAC_NR1_ET.mat \
        --subject YAC --task NR --block 1 \
        --output-dir data/interim/zuco2/YAC
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


EXPECTED_HEADERS = [
    "TIME",
    "L_GAZE_X",
    "L_GAZE_Y",
    "L_AREA",
    "targetpos_x",
    "targetpos_y",
    "targetdistance",
]


def to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(x) for x in value.ravel())
        if value.dtype == object and value.size == 1:
            return to_text(value.item())
        return " ".join(to_text(x) for x in value.ravel())
    return str(value)


def get_headers(mat: dict[str, Any]) -> list[str]:
    return [
        to_text(value).strip()
        for value in np.asarray(mat["colheader"], dtype=object).ravel()
    ]


def pair_trials(event: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return (start_event_index, end_event_index, start_ms, end_ms)."""
    pairs: list[tuple[int, int, int, int]] = []
    pending: tuple[int, int] | None = None

    for event_index, row in enumerate(np.asarray(event)):
        timestamp = int(row[0])
        code = int(row[1])

        if code == 10:
            if pending is not None:
                raise ValueError(
                    f"Unclosed code 10 at event {pending[0]} before "
                    f"another code 10 at event {event_index}."
                )
            pending = (event_index, timestamp)
        elif code == 11:
            if pending is None:
                raise ValueError(
                    f"Code 11 at event {event_index} has no preceding code 10."
                )
            start_index, start_ms = pending
            if timestamp <= start_ms:
                raise ValueError(
                    f"Invalid interval: {start_ms} -> {timestamp}."
                )
            pairs.append((start_index, event_index, start_ms, timestamp))
            pending = None

    if pending is not None:
        raise ValueError(f"Final code 10 at event {pending[0]} is unclosed.")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--task", choices=["NR", "TSR"], required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        default="data/interim/zuco2",
    )
    args = parser.parse_args()

    path = Path(args.mat_file)
    if not path.is_file():
        raise FileNotFoundError(path)

    mat = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
        chars_as_strings=True,
    )

    headers = get_headers(mat)
    if headers != EXPECTED_HEADERS:
        raise ValueError(
            "Unexpected column headers.\n"
            f"Expected: {EXPECTED_HEADERS}\n"
            f"Actual:   {headers}"
        )

    data = np.asarray(mat["data"], dtype=float)
    event = np.asarray(mat["event"])
    if data.ndim != 2 or data.shape[1] != len(headers):
        raise ValueError(f"Unexpected data shape: {data.shape}")

    time = data[:, 0]
    if np.any(np.diff(time) < 0):
        raise ValueError("TIME is not monotonically nondecreasing.")

    pairs = pair_trials(event)
    subject = args.subject.upper()
    task = args.task.upper()
    block = int(args.block)

    sample_frames: list[pd.DataFrame] = []
    trial_rows: list[dict[str, Any]] = []

    for trial_index, (
        start_event_index,
        end_event_index,
        start_ms,
        end_ms,
    ) in enumerate(pairs):
        left = int(np.searchsorted(time, start_ms, side="left"))
        right = int(np.searchsorted(time, end_ms, side="right"))
        segment = data[left:right]

        if segment.size == 0:
            raise ValueError(
                f"No samples found for trial {trial_index}: "
                f"{start_ms} -> {end_ms}"
            )

        frame = pd.DataFrame(segment, columns=headers)
        frame = frame.rename(
            columns={
                "TIME": "timestamp_ms_abs",
                "L_GAZE_X": "x_px",
                "L_GAZE_Y": "y_px",
                "L_AREA": "pupil_area",
            }
        )
        frame.insert(0, "subject_id", subject)
        frame.insert(1, "task", task)
        frame.insert(2, "block_id", block)
        frame.insert(3, "trial_index_in_block", trial_index)
        frame.insert(
            4,
            "trial_id",
            f"{subject}_{task}{block}_T{trial_index:03d}",
        )
        frame.insert(5, "sample_index_in_trial", np.arange(len(frame)))
        frame.insert(
            7,
            "timestamp_ms_rel",
            frame["timestamp_ms_abs"] - start_ms,
        )

        finite_xy = (
            np.isfinite(frame["x_px"].to_numpy())
            & np.isfinite(frame["y_px"].to_numpy())
        )
        positive_xy = (
            finite_xy
            & (frame["x_px"].to_numpy() > 0)
            & (frame["y_px"].to_numpy() > 0)
        )
        valid_pupil = (
            np.isfinite(frame["pupil_area"].to_numpy())
            & (frame["pupil_area"].to_numpy() > 0)
        )

        frame["finite_xy"] = finite_xy
        frame["positive_xy"] = positive_xy
        frame["valid_pupil"] = valid_pupil
        frame["valid_basic"] = positive_xy & valid_pupil

        dt = np.diff(frame["timestamp_ms_abs"].to_numpy())
        positive_dt = dt[dt > 0]
        median_dt = (
            float(np.median(positive_dt))
            if positive_dt.size
            else math.nan
        )

        duration_ms = end_ms - start_ms
        expected_samples = duration_ms / 2.0

        trial_rows.append(
            {
                "subject_id": subject,
                "task": task,
                "block_id": block,
                "trial_index_in_block": trial_index,
                "trial_id": f"{subject}_{task}{block}_T{trial_index:03d}",
                "start_event_index": start_event_index,
                "end_event_index": end_event_index,
                "start_timestamp_ms": start_ms,
                "end_timestamp_ms": end_ms,
                "duration_ms": duration_ms,
                "duration_sec": duration_ms / 1000.0,
                "sample_start_index_session": left,
                "sample_end_index_session_exclusive": right,
                "n_samples": len(frame),
                "expected_samples_at_500hz": expected_samples,
                "sample_coverage_ratio": (
                    len(frame) / expected_samples
                    if expected_samples > 0 else math.nan
                ),
                "median_dt_ms": median_dt,
                "estimated_hz": (
                    1000.0 / median_dt
                    if math.isfinite(median_dt) and median_dt > 0
                    else math.nan
                ),
                "finite_xy_rate": float(np.mean(finite_xy)),
                "positive_xy_rate": float(np.mean(positive_xy)),
                "valid_pupil_rate": float(np.mean(valid_pupil)),
                "valid_basic_rate": float(np.mean(positive_xy & valid_pupil)),
                "x_min": float(np.nanmin(frame["x_px"])),
                "x_max": float(np.nanmax(frame["x_px"])),
                "y_min": float(np.nanmin(frame["y_px"])),
                "y_max": float(np.nanmax(frame["y_px"])),
            }
        )
        sample_frames.append(frame)

    samples = pd.concat(sample_frames, ignore_index=True)
    trials = pd.DataFrame(trial_rows)

    non_trial_events = pd.DataFrame(event, columns=["timestamp_ms", "event_code"])
    non_trial_events["event_code"] = non_trial_events["event_code"].astype(int)
    non_trial_events = non_trial_events[
        ~non_trial_events["event_code"].isin([10, 11])
    ].copy()
    non_trial_events.insert(0, "subject_id", subject)
    non_trial_events.insert(1, "task", task)
    non_trial_events.insert(2, "block_id", block)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{task}{block}"

    samples_path = output_dir / f"{stem}_samples.parquet"
    trials_path = output_dir / f"{stem}_trials.csv"
    events_path = output_dir / f"{stem}_non_trial_events.csv"

    try:
        samples.to_parquet(samples_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Parquet output requires pyarrow. Install it with:\n"
            "python -m pip install pyarrow"
        ) from exc

    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    non_trial_events.to_csv(events_path, index=False, encoding="utf-8-sig")

    print("=== EXPORT SUMMARY ===")
    print(f"source: {path}")
    print(f"trials: {len(trials)}")
    print(f"samples: {len(samples)}")
    print(f"median duration sec: {trials['duration_sec'].median():.3f}")
    print(f"median estimated Hz: {trials['estimated_hz'].median():.3f}")
    print(f"median valid_basic_rate: {trials['valid_basic_rate'].median():.6f}")
    print(f"sample file: {samples_path}")
    print(f"trial file: {trials_path}")
    print(f"other events: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
