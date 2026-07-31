#!/usr/bin/env python
"""Export all sentence-reading intervals from a ZuCo ET session.

ZuCo 2.0 uses two reading marker patterns:

- TR10 -> TR11: ordinary sentence-reading trial
- TR12 -> TR13: sentence-reading trial followed by an additional response/
  control phase ending at TR15

Only the reading intervals 10->11 and 12->13 are exported. The interval
13->15 is not included in the gaze input.

Outputs:
1. <SUBJECT>_<TASK><BLOCK>_samples.parquet
2. <SUBJECT>_<TASK><BLOCK>_trials.csv
3. <SUBJECT>_<TASK><BLOCK>_non_reading_events.csv

No fixation, saccade, or blink labels are used in the sample input.
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

READING_PAIRS = {
    10: 11,
    12: 13,
}

PAIR_NAMES = {
    (10, 11): "TR10_TR11",
    (12, 13): "TR12_TR13",
}


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


def pair_reading_trials(
    event: np.ndarray,
) -> list[dict[str, int | str]]:
    """Pair 10->11 and 12->13 intervals in chronological order."""
    event = np.asarray(event)
    open_trial: dict[str, int] | None = None
    pairs: list[dict[str, int | str]] = []

    for event_index, row in enumerate(event):
        timestamp_ms = int(row[0])
        event_code = int(row[1])

        if event_code in READING_PAIRS:
            if open_trial is not None:
                raise ValueError(
                    "A reading trial started before the previous reading "
                    f"trial was closed: open={open_trial}, "
                    f"new_event_index={event_index}, code={event_code}."
                )
            open_trial = {
                "start_event_index": event_index,
                "start_timestamp_ms": timestamp_ms,
                "start_event_code": event_code,
                "expected_end_code": READING_PAIRS[event_code],
            }
            continue

        if open_trial is None:
            continue

        expected_end_code = int(open_trial["expected_end_code"])
        if event_code != expected_end_code:
            continue

        start_ms = int(open_trial["start_timestamp_ms"])
        if timestamp_ms <= start_ms:
            raise ValueError(
                f"Non-positive reading interval: {start_ms}->{timestamp_ms}"
            )

        start_code = int(open_trial["start_event_code"])
        pairs.append(
            {
                "start_event_index": int(open_trial["start_event_index"]),
                "end_event_index": event_index,
                "start_timestamp_ms": start_ms,
                "end_timestamp_ms": timestamp_ms,
                "start_event_code": start_code,
                "end_event_code": event_code,
                "marker_pair": PAIR_NAMES[(start_code, event_code)],
            }
        )
        open_trial = None

    if open_trial is not None:
        raise ValueError(f"Unclosed reading trial: {open_trial}")

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

    session_time = data[:, 0]
    if np.any(np.diff(session_time) < 0):
        raise ValueError("TIME is not monotonically nondecreasing.")

    pairs = pair_reading_trials(event)
    subject = args.subject.upper()
    task = args.task.upper()
    block = int(args.block)

    sample_frames: list[pd.DataFrame] = []
    trial_rows: list[dict[str, Any]] = []

    for trial_index, pair in enumerate(pairs):
        start_ms = int(pair["start_timestamp_ms"])
        end_ms = int(pair["end_timestamp_ms"])

        left = int(np.searchsorted(session_time, start_ms, side="left"))
        right = int(np.searchsorted(session_time, end_ms, side="right"))
        segment = data[left:right]

        if segment.size == 0:
            raise ValueError(
                f"No samples for trial {trial_index}: {start_ms}->{end_ms}"
            )

        trial_id = f"{subject}_{task}{block}_T{trial_index:03d}"

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
        frame.insert(4, "trial_id", trial_id)
        frame.insert(5, "marker_pair", pair["marker_pair"])
        frame.insert(6, "start_event_code", int(pair["start_event_code"]))
        frame.insert(7, "end_event_code", int(pair["end_event_code"]))
        frame.insert(8, "sample_index_in_trial", np.arange(len(frame)))
        frame.insert(
            10,
            "timestamp_ms_rel",
            frame["timestamp_ms_abs"] - start_ms,
        )

        x = frame["x_px"].to_numpy()
        y = frame["y_px"].to_numpy()
        pupil = frame["pupil_area"].to_numpy()

        finite_xy = np.isfinite(x) & np.isfinite(y)
        positive_xy = finite_xy & (x > 0) & (y > 0)
        valid_pupil = np.isfinite(pupil) & (pupil > 0)
        valid_basic = positive_xy & valid_pupil

        frame["finite_xy"] = finite_xy
        frame["positive_xy"] = positive_xy
        frame["valid_pupil"] = valid_pupil
        frame["valid_basic"] = valid_basic

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
                "trial_id": trial_id,
                "marker_pair": pair["marker_pair"],
                "start_event_code": int(pair["start_event_code"]),
                "end_event_code": int(pair["end_event_code"]),
                "start_event_index": int(pair["start_event_index"]),
                "end_event_index": int(pair["end_event_index"]),
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
                    if expected_samples > 0
                    else math.nan
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
                "valid_basic_rate": float(np.mean(valid_basic)),
                "x_min": float(np.nanmin(x)),
                "x_max": float(np.nanmax(x)),
                "y_min": float(np.nanmin(y)),
                "y_max": float(np.nanmax(y)),
            }
        )

        sample_frames.append(frame)

    samples = pd.concat(sample_frames, ignore_index=True)
    trials = pd.DataFrame(trial_rows)

    reading_codes = set(READING_PAIRS) | set(READING_PAIRS.values())
    non_reading_events = pd.DataFrame(
        event,
        columns=["timestamp_ms", "event_code"],
    )
    non_reading_events["event_code"] = (
        non_reading_events["event_code"].astype(int)
    )
    non_reading_events = non_reading_events[
        ~non_reading_events["event_code"].isin(reading_codes)
    ].copy()
    non_reading_events.insert(0, "subject_id", subject)
    non_reading_events.insert(1, "task", task)
    non_reading_events.insert(2, "block_id", block)

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
            "Parquet output requires pyarrow:\n"
            "python -m pip install pyarrow"
        ) from exc

    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    non_reading_events.to_csv(
        events_path,
        index=False,
        encoding="utf-8-sig",
    )

    pair_counts = trials["marker_pair"].value_counts().sort_index()

    print("=== EXPORT SUMMARY ===")
    print(f"source: {path}")
    print(f"trials: {len(trials)}")
    for marker_pair, count in pair_counts.items():
        print(f"{marker_pair}: {count}")
    print(f"samples: {len(samples)}")
    print(f"median duration sec: {trials['duration_sec'].median():.3f}")
    print(f"median estimated Hz: {trials['estimated_hz'].median():.3f}")
    print(
        "median valid_basic_rate: "
        f"{trials['valid_basic_rate'].median():.6f}"
    )
    print(f"sample file: {samples_path}")
    print(f"trial file: {trials_path}")
    print(f"other events: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
