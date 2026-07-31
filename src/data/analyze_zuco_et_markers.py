#!/usr/bin/env python
"""Analyze one ZuCo session-level ET file and compare Raw vs corrected.

Outputs:
- event_code_summary.csv
- parsed_messages.csv
- event_message_context.csv
- candidate_trials_code10_11.csv
- raw_corrected_data_comparison.csv
- raw_corrected_object_comparison.txt

The script assumes classic MATLAB MAT files such as:
    YAC_NR1_ET.mat
    YAC_NR1_corrected_ET.mat
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat


MSG_RE = re.compile(r"^\s*MSG[\t ]+(\d+)[\t ]*(.*)$", re.IGNORECASE)


def load_mat(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
        chars_as_strings=True,
    )


def to_text(value: Any) -> str:
    """Convert a MATLAB-loaded message value to a normal Python string."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.str_):
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


def parse_messages(values: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flat = np.asarray(values, dtype=object).ravel()

    current_timestamp: int | None = None
    for index, value in enumerate(flat):
        raw = to_text(value).replace("\r", "").rstrip("\n")
        match = MSG_RE.match(raw)

        if match:
            timestamp = int(match.group(1))
            text = match.group(2).strip()
            current_timestamp = timestamp
            is_continuation = 0
        else:
            # Calibration exports sometimes put continuation text on a line
            # without the MSG prefix. Retain it with the preceding timestamp.
            timestamp = current_timestamp
            text = raw.strip()
            is_continuation = 1

        rows.append(
            {
                "message_index": index,
                "timestamp_ms": timestamp if timestamp is not None else "",
                "text": text,
                "raw": raw,
                "is_continuation": is_continuation,
                "is_calibration": int(
                    "!CAL" in text.upper()
                    or "CALIB" in text.upper()
                    or "VALIDATION" in text.upper()
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_event_codes(event: np.ndarray) -> list[dict[str, Any]]:
    event = np.asarray(event)
    codes, counts = np.unique(event[:, 1].astype(int), return_counts=True)
    return [
        {"event_code": int(code), "count": int(count)}
        for code, count in zip(codes, counts)
    ]


def nearest_messages(
    event: np.ndarray,
    messages: list[dict[str, Any]],
    window_ms: int,
) -> list[dict[str, Any]]:
    timestamped = [
        row for row in messages
        if row["timestamp_ms"] != ""
    ]
    rows: list[dict[str, Any]] = []

    for event_index, (timestamp, code) in enumerate(np.asarray(event)):
        timestamp = int(timestamp)
        code = int(code)

        candidates = []
        for message in timestamped:
            delta = int(message["timestamp_ms"]) - timestamp
            if abs(delta) <= window_ms:
                candidates.append((abs(delta), delta, message))

        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates:
            rows.append(
                {
                    "event_index": event_index,
                    "event_timestamp_ms": timestamp,
                    "event_code": code,
                    "message_timestamp_ms": "",
                    "delta_ms": "",
                    "message_text": "",
                    "message_index": "",
                }
            )
            continue

        for _, delta, message in candidates[:10]:
            rows.append(
                {
                    "event_index": event_index,
                    "event_timestamp_ms": timestamp,
                    "event_code": code,
                    "message_timestamp_ms": message["timestamp_ms"],
                    "delta_ms": delta,
                    "message_text": message["text"],
                    "message_index": message["message_index"],
                }
            )
    return rows


def pair_code_10_11(
    event: np.ndarray,
    data: np.ndarray,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pair each code 10 with the next code 11 before another code 10."""
    event = np.asarray(event)
    data = np.asarray(data)

    time = data[:, 0].astype(float)
    x = data[:, 1].astype(float)
    y = data[:, 2].astype(float)
    pupil = data[:, 3].astype(float)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    pending_start: tuple[int, int] | None = None

    for event_index, (timestamp_value, code_value) in enumerate(event):
        timestamp = int(timestamp_value)
        code = int(code_value)

        if code == 10:
            if pending_start is not None:
                warnings.append(
                    f"Code 10 at event {event_index} occurred before closing "
                    f"the previous start at event {pending_start[0]}."
                )
            pending_start = (event_index, timestamp)
            continue

        if code != 11:
            continue

        if pending_start is None:
            warnings.append(
                f"Code 11 at event {event_index} has no preceding code 10."
            )
            continue

        start_event_index, start_ms = pending_start
        end_ms = timestamp
        pending_start = None

        if end_ms <= start_ms:
            warnings.append(
                f"Non-positive interval: start={start_ms}, end={end_ms}."
            )
            continue

        # Keep samples from the first timestamp >= start to the last <= end.
        left = int(np.searchsorted(time, start_ms, side="left"))
        right = int(np.searchsorted(time, end_ms, side="right"))
        segment_x = x[left:right]
        segment_y = y[left:right]
        segment_pupil = pupil[left:right]

        finite_xy = np.isfinite(segment_x) & np.isfinite(segment_y)
        positive_xy = finite_xy & (segment_x > 0) & (segment_y > 0)
        valid_pupil = np.isfinite(segment_pupil) & (segment_pupil > 0)

        n_samples = max(0, right - left)
        duration_ms = end_ms - start_ms
        expected_samples_500hz = duration_ms / 2.0

        rows.append(
            {
                "trial_index_in_block": len(rows),
                "start_event_index": start_event_index,
                "end_event_index": event_index,
                "start_timestamp_ms": start_ms,
                "end_timestamp_ms": end_ms,
                "duration_ms": duration_ms,
                "duration_sec": duration_ms / 1000.0,
                "sample_start_index": left,
                "sample_end_index_exclusive": right,
                "n_samples": n_samples,
                "expected_samples_at_500hz": expected_samples_500hz,
                "sample_coverage_ratio": (
                    n_samples / expected_samples_500hz
                    if expected_samples_500hz > 0
                    else math.nan
                ),
                "finite_xy_rate": (
                    float(np.mean(finite_xy)) if n_samples else math.nan
                ),
                "positive_xy_rate": (
                    float(np.mean(positive_xy)) if n_samples else math.nan
                ),
                "valid_pupil_rate": (
                    float(np.mean(valid_pupil)) if n_samples else math.nan
                ),
                "x_min": (
                    float(np.nanmin(segment_x[finite_xy]))
                    if np.any(finite_xy) else math.nan
                ),
                "x_max": (
                    float(np.nanmax(segment_x[finite_xy]))
                    if np.any(finite_xy) else math.nan
                ),
                "y_min": (
                    float(np.nanmin(segment_y[finite_xy]))
                    if np.any(finite_xy) else math.nan
                ),
                "y_max": (
                    float(np.nanmax(segment_y[finite_xy]))
                    if np.any(finite_xy) else math.nan
                ),
            }
        )

    if pending_start is not None:
        warnings.append(
            f"Final code 10 at event {pending_start[0]} has no following code 11."
        )

    return rows, warnings


def compare_numeric_arrays(
    raw: np.ndarray,
    corrected: np.ndarray,
    headers: list[str],
) -> list[dict[str, Any]]:
    raw = np.asarray(raw)
    corrected = np.asarray(corrected)

    rows: list[dict[str, Any]] = []
    n_columns = max(
        raw.shape[1] if raw.ndim == 2 else 0,
        corrected.shape[1] if corrected.ndim == 2 else 0,
    )

    for column in range(n_columns):
        name = headers[column] if column < len(headers) else f"column_{column}"
        if (
            raw.ndim != 2
            or corrected.ndim != 2
            or column >= raw.shape[1]
            or column >= corrected.shape[1]
        ):
            rows.append(
                {
                    "column_index": column,
                    "column_name": name,
                    "raw_shape": str(raw.shape),
                    "corrected_shape": str(corrected.shape),
                    "same_shape": int(raw.shape == corrected.shape),
                    "n_compared": 0,
                    "n_different": "",
                    "different_rate": "",
                    "mean_abs_difference": "",
                    "max_abs_difference": "",
                    "exact_equal_including_nan": 0,
                }
            )
            continue

        n = min(raw.shape[0], corrected.shape[0])
        a = raw[:n, column].astype(float)
        b = corrected[:n, column].astype(float)

        both_nan = np.isnan(a) & np.isnan(b)
        equal = (a == b) | both_nan
        finite_both = np.isfinite(a) & np.isfinite(b)
        differences = np.abs(a[finite_both] - b[finite_both])

        rows.append(
            {
                "column_index": column,
                "column_name": name,
                "raw_shape": str(raw.shape),
                "corrected_shape": str(corrected.shape),
                "same_shape": int(raw.shape == corrected.shape),
                "n_compared": n,
                "n_different": int(np.sum(~equal)),
                "different_rate": float(np.mean(~equal)) if n else math.nan,
                "mean_abs_difference": (
                    float(np.mean(differences)) if differences.size else math.nan
                ),
                "max_abs_difference": (
                    float(np.max(differences)) if differences.size else math.nan
                ),
                "exact_equal_including_nan": int(np.all(equal)),
            }
        )
    return rows


def get_headers(mat: dict[str, Any]) -> list[str]:
    values = np.asarray(mat["colheader"], dtype=object).ravel()
    return [to_text(value).strip() for value in values]


def compare_other_objects(
    raw_mat: dict[str, Any],
    corrected_mat: dict[str, Any],
) -> list[str]:
    lines: list[str] = []

    raw_messages = [to_text(x) for x in np.asarray(raw_mat["messages"], dtype=object).ravel()]
    corrected_messages = [to_text(x) for x in np.asarray(corrected_mat["messages"], dtype=object).ravel()]
    lines.append(f"messages_exact_equal={raw_messages == corrected_messages}")
    lines.append(f"raw_message_count={len(raw_messages)}")
    lines.append(f"corrected_message_count={len(corrected_messages)}")

    raw_event = np.asarray(raw_mat["event"])
    corrected_event = np.asarray(corrected_mat["event"])
    lines.append(
        "event_exact_equal="
        + str(
            raw_event.shape == corrected_event.shape
            and np.array_equal(raw_event, corrected_event)
        )
    )
    lines.append(f"raw_event_shape={raw_event.shape}")
    lines.append(f"corrected_event_shape={corrected_event.shape}")

    for event_name in ("saccades", "fixations", "blinks"):
        raw_obj = getattr(raw_mat["eyeevent"], event_name)
        corrected_obj = getattr(corrected_mat["eyeevent"], event_name)
        raw_data = np.asarray(raw_obj.data)
        corrected_data = np.asarray(corrected_obj.data)

        lines.append("")
        lines.append(f"[{event_name}]")
        lines.append(f"raw_shape={raw_data.shape}")
        lines.append(f"corrected_shape={corrected_data.shape}")
        lines.append(
            "exact_equal="
            + str(
                raw_data.shape == corrected_data.shape
                and np.array_equal(raw_data, corrected_data, equal_nan=True)
            )
        )

        # For fixations, check whether corrected start/end timestamps are a
        # subset of raw start/end timestamps.
        if raw_data.ndim == 2 and corrected_data.ndim == 2 and raw_data.shape[1] >= 2 and corrected_data.shape[1] >= 2:
            raw_keys = {
                (int(round(row[0])), int(round(row[1])))
                for row in raw_data
                if np.isfinite(row[0]) and np.isfinite(row[1])
            }
            corrected_keys = {
                (int(round(row[0])), int(round(row[1])))
                for row in corrected_data
                if np.isfinite(row[0]) and np.isfinite(row[1])
            }
            lines.append(
                f"corrected_time_intervals_in_raw="
                f"{len(corrected_keys & raw_keys)}/{len(corrected_keys)}"
            )
            lines.append(
                f"raw_intervals_removed={len(raw_keys - corrected_keys)}"
            )
            lines.append(
                f"corrected_intervals_added={len(corrected_keys - raw_keys)}"
            )

    return lines


def print_summary(
    code_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    print("=== EVENT CODE COUNTS ===")
    for row in code_rows:
        print(f"code {row['event_code']}: {row['count']}")

    print("\n=== CODE 10 -> 11 CANDIDATE TRIALS ===")
    print(f"paired trials: {len(trial_rows)}")
    if trial_rows:
        durations = np.asarray([row["duration_sec"] for row in trial_rows], dtype=float)
        print(f"duration median: {np.median(durations):.3f} sec")
        print(f"duration min: {np.min(durations):.3f} sec")
        print(f"duration max: {np.max(durations):.3f} sec")
        print(
            "sample coverage median: "
            f"{np.median([row['sample_coverage_ratio'] for row in trial_rows]):.6f}"
        )

    if warnings:
        print("\n=== PAIRING WARNINGS ===")
        for warning in warnings:
            print(warning)

    print("\n=== RAW VS CORRECTED CONTINUOUS DATA ===")
    for row in comparison_rows:
        print(
            f"{row['column_name']}: "
            f"different={row['n_different']} "
            f"rate={row['different_rate']} "
            f"mean_abs_diff={row['mean_abs_difference']} "
            f"max_abs_diff={row['max_abs_difference']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("corrected_file")
    parser.add_argument(
        "--output-dir",
        default="outputs/tables/YAC_NR1_marker_analysis",
    )
    parser.add_argument("--message-window-ms", type=int, default=250)
    args = parser.parse_args()

    raw_path = Path(args.raw_file)
    corrected_path = Path(args.corrected_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_mat = load_mat(raw_path)
    corrected_mat = load_mat(corrected_path)

    raw_data = np.asarray(raw_mat["data"])
    corrected_data = np.asarray(corrected_mat["data"])
    raw_event = np.asarray(raw_mat["event"])

    headers = get_headers(raw_mat)
    messages = parse_messages(raw_mat["messages"])
    code_rows = summarize_event_codes(raw_event)
    context_rows = nearest_messages(
        raw_event,
        messages,
        window_ms=args.message_window_ms,
    )
    trial_rows, warnings = pair_code_10_11(raw_event, raw_data)
    comparison_rows = compare_numeric_arrays(
        raw_data,
        corrected_data,
        headers,
    )
    object_comparison_lines = compare_other_objects(raw_mat, corrected_mat)

    write_csv(output_dir / "event_code_summary.csv", code_rows)
    write_csv(output_dir / "parsed_messages.csv", messages)
    write_csv(output_dir / "event_message_context.csv", context_rows)
    write_csv(output_dir / "candidate_trials_code10_11.csv", trial_rows)
    write_csv(
        output_dir / "raw_corrected_data_comparison.csv",
        comparison_rows,
    )
    (output_dir / "raw_corrected_object_comparison.txt").write_text(
        "\n".join(object_comparison_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pairing_warnings.txt").write_text(
        "\n".join(warnings) + ("\n" if warnings else ""),
        encoding="utf-8",
    )

    print_summary(code_rows, trial_rows, comparison_rows, warnings)
    print("\nWritten to:")
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
