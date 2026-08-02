#!/usr/bin/env python
r"""
Extract regular first-reading paragraph gaze samples from OneStop EyeLink ASC ZIPs.

The extractor is intentionally participant-wise and memory bounded. It reads ASC
archives directly, identifies paragraph intervals using:

    MSG <time> NULL_ACTION_PARAG
    MSG <time> UPDATE_PARAG_RT

It maps each block to the canonical first-reading manifest using the ASC
!V TRIAL_VAR field Trial_Index_, excludes practice and rereading blocks through
that manifest match, and resamples monocular gaze to a regular 30 Hz grid by
nearest native sample.

Run from the gaze_reading_goal repository root:

python src\data\extract_onestop_regular_raw_samples.py ^
  --input-dir data\raw\onestop\asc_samples ^
  --manifest data\interim\onestop\onestop_regular_trial_manifest.csv ^
  --output-dir data\interim\onestop\raw_samples_30hz ^
  --audit-dir outputs\onestop_raw_extraction_sample

The same script can later be pointed at the directory containing all ASC ZIPs.

Outputs:
- one <participant_id>_samples_30hz.parquet per ASC archive;
- onestop_raw_extraction_trial_summary.csv;
- onestop_raw_extraction_excluded_blocks.csv;
- onestop_raw_extraction_audit.json.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


RAW_STEM_RE = re.compile(
    r"^(?P<stem>[lL]\d+_\d+)\.asc(?:\.zip)?$",
    re.IGNORECASE,
)
MSG_RE = re.compile(r"^MSG\s+(\d+)\s*(.*)$", re.IGNORECASE)
TRIALID_RE = re.compile(r"^TRIALID\s+(.+?)\s*$", re.IGNORECASE)
TRIAL_VAR_RE = re.compile(
    r"^!V\s+TRIAL_VAR\s+(?P<name>\S+)(?:\s+(?P<value>.*?))?\s*$",
    re.IGNORECASE,
)

PARAGRAPH_ONSET = "NULL_ACTION_PARAG"
PARAGRAPH_OFFSET = "UPDATE_PARAG_RT"

MANIFEST_COLUMNS = [
    "onestop_trial_id",
    "participant_id",
    "regime",
    "label",
    "label_name",
    "trial_index",
    "stimulus_id",
    "task_context_id",
    "list_number",
    "question_preview",
    "article_batch",
    "article_id",
    "article_index",
    "paragraph_id",
    "difficulty_level",
    "onestopqa_question_id",
    "article_title",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/onestop/asc_samples"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/interim/onestop/onestop_regular_trial_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/onestop/raw_samples_30hz"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/onestop_raw_extraction_sample"),
    )
    parser.add_argument("--target-hz", type=float, default=30.0)
    parser.add_argument(
        "--max-nearest-gap-ms",
        type=float,
        default=10.0,
        help=(
            "A resampled point is invalidated when its nearest native sample "
            "is farther away than this tolerance."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing participant parquet files.",
    )
    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def parse_bool(value: Any) -> bool | None:
    text = normalize_scalar(value).lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def archive_stem(path: Path) -> str:
    match = RAW_STEM_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported ASC filename: {path}")
    return match.group("stem").lower()


def find_inputs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.lower().endswith(".asc.zip")
            or path.name.lower().endswith(".asc")
        )
    )


def choose_zip_member(zf: zipfile.ZipFile) -> str:
    members = [
        name
        for name in zf.namelist()
        if not name.endswith("/")
        and "__MACOSX" not in name
        and not Path(name).name.startswith("._")
        and name.lower().endswith(".asc")
    ]
    if len(members) != 1:
        raise RuntimeError(
            f"Expected exactly one ASC member, found {len(members)}: {members}"
        )
    return members[0]


def decode_stream(binary: io.BufferedReader) -> Iterator[str]:
    wrapper = io.TextIOWrapper(
        binary,
        encoding="latin-1",
        errors="replace",
        newline="",
    )
    try:
        for line in wrapper:
            yield line.rstrip("\r\n")
    finally:
        wrapper.detach()


def iter_lines(path: Path) -> tuple[str, Iterator[str]]:
    if path.name.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path, "r")
        member = choose_zip_member(zf)
        raw = zf.open(member, "r")

        def generator() -> Iterator[str]:
            try:
                yield from decode_stream(raw)
            finally:
                raw.close()
                zf.close()

        return member, generator()

    raw = path.open("rb")

    def generator() -> Iterator[str]:
        try:
            yield from decode_stream(raw)
        finally:
            raw.close()

    return path.name, generator()


def parse_msg(line: str) -> tuple[int, str] | None:
    match = MSG_RE.match(line.strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2).strip()


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype="string", low_memory=False)
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"Manifest missing required columns: {sorted(missing)}"
        )
    manifest = manifest[MANIFEST_COLUMNS].copy()
    manifest["participant_id"] = (
        manifest["participant_id"].astype("string").str.strip().str.lower()
    )
    manifest["trial_index_norm"] = manifest["trial_index"].map(normalize_scalar)

    duplicate_key = manifest.duplicated(
        ["participant_id", "trial_index_norm"],
        keep=False,
    )
    if duplicate_key.any():
        bad = manifest.loc[
            duplicate_key,
            ["participant_id", "trial_index", "onestop_trial_id"],
        ]
        raise RuntimeError(
            "Manifest participant/trial_index keys are not unique:\n"
            + bad.to_string(index=False)
        )
    return manifest


def new_block(stem: str, trialid_raw: str, timestamp: int) -> dict[str, Any]:
    return {
        "stem": stem,
        "trialid_raw": trialid_raw,
        "trialid_timestamp": timestamp,
        "trial_vars": {},
        "paragraph_onset_timestamp": None,
        "paragraph_offset_timestamp": None,
        "sample_timestamp_ms": [],
        "x_px": [],
        "y_px": [],
        "pupil": [],
        "status": [],
        "malformed_sample_rows": 0,
    }


def choose_var(vars_: dict[str, str], names: list[str]) -> str:
    lowered = {key.lower(): value for key, value in vars_.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            normalized = normalize_scalar(value)
            if normalized:
                return normalized
    return ""


def trial_index_candidates(
    vars_: dict[str, str],
    trialid_raw: str,
) -> list[tuple[str, str]]:
    """Return trial-index candidates for audit, in semantic priority order."""
    raw_candidates = [
        ("trial", choose_var(vars_, ["trial"])),
        (
            "trial_index",
            choose_var(vars_, ["trial_index", "TRIAL_INDEX"]),
        ),
        (
            "trial_id",
            choose_var(vars_, ["trial_id", "TRIAL_ID"]),
        ),
        ("trialid_raw", normalize_scalar(trialid_raw)),
        ("Trial_Index_", choose_var(vars_, ["Trial_Index_"])),
    ]

    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[str, str]] = []
    for source, value in raw_candidates:
        normalized = normalize_scalar(value)
        key = (source, normalized)
        if not normalized or key in seen:
            continue
        seen.add(key)
        candidates.append(key)
    return candidates


def resolve_manifest_match(
    vars_: dict[str, str],
    trialid_raw: str,
    manifest_for_stem: pd.DataFrame,
) -> tuple[str, str, pd.DataFrame, list[dict[str, Any]]]:
    """
    Resolve an ASC block to the canonical manifest.

    OneStop's processed report trial number is the TRIAL_VAR field ``trial``.
    ``Trial_Index_`` and the TRIALID message are execution/internal indices and
    are retained only for audit. They are not allowed to override or conflict
    with a valid canonical ``trial`` match.
    """
    candidates = trial_index_candidates(vars_, trialid_raw)
    candidate_audit: list[dict[str, Any]] = []

    for source, value in candidates:
        frame = manifest_for_stem[
            manifest_for_stem["trial_index_norm"].eq(value)
        ]
        candidate_audit.append(
            {
                "source": source,
                "value": value,
                "manifest_match_count": int(len(frame)),
            }
        )

    canonical_trial = choose_var(vars_, ["trial"])
    if canonical_trial:
        canonical_frame = manifest_for_stem[
            manifest_for_stem["trial_index_norm"].eq(canonical_trial)
        ]
        return (
            canonical_trial,
            "trial",
            canonical_frame,
            candidate_audit,
        )

    # Defensive fallback for an unexpected export lacking the canonical field.
    # Accept the first uniquely matched candidate in the priority order above.
    for source, value in candidates:
        frame = manifest_for_stem[
            manifest_for_stem["trial_index_norm"].eq(value)
        ]
        if len(frame) == 1:
            return value, f"fallback:{source}", frame, candidate_audit

    fallback_source, fallback_value = (
        candidates[0] if candidates else ("none", "")
    )
    return (
        fallback_value,
        fallback_source,
        manifest_for_stem.iloc[0:0],
        candidate_audit,
    )


def parse_float_token(token: str) -> float:
    if token == ".":
        return float("nan")
    try:
        return float(token)
    except ValueError:
        return float("nan")


def append_sample(block: dict[str, Any], stripped: str) -> None:
    tokens = stripped.split()
    if len(tokens) < 4:
        block["malformed_sample_rows"] += 1
        return
    try:
        timestamp = int(tokens[0])
    except ValueError:
        block["malformed_sample_rows"] += 1
        return

    onset = block["paragraph_onset_timestamp"]
    offset = block["paragraph_offset_timestamp"]
    if onset is None or timestamp < onset:
        return
    if offset is not None and timestamp >= offset:
        return

    block["sample_timestamp_ms"].append(timestamp)
    block["x_px"].append(parse_float_token(tokens[1]))
    block["y_px"].append(parse_float_token(tokens[2]))
    block["pupil"].append(parse_float_token(tokens[3]))
    block["status"].append(tokens[4] if len(tokens) >= 5 else "")


def nearest_resample(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    pupil: np.ndarray,
    status: np.ndarray,
    onset_ms: int,
    offset_ms: int,
    target_hz: float,
    max_nearest_gap_ms: float,
) -> pd.DataFrame:
    duration_ms = offset_ms - onset_ms
    if duration_ms <= 0:
        return pd.DataFrame()

    step_ms = 1000.0 / target_hz
    n_points = int(math.ceil(duration_ms / step_ms))
    target = onset_ms + np.arange(n_points, dtype=float) * step_ms
    target = target[target < offset_ms]

    if timestamps.size == 0:
        return pd.DataFrame(
            {
                "sample_index_30hz": np.arange(target.size, dtype=np.int32),
                "target_timestamp_ms": target,
                "elapsed_s": (target - onset_ms) / 1000.0,
                "source_timestamp_ms": pd.array(
                    [pd.NA] * target.size,
                    dtype="Int64",
                ),
                "source_time_error_ms": np.nan,
                "x_px": np.nan,
                "y_px": np.nan,
                "pupil": np.nan,
                "status": "",
                "valid": False,
            }
        )

    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    x = x[order]
    y = y[order]
    pupil = pupil[order]
    status = status[order]

    right = np.searchsorted(timestamps, target, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)

    right_diff = np.abs(timestamps[right] - target)
    left_diff = np.abs(timestamps[left] - target)
    choose_left = left_diff <= right_diff
    nearest = np.where(choose_left, left, right)

    source_t = timestamps[nearest]
    error = np.abs(source_t - target)
    within_tolerance = error <= max_nearest_gap_ms
    finite_xy = np.isfinite(x[nearest]) & np.isfinite(y[nearest])
    valid = within_tolerance & finite_xy

    out_x = x[nearest].copy()
    out_y = y[nearest].copy()
    out_pupil = pupil[nearest].copy()
    out_status = status[nearest].astype(object).copy()

    out_x[~within_tolerance] = np.nan
    out_y[~within_tolerance] = np.nan
    out_pupil[~within_tolerance] = np.nan
    out_status[~within_tolerance] = ""

    return pd.DataFrame(
        {
            "sample_index_30hz": np.arange(target.size, dtype=np.int32),
            "target_timestamp_ms": target,
            "elapsed_s": (target - onset_ms) / 1000.0,
            "source_timestamp_ms": source_t.astype(np.int64),
            "source_time_error_ms": error,
            "x_px": out_x,
            "y_px": out_y,
            "pupil": out_pupil,
            "status": out_status,
            "valid": valid,
        }
    )


def block_to_outputs(
    block: dict[str, Any],
    manifest_for_stem: pd.DataFrame,
    target_hz: float,
    max_nearest_gap_ms: float,
) -> tuple[pd.DataFrame | None, dict[str, Any], dict[str, Any] | None]:
    vars_ = block["trial_vars"]
    (
        parsed_trial_index,
        trial_index_source,
        manifest_matches,
        trial_index_candidates_audit,
    ) = resolve_manifest_match(
        vars_=vars_,
        trialid_raw=block["trialid_raw"],
        manifest_for_stem=manifest_for_stem,
    )

    practice = choose_var(
        vars_,
        ["practice", "practice_trial", "is_practice"],
    )
    reread = choose_var(
        vars_,
        ["reread", "repeated_reading", "repeated_reading_trial"],
    )

    onset = block["paragraph_onset_timestamp"]
    offset = block["paragraph_offset_timestamp"]
    has_complete_paragraph = (
        onset is not None and offset is not None and offset > onset
    )

    exclusion_reason = None
    if not has_complete_paragraph:
        exclusion_reason = "no_complete_paragraph_interval"
    elif parse_bool(practice) is True:
        exclusion_reason = "practice_trial"
    elif parse_bool(reread) is True:
        exclusion_reason = "reread_trial"
    elif len(manifest_matches) == 0:
        exclusion_reason = "not_in_regular_manifest"
    elif len(manifest_matches) > 1:
        exclusion_reason = "duplicate_manifest_match"

    base_audit = {
        "participant_id": block["stem"],
        "trialid_raw": block["trialid_raw"],
        "parsed_trial_index": parsed_trial_index,
        "trial_index_source": trial_index_source,
        "trial_index_candidates_json": json.dumps(
            trial_index_candidates_audit,
            ensure_ascii=False,
        ),
        "weaker_identifier_disagrees": any(
            item["source"] != "trial"
            and item["manifest_match_count"] == 1
            and item["value"] != parsed_trial_index
            for item in trial_index_candidates_audit
        ),
        "trial_var_trial": choose_var(vars_, ["trial"]),
        "trial_var_trial_index_internal": choose_var(
            vars_,
            ["Trial_Index_"],
        ),
        "trial_var_recycled": choose_var(
            vars_,
            ["Trial_Recycled_"],
        ),
        "practice_from_asc": practice,
        "practice_bool_from_asc": parse_bool(practice),
        "reread_from_asc": reread,
        "reread_bool_from_asc": parse_bool(reread),
        "paragraph_onset_timestamp": onset,
        "paragraph_offset_timestamp": offset,
        "paragraph_duration_ms": (
            offset - onset if has_complete_paragraph else None
        ),
        "manifest_match_count": int(len(manifest_matches)),
        "malformed_sample_rows": int(block["malformed_sample_rows"]),
    }

    if exclusion_reason is not None:
        excluded = {
            **base_audit,
            "exclusion_reason": exclusion_reason,
            "raw_sample_count": len(block["sample_timestamp_ms"]),
        }
        return None, base_audit, excluded

    manifest_row = manifest_matches.iloc[0]

    timestamps = np.asarray(
        block["sample_timestamp_ms"],
        dtype=np.int64,
    )
    x = np.asarray(block["x_px"], dtype=float)
    y = np.asarray(block["y_px"], dtype=float)
    pupil = np.asarray(block["pupil"], dtype=float)
    status = np.asarray(block["status"], dtype=str)

    native_valid = np.isfinite(x) & np.isfinite(y)
    unique_timestamps = np.unique(timestamps)
    duplicate_timestamp_count = len(timestamps) - len(unique_timestamps)

    positive_diffs = np.diff(unique_timestamps)
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt_ms = (
        float(np.median(positive_diffs))
        if positive_diffs.size > 0
        else None
    )
    native_rate_hz = (
        1000.0 / median_dt_ms
        if median_dt_ms is not None and median_dt_ms > 0
        else None
    )

    resampled = nearest_resample(
        timestamps=timestamps,
        x=x,
        y=y,
        pupil=pupil,
        status=status,
        onset_ms=int(onset),
        offset_ms=int(offset),
        target_hz=target_hz,
        max_nearest_gap_ms=max_nearest_gap_ms,
    )

    metadata = {
        column: manifest_row[column]
        for column in MANIFEST_COLUMNS
        if column in manifest_row.index
    }
    metadata["trial_index"] = normalize_scalar(metadata["trial_index"])

    for column, value in metadata.items():
        resampled[column] = value
    resampled["paragraph_onset_timestamp"] = int(onset)
    resampled["paragraph_offset_timestamp"] = int(offset)
    resampled["target_hz"] = float(target_hz)

    ordered = [
        "onestop_trial_id",
        "participant_id",
        "regime",
        "label",
        "label_name",
        "trial_index",
        "stimulus_id",
        "task_context_id",
        "sample_index_30hz",
        "target_timestamp_ms",
        "elapsed_s",
        "source_timestamp_ms",
        "source_time_error_ms",
        "x_px",
        "y_px",
        "pupil",
        "status",
        "valid",
        "paragraph_onset_timestamp",
        "paragraph_offset_timestamp",
        "target_hz",
        "list_number",
        "question_preview",
        "article_batch",
        "article_id",
        "article_index",
        "paragraph_id",
        "difficulty_level",
        "onestopqa_question_id",
        "article_title",
    ]
    ordered = [column for column in ordered if column in resampled.columns]
    resampled = resampled[ordered]

    summary = {
        **base_audit,
        "onestop_trial_id": manifest_row["onestop_trial_id"],
        "regime": manifest_row["regime"],
        "label": int(manifest_row["label"]),
        "label_name": manifest_row["label_name"],
        "stimulus_id": manifest_row["stimulus_id"],
        "task_context_id": manifest_row["task_context_id"],
        "raw_sample_count": int(len(timestamps)),
        "raw_valid_sample_count": int(native_valid.sum()),
        "raw_valid_rate": (
            float(native_valid.mean()) if len(native_valid) else None
        ),
        "raw_duplicate_timestamp_count": int(duplicate_timestamp_count),
        "raw_median_dt_ms": median_dt_ms,
        "raw_inferred_rate_hz": native_rate_hz,
        "resampled_sample_count": int(len(resampled)),
        "resampled_valid_count": int(resampled["valid"].sum()),
        "resampled_valid_rate": (
            float(resampled["valid"].mean()) if len(resampled) else None
        ),
        "resampled_source_time_error_q95_ms": (
            float(resampled["source_time_error_ms"].quantile(0.95))
            if len(resampled)
            else None
        ),
    }
    return resampled, summary, None


def extract_file(
    path: Path,
    manifest_for_stem: pd.DataFrame,
    target_hz: float,
    max_nearest_gap_ms: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    stem = archive_stem(path)
    _, lines = iter_lines(path)

    current: dict[str, Any] | None = None
    trial_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    def finalize_current() -> None:
        nonlocal current
        if current is None:
            return
        frame, summary, excluded_row = block_to_outputs(
            block=current,
            manifest_for_stem=manifest_for_stem,
            target_hz=target_hz,
            max_nearest_gap_ms=max_nearest_gap_ms,
        )
        if frame is not None:
            trial_frames.append(frame)
            summaries.append(summary)
        if excluded_row is not None:
            excluded.append(excluded_row)
        current = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        first_token = stripped.split(None, 1)[0]
        if first_token.isdigit():
            if current is not None:
                append_sample(current, stripped)
            continue

        if first_token.upper() != "MSG":
            continue

        parsed = parse_msg(stripped)
        if parsed is None:
            continue
        timestamp, text = parsed

        trialid_match = TRIALID_RE.match(text)
        if trialid_match:
            finalize_current()
            current = new_block(
                stem=stem,
                trialid_raw=trialid_match.group(1).strip(),
                timestamp=timestamp,
            )
            continue

        if current is None:
            continue

        var_match = TRIAL_VAR_RE.match(text)
        if var_match:
            name = var_match.group("name").strip()
            value = (var_match.group("value") or "").strip()
            current["trial_vars"][name] = value
            continue

        action = text.split(None, 1)[0].upper() if text else ""
        if action == PARAGRAPH_ONSET:
            if current["paragraph_onset_timestamp"] is None:
                current["paragraph_onset_timestamp"] = timestamp
            continue
        if action == PARAGRAPH_OFFSET:
            if current["paragraph_offset_timestamp"] is None:
                current["paragraph_offset_timestamp"] = timestamp
            continue

    finalize_current()

    if trial_frames:
        samples = pd.concat(trial_frames, ignore_index=True)
    else:
        samples = pd.DataFrame()
    return samples, summaries, excluded


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    if args.target_hz <= 0:
        print("ERROR: --target-hz must be positive.", file=sys.stderr)
        return 2

    files = find_inputs(args.input_dir)
    if not files:
        print(
            f"ERROR: no ASC files found under {args.input_dir}",
            file=sys.stderr,
        )
        return 3

    manifest = load_manifest(args.manifest)

    all_summaries: list[dict[str, Any]] = []
    all_excluded: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []

    for index, path in enumerate(files, start=1):
        stem = archive_stem(path)
        participant_manifest = manifest[
            manifest["participant_id"].eq(stem)
        ].copy()
        if participant_manifest.empty:
            raise RuntimeError(
                f"ASC participant {stem} was not found in the manifest."
            )

        output_path = args.output_dir / f"{stem}_samples_30hz.parquet"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Use --overwrite to replace it."
            )

        print(
            f"[{index}/{len(files)}] Extracting {stem}: "
            f"expected_trials={participant_manifest['onestop_trial_id'].nunique()}"
        )
        samples, summaries, excluded = extract_file(
            path=path,
            manifest_for_stem=participant_manifest,
            target_hz=args.target_hz,
            max_nearest_gap_ms=args.max_nearest_gap_ms,
        )

        if samples.empty:
            raise RuntimeError(f"No regular samples extracted from {path}.")

        samples.to_parquet(
            output_path,
            index=False,
            compression="zstd",
        )

        extracted_trial_ids = {
            row["onestop_trial_id"] for row in summaries
        }
        expected_trial_ids = set(
            participant_manifest["onestop_trial_id"].astype(str)
        )
        missing_trial_ids = sorted(expected_trial_ids - extracted_trial_ids)
        unexpected_trial_ids = sorted(extracted_trial_ids - expected_trial_ids)

        participant_rows.append(
            {
                "participant_id": stem,
                "regime": participant_manifest["regime"].iloc[0],
                "expected_trials": len(expected_trial_ids),
                "extracted_trials": len(extracted_trial_ids),
                "missing_trials": len(missing_trial_ids),
                "unexpected_trials": len(unexpected_trial_ids),
                "resampled_rows": len(samples),
                "output_path": str(output_path),
                "missing_trial_ids": json.dumps(missing_trial_ids),
                "unexpected_trial_ids": json.dumps(unexpected_trial_ids),
            }
        )
        all_summaries.extend(summaries)
        all_excluded.extend(excluded)

    summary_df = pd.DataFrame(all_summaries)
    excluded_df = pd.DataFrame(all_excluded)
    participant_df = pd.DataFrame(participant_rows)

    summary_path = (
        args.audit_dir / "onestop_raw_extraction_trial_summary.csv"
    )
    excluded_path = (
        args.audit_dir / "onestop_raw_extraction_excluded_blocks.csv"
    )
    participant_path = (
        args.audit_dir / "onestop_raw_extraction_participant_summary.csv"
    )
    audit_path = args.audit_dir / "onestop_raw_extraction_audit.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    excluded_df.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    participant_df.to_csv(
        participant_path,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_trial_ids = int(
        summary_df["onestop_trial_id"].duplicated().sum()
    )
    total_expected = int(participant_df["expected_trials"].sum())
    total_extracted = int(participant_df["extracted_trials"].sum())
    total_missing = int(participant_df["missing_trials"].sum())
    total_unexpected = int(participant_df["unexpected_trials"].sum())

    audit = {
        "input_files": len(files),
        "target_hz": args.target_hz,
        "max_nearest_gap_ms": args.max_nearest_gap_ms,
        "expected_trials": total_expected,
        "extracted_trials": total_extracted,
        "missing_trials": total_missing,
        "unexpected_trials": total_unexpected,
        "duplicate_extracted_trial_ids": duplicate_trial_ids,
        "excluded_blocks": int(len(excluded_df)),
        "resampled_rows": int(
            participant_df["resampled_rows"].sum()
        ),
        "raw_valid_rate_median": (
            float(summary_df["raw_valid_rate"].median())
            if not summary_df.empty
            else None
        ),
        "resampled_valid_rate_median": (
            float(summary_df["resampled_valid_rate"].median())
            if not summary_df.empty
            else None
        ),
        "raw_inferred_rate_hz_median": (
            float(summary_df["raw_inferred_rate_hz"].median())
            if not summary_df.empty
            else None
        ),
        "paragraph_duration_ms_median": (
            float(summary_df["paragraph_duration_ms"].median())
            if not summary_df.empty
            else None
        ),
        "source_time_error_q95_ms_median": (
            float(
                summary_df[
                    "resampled_source_time_error_q95_ms"
                ].median()
            )
            if not summary_df.empty
            else None
        ),
        "outputs": {
            "sample_directory": str(args.output_dir),
            "trial_summary": str(summary_path),
            "excluded_blocks": str(excluded_path),
            "participant_summary": str(participant_path),
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ONESTOP RAW EXTRACTION SUMMARY ===")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    print("\n=== PARTICIPANT SUMMARY ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(participant_df.to_string(index=False))

    print("\n=== EXCLUDED BLOCK REASONS ===")
    if excluded_df.empty:
        print("None")
    else:
        print(
            excluded_df["exclusion_reason"]
            .value_counts(dropna=False)
            .rename_axis("reason")
            .reset_index(name="count")
            .to_string(index=False)
        )
        if "reread_bool_from_asc" in excluded_df.columns:
            print("\nExcluded practice/reread combinations:")
            print(
                excluded_df.groupby(
                    [
                        "practice_bool_from_asc",
                        "reread_bool_from_asc",
                    ],
                    dropna=False,
                )
                .size()
                .rename("count")
                .reset_index()
                .to_string(index=False)
            )

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {excluded_path}")
    print(f"Saved: {audit_path}")

    failed = (
        total_extracted != total_expected
        or total_missing != 0
        or total_unexpected != 0
        or duplicate_trial_ids != 0
    )
    if failed:
        print(
            "AUDIT STATUS: FAIL — trial coverage mismatch.",
            file=sys.stderr,
        )
        return 4

    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
