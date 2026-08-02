#!/usr/bin/env python
"""
Audit the structure of sampled OneStop EyeLink ASC archives before writing
the full paragraph extractor.

Run from the gaze_reading_goal repository root:

python src\data\audit_onestop_asc_structure.py ^
  --input-dir data\raw\onestop\asc_samples ^
  --manifest data\interim\onestop\onestop_regular_trial_manifest.csv ^
  --output-dir outputs\onestop_asc_structure

The script reads .asc.zip archives directly. It does not modify or extract them.

Outputs:
- onestop_asc_structure_summary.csv
- onestop_asc_structure_audit.json
- onestop_asc_candidate_messages.csv
- onestop_asc_message_prefixes.csv
- one human-readable message excerpt per participant
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


SAMPLE_RE = re.compile(r"^\s*(\d+)\s+")
RAW_STEM_RE = re.compile(r"^(?P<stem>[lL]\d+_\d+)\.asc(?:\.zip)?$", re.IGNORECASE)
MSG_RE = re.compile(r"^MSG\s+(\d+)\s*(.*)$", re.IGNORECASE)

KEYWORD_RE = re.compile(
    r"trial|paragraph|article|question|preview|practice|repeat|reading|"
    r"condition|display|screen|stimulus|fixation|drift|calibration|validation|"
    r"recording|start|end|response|answer",
    re.IGNORECASE,
)

TRIAL_INDEX_PATTERNS = [
    re.compile(r"\bTRIALID\b\s*[:=]?\s*([A-Za-z0-9_.-]+)", re.IGNORECASE),
    re.compile(r"\btrial[_ ]?index\b\s*[:=]?\s*([A-Za-z0-9_.-]+)", re.IGNORECASE),
    re.compile(r"\btrial\b\s*[:=]\s*([A-Za-z0-9_.-]+)", re.IGNORECASE),
]

DISPLAY_PATTERNS = [
    re.compile(
        r"(?:DISPLAY_COORDS|GAZE_COORDS)\s+"
        r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"!V\s+(?:DISPLAY_COORDS|GAZE_COORDS)\s+"
        r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
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
        default=Path("data/interim/onestop/onestop_regular_trial_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/onestop_asc_structure"),
    )
    parser.add_argument(
        "--max-candidate-messages",
        type=int,
        default=5000,
        help="Maximum keyword-bearing MSG lines retained per file.",
    )
    parser.add_argument(
        "--excerpt-lines",
        type=int,
        default=600,
        help="Maximum selected lines written to each readable excerpt.",
    )
    return parser.parse_args()


def normalize_integer(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def archive_stem(path: Path) -> str:
    name = path.name
    match = RAW_STEM_RE.match(name)
    if match:
        return match.group("stem").lower()
    if name.lower().endswith(".asc.zip"):
        return name[:-8].lower()
    if name.lower().endswith(".asc"):
        return name[:-4].lower()
    raise ValueError(f"Unsupported ASC filename: {path}")


def find_inputs(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith(".asc.zip") or lower.endswith(".asc"):
            files.append(path)
    return sorted(files)


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
    # EyeLink ASC is generally ASCII-compatible. latin-1 gives a lossless
    # one-byte fallback while preserving all standard message text.
    wrapper = io.TextIOWrapper(binary, encoding="latin-1", errors="replace", newline="")
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

    handle = path.open("rb")

    def generator() -> Iterator[str]:
        try:
            yield from decode_stream(handle)
        finally:
            handle.close()

    return path.name, generator()


def message_prefix(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "<EMPTY>"
    tokens = cleaned.split()
    if not tokens:
        return "<EMPTY>"

    first = tokens[0]
    first = re.sub(r"[:=].*$", "", first)
    if first.upper() in {"!V", "!MODE", "!CAL", "!CMD"} and len(tokens) > 1:
        second = re.sub(r"[:=].*$", "", tokens[1])
        return f"{first.upper()} {second.upper()}"
    return first.upper()


def extract_trial_candidate(text: str) -> str | None:
    for pattern in TRIAL_INDEX_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def infer_sample_fields(tokens: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": None,
        "token_count": len(tokens),
        "first_gaze_x": None,
        "first_gaze_y": None,
        "first_pupil": None,
        "first_gaze_missing": None,
    }
    if not tokens:
        return result

    try:
        result["timestamp"] = int(tokens[0])
    except ValueError:
        return result

    if len(tokens) >= 4:
        x, y, pupil = tokens[1], tokens[2], tokens[3]
        result["first_gaze_missing"] = x == "." or y == "."
        for key, token in (
            ("first_gaze_x", x),
            ("first_gaze_y", y),
            ("first_pupil", pupil),
        ):
            try:
                result[key] = float(token)
            except ValueError:
                result[key] = None
    return result


def audit_one(
    path: Path,
    participant_meta: dict[str, Any] | None,
    max_messages: int,
    excerpt_limit: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    stem = archive_stem(path)
    member, lines = iter_lines(path)

    line_count = 0
    sample_count = 0
    msg_count = 0
    start_count = 0
    end_count = 0
    input_count = 0
    fixation_event_count = 0
    saccade_event_count = 0
    blink_event_count = 0

    first_sample_time = None
    last_sample_time = None
    sample_token_counts: Counter[int] = Counter()
    first_eye_missing_count = 0
    first_eye_observed_count = 0

    display_coords: set[tuple[float, float, float, float]] = set()
    sample_config_lines: list[str] = []
    event_config_lines: list[str] = []
    mode_lines: list[str] = []
    recording_lines: list[str] = []

    candidate_messages: list[dict[str, Any]] = []
    prefixes: Counter[str] = Counter()
    trial_candidates: Counter[str] = Counter()
    excerpt: list[str] = []

    for line in lines:
        line_count += 1
        stripped = line.strip()
        if not stripped:
            continue

        upper = stripped.upper()
        record_type = stripped.split(None, 1)[0].upper()

        if SAMPLE_RE.match(stripped):
            tokens = stripped.split()
            info = infer_sample_fields(tokens)
            if info["timestamp"] is not None:
                sample_count += 1
                sample_token_counts[info["token_count"]] += 1
                if first_sample_time is None:
                    first_sample_time = info["timestamp"]
                last_sample_time = info["timestamp"]
                if info["first_gaze_missing"] is True:
                    first_eye_missing_count += 1
                elif info["first_gaze_missing"] is False:
                    first_eye_observed_count += 1
            continue

        # EyeLink ASC uses either spaces or tabs as separators.
        if record_type == "MSG":
            msg_count += 1
            match = MSG_RE.match(stripped)
            timestamp = None
            text = stripped
            if match:
                timestamp = int(match.group(1))
                text = match.group(2).strip()

            prefix = message_prefix(text)
            prefixes[prefix] += 1

            trial_candidate = extract_trial_candidate(text)
            if trial_candidate is not None:
                trial_candidates[trial_candidate] += 1

            keep = bool(KEYWORD_RE.search(text))
            if keep and len(candidate_messages) < max_messages:
                candidate_messages.append(
                    {
                        "stem": stem,
                        "regime": (
                            participant_meta.get("regime")
                            if participant_meta is not None
                            else None
                        ),
                        "participant_id": (
                            participant_meta.get("participant_id")
                            if participant_meta is not None
                            else stem
                        ),
                        "timestamp": timestamp,
                        "prefix": prefix,
                        "trial_candidate": trial_candidate,
                        "message": text,
                    }
                )
            if keep and len(excerpt) < excerpt_limit:
                excerpt.append(stripped)
            continue

        if record_type == "START":
            start_count += 1
            if len(excerpt) < excerpt_limit:
                excerpt.append(stripped)
        elif record_type == "END":
            end_count += 1
            if len(excerpt) < excerpt_limit:
                excerpt.append(stripped)
        elif record_type == "INPUT":
            input_count += 1
        elif record_type == "EFIX":
            fixation_event_count += 1
        elif record_type == "ESACC":
            saccade_event_count += 1
        elif record_type == "EBLINK":
            blink_event_count += 1

        if upper.startswith("SAMPLES"):
            if len(sample_config_lines) < 50:
                sample_config_lines.append(stripped)
            if len(excerpt) < excerpt_limit:
                excerpt.append(stripped)
        elif upper.startswith("EVENTS"):
            if len(event_config_lines) < 50:
                event_config_lines.append(stripped)
            if len(excerpt) < excerpt_limit:
                excerpt.append(stripped)
        elif upper.startswith("!MODE"):
            if len(mode_lines) < 50:
                mode_lines.append(stripped)
            if len(excerpt) < excerpt_limit:
                excerpt.append(stripped)
        elif "RECORD" in upper or "TRACKER" in upper or "PUPIL" in upper:
            if len(recording_lines) < 100:
                recording_lines.append(stripped)

        for pattern in DISPLAY_PATTERNS:
            display_match = pattern.search(stripped)
            if display_match:
                display_coords.add(tuple(float(v) for v in display_match.groups()))

    observed_total = first_eye_missing_count + first_eye_observed_count
    missing_rate = (
        first_eye_missing_count / observed_total if observed_total > 0 else None
    )

    expected_trial_indices: list[str] = []
    expected_trial_count = None
    if participant_meta is not None:
        expected_trial_indices = participant_meta.get("expected_trial_indices", [])
        expected_trial_count = participant_meta.get("expected_trial_count")

    excerpt_path = output_dir / f"{stem}_message_excerpt.txt"
    excerpt_header = [
        f"archive={path}",
        f"member={member}",
        f"stem={stem}",
        f"regime={participant_meta.get('regime') if participant_meta else None}",
        f"expected_regular_trials={expected_trial_count}",
        f"expected_trial_indices={','.join(expected_trial_indices)}",
        "",
    ]
    excerpt_path.write_text(
        "\n".join(excerpt_header + excerpt) + "\n",
        encoding="utf-8",
    )

    summary = {
        "archive_path": str(path),
        "archive_size_bytes": path.stat().st_size,
        "asc_member": member,
        "stem": stem,
        "regime": participant_meta.get("regime") if participant_meta else None,
        "label_name": (
            participant_meta.get("label_name") if participant_meta else None
        ),
        "expected_regular_trials": expected_trial_count,
        "line_count": line_count,
        "sample_count": sample_count,
        "msg_count": msg_count,
        "start_count": start_count,
        "end_count": end_count,
        "input_count": input_count,
        "efix_count": fixation_event_count,
        "esacc_count": saccade_event_count,
        "eblink_count": blink_event_count,
        "first_sample_time": first_sample_time,
        "last_sample_time": last_sample_time,
        "sample_duration_ms": (
            last_sample_time - first_sample_time
            if first_sample_time is not None and last_sample_time is not None
            else None
        ),
        "sample_token_count_distribution": dict(sample_token_counts),
        "first_eye_missing_rate": missing_rate,
        "display_coords": [list(v) for v in sorted(display_coords)],
        "sample_config_lines": sample_config_lines,
        "event_config_lines": event_config_lines,
        "mode_lines": mode_lines,
        "recording_lines": recording_lines,
        "trial_candidate_counts": dict(trial_candidates),
        "candidate_message_count_retained": len(candidate_messages),
        "excerpt_path": str(excerpt_path),
    }
    return summary, candidate_messages, prefixes


def load_manifest_metadata(path: Path) -> dict[str, dict[str, Any]]:
    manifest = pd.read_csv(path, dtype="string", low_memory=False)
    required = {
        "participant_id",
        "regime",
        "label_name",
        "trial_index",
        "onestop_trial_id",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    result: dict[str, dict[str, Any]] = {}
    for participant_id, frame in manifest.groupby("participant_id", observed=True):
        stem = str(participant_id).strip().lower()
        indices = sorted(
            {normalize_integer(v) for v in frame["trial_index"] if normalize_integer(v)},
            key=lambda x: int(x) if x.isdigit() else x,
        )
        result[stem] = {
            "participant_id": str(participant_id),
            "regime": str(frame["regime"].iloc[0]),
            "label_name": str(frame["label_name"].iloc[0]),
            "expected_trial_count": int(frame["onestop_trial_id"].nunique()),
            "expected_trial_indices": indices,
        }
    return result


def json_default(value: Any) -> Any:
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = find_inputs(args.input_dir)
    if not files:
        print(
            f"ERROR: no .asc.zip or .asc files found under {args.input_dir}",
            file=sys.stderr,
        )
        return 2

    manifest_meta = load_manifest_metadata(args.manifest)

    summaries = []
    all_messages: list[dict[str, Any]] = []
    all_prefix_rows: list[dict[str, Any]] = []

    for index, path in enumerate(files, start=1):
        stem = archive_stem(path)
        participant_meta = manifest_meta.get(stem)
        print(f"[{index}/{len(files)}] Auditing {path}")
        summary, messages, prefixes = audit_one(
            path=path,
            participant_meta=participant_meta,
            max_messages=args.max_candidate_messages,
            excerpt_limit=args.excerpt_lines,
            output_dir=args.output_dir,
        )
        summaries.append(summary)
        all_messages.extend(messages)
        for prefix, count in prefixes.most_common():
            all_prefix_rows.append(
                {
                    "stem": stem,
                    "regime": (
                        participant_meta.get("regime")
                        if participant_meta is not None
                        else None
                    ),
                    "prefix": prefix,
                    "count": count,
                }
            )

    summary_rows = []
    for item in summaries:
        summary_rows.append(
            {
                "stem": item["stem"],
                "regime": item["regime"],
                "label_name": item["label_name"],
                "archive_path": item["archive_path"],
                "archive_size_bytes": item["archive_size_bytes"],
                "asc_member": item["asc_member"],
                "expected_regular_trials": item["expected_regular_trials"],
                "line_count": item["line_count"],
                "sample_count": item["sample_count"],
                "msg_count": item["msg_count"],
                "start_count": item["start_count"],
                "end_count": item["end_count"],
                "efix_count": item["efix_count"],
                "esacc_count": item["esacc_count"],
                "eblink_count": item["eblink_count"],
                "first_sample_time": item["first_sample_time"],
                "last_sample_time": item["last_sample_time"],
                "sample_duration_ms": item["sample_duration_ms"],
                "first_eye_missing_rate": item["first_eye_missing_rate"],
                "display_coords": json.dumps(item["display_coords"]),
                "sample_token_count_distribution": json.dumps(
                    item["sample_token_count_distribution"]
                ),
                "trial_candidate_count": len(item["trial_candidate_counts"]),
                "candidate_message_count_retained": item[
                    "candidate_message_count_retained"
                ],
                "excerpt_path": item["excerpt_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    messages_df = pd.DataFrame(all_messages)
    prefixes_df = pd.DataFrame(all_prefix_rows)

    summary_path = args.output_dir / "onestop_asc_structure_summary.csv"
    messages_path = args.output_dir / "onestop_asc_candidate_messages.csv"
    prefixes_path = args.output_dir / "onestop_asc_message_prefixes.csv"
    audit_path = args.output_dir / "onestop_asc_structure_audit.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    messages_df.to_csv(messages_path, index=False, encoding="utf-8-sig")
    prefixes_df.to_csv(prefixes_path, index=False, encoding="utf-8-sig")
    audit_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("\n=== ONESTOP ASC STRUCTURE SUMMARY ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(summary_df.to_string(index=False))

    print("\n=== TOP MESSAGE PREFIXES ===")
    if prefixes_df.empty:
        print("No MSG prefixes found.")
    else:
        top = (
            prefixes_df.groupby(["regime", "prefix"], dropna=False)["count"]
            .sum()
            .reset_index()
            .sort_values(["regime", "count"], ascending=[True, False])
            .groupby("regime", dropna=False)
            .head(30)
        )
        print(top.to_string(index=False))

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {messages_path}")
    print(f"Saved: {audit_path}")

    unknown = [item["stem"] for item in summaries if item["regime"] is None]
    if unknown:
        print(
            f"WARNING: ASC stems not found in manifest: {unknown}",
            file=sys.stderr,
        )
        return 3

    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
