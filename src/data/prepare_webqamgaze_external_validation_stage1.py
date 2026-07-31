#!/usr/bin/env python
"""Prepare WebQAmGaze English data for frozen ZuCo external validation: stage 1.

This stage performs schema-wide discovery and constructs a conservative preliminary
trial/gaze manifest. It deliberately does NOT fit, tune, calibrate, or apply any
classifier. WebQAmGaze labels are used only to describe the external test set.

Main outputs
------------
- english_participant_audit.csv
- english_candidate_records.csv
- english_gaze_array_candidates.csv
- english_relevant_field_values.csv
- english_experiment_file_schemas.csv
- preliminary_external_trial_manifest.csv
- stage1_manifest.json

The preliminary trial manifest only contains records whose NR/IS condition can be
inferred conservatively and which can be linked to at least one gaze-array candidate.
Ambiguous records remain in the audit tables rather than being silently discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

LANG_PATTERN = re.compile(r"_(EN|DE|ES|TR|GR|RU)_", re.IGNORECASE)
SET_PATTERN = re.compile(
    r"(?P<set_name>(?:mturk|vol(?:unteer)?)_[A-Z]{2}_v\d+)",
    re.IGNORECASE,
)

RELEVANT_KEY_PATTERN = re.compile(
    r"gaze|webgazer|fixat|sacc|sample|timestamp|time|duration|trial|task|"
    r"condition|question|answer|screen|resolution|width|height|target|word|"
    r"sentence|paragraph|passage|text|stimulus|article|context|worker|"
    r"participant|subject|browser|window|viewport|validation|calibration|"
    r"accuracy|correct|response|rt|data",
    re.IGNORECASE,
)
GAZE_CONTAINER_PATTERN = re.compile(
    r"gaze|webgazer|eye.?track|sample|prediction|raw.?data",
    re.IGNORECASE,
)
READING_RECORD_PATTERN = re.compile(
    r"read|passage|paragraph|article|context|text|stimulus|question|"
    r"information.?seek|normal.?read|webgazer",
    re.IGNORECASE,
)

X_NAMES = {
    "x", "xpos", "x_pos", "xposition", "x_position", "gazex", "gaze_x",
    "predictionx", "prediction_x", "clientx", "client_x", "screenx", "screen_x",
}
Y_NAMES = {
    "y", "ypos", "y_pos", "yposition", "y_position", "gazey", "gaze_y",
    "predictiony", "prediction_y", "clienty", "client_y", "screeny", "screen_y",
}
TIME_NAMES = {
    "t", "time", "timestamp", "ts", "elapsedtime", "elapsed_time",
    "timesec", "time_sec", "timems", "time_ms", "clock", "duration",
}

# Conservative task-label signals. Generic "read" alone is intentionally not enough.
NR_PATTERNS = [
    re.compile(r"(^|[^a-z])nr([^a-z]|$)", re.IGNORECASE),
    re.compile(r"normal[\s_\-]*read", re.IGNORECASE),
    re.compile(r"natural[\s_\-]*read", re.IGNORECASE),
    re.compile(r"ordinary[\s_\-]*read", re.IGNORECASE),
]
IS_PATTERNS = [
    re.compile(r"(^|[^a-z])is([^a-z]|$)", re.IGNORECASE),
    re.compile(r"information[\s_\-]*seek", re.IGNORECASE),
    re.compile(r"target(?:ed)?[\s_\-]*read", re.IGNORECASE),
    re.compile(r"search[\s_\-]*(?:task|read)", re.IGNORECASE),
]

SCALAR_TYPES = (str, int, float, bool, type(None))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_load(path: Path) -> tuple[Any | None, str | None]:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with path.open("r", encoding=encoding) as handle:
                return json.load(handle), None
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
    return None, "UnicodeDecodeError: unable to decode as UTF-8/UTF-8-SIG"


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: set[str] = set()
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def infer_language(name: str) -> str:
    match = LANG_PATTERN.search(name)
    return match.group(1).upper() if match else "UNKNOWN"


def split_participant_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    match = SET_PATTERN.search(stem)
    if not match:
        return stem, "UNKNOWN"
    participant = stem[: match.start()].rstrip("_")
    return participant, match.group("set_name")


def json_path(parent: str, key: Any, is_index: bool = False) -> str:
    if is_index:
        return f"{parent}[{key}]" if parent else f"$[{key}]"
    text = str(key)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return f"{parent}.{text}" if parent else f"$.{text}"
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'{parent}["{escaped}"]' if parent else f'$["{escaped}"]'


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def compact(value: Any, max_chars: int = 180) -> str:
    if isinstance(value, str):
        text = value.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
    elif isinstance(value, SCALAR_TYPES):
        text = repr(value)
    elif isinstance(value, dict):
        text = f"dict(n={len(value)}, keys={list(value)[:12]})"
    elif isinstance(value, list):
        text = f"list(n={len(value)})"
    else:
        text = type(value).__name__
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def iter_nodes(
    value: Any,
    path: str = "$",
    depth: int = 0,
    max_depth: int = 14,
) -> Iterator[tuple[str, Any, int]]:
    yield path, value, depth
    if depth >= max_depth:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_nodes(child, json_path(path, key), depth + 1, max_depth)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_nodes(child, json_path(path, index, is_index=True), depth + 1, max_depth)


def scalar_items(
    value: Any,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 2,
    max_items: int = 100,
) -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []

    def visit(current: Any, path: str, level: int) -> None:
        if len(output) >= max_items:
            return
        if isinstance(current, SCALAR_TYPES):
            output.append((path or "$", current))
            return
        if level >= max_depth:
            return
        if isinstance(current, dict):
            for key, child in current.items():
                visit(child, f"{path}.{key}" if path else str(key), level + 1)
                if len(output) >= max_items:
                    return
        elif isinstance(current, list) and len(current) <= 12:
            for index, child in enumerate(current):
                visit(child, f"{path}[{index}]" if path else f"[{index}]", level + 1)
                if len(output) >= max_items:
                    return

    visit(value, prefix, depth)
    return output


def infer_condition_from_items(items: Sequence[tuple[str, Any]]) -> tuple[str, str, bool]:
    evidence_nr: list[str] = []
    evidence_is: list[str] = []

    for key_path, value in items:
        key_text = str(key_path)
        value_text = str(value)
        combined = f"{key_text}={value_text}"

        for pattern in NR_PATTERNS:
            if pattern.search(combined):
                evidence_nr.append(compact(combined, 120))
                break
        for pattern in IS_PATTERNS:
            if pattern.search(combined):
                evidence_is.append(compact(combined, 120))
                break

        # Explicit condition/task fields with compact symbolic values.
        last_key = normalize_key(key_path.split(".")[-1].split("[")[0])
        symbolic = re.sub(r"[^a-z]", "", value_text.lower())
        if last_key in {
            "task", "condition", "readingtask", "readingcondition",
            "experimentcondition", "trialcondition", "tasktype",
        }:
            if symbolic in {"nr", "normal", "normalreading", "naturalreading"}:
                evidence_nr.append(compact(combined, 120))
            elif symbolic in {
                "is", "informationseeking", "informationsearch",
                "targetedreading", "search",
            }:
                evidence_is.append(compact(combined, 120))

    nr = bool(evidence_nr)
    is_ = bool(evidence_is)
    if nr and not is_:
        return "NR", " | ".join(dict.fromkeys(evidence_nr))[:500], False
    if is_ and not nr:
        return "IS", " | ".join(dict.fromkeys(evidence_is))[:500], False
    if nr and is_:
        evidence = (
            "NR: " + " | ".join(dict.fromkeys(evidence_nr))[:220]
            + " || IS: " + " | ".join(dict.fromkeys(evidence_is))[:220]
        )
        return "AMBIGUOUS", evidence, True
    return "", "", False


def identify_axis_field(keys: Iterable[Any], candidates: set[str]) -> str:
    normalized = {normalize_key(key): str(key) for key in keys}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in normalized:
            return normalized[key]
    return ""


def numeric_fraction(values: Iterable[Any]) -> float:
    total = 0
    numeric = 0
    for value in values:
        total += 1
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
            if math.isfinite(number):
                numeric += 1
        except (TypeError, ValueError):
            pass
    return numeric / total if total else 0.0


def dict_list_gaze_candidate(
    value: list[Any],
) -> tuple[bool, dict[str, Any]]:
    if len(value) < 3:
        return False, {}
    sample = [item for item in value[: min(len(value), 200)] if isinstance(item, dict)]
    if len(sample) < max(3, min(10, len(value)) // 2):
        return False, {}

    key_counter: Counter[str] = Counter()
    original_by_normalized: dict[str, str] = {}
    for item in sample:
        for key in item:
            normalized = normalize_key(key)
            key_counter[normalized] += 1
            original_by_normalized.setdefault(normalized, str(key))

    common_keys = [
        original_by_normalized[key]
        for key, count in key_counter.items()
        if count >= max(2, int(0.6 * len(sample)))
    ]
    x_field = identify_axis_field(common_keys, X_NAMES)
    y_field = identify_axis_field(common_keys, Y_NAMES)
    time_field = identify_axis_field(common_keys, TIME_NAMES)

    # Fallback: fields containing axis names.
    if not x_field:
        for key in common_keys:
            norm = normalize_key(key)
            if norm.endswith("x") and ("gaze" in norm or "prediction" in norm):
                x_field = key
                break
    if not y_field:
        for key in common_keys:
            norm = normalize_key(key)
            if norm.endswith("y") and ("gaze" in norm or "prediction" in norm):
                y_field = key
                break

    if not (x_field and y_field):
        return False, {}

    x_fraction = numeric_fraction(item.get(x_field) for item in sample)
    y_fraction = numeric_fraction(item.get(y_field) for item in sample)
    if min(x_fraction, y_fraction) < 0.5:
        return False, {}

    time_fraction = (
        numeric_fraction(item.get(time_field) for item in sample)
        if time_field else 0.0
    )
    return True, {
        "container_kind": "list_of_dict",
        "n_samples": len(value),
        "sample_dict_fraction": len(sample) / min(len(value), 200),
        "common_keys": "|".join(common_keys[:80]),
        "x_field": x_field,
        "y_field": y_field,
        "time_field": time_field,
        "x_numeric_fraction": x_fraction,
        "y_numeric_fraction": y_fraction,
        "time_numeric_fraction": time_fraction,
    }


def sequence_gaze_candidate(value: list[Any]) -> tuple[bool, dict[str, Any]]:
    if len(value) < 3:
        return False, {}
    sample = value[: min(len(value), 200)]
    sequences = [
        item for item in sample
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]
    if len(sequences) < max(3, int(0.7 * len(sample))):
        return False, {}
    xy_ok = 0
    third_ok = 0
    for item in sequences:
        try:
            x = float(item[0])
            y = float(item[1])
            if math.isfinite(x) and math.isfinite(y):
                xy_ok += 1
            if len(item) >= 3 and math.isfinite(float(item[2])):
                third_ok += 1
        except (TypeError, ValueError):
            continue
    if xy_ok / len(sequences) < 0.7:
        return False, {}
    return True, {
        "container_kind": "list_of_sequences",
        "n_samples": len(value),
        "sample_dict_fraction": 0.0,
        "common_keys": "",
        "x_field": "[0]",
        "y_field": "[1]",
        "time_field": "[2]" if third_ok / len(sequences) >= 0.5 else "",
        "x_numeric_fraction": xy_ok / len(sequences),
        "y_numeric_fraction": xy_ok / len(sequences),
        "time_numeric_fraction": third_ok / len(sequences),
    }


def find_gaze_arrays(value: Any) -> list[tuple[str, list[Any], dict[str, Any], int]]:
    output: list[tuple[str, list[Any], dict[str, Any], int]] = []
    for path, node, depth in iter_nodes(value):
        if not isinstance(node, list):
            continue
        is_candidate, metadata = dict_list_gaze_candidate(node)
        if not is_candidate:
            is_candidate, metadata = sequence_gaze_candidate(node)
        if is_candidate:
            metadata["path_name_signal"] = bool(GAZE_CONTAINER_PATTERN.search(path))
            output.append((path, node, metadata, depth))
    return output


def nearest_common_prefix(path_a: str, path_b: str) -> int:
    # Character-level score is sufficient for ranking links within one JSON document.
    count = 0
    for left, right in zip(path_a, path_b):
        if left != right:
            break
        count += 1
    return count


def looks_like_candidate_record(path: str, node: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    items = scalar_items(node, max_depth=2, max_items=120)
    key_text = " ".join(map(str, node.keys()))
    scalar_text = " ".join(f"{key}={value}" for key, value in items)
    combined = f"{path} {key_text} {scalar_text}"

    condition, evidence, conflict = infer_condition_from_items(items)
    relevant_key_hits = sum(bool(RELEVANT_KEY_PATTERN.search(str(key))) for key in node)
    reading_signal = bool(READING_RECORD_PATTERN.search(combined))
    has_question = bool(re.search(r"question|query", combined, re.IGNORECASE))
    has_text = bool(
        re.search(
            r"passage|paragraph|article|context|stimulus|reading.?text",
            combined,
            re.IGNORECASE,
        )
    )
    has_timing = bool(re.search(r"timestamp|duration|response.?time|\brt\b", combined, re.IGNORECASE))
    candidate = bool(
        condition
        or (reading_signal and relevant_key_hits >= 2)
        or (has_question and has_text)
    )
    return candidate, {
        "condition": condition,
        "condition_evidence": evidence,
        "condition_conflict": conflict,
        "relevant_key_hits": relevant_key_hits,
        "reading_signal": reading_signal,
        "has_question_signal": has_question,
        "has_text_signal": has_text,
        "has_timing_signal": has_timing,
        "scalar_preview": compact(scalar_text, 1000),
    }


def extract_named_scalar(
    items: Sequence[tuple[str, Any]],
    patterns: Sequence[re.Pattern[str]],
) -> tuple[str, str]:
    for key_path, value in items:
        if not isinstance(value, SCALAR_TYPES):
            continue
        if any(pattern.search(key_path) for pattern in patterns):
            return key_path, compact(value, 500)
    return "", ""


ID_PATTERNS = [
    re.compile(r"(?:^|\.)(?:trial_?id|trialindex|trial_index|item_?id|text_?id|passage_?id)$", re.I),
    re.compile(r"(?:^|\.)(?:stimulus_?id|article_?id|context_?id)$", re.I),
]
QUESTION_PATTERNS = [
    re.compile(r"question", re.I),
    re.compile(r"(?:^|\.)(?:query)$", re.I),
]
TEXT_PATTERNS = [
    re.compile(r"passage|paragraph|article|context|stimulus|reading.?text", re.I),
]
SCREEN_WIDTH_PATTERNS = [
    re.compile(r"(?:screen|window|viewport).*(?:width)", re.I),
    re.compile(r"(?:^|\.)(?:screenw|screen_width|width)$", re.I),
]
SCREEN_HEIGHT_PATTERNS = [
    re.compile(r"(?:screen|window|viewport).*(?:height)", re.I),
    re.compile(r"(?:^|\.)(?:screenh|screen_height|height)$", re.I),
]
RT_PATTERNS = [
    re.compile(r"(?:^|\.)(?:rt|response_?time|duration|reading_?time)$", re.I),
]


def csv_schema(path: Path, root: Path, set_name: str, language: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "set_name": set_name,
        "language": language,
        "relative_path": path.relative_to(root).as_posix(),
        "filename": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "parse_status": "",
        "n_rows_estimate": "",
        "columns": "",
        "error": "",
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        row["parse_status"] = "ok"
        row["n_rows_estimate"] = count
        row["columns"] = "|".join(header)
    except Exception as exc:  # noqa: BLE001
        row["parse_status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/webqamgaze_repository_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage1"),
    )
    parser.add_argument("--language", default="EN")
    parser.add_argument(
        "--max-json-depth",
        type=int,
        default=14,
        help="Maximum recursion depth for participant JSON discovery.",
    )
    parser.add_argument(
        "--max-relevant-values-per-field",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository root not found: {root}")

    preprocessed_root = root / "scripts" / "pre_processed_data"
    if not preprocessed_root.is_dir():
        raise FileNotFoundError(f"Missing pre_processed_data: {preprocessed_root}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    language = args.language.upper()
    participant_paths = sorted(
        path for path in preprocessed_root.glob("*.json")
        if infer_language(path.name) == language
    )
    if not participant_paths:
        raise ValueError(f"No participant JSON files found for language={language}")

    participant_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    gaze_rows: list[dict[str, Any]] = []
    preliminary_rows: list[dict[str, Any]] = []

    relevant_values: dict[str, Counter[str]] = defaultdict(Counter)
    relevant_sources: dict[str, set[str]] = defaultdict(set)

    parsed_count = 0
    participants_with_gaze = 0
    participants_with_labeled_records = 0
    parse_errors = 0

    for participant_path in participant_paths:
        participant_id, set_name = split_participant_filename(participant_path)
        value, error = safe_json_load(participant_path)
        base = {
            "participant_id": participant_id,
            "set_name": set_name,
            "language": language,
            "filename": participant_path.name,
            "relative_path": participant_path.relative_to(root).as_posix(),
            "size_bytes": participant_path.stat().st_size,
        }
        if error is not None:
            parse_errors += 1
            participant_rows.append(
                {
                    **base,
                    "parse_status": "error",
                    "top_level_type": "",
                    "top_level_length": "",
                    "top_level_keys": "",
                    "n_candidate_records": 0,
                    "n_labeled_candidate_records": 0,
                    "n_gaze_array_candidates": 0,
                    "n_preliminary_linked_trials": 0,
                    "error": error,
                }
            )
            continue

        parsed_count += 1
        top_type = type(value).__name__
        top_length = len(value) if isinstance(value, (dict, list)) else ""
        top_keys = "|".join(map(str, list(value.keys())[:100])) if isinstance(value, dict) else ""

        gaze_candidates = find_gaze_arrays(value)
        if gaze_candidates:
            participants_with_gaze += 1

        gaze_meta_for_linking: list[dict[str, Any]] = []
        for gaze_index, (path, array, metadata, depth) in enumerate(gaze_candidates):
            row = {
                **base,
                "gaze_candidate_index": gaze_index,
                "json_path": path,
                "depth": depth,
                **metadata,
            }
            gaze_rows.append(row)
            gaze_meta_for_linking.append(row)

        candidate_count = 0
        labeled_count = 0
        linked_count = 0

        for path, node, depth in iter_nodes(value, max_depth=args.max_json_depth):
            # Gather relevant scalar values over the full JSON.
            if isinstance(node, SCALAR_TYPES):
                last_component = path.rsplit(".", 1)[-1]
                if RELEVANT_KEY_PATTERN.search(last_component):
                    relevant_values[path][compact(node, 300)] += 1
                    relevant_sources[path].add(participant_path.name)

            if not isinstance(node, dict):
                continue
            is_candidate, meta = looks_like_candidate_record(path, node)
            if not is_candidate:
                continue

            candidate_count += 1
            items = scalar_items(node, max_depth=3, max_items=220)
            condition = str(meta["condition"])
            if condition in {"NR", "IS"}:
                labeled_count += 1

            trial_id_path, trial_id = extract_named_scalar(items, ID_PATTERNS)
            question_path, question = extract_named_scalar(items, QUESTION_PATTERNS)
            text_path, text_value = extract_named_scalar(items, TEXT_PATTERNS)
            width_path, screen_width = extract_named_scalar(items, SCREEN_WIDTH_PATTERNS)
            height_path, screen_height = extract_named_scalar(items, SCREEN_HEIGHT_PATTERNS)
            rt_path, response_time = extract_named_scalar(items, RT_PATTERNS)

            best_gaze: dict[str, Any] | None = None
            best_score = -1
            for candidate in gaze_meta_for_linking:
                score = nearest_common_prefix(path, str(candidate["json_path"]))
                if score > best_score:
                    best_score = score
                    best_gaze = candidate

            record_index = len(record_rows)
            record_row = {
                **base,
                "record_index_global": record_index,
                "json_path": path,
                "depth": depth,
                "record_keys": "|".join(map(str, list(node.keys())[:150])),
                **meta,
                "trial_id_field": trial_id_path,
                "trial_id_value": trial_id,
                "question_field": question_path,
                "question_preview": question,
                "text_field": text_path,
                "text_preview": text_value,
                "screen_width_field": width_path,
                "screen_width_value": screen_width,
                "screen_height_field": height_path,
                "screen_height_value": screen_height,
                "response_time_field": rt_path,
                "response_time_value": response_time,
                "nearest_gaze_json_path": (
                    best_gaze["json_path"] if best_gaze is not None else ""
                ),
                "nearest_gaze_n_samples": (
                    best_gaze["n_samples"] if best_gaze is not None else ""
                ),
                "nearest_gaze_x_field": (
                    best_gaze["x_field"] if best_gaze is not None else ""
                ),
                "nearest_gaze_y_field": (
                    best_gaze["y_field"] if best_gaze is not None else ""
                ),
                "nearest_gaze_time_field": (
                    best_gaze["time_field"] if best_gaze is not None else ""
                ),
                "record_gaze_common_prefix_chars": max(best_score, 0),
            }
            record_rows.append(record_row)

            if condition in {"NR", "IS"} and best_gaze is not None:
                label_nr = 1 if condition == "NR" else 0
                external_label = (
                    "normal_reading" if condition == "NR"
                    else "information_seeking"
                )
                preliminary_trial_id = (
                    trial_id
                    if trial_id
                    else f"{participant_id}::{set_name}::record_{candidate_count:04d}"
                )
                preliminary_rows.append(
                    {
                        "external_trial_id": preliminary_trial_id,
                        "participant_id": participant_id,
                        "set_name": set_name,
                        "language": language,
                        "external_task": condition,
                        "external_label_name": external_label,
                        "label_nr": label_nr,
                        "source_json": participant_path.relative_to(root).as_posix(),
                        "trial_record_json_path": path,
                        "gaze_json_path": best_gaze["json_path"],
                        "n_gaze_samples_reported": best_gaze["n_samples"],
                        "gaze_container_kind": best_gaze["container_kind"],
                        "gaze_x_field": best_gaze["x_field"],
                        "gaze_y_field": best_gaze["y_field"],
                        "gaze_time_field": best_gaze["time_field"],
                        "condition_evidence": meta["condition_evidence"],
                        "trial_id_source_field": trial_id_path,
                        "question_source_field": question_path,
                        "question_preview": question,
                        "text_source_field": text_path,
                        "text_preview": text_value,
                        "screen_width_source_field": width_path,
                        "screen_width_value": screen_width,
                        "screen_height_source_field": height_path,
                        "screen_height_value": screen_height,
                        "response_time_source_field": rt_path,
                        "response_time_value": response_time,
                        "record_gaze_common_prefix_chars": max(best_score, 0),
                        "manifest_status": "preliminary_requires_schema_confirmation",
                    }
                )
                linked_count += 1

        if labeled_count:
            participants_with_labeled_records += 1

        participant_rows.append(
            {
                **base,
                "parse_status": "ok",
                "top_level_type": top_type,
                "top_level_length": top_length,
                "top_level_keys": top_keys,
                "n_candidate_records": candidate_count,
                "n_labeled_candidate_records": labeled_count,
                "n_gaze_array_candidates": len(gaze_candidates),
                "n_preliminary_linked_trials": linked_count,
                "error": "",
            }
        )

    relevant_rows: list[dict[str, Any]] = []
    for path, counts in sorted(relevant_values.items()):
        for value, count in counts.most_common(args.max_relevant_values_per_field):
            relevant_rows.append(
                {
                    "json_path": path,
                    "value_preview": value,
                    "n_occurrences": count,
                    "n_source_files": len(relevant_sources[path]),
                    "example_source_files": "|".join(sorted(relevant_sources[path])[:8]),
                }
            )

    experiment_rows: list[dict[str, Any]] = []
    experiment_root = root / "scripts" / "experiment_data"
    if experiment_root.is_dir():
        for set_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
            set_language = infer_language(set_dir.name)
            if set_language != language:
                continue
            for file_path in sorted(set_dir.rglob("*")):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() in {".csv", ".tsv"}:
                    experiment_rows.append(
                        csv_schema(
                            file_path,
                            root=root,
                            set_name=set_dir.name,
                            language=set_language,
                        )
                    )

    # Duplicate diagnostics for preliminary trial IDs and participant-condition counts.
    duplicate_counter = Counter(row["external_trial_id"] for row in preliminary_rows)
    for row in preliminary_rows:
        row["external_trial_id_duplicate_count"] = duplicate_counter[row["external_trial_id"]]

    condition_counts = Counter(row["external_task"] for row in preliminary_rows)
    participant_condition_counts: Counter[tuple[str, str]] = Counter(
        (row["participant_id"], row["external_task"]) for row in preliminary_rows
    )
    for row in preliminary_rows:
        row["participant_task_trial_count"] = participant_condition_counts[
            (row["participant_id"], row["external_task"])
        ]

    paths = {
        "participants": output_dir / "english_participant_audit.csv",
        "records": output_dir / "english_candidate_records.csv",
        "gaze_arrays": output_dir / "english_gaze_array_candidates.csv",
        "relevant_values": output_dir / "english_relevant_field_values.csv",
        "experiment_schemas": output_dir / "english_experiment_file_schemas.csv",
        "preliminary_trials": output_dir / "preliminary_external_trial_manifest.csv",
        "manifest": output_dir / "stage1_manifest.json",
    }
    write_csv(paths["participants"], participant_rows)
    write_csv(paths["records"], record_rows)
    write_csv(paths["gaze_arrays"], gaze_rows)
    write_csv(paths["relevant_values"], relevant_rows)
    write_csv(paths["experiment_schemas"], experiment_rows)
    write_csv(paths["preliminary_trials"], preliminary_rows)

    manifest = {
        "stage": "webqamgaze_external_validation_stage1",
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "repository_root": str(root),
        "repository_git_commit": git_commit(root),
        "language": language,
        "purpose": (
            "Full-schema discovery and conservative preliminary trial/gaze "
            "linking before frozen ZuCo model application"
        ),
        "n_participant_json_files": len(participant_paths),
        "n_participant_json_parsed": parsed_count,
        "n_participant_json_parse_errors": parse_errors,
        "n_participants_with_gaze_candidates": participants_with_gaze,
        "n_participants_with_labeled_candidate_records": participants_with_labeled_records,
        "n_candidate_records": len(record_rows),
        "n_gaze_array_candidates": len(gaze_rows),
        "n_preliminary_linked_trials": len(preliminary_rows),
        "preliminary_task_counts": dict(sorted(condition_counts.items())),
        "n_unique_preliminary_trial_ids": len(duplicate_counter),
        "n_duplicate_preliminary_trial_ids": sum(
            count > 1 for count in duplicate_counter.values()
        ),
        "constraints": [
            "No classifier is fitted or applied in stage 1.",
            "No WebQAmGaze label is used for training, threshold selection, feature selection, or preprocessing estimation.",
            "Task mapping is fixed as NR=1 and IS=0.",
            "Ambiguous task records are retained in audit outputs and excluded from the preliminary trial manifest.",
            "The preliminary record-to-gaze links require confirmation before feature extraction.",
            "Frozen ZuCo artifacts remain unchanged.",
        ],
        "outputs": {key: str(path) for key, path in paths.items() if key != "manifest"},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE EXTERNAL VALIDATION STAGE 1 ===")
    print(f"Root: {root}")
    print(f"Git commit: {manifest['repository_git_commit']}")
    print(f"Language: {language}")
    print(f"Participant JSON files: {len(participant_paths)}")
    print(f"Parsed: {parsed_count}; errors: {parse_errors}")
    print(
        "Participants with gaze candidates: "
        f"{participants_with_gaze}/{parsed_count}"
    )
    print(
        "Participants with labeled candidate records: "
        f"{participants_with_labeled_records}/{parsed_count}"
    )
    print(f"Candidate records: {len(record_rows)}")
    print(f"Gaze-array candidates: {len(gaze_rows)}")
    print(f"Preliminary linked trials: {len(preliminary_rows)}")
    print("Preliminary task counts:")
    for task, count in sorted(condition_counts.items()):
        print(f"  {task}: {count}")
    print(f"Duplicate preliminary trial IDs: {manifest['n_duplicate_preliminary_trial_ids']}")
    print("Written:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
