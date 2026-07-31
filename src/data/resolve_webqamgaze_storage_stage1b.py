#!/usr/bin/env python
"""Resolve the actual WebQAmGaze English gaze-data storage schema.

This is a diagnostic/recovery stage after a generic JSON-array scan found no
gaze arrays. WebQAmGaze stores data in more than one representation:

1. participant-level pre_processed_data/*.json files, often serialized from
   pandas objects (dict-of-columns / dict-of-index structures);
2. experiment_data/**/trialdata.csv files, whose cells may contain nested JSON
   payloads exported by psiTurk/jsPsych.

The script inspects both locations, decodes nested JSON/Python-literal strings,
detects list-of-record and pandas-style columnar gaze tables, and reports the
actual trial/task/gaze schemas. It does not train, tune, calibrate, or apply any
model.

Outputs
-------
participant_json_structure.csv
participant_json_table_candidates.csv
participant_json_key_inventory.csv
english_collection_files.csv
trialdata_file_summary.csv
trialdata_column_inventory.csv
trialdata_payload_summary.csv
raw_gaze_trial_candidates.csv
condition_value_inventory.csv
representative_payloads.json
stage1b_manifest.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd


LANG_PATTERN = re.compile(r"(?:^|_)(EN)(?:_|$)", re.IGNORECASE)
SET_PATTERN = re.compile(r"(mturk_EN_v\d+)", re.IGNORECASE)
NUMERIC_KEY_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")

X_ALIASES = {
    "x", "gx", "gazex", "gaze_x", "predictionx", "prediction_x",
    "xpos", "x_pos", "xposition", "x_position", "clientx", "client_x",
    "screenx", "screen_x", "rawx", "raw_x",
}
Y_ALIASES = {
    "y", "gy", "gazey", "gaze_y", "predictiony", "prediction_y",
    "ypos", "y_pos", "yposition", "y_position", "clienty", "client_y",
    "screeny", "screen_y", "rawy", "raw_y",
}
T_ALIASES = {
    "t", "gt", "time", "timestamp", "ts", "elapsedtime", "elapsed_time",
    "timems", "time_ms", "timesec", "time_sec", "clock", "sampletime",
    "sample_time",
}
GAZE_NAME_PATTERN = re.compile(
    r"webgazer|gaze|eye.?track|prediction|sample.?data|raw.?point",
    re.IGNORECASE,
)
TASK_KEY_PATTERN = re.compile(
    r"task|condition|paradigm|reading.?type|trial.?type|phase|block",
    re.IGNORECASE,
)
TRIAL_KEY_PATTERN = re.compile(
    r"trial|question|passage|paragraph|article|stimulus|image|text|item",
    re.IGNORECASE,
)
PARTICIPANT_KEY_PATTERN = re.compile(
    r"participant|subject|worker|uniqueid|workerid|prolific|assignment",
    re.IGNORECASE,
)
SCREEN_KEY_PATTERN = re.compile(
    r"screen|resolution|viewport|window|width|height",
    re.IGNORECASE,
)

NR_PATTERNS = [
    re.compile(r"(^|[^a-z])nr([^a-z]|$)", re.I),
    re.compile(r"normal[\s_\-]*read", re.I),
    re.compile(r"natural[\s_\-]*read", re.I),
]
IS_PATTERNS = [
    re.compile(r"(^|[^a-z])is([^a-z]|$)", re.I),
    re.compile(r"information[\s_\-]*seek", re.I),
    re.compile(r"target(?:ed)?[\s_\-]*read", re.I),
    re.compile(r"search[\s_\-]*(?:read|task)", re.I),
]

SCALAR_TYPES = (str, int, float, bool, type(None))


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


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def compact(value: Any, max_chars: int = 250) -> str:
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.replace("\r", " ").replace("\n", " ")).strip()
    elif isinstance(value, SCALAR_TYPES):
        text = repr(value)
    elif isinstance(value, dict):
        text = f"dict(n={len(value)}, keys={list(value)[:12]})"
    elif isinstance(value, list):
        text = f"list(n={len(value)})"
    else:
        text = type(value).__name__
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def stable_signature(columns: Sequence[Any]) -> str:
    return "|".join(str(column) for column in columns)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_path(parent: str, key: Any, index: bool = False) -> str:
    if index:
        return f"{parent}[{key}]"
    text = str(key)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return f"{parent}.{text}"
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'{parent}["{escaped}"]'


def iter_nodes(value: Any, path: str = "$", max_depth: int = 12) -> Iterator[tuple[str, Any, int]]:
    stack: list[tuple[str, Any, int]] = [(path, value, 0)]
    while stack:
        current_path, current, depth = stack.pop()
        yield current_path, current, depth
        if depth >= max_depth:
            continue
        if isinstance(current, dict):
            for key, child in reversed(list(current.items())):
                stack.append((json_path(current_path, key), child, depth + 1))
        elif isinstance(current, list):
            for index in range(len(current) - 1, -1, -1):
                stack.append((json_path(current_path, index, index=True), current[index], depth + 1))


def safe_json_load(path: Path) -> tuple[Any | None, str | None]:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with path.open("r", encoding=encoding) as handle:
                return json.load(handle), None
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
    return None, "unable to decode as UTF-8"


def looks_encoded(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    pairs = {("{", "}"), ("[", "]"), ("(", ")")}
    return (stripped[0], stripped[-1]) in pairs


def decode_string_payload(text: str, max_chars: int) -> tuple[Any | None, str]:
    stripped = text.strip()
    if len(stripped) > max_chars or not looks_encoded(stripped):
        return None, ""
    try:
        return json.loads(stripped), "json"
    except Exception:  # noqa: BLE001
        pass
    try:
        value = ast.literal_eval(stripped)
        if isinstance(value, (dict, list, tuple)):
            return value, "python_literal"
    except Exception:  # noqa: BLE001
        pass
    return None, ""


def recursively_decode(
    value: Any,
    *,
    max_string_chars: int,
    max_depth: int = 5,
) -> tuple[Any, list[dict[str, Any]]]:
    decoded_rows: list[dict[str, Any]] = []

    def visit(current: Any, path: str, depth: int) -> Any:
        if depth > max_depth:
            return current
        if isinstance(current, str):
            decoded, method = decode_string_payload(current, max_string_chars)
            if method:
                decoded_rows.append(
                    {
                        "json_path": path,
                        "decode_method": method,
                        "encoded_length": len(current),
                        "decoded_type": type(decoded).__name__,
                        "decoded_length": (
                            len(decoded) if isinstance(decoded, (dict, list, tuple)) else ""
                        ),
                        "encoded_preview": compact(current, 300),
                    }
                )
                return visit(decoded, path + "::<decoded>", depth + 1)
            return current
        if isinstance(current, dict):
            return {
                key: visit(child, json_path(path, key), depth + 1)
                for key, child in current.items()
            }
        if isinstance(current, list):
            return [
                visit(child, json_path(path, index, index=True), depth + 1)
                for index, child in enumerate(current)
            ]
        if isinstance(current, tuple):
            return [
                visit(child, json_path(path, index, index=True), depth + 1)
                for index, child in enumerate(current)
            ]
        return current

    return visit(value, "$", 0), decoded_rows


def numeric_fraction(values: Iterable[Any]) -> float:
    total = 0
    finite = 0
    for value in values:
        total += 1
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
            if math.isfinite(number):
                finite += 1
        except (TypeError, ValueError):
            pass
    return finite / total if total else 0.0


def sequence_from_column(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        # Pandas default orient commonly uses row-index keys.
        if not value:
            return []
        keys = list(value)
        numeric_like = sum(bool(NUMERIC_KEY_PATTERN.match(str(key))) for key in keys)
        if numeric_like / len(keys) >= 0.8:
            def sort_key(item: tuple[Any, Any]) -> tuple[int, float | str]:
                key = str(item[0])
                try:
                    return 0, float(key)
                except ValueError:
                    return 1, key
            return [child for _, child in sorted(value.items(), key=sort_key)]
    return None


def choose_axis(columns: Sequence[Any], aliases: set[str], axis: str) -> str:
    mapping = {normalize_key(column): str(column) for column in columns}
    for alias in aliases:
        normalized = normalize_key(alias)
        if normalized in mapping:
            return mapping[normalized]
    for column in columns:
        normalized = normalize_key(column)
        if axis == "x" and normalized.endswith("x") and (
            "gaze" in normalized or "webgazer" in normalized or "prediction" in normalized
        ):
            return str(column)
        if axis == "y" and normalized.endswith("y") and (
            "gaze" in normalized or "webgazer" in normalized or "prediction" in normalized
        ):
            return str(column)
        if axis == "t" and (
            normalized.endswith("time") or normalized.endswith("timestamp")
        ):
            return str(column)
    return ""


def rows_from_list_of_records(value: Any) -> tuple[list[dict[str, Any]] | None, str]:
    if not isinstance(value, list) or len(value) < 2:
        return None, ""
    records = [item for item in value if isinstance(item, dict)]
    if len(records) / len(value) < 0.7:
        return None, ""
    return records, "list_of_records"


def rows_from_dict_of_records(value: Any) -> tuple[list[dict[str, Any]] | None, str]:
    if not isinstance(value, dict) or len(value) < 2:
        return None, ""
    items = list(value.items())
    if sum(isinstance(child, dict) for _, child in items) / len(items) < 0.8:
        return None, ""
    # To avoid confusing arbitrary nested objects with a table, require mostly
    # numeric/index-like row keys or strongly overlapping child keys.
    numeric_keys = sum(bool(NUMERIC_KEY_PATTERN.match(str(key))) for key, _ in items)
    child_key_sets = [set(map(str, child.keys())) for _, child in items if isinstance(child, dict)]
    common = set.intersection(*child_key_sets[: min(20, len(child_key_sets))]) if child_key_sets else set()
    if numeric_keys / len(items) < 0.7 and len(common) < 2:
        return None, ""
    return [dict(child) for _, child in items if isinstance(child, dict)], "dict_of_records"


def rows_from_dict_of_columns(value: Any) -> tuple[list[dict[str, Any]] | None, str]:
    if not isinstance(value, dict) or len(value) < 2:
        return None, ""
    sequences: dict[str, list[Any]] = {}
    for key, child in value.items():
        sequence = sequence_from_column(child)
        if sequence is not None:
            sequences[str(key)] = sequence
    if len(sequences) < 2:
        return None, ""
    lengths = Counter(len(sequence) for sequence in sequences.values())
    n_rows, count = lengths.most_common(1)[0]
    if n_rows < 2 or count < 2:
        return None, ""
    selected = {
        key: sequence for key, sequence in sequences.items()
        if len(sequence) == n_rows
    }
    rows = [
        {key: sequence[index] for key, sequence in selected.items()}
        for index in range(n_rows)
    ]
    return rows, "dict_of_columns"


def rows_from_split_orient(value: Any) -> tuple[list[dict[str, Any]] | None, str]:
    if not isinstance(value, dict):
        return None, ""
    columns = value.get("columns")
    data = value.get("data")
    if not isinstance(columns, list) or not isinstance(data, list) or len(data) < 2:
        return None, ""
    if not all(isinstance(row, list) for row in data[: min(20, len(data))]):
        return None, ""
    rows = [
        {str(column): row[index] if index < len(row) else None
         for index, column in enumerate(columns)}
        for row in data
    ]
    return rows, "pandas_split"


def detect_tables(value: Any, max_depth: int = 10) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    converters = (
        rows_from_split_orient,
        rows_from_list_of_records,
        rows_from_dict_of_columns,
        rows_from_dict_of_records,
    )
    for path, node, depth in iter_nodes(value, max_depth=max_depth):
        for converter in converters:
            rows, orient = converter(node)
            if rows is None:
                continue
            key = (path, orient)
            if key in seen:
                continue
            seen.add(key)
            sample = rows[: min(200, len(rows))]
            column_counter: Counter[str] = Counter()
            for row in sample:
                column_counter.update(map(str, row.keys()))
            columns = [
                column for column, _ in column_counter.most_common()
            ]
            x_field = choose_axis(columns, X_ALIASES, "x")
            y_field = choose_axis(columns, Y_ALIASES, "y")
            t_field = choose_axis(columns, T_ALIASES, "t")
            gaze_name_signal = bool(GAZE_NAME_PATTERN.search(path + " " + " ".join(columns)))
            x_fraction = (
                numeric_fraction(row.get(x_field) for row in sample)
                if x_field else 0.0
            )
            y_fraction = (
                numeric_fraction(row.get(y_field) for row in sample)
                if y_field else 0.0
            )
            t_fraction = (
                numeric_fraction(row.get(t_field) for row in sample)
                if t_field else 0.0
            )
            is_gaze = bool(
                x_field and y_field
                and min(x_fraction, y_fraction) >= 0.5
                and (gaze_name_signal or t_field)
            )
            output.append(
                {
                    "json_path": path,
                    "depth": depth,
                    "orientation": orient,
                    "n_rows": len(rows),
                    "n_columns": len(columns),
                    "columns": stable_signature(columns[:150]),
                    "x_field": x_field,
                    "y_field": y_field,
                    "time_field": t_field,
                    "x_numeric_fraction": x_fraction,
                    "y_numeric_fraction": y_fraction,
                    "time_numeric_fraction": t_fraction,
                    "gaze_name_signal": gaze_name_signal,
                    "is_gaze_candidate": is_gaze,
                    "_rows": rows,
                }
            )
            break
    return output


def infer_condition(pairs: Sequence[tuple[str, Any]]) -> tuple[str, str, bool]:
    nr: list[str] = []
    is_values: list[str] = []
    for key, value in pairs:
        combined = f"{key}={value}"
        for pattern in NR_PATTERNS:
            if pattern.search(combined):
                nr.append(compact(combined, 160))
                break
        for pattern in IS_PATTERNS:
            if pattern.search(combined):
                is_values.append(compact(combined, 160))
                break
    if nr and not is_values:
        return "NR", " | ".join(dict.fromkeys(nr))[:500], False
    if is_values and not nr:
        return "IS", " | ".join(dict.fromkeys(is_values))[:500], False
    if nr and is_values:
        return (
            "AMBIGUOUS",
            "NR: " + " | ".join(dict.fromkeys(nr))[:220]
            + " || IS: " + " | ".join(dict.fromkeys(is_values))[:220],
            True,
        )
    return "", "", False


def scalar_pairs(value: Any, max_depth: int = 5, max_items: int = 300) -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    for path, node, _ in iter_nodes(value, max_depth=max_depth):
        if isinstance(node, SCALAR_TYPES):
            output.append((path, node))
            if len(output) >= max_items:
                break
    return output


def extract_metadata_preview(value: Any) -> dict[str, str]:
    pairs = scalar_pairs(value, max_depth=5, max_items=500)
    selected: dict[str, str] = {}
    for path, node in pairs:
        last = path.rsplit(".", 1)[-1]
        if (
            TASK_KEY_PATTERN.search(last)
            or TRIAL_KEY_PATTERN.search(last)
            or PARTICIPANT_KEY_PATTERN.search(last)
            or SCREEN_KEY_PATTERN.search(last)
        ):
            selected[path] = compact(node, 300)
        if len(selected) >= 60:
            break
    return selected


def parse_nested_payload(value: Any, max_chars: int) -> tuple[Any | None, str]:
    if isinstance(value, (dict, list)):
        return value, "already_structured"
    if not isinstance(value, str):
        return None, ""
    return decode_string_payload(value, max_chars)


def read_csv_flexible(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    attempts = [
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "utf-8", "sep": None, "engine": "python"},
        {"encoding": "latin-1", "sep": None, "engine": "python"},
    ]
    errors: list[str] = []
    for kwargs in attempts:
        try:
            return pd.read_csv(path, low_memory=False, **kwargs), None
        except TypeError:
            # low_memory is unsupported with some Python-engine paths.
            try:
                return pd.read_csv(path, **kwargs), None
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    return None, " || ".join(errors)


def likely_payload_columns(frame: pd.DataFrame) -> list[str]:
    priority: list[tuple[float, str]] = []
    for column in frame.columns:
        series = frame[column]
        sample = series.dropna().astype(str).head(100)
        if sample.empty:
            continue
        starts = sample.str.strip().str[:1].isin(["{", "[", "("]).mean()
        name_signal = bool(
            re.search(
                r"trialdata|datastring|data|payload|json|response",
                str(column),
                re.I,
            )
        )
        average_length = sample.str.len().mean()
        score = float(starts) * 3.0 + float(name_signal) * 2.0
        if average_length > 100:
            score += 1.0
        if score >= 1.5:
            priority.append((score, str(column)))
    return [column for _, column in sorted(priority, reverse=True)]


def participant_id_from_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    match = SET_PATTERN.search(stem)
    if not match:
        return stem, ""
    participant = stem[: match.start()].rstrip("_")
    return participant, match.group(1)


def collection_set(path: Path) -> str:
    for part in path.parts:
        match = SET_PATTERN.fullmatch(part)
        if match:
            return match.group(1)
    match = SET_PATTERN.search(path.as_posix())
    return match.group(1) if match else ""


def condition_inventory_from_pairs(
    pairs: Sequence[tuple[str, Any]],
    *,
    source_type: str,
    source_file: str,
    participant_id: str,
    accumulator: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    for path, value in pairs:
        if not TASK_KEY_PATTERN.search(path):
            continue
        key = (source_type, path, compact(value, 300))
        row = accumulator.setdefault(
            key,
            {
                "source_type": source_type,
                "field_path": path,
                "value_preview": compact(value, 300),
                "n_occurrences": 0,
                "source_files": set(),
                "participant_ids": set(),
            },
        )
        row["n_occurrences"] += 1
        row["source_files"].add(source_file)
        if participant_id:
            row["participant_ids"].add(participant_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage1b"),
    )
    parser.add_argument("--language", default="EN")
    parser.add_argument(
        "--max-decode-string-chars",
        type=int,
        default=20_000_000,
    )
    parser.add_argument(
        "--max-representative-payloads",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    language = args.language.upper()
    if language != "EN":
        raise ValueError("This recovery script currently targets the English subset.")

    preprocessed_root = root / "scripts" / "pre_processed_data"
    experiment_root = root / "scripts" / "experiment_data"
    if not preprocessed_root.is_dir():
        raise FileNotFoundError(preprocessed_root)
    if not experiment_root.is_dir():
        raise FileNotFoundError(experiment_root)

    participant_files = sorted(
        path for path in preprocessed_root.glob("*_EN_*.json")
        if path.is_file()
    )
    known_participants: set[str] = set()
    for path in participant_files:
        participant_id, _ = participant_id_from_filename(path)
        known_participants.add(participant_id)

    participant_structure_rows: list[dict[str, Any]] = []
    participant_table_rows: list[dict[str, Any]] = []
    participant_key_counter: Counter[tuple[str, str]] = Counter()
    decoded_string_rows: list[dict[str, Any]] = []
    condition_inventory: dict[tuple[str, str, str], dict[str, Any]] = {}
    representative_payloads: list[dict[str, Any]] = []

    participant_gaze_candidates = 0
    participant_files_with_gaze = 0

    for participant_path in participant_files:
        participant_id, set_name = participant_id_from_filename(participant_path)
        value, error = safe_json_load(participant_path)
        base = {
            "participant_id": participant_id,
            "set_name": set_name,
            "source_file": participant_path.relative_to(root).as_posix(),
            "size_bytes": participant_path.stat().st_size,
        }
        if error:
            participant_structure_rows.append(
                {
                    **base,
                    "parse_status": "error",
                    "top_level_type": "",
                    "top_level_length": "",
                    "top_level_keys": "",
                    "n_decoded_string_payloads": 0,
                    "n_table_candidates": 0,
                    "n_gaze_table_candidates": 0,
                    "error": error,
                }
            )
            continue

        decoded, decoded_rows = recursively_decode(
            value,
            max_string_chars=args.max_decode_string_chars,
        )
        for row in decoded_rows:
            decoded_string_rows.append({**base, **row})

        if isinstance(value, dict):
            for key, child in value.items():
                participant_key_counter[(str(key), type(child).__name__)] += 1

        tables = detect_tables(decoded)
        gaze_tables = [table for table in tables if table["is_gaze_candidate"]]
        participant_gaze_candidates += len(gaze_tables)
        if gaze_tables:
            participant_files_with_gaze += 1

        for table_index, table in enumerate(tables):
            table_row = {
                **base,
                "table_index": table_index,
                **{key: value for key, value in table.items() if key != "_rows"},
            }
            participant_table_rows.append(table_row)

            rows = table["_rows"]
            for row_index, record in enumerate(rows[: min(1000, len(rows))]):
                condition_inventory_from_pairs(
                    [(str(key), value) for key, value in record.items()],
                    source_type="participant_table",
                    source_file=base["source_file"],
                    participant_id=participant_id,
                    accumulator=condition_inventory,
                )

        condition_inventory_from_pairs(
            scalar_pairs(decoded, max_depth=8, max_items=3000),
            source_type="participant_json",
            source_file=base["source_file"],
            participant_id=participant_id,
            accumulator=condition_inventory,
        )

        if len(representative_payloads) < args.max_representative_payloads:
            representative_payloads.append(
                {
                    "source_type": "participant_json",
                    "participant_id": participant_id,
                    "set_name": set_name,
                    "source_file": base["source_file"],
                    "top_level_type": type(value).__name__,
                    "top_level_keys": list(value.keys())[:100] if isinstance(value, dict) else [],
                    "decoded_string_payloads": decoded_rows[:20],
                    "table_candidates": [
                        {key: val for key, val in table.items() if key != "_rows"}
                        for table in tables[:20]
                    ],
                    "metadata_preview": extract_metadata_preview(decoded),
                }
            )

        participant_structure_rows.append(
            {
                **base,
                "parse_status": "ok",
                "top_level_type": type(value).__name__,
                "top_level_length": (
                    len(value) if isinstance(value, (dict, list)) else ""
                ),
                "top_level_keys": (
                    stable_signature(list(value.keys())[:150])
                    if isinstance(value, dict) else ""
                ),
                "n_decoded_string_payloads": len(decoded_rows),
                "n_table_candidates": len(tables),
                "n_gaze_table_candidates": len(gaze_tables),
                "gaze_table_paths": "|".join(
                    str(table["json_path"]) for table in gaze_tables[:30]
                ),
                "error": "",
            }
        )

    participant_key_rows = [
        {
            "top_level_key": key,
            "value_type": value_type,
            "n_participant_files": count,
        }
        for (key, value_type), count in participant_key_counter.most_common()
    ]

    # Inventory English experiment-set files and prioritize raw collection CSVs.
    english_set_dirs = sorted(
        path for path in experiment_root.iterdir()
        if path.is_dir() and SET_PATTERN.fullmatch(path.name)
    )
    collection_rows: list[dict[str, Any]] = []
    raw_csv_paths: list[Path] = []
    for set_dir in english_set_dirs:
        for path in sorted(set_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            filename_lower = path.name.lower()
            is_collection_csv = path.suffix.lower() == ".csv" and (
                filename_lower == "trialdata.csv"
                or "trialdata" in filename_lower
                or "data" in filename_lower
            )
            collection_rows.append(
                {
                    "set_name": set_dir.name,
                    "source_file": relative,
                    "filename": path.name,
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "is_trialdata_named": "trialdata" in filename_lower,
                    "is_collection_csv_candidate": is_collection_csv,
                }
            )
            if is_collection_csv:
                raw_csv_paths.append(path)

    trial_file_rows: list[dict[str, Any]] = []
    trial_column_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    raw_gaze_rows: list[dict[str, Any]] = []

    total_csv_rows = 0
    total_decoded_payloads = 0
    raw_gaze_candidate_count = 0
    raw_gaze_participants: set[str] = set()
    raw_gaze_signatures: Counter[tuple[str, str, str, str]] = Counter()

    for csv_path in sorted(set(raw_csv_paths)):
        frame, error = read_csv_flexible(csv_path)
        relative = csv_path.relative_to(root).as_posix()
        set_name = collection_set(csv_path)
        if frame is None:
            trial_file_rows.append(
                {
                    "set_name": set_name,
                    "source_file": relative,
                    "parse_status": "error",
                    "n_rows": "",
                    "n_columns": "",
                    "columns": "",
                    "payload_columns": "",
                    "n_decoded_payloads": 0,
                    "n_gaze_payloads": 0,
                    "error": error,
                }
            )
            continue

        total_csv_rows += len(frame)
        payload_columns = likely_payload_columns(frame)
        for column in frame.columns:
            series = frame[column]
            nonmissing = int(series.notna().sum())
            examples = [compact(value, 220) for value in series.dropna().head(3).tolist()]
            trial_column_rows.append(
                {
                    "set_name": set_name,
                    "source_file": relative,
                    "column": str(column),
                    "dtype": str(series.dtype),
                    "n_rows": len(frame),
                    "n_nonmissing": nonmissing,
                    "n_unique_capped": int(series.nunique(dropna=True))
                    if nonmissing <= 100_000 else "",
                    "example_values": " || ".join(examples),
                    "is_payload_candidate": str(column) in payload_columns,
                }
            )

        file_payload_count = 0
        file_gaze_count = 0

        participant_columns = [
            str(column) for column in frame.columns
            if PARTICIPANT_KEY_PATTERN.search(str(column))
        ]

        for row_index, source_row in frame.iterrows():
            participant_value = ""
            participant_source_column = ""
            for column in participant_columns:
                raw_value = source_row.get(column)
                if pd.notna(raw_value):
                    participant_value = str(raw_value)
                    participant_source_column = column
                    break

            matched_participant = ""
            if participant_value:
                for known in known_participants:
                    if known in participant_value:
                        matched_participant = known
                        break
                if not matched_participant:
                    prefix = participant_value.split(":", 1)[0]
                    if prefix in known_participants:
                        matched_participant = prefix
            participant_id = matched_participant or participant_value

            for column in payload_columns:
                raw_value = source_row.get(column)
                if pd.isna(raw_value):
                    continue
                payload, decode_method = parse_nested_payload(
                    raw_value,
                    args.max_decode_string_chars,
                )
                if payload is None:
                    continue

                decoded_payload, nested_decoded = recursively_decode(
                    payload,
                    max_string_chars=args.max_decode_string_chars,
                    max_depth=5,
                )
                total_decoded_payloads += 1
                file_payload_count += 1

                pairs = scalar_pairs(decoded_payload, max_depth=7, max_items=3000)
                condition, condition_evidence, condition_conflict = infer_condition(pairs)
                metadata = extract_metadata_preview(decoded_payload)
                tables = detect_tables(decoded_payload, max_depth=12)
                gaze_tables = [table for table in tables if table["is_gaze_candidate"]]

                condition_inventory_from_pairs(
                    pairs,
                    source_type="trialdata_payload",
                    source_file=relative,
                    participant_id=participant_id,
                    accumulator=condition_inventory,
                )

                payload_row_id = len(payload_rows)
                payload_rows.append(
                    {
                        "payload_row_id": payload_row_id,
                        "set_name": set_name,
                        "source_file": relative,
                        "csv_row_index": int(row_index),
                        "payload_column": column,
                        "decode_method": decode_method,
                        "participant_id": participant_id,
                        "participant_source_column": participant_source_column,
                        "matched_preprocessed_participant": matched_participant,
                        "payload_type": type(decoded_payload).__name__,
                        "payload_length": (
                            len(decoded_payload)
                            if isinstance(decoded_payload, (dict, list)) else ""
                        ),
                        "n_nested_decoded_strings": len(nested_decoded),
                        "n_table_candidates": len(tables),
                        "n_gaze_table_candidates": len(gaze_tables),
                        "condition_inferred": condition,
                        "condition_evidence": condition_evidence,
                        "condition_conflict": condition_conflict,
                        "metadata_preview_json": json.dumps(
                            metadata, ensure_ascii=False
                        ),
                        "top_level_keys": (
                            stable_signature(list(decoded_payload.keys())[:150])
                            if isinstance(decoded_payload, dict) else ""
                        ),
                    }
                )

                if gaze_tables:
                    file_gaze_count += 1
                    if participant_id:
                        raw_gaze_participants.add(participant_id)

                for gaze_index, table in enumerate(gaze_tables):
                    raw_gaze_candidate_count += 1
                    raw_gaze_signatures[
                        (
                            str(table["orientation"]),
                            str(table["x_field"]),
                            str(table["y_field"]),
                            str(table["time_field"]),
                        )
                    ] += 1
                    table_rows = table["_rows"]
                    x_values = [row.get(table["x_field"]) for row in table_rows[:500]]
                    y_values = [row.get(table["y_field"]) for row in table_rows[:500]]
                    t_values = (
                        [row.get(table["time_field"]) for row in table_rows[:500]]
                        if table["time_field"] else []
                    )

                    def finite_range(values: list[Any]) -> tuple[Any, Any]:
                        numbers: list[float] = []
                        for value in values:
                            try:
                                number = float(value)
                                if math.isfinite(number):
                                    numbers.append(number)
                            except (TypeError, ValueError):
                                pass
                        return (
                            min(numbers) if numbers else "",
                            max(numbers) if numbers else "",
                        )

                    x_min, x_max = finite_range(x_values)
                    y_min, y_max = finite_range(y_values)
                    t_min, t_max = finite_range(t_values)

                    raw_gaze_rows.append(
                        {
                            "payload_row_id": payload_row_id,
                            "gaze_candidate_index": gaze_index,
                            "set_name": set_name,
                            "source_file": relative,
                            "csv_row_index": int(row_index),
                            "payload_column": column,
                            "participant_id": participant_id,
                            "matched_preprocessed_participant": matched_participant,
                            "condition_inferred": condition,
                            "condition_evidence": condition_evidence,
                            **{
                                key: value for key, value in table.items()
                                if key != "_rows"
                            },
                            "x_min_first500": x_min,
                            "x_max_first500": x_max,
                            "y_min_first500": y_min,
                            "y_max_first500": y_max,
                            "time_min_first500": t_min,
                            "time_max_first500": t_max,
                            "metadata_preview_json": json.dumps(
                                metadata, ensure_ascii=False
                            ),
                        }
                    )

                if (
                    len(representative_payloads)
                    < args.max_representative_payloads
                    and (gaze_tables or condition or len(representative_payloads) < 5)
                ):
                    representative_payloads.append(
                        {
                            "source_type": "trialdata_payload",
                            "set_name": set_name,
                            "source_file": relative,
                            "csv_row_index": int(row_index),
                            "payload_column": column,
                            "participant_id": participant_id,
                            "decode_method": decode_method,
                            "top_level_type": type(decoded_payload).__name__,
                            "top_level_keys": (
                                list(decoded_payload.keys())[:150]
                                if isinstance(decoded_payload, dict) else []
                            ),
                            "condition_inferred": condition,
                            "condition_evidence": condition_evidence,
                            "metadata_preview": metadata,
                            "nested_decoded_strings": nested_decoded[:30],
                            "table_candidates": [
                                {
                                    key: value
                                    for key, value in table.items()
                                    if key != "_rows"
                                }
                                for table in tables[:30]
                            ],
                        }
                    )

        trial_file_rows.append(
            {
                "set_name": set_name,
                "source_file": relative,
                "parse_status": "ok",
                "n_rows": len(frame),
                "n_columns": len(frame.columns),
                "columns": stable_signature(frame.columns),
                "payload_columns": stable_signature(payload_columns),
                "n_decoded_payloads": file_payload_count,
                "n_gaze_payloads": file_gaze_count,
                "error": "",
            }
        )

    condition_rows: list[dict[str, Any]] = []
    for row in condition_inventory.values():
        condition_rows.append(
            {
                **{
                    key: value for key, value in row.items()
                    if key not in {"source_files", "participant_ids"}
                },
                "n_source_files": len(row["source_files"]),
                "example_source_files": "|".join(sorted(row["source_files"])[:8]),
                "n_participant_ids": len(row["participant_ids"]),
                "example_participant_ids": "|".join(sorted(row["participant_ids"])[:8]),
            }
        )
    condition_rows.sort(
        key=lambda row: (
            str(row["source_type"]),
            str(row["field_path"]),
            -int(row["n_occurrences"]),
        )
    )

    paths = {
        "participant_structure": output_dir / "participant_json_structure.csv",
        "participant_tables": output_dir / "participant_json_table_candidates.csv",
        "participant_keys": output_dir / "participant_json_key_inventory.csv",
        "decoded_strings": output_dir / "decoded_string_payloads.csv",
        "collection_files": output_dir / "english_collection_files.csv",
        "trialdata_files": output_dir / "trialdata_file_summary.csv",
        "trialdata_columns": output_dir / "trialdata_column_inventory.csv",
        "payloads": output_dir / "trialdata_payload_summary.csv",
        "raw_gaze": output_dir / "raw_gaze_trial_candidates.csv",
        "conditions": output_dir / "condition_value_inventory.csv",
        "representatives": output_dir / "representative_payloads.json",
        "manifest": output_dir / "stage1b_manifest.json",
    }

    write_csv(paths["participant_structure"], participant_structure_rows)
    write_csv(paths["participant_tables"], participant_table_rows)
    write_csv(paths["participant_keys"], participant_key_rows)
    write_csv(paths["decoded_strings"], decoded_string_rows)
    write_csv(paths["collection_files"], collection_rows)
    write_csv(paths["trialdata_files"], trial_file_rows)
    write_csv(paths["trialdata_columns"], trial_column_rows)
    write_csv(paths["payloads"], payload_rows)
    write_csv(paths["raw_gaze"], raw_gaze_rows)
    write_csv(paths["conditions"], condition_rows)
    paths["representatives"].write_text(
        json.dumps(representative_payloads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    signature_rows = [
        {
            "orientation": orientation,
            "x_field": x_field,
            "y_field": y_field,
            "time_field": time_field,
            "count": count,
        }
        for (orientation, x_field, y_field, time_field), count
        in raw_gaze_signatures.most_common()
    ]

    manifest = {
        "stage": "webqamgaze_external_validation_stage1b_storage_resolution",
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "repository_root": str(root),
        "repository_git_commit": git_commit(root),
        "language": language,
        "n_preprocessed_participant_json_files": len(participant_files),
        "n_participant_files_with_gaze_tables": participant_files_with_gaze,
        "n_participant_json_gaze_table_candidates": participant_gaze_candidates,
        "n_english_experiment_set_directories": len(english_set_dirs),
        "n_collection_csv_candidates": len(set(raw_csv_paths)),
        "n_total_collection_csv_rows": total_csv_rows,
        "n_decoded_trialdata_payloads": total_decoded_payloads,
        "n_raw_gaze_trial_candidates": raw_gaze_candidate_count,
        "n_unique_raw_gaze_participant_ids": len(raw_gaze_participants),
        "raw_gaze_signatures": signature_rows,
        "constraints": [
            "No model was fitted or applied.",
            "No external label was used for tuning, feature selection, preprocessing estimation, or threshold calibration.",
            "NR remains mapped to 1 and IS remains mapped to 0 only after explicit schema confirmation.",
            "Detected structures are candidates and must be confirmed before final trial extraction.",
            "Frozen ZuCo model artifacts remain unchanged.",
        ],
        "outputs": {
            key: str(path) for key, path in paths.items() if key != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE EXTERNAL VALIDATION STAGE 1B ===")
    print(f"Root: {root}")
    print(f"Git commit: {manifest['repository_git_commit']}")
    print(f"Participant JSON files: {len(participant_files)}")
    print(
        "Participant JSON files with gaze tables: "
        f"{participant_files_with_gaze}/{len(participant_files)}"
    )
    print(
        "Participant-JSON gaze table candidates: "
        f"{participant_gaze_candidates}"
    )
    print(f"English experiment set directories: {len(english_set_dirs)}")
    print(f"Collection CSV candidates: {len(set(raw_csv_paths))}")
    print(f"Collection CSV rows: {total_csv_rows}")
    print(f"Decoded trialdata payloads: {total_decoded_payloads}")
    print(f"Raw gaze trial candidates: {raw_gaze_candidate_count}")
    print(
        "Unique raw-gaze participant identifiers: "
        f"{len(raw_gaze_participants)}"
    )
    print("Top raw-gaze signatures:")
    if signature_rows:
        for row in signature_rows[:10]:
            print(
                "  "
                f"{row['orientation']}: "
                f"x={row['x_field']}, y={row['y_field']}, "
                f"t={row['time_field'] or '<none>'}, "
                f"count={row['count']}"
            )
    else:
        print("  <none>")
    print("Written:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
