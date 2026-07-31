#!/usr/bin/env python
"""Inspect the official WebQAmGaze participant-Series schema and loading code.

This is a targeted diagnostic after generic nested-table scans failed. The
WebQAmGaze repository documents participant files as pandas Series objects and
ships an official loading notebook. This script:

1. reads every English participant JSON with pandas.read_json(..., typ="series");
2. inventories Series index names, value types, lengths, and schema signatures;
3. deeply inspects one representative participant without dumping full gaze data;
4. extracts relevant code lines from the official loading notebook and scripts;
5. checks experiment_data CSVs for header-only/template files;
6. searches structured and string values for likely gaze/fixation/trial fields.

It does not train, tune, calibrate, or apply any model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd


RELEVANT_CODE = re.compile(
    r"read_json|to_json|pre_processed|fixation_data|webgazer|gaze|fixation|"
    r"trial|question|set_name|worker|participant|screen|resolution|"
    r"target|word_bound|duration|timestamp",
    re.IGNORECASE,
)
RELEVANT_FIELD = re.compile(
    r"gaze|webgazer|fixat|sample|point|coord|timestamp|time|duration|"
    r"trial|task|condition|question|answer|screen|resolution|width|height|"
    r"target|word|sentence|paragraph|passage|text|stimulus|image|"
    r"participant|worker|subject|set_name|correct",
    re.IGNORECASE,
)
GAZE_TOKEN = re.compile(
    r"\b(?:gaze|webgazer|fixation|fixations|x|y|t|timestamp|duration)\b",
    re.IGNORECASE,
)
NUMERIC_KEY = re.compile(r"^-?\d+(?:\.\d+)?$")

SCALAR_TYPES = (str, int, float, bool, type(None))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact(value: Any, limit: int = 300) -> str:
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.replace("\r", " ").replace("\n", " ")).strip()
    else:
        try:
            text = repr(value)
        except Exception:  # noqa: BLE001
            text = f"<{type(value).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def safe_len(value: Any) -> int | str:
    try:
        return len(value)
    except Exception:  # noqa: BLE001
        return ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with path.open("r", encoding=encoding) as handle:
                return json.load(handle)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "could not decode")


def read_series(path: Path) -> tuple[pd.Series | None, str]:
    attempts: list[tuple[str, Any]] = [
        ("pd.read_json_typ_series", lambda: pd.read_json(path, typ="series")),
        (
            "pd.Series_json_load",
            lambda: pd.Series(load_json(path)),
        ),
    ]
    errors: list[str] = []
    for name, loader in attempts:
        try:
            result = loader()
            if isinstance(result, pd.Series):
                return result, name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return None, " || ".join(errors)


def try_decode_string(value: str, max_chars: int = 5_000_000) -> tuple[Any | None, str]:
    text = value.strip()
    if not text or len(text) > max_chars:
        return None, ""
    if text[0] not in "[{(" or text[-1] not in "]})":
        return None, ""
    try:
        return json.loads(text), "json"
    except Exception:  # noqa: BLE001
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list, tuple)):
            return parsed, "python_literal"
    except Exception:  # noqa: BLE001
        pass
    return None, ""


def summarize_value(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "value_type": type(value).__name__,
        "value_length": safe_len(value),
        "preview": compact(value, 500),
        "decoded_type": "",
        "decoded_length": "",
        "decode_method": "",
        "dict_keys": "",
        "list_item_types": "",
        "contains_relevant_token": False,
    }

    current = value
    if isinstance(value, str):
        decoded, method = try_decode_string(value)
        if method:
            current = decoded
            result["decode_method"] = method
            result["decoded_type"] = type(decoded).__name__
            result["decoded_length"] = safe_len(decoded)

    if isinstance(current, dict):
        result["dict_keys"] = "|".join(map(str, list(current.keys())[:100]))
    elif isinstance(current, (list, tuple)):
        type_counts = Counter(type(item).__name__ for item in current[:200])
        result["list_item_types"] = "|".join(
            f"{name}:{count}" for name, count in type_counts.most_common()
        )

    try:
        probe = compact(current, 4000)
        result["contains_relevant_token"] = bool(RELEVANT_FIELD.search(probe))
    except Exception:  # noqa: BLE001
        pass
    return result


def iter_nodes(
    value: Any,
    path: str = "$",
    max_depth: int = 9,
    max_nodes: int = 50_000,
) -> Iterator[tuple[str, Any, int]]:
    stack: list[tuple[str, Any, int]] = [(path, value, 0)]
    seen_nodes = 0
    while stack and seen_nodes < max_nodes:
        current_path, current, depth = stack.pop()
        seen_nodes += 1
        yield current_path, current, depth
        if depth >= max_depth:
            continue

        if isinstance(current, str):
            decoded, method = try_decode_string(current)
            if method:
                stack.append((current_path + "::<decoded>", decoded, depth + 1))
            continue

        if isinstance(current, pd.Series):
            for key, child in reversed(list(current.items())):
                stack.append((f"{current_path}[{key!r}]", child, depth + 1))
        elif isinstance(current, dict):
            for key, child in reversed(list(current.items())):
                stack.append((f"{current_path}[{key!r}]", child, depth + 1))
        elif isinstance(current, (list, tuple)):
            # Cap very large arrays but inspect both beginning and end.
            indexes = list(range(min(len(current), 500)))
            if len(current) > 500:
                indexes += list(range(max(500, len(current) - 20), len(current)))
            for index in reversed(indexes):
                stack.append((f"{current_path}[{index}]", current[index], depth + 1))


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


def inspect_candidate_container(path: str, value: Any) -> dict[str, Any] | None:
    # List of dict records
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        sample = list(value[: min(300, len(value))])
        dicts = [item for item in sample if isinstance(item, dict)]
        if len(dicts) / len(sample) >= 0.7:
            key_counts: Counter[str] = Counter()
            for item in dicts:
                key_counts.update(map(str, item.keys()))
            keys = [key for key, _ in key_counts.most_common()]
            normalized = {re.sub(r"[^a-z0-9]", "", key.lower()): key for key in keys}
            x_key = next(
                (
                    original for normalized_key, original in normalized.items()
                    if normalized_key in {"x", "gx", "gazex", "predictionx", "xpos"}
                    or normalized_key.endswith("gazex")
                ),
                "",
            )
            y_key = next(
                (
                    original for normalized_key, original in normalized.items()
                    if normalized_key in {"y", "gy", "gazey", "predictiony", "ypos"}
                    or normalized_key.endswith("gazey")
                ),
                "",
            )
            t_key = next(
                (
                    original for normalized_key, original in normalized.items()
                    if normalized_key in {
                        "t", "gt", "time", "timestamp", "ts", "elapsedtime",
                        "timems", "sampletime",
                    }
                    or normalized_key.endswith("timestamp")
                ),
                "",
            )
            if x_key and y_key:
                return {
                    "json_path": path,
                    "container_kind": "list_of_dict",
                    "n_rows": len(value),
                    "keys": "|".join(keys[:100]),
                    "x_field": x_key,
                    "y_field": y_key,
                    "time_field": t_key,
                    "x_numeric_fraction": numeric_fraction(
                        item.get(x_key) for item in dicts
                    ),
                    "y_numeric_fraction": numeric_fraction(
                        item.get(y_key) for item in dicts
                    ),
                    "time_numeric_fraction": (
                        numeric_fraction(item.get(t_key) for item in dicts)
                        if t_key else 0.0
                    ),
                }

        sequences = [
            item for item in sample
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        if sequences and len(sequences) / len(sample) >= 0.7:
            first_fraction = numeric_fraction(item[0] for item in sequences)
            second_fraction = numeric_fraction(item[1] for item in sequences)
            third_fraction = numeric_fraction(
                item[2] for item in sequences if len(item) >= 3
            )
            if min(first_fraction, second_fraction) >= 0.7:
                return {
                    "json_path": path,
                    "container_kind": "list_of_numeric_sequences",
                    "n_rows": len(value),
                    "keys": "",
                    "x_field": "[0]",
                    "y_field": "[1]",
                    "time_field": "[2]" if third_fraction >= 0.7 else "",
                    "x_numeric_fraction": first_fraction,
                    "y_numeric_fraction": second_fraction,
                    "time_numeric_fraction": third_fraction,
                }

    # Dict of columns / pandas orient=columns
    if isinstance(value, dict) and len(value) >= 2:
        columns: dict[str, list[Any]] = {}
        for key, child in value.items():
            if isinstance(child, list):
                columns[str(key)] = child
            elif isinstance(child, dict) and child:
                keys = list(child.keys())
                numeric_keys = sum(bool(NUMERIC_KEY.match(str(item))) for item in keys)
                if numeric_keys / len(keys) >= 0.8:
                    def sort_key(item: tuple[Any, Any]) -> tuple[int, Any]:
                        try:
                            return 0, float(item[0])
                        except (TypeError, ValueError):
                            return 1, str(item[0])
                    columns[str(key)] = [
                        item_value
                        for _, item_value in sorted(child.items(), key=sort_key)
                    ]
        if len(columns) >= 2:
            lengths = Counter(len(column) for column in columns.values())
            row_count, matching_columns = lengths.most_common(1)[0]
            if row_count >= 2 and matching_columns >= 2:
                names = list(columns)
                normalized = {
                    re.sub(r"[^a-z0-9]", "", name.lower()): name
                    for name in names
                }
                x_key = next(
                    (
                        original for key, original in normalized.items()
                        if key in {"x", "gx", "gazex", "predictionx", "xpos"}
                        or key.endswith("gazex")
                    ),
                    "",
                )
                y_key = next(
                    (
                        original for key, original in normalized.items()
                        if key in {"y", "gy", "gazey", "predictiony", "ypos"}
                        or key.endswith("gazey")
                    ),
                    "",
                )
                t_key = next(
                    (
                        original for key, original in normalized.items()
                        if key in {
                            "t", "gt", "time", "timestamp", "ts",
                            "elapsedtime", "timems", "sampletime",
                        }
                        or key.endswith("timestamp")
                    ),
                    "",
                )
                if x_key and y_key:
                    return {
                        "json_path": path,
                        "container_kind": "dict_of_columns",
                        "n_rows": row_count,
                        "keys": "|".join(names[:100]),
                        "x_field": x_key,
                        "y_field": y_key,
                        "time_field": t_key,
                        "x_numeric_fraction": numeric_fraction(columns[x_key]),
                        "y_numeric_fraction": numeric_fraction(columns[y_key]),
                        "time_numeric_fraction": (
                            numeric_fraction(columns[t_key]) if t_key else 0.0
                        ),
                    }
    return None


def inspect_official_code(root: Path) -> list[dict[str, Any]]:
    paths = [
        root / "scripts" / "Loading Eye-Tracking Data + Word Boundaries.ipynb",
        root / "scripts" / "study.py",
        root / "scripts" / "experiment.py",
        root / "scripts" / "utils.py",
        root / "scripts" / "DataAnalysis.ipynb",
        root / "scripts" / "DataAnalysisV2.ipynb",
        root / "scripts" / "DataAnalysisV3.ipynb",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "source_file": relative,
                        "cell_index": "",
                        "line_number_in_cell": "",
                        "line": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            for cell_index, cell in enumerate(notebook.get("cells", [])):
                source = cell.get("source", [])
                if isinstance(source, str):
                    source = source.splitlines(keepends=True)
                for line_number, line in enumerate(source, start=1):
                    text = str(line).rstrip("\r\n")
                    if RELEVANT_CODE.search(text):
                        rows.append(
                            {
                                "source_file": relative,
                                "cell_index": cell_index,
                                "line_number_in_cell": line_number,
                                "line": text,
                                "error": "",
                            }
                        )
        else:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = path.read_text(encoding="latin-1").splitlines()
            for line_number, text in enumerate(lines, start=1):
                if RELEVANT_CODE.search(text):
                    rows.append(
                        {
                            "source_file": relative,
                            "cell_index": "",
                            "line_number_in_cell": line_number,
                            "line": text,
                            "error": "",
                        }
                    )
    return rows


def inspect_collection_csvs(root: Path) -> list[dict[str, Any]]:
    experiment_root = root / "scripts" / "experiment_data"
    rows: list[dict[str, Any]] = []
    if not experiment_root.is_dir():
        return rows
    for path in sorted(experiment_root.rglob("*.csv")):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        first_lines: list[str] = []
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for _ in range(3):
                    line = handle.readline()
                    if not line:
                        break
                    first_lines.append(compact(line, 500))
        except Exception as exc:  # noqa: BLE001
            first_lines = [f"{type(exc).__name__}: {exc}"]

        row_count = ""
        column_count = ""
        columns = ""
        parse_error = ""
        try:
            frame = pd.read_csv(path, nrows=10)
            column_count = len(frame.columns)
            columns = "|".join(map(str, frame.columns))
            # Count physical records cheaply enough for these repository files.
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                physical_lines = sum(1 for _ in handle)
            row_count = max(0, physical_lines - 1)
        except Exception as exc:  # noqa: BLE001
            parse_error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "source_file": relative,
                "size_bytes": size,
                "physical_data_rows": row_count,
                "column_count": column_count,
                "columns": columns,
                "first_lines": " || ".join(first_lines),
                "is_git_lfs_pointer": (
                    bool(first_lines)
                    and "git-lfs.github.com/spec/v1" in first_lines[0]
                ),
                "parse_error": parse_error,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage1c"),
    )
    parser.add_argument(
        "--representative-file",
        type=Path,
        default=None,
        help="Optional participant JSON path. Otherwise the largest English file is used.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed = root / "scripts" / "pre_processed_data"
    participant_files = sorted(preprocessed.glob("*_EN_*.json"))
    if not participant_files:
        raise FileNotFoundError("No English participant JSON files found.")

    if args.representative_file is not None:
        representative = args.representative_file.resolve()
    else:
        representative = max(participant_files, key=lambda path: path.stat().st_size)

    series_index_counter: Counter[str] = Counter()
    series_index_type_counter: Counter[tuple[str, str]] = Counter()
    schema_counter: Counter[str] = Counter()
    participant_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []
    read_errors = 0

    representative_series: pd.Series | None = None
    representative_method = ""

    for path in participant_files:
        series, method = read_series(path)
        participant_id = path.stem.split("_mturk_EN_", 1)[0]
        set_match = re.search(r"(mturk_EN_v\d+)", path.stem, re.I)
        set_name = set_match.group(1) if set_match else ""
        if series is None:
            read_errors += 1
            participant_rows.append(
                {
                    "participant_id": participant_id,
                    "set_name": set_name,
                    "source_file": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "read_status": "error",
                    "read_method": method,
                    "series_length": "",
                    "series_index": "",
                    "schema_signature": "",
                    "n_candidate_containers": 0,
                }
            )
            continue

        if path.resolve() == representative.resolve():
            representative_series = series
            representative_method = method

        index_names = [str(index) for index in series.index]
        signature_parts: list[str] = []
        for index, value in series.items():
            index_name = str(index)
            value_type = type(value).__name__
            series_index_counter[index_name] += 1
            series_index_type_counter[(index_name, value_type)] += 1
            signature_parts.append(f"{index_name}:{value_type}")

            if path.resolve() == representative.resolve():
                value_rows.append(
                    {
                        "series_index": index_name,
                        **summarize_value(value),
                    }
                )

        signature = "|".join(signature_parts)
        schema_counter[signature] += 1

        participant_candidate_count = 0
        for json_path, node, depth in iter_nodes(series):
            candidate = inspect_candidate_container(json_path, node)
            if candidate is not None:
                candidate_key = (
                    participant_id,
                    candidate["json_path"],
                    candidate["container_kind"],
                )
                candidate_row = {
                    "participant_id": participant_id,
                    "set_name": set_name,
                    "source_file": path.relative_to(root).as_posix(),
                    "depth": depth,
                    **candidate,
                }
                all_candidate_rows.append(candidate_row)
                participant_candidate_count += 1

        participant_rows.append(
            {
                "participant_id": participant_id,
                "set_name": set_name,
                "source_file": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "read_status": "ok",
                "read_method": method,
                "series_length": len(series),
                "series_index": "|".join(index_names),
                "schema_signature": signature,
                "n_candidate_containers": participant_candidate_count,
            }
        )

    if representative_series is None:
        representative_series, representative_method = read_series(representative)
    if representative_series is None:
        raise RuntimeError(f"Could not read representative file: {representative}")

    index_inventory_rows = [
        {
            "series_index": index_name,
            "value_type": value_type,
            "n_participant_files": count,
        }
        for (index_name, value_type), count
        in series_index_type_counter.most_common()
    ]
    schema_rows = [
        {
            "schema_signature": signature,
            "n_participant_files": count,
        }
        for signature, count in schema_counter.most_common()
    ]

    official_code_rows = inspect_official_code(root)
    collection_rows = inspect_collection_csvs(root)

    raw_text = representative.read_text(encoding="utf-8")
    representative_report = {
        "source_file": representative.relative_to(root).as_posix(),
        "size_bytes": representative.stat().st_size,
        "raw_first_1000_chars": raw_text[:1000],
        "raw_last_1000_chars": raw_text[-1000:],
        "is_git_lfs_pointer": raw_text.startswith(
            "version https://git-lfs.github.com/spec/v1"
        ),
        "read_method": representative_method,
        "series_length": len(representative_series),
        "series_index": [str(index) for index in representative_series.index],
        "series_dtype": str(representative_series.dtype),
        "value_summaries": value_rows,
    }

    paths = {
        "participants": output_dir / "participant_series_audit.csv",
        "index_inventory": output_dir / "participant_series_index_inventory.csv",
        "schemas": output_dir / "participant_series_schema_signatures.csv",
        "representative_values": output_dir / "representative_series_values.csv",
        "candidate_containers": output_dir / "participant_candidate_gaze_containers.csv",
        "official_code": output_dir / "official_loading_code_matches.csv",
        "collection_csvs": output_dir / "experiment_csv_physical_audit.csv",
        "representative_report": output_dir / "representative_participant_report.json",
        "manifest": output_dir / "stage1c_manifest.json",
    }

    write_csv(paths["participants"], participant_rows)
    write_csv(paths["index_inventory"], index_inventory_rows)
    write_csv(paths["schemas"], schema_rows)
    write_csv(paths["representative_values"], value_rows)
    write_csv(paths["candidate_containers"], all_candidate_rows)
    write_csv(paths["official_code"], official_code_rows)
    write_csv(paths["collection_csvs"], collection_rows)
    paths["representative_report"].write_text(
        json.dumps(representative_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    nonempty_csvs = sum(
        bool(row["physical_data_rows"])
        and int(row["physical_data_rows"]) > 0
        for row in collection_rows
        if row["physical_data_rows"] != ""
    )
    lfs_csvs = sum(bool(row["is_git_lfs_pointer"]) for row in collection_rows)
    participant_with_candidates = sum(
        int(row["n_candidate_containers"]) > 0
        for row in participant_rows
        if row["read_status"] == "ok"
    )

    manifest = {
        "stage": "webqamgaze_external_validation_stage1c_official_schema",
        "created_at_utc": utc_now_iso(),
        "repository_root": str(root),
        "n_english_participant_files": len(participant_files),
        "n_read_errors": read_errors,
        "n_unique_series_schema_signatures": len(schema_counter),
        "n_unique_series_index_names": len(series_index_counter),
        "representative_file": str(representative),
        "representative_series_length": len(representative_series),
        "representative_series_index": [
            str(index) for index in representative_series.index
        ],
        "n_candidate_gaze_containers": len(all_candidate_rows),
        "n_participants_with_candidate_containers": participant_with_candidates,
        "n_official_loading_code_matches": len(official_code_rows),
        "n_experiment_csv_files": len(collection_rows),
        "n_nonempty_experiment_csv_files": nonempty_csvs,
        "n_git_lfs_pointer_csv_files": lfs_csvs,
        "constraints": [
            "No model fitting or application.",
            "No external-label-based model or threshold selection.",
            "The purpose is only to identify the repository's official data schema.",
            "Frozen ZuCo artifacts remain unchanged.",
        ],
        "outputs": {
            key: str(path) for key, path in paths.items() if key != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE EXTERNAL VALIDATION STAGE 1C ===")
    print(f"Root: {root}")
    print(f"English participant files: {len(participant_files)}")
    print(f"Read errors: {read_errors}")
    print(f"Unique Series schema signatures: {len(schema_counter)}")
    print(f"Unique Series index names: {len(series_index_counter)}")
    print(
        "Representative file: "
        f"{representative.relative_to(root).as_posix()}"
    )
    print(f"Representative read method: {representative_method}")
    print(f"Representative Series length: {len(representative_series)}")
    print("Representative Series index:")
    for index in representative_series.index:
        value = representative_series.loc[index]
        print(
            f"  {index!s}: type={type(value).__name__}, "
            f"len={safe_len(value)}, preview={compact(value, 180)}"
        )
    print("Most common Series index/type combinations:")
    for row in index_inventory_rows[:30]:
        print(
            f"  {row['series_index']}: {row['value_type']} "
            f"({row['n_participant_files']} files)"
        )
    print(f"Candidate gaze containers: {len(all_candidate_rows)}")
    print(
        "Participants with candidate containers: "
        f"{participant_with_candidates}/{len(participant_files) - read_errors}"
    )
    if all_candidate_rows:
        print("Top candidate container signatures:")
        signature_counts = Counter(
            (
                row["container_kind"],
                row["x_field"],
                row["y_field"],
                row["time_field"],
            )
            for row in all_candidate_rows
        )
        for signature, count in signature_counts.most_common(10):
            print(
                f"  kind={signature[0]}, x={signature[1]}, "
                f"y={signature[2]}, t={signature[3] or '<none>'}, "
                f"count={count}"
            )
    print(f"Official loading-code matches: {len(official_code_rows)}")
    print("First relevant official-code lines:")
    for row in official_code_rows[:40]:
        location = (
            f"cell {row['cell_index']}, line {row['line_number_in_cell']}"
            if row["cell_index"] != ""
            else f"line {row['line_number_in_cell']}"
        )
        print(f"  {row['source_file']} [{location}]: {row['line']}")
    print(f"Experiment CSV files: {len(collection_rows)}")
    print(f"Nonempty experiment CSV files: {nonempty_csvs}")
    print(f"Git LFS pointer CSV files: {lfs_csvs}")
    print("Written:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
