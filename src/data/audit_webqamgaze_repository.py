#!/usr/bin/env python
"""Audit a local WebQAmGaze checkout before frozen-model external validation.

This script deliberately performs no model fitting and no label-dependent tuning.
It inventories the repository, participant JSON files, English subsets, experiment
folders, and candidate fields needed to build a frozen ZuCo -> WebQAmGaze
external-validation table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LANG_PATTERN = re.compile(r"_(EN|DE|ES|TR|GR|RU)_", re.IGNORECASE)
SET_PATTERN = re.compile(
    r"(?P<set_name>(?:mturk|vol(?:unteer)?)_[A-Z]{2}_v\d+)$",
    re.IGNORECASE,
)
CANDIDATE_FIELD_PATTERN = re.compile(
    r"gaze|webgazer|fixat|sacc|sample|timestamp|time|duration|trial|task|condition|"
    r"question|answer|screen|resolution|width|height|target|word|sentence|paragraph|"
    r"\bx\b|\by\b|x_pos|y_pos|xpos|ypos|coord",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
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
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
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
    set_name = match.group("set_name")
    participant = stem[: match.start()].rstrip("_")
    return participant, set_name


def safe_json_load(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                return json.load(handle), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def compact_value(value: Any, max_chars: int = 160) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = repr(value)
    elif isinstance(value, dict):
        text = f"dict(n={len(value)}, keys={list(value)[:8]})"
    elif isinstance(value, list):
        text = f"list(n={len(value)})"
    else:
        text = type(value).__name__
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def recursive_inventory(
    value: Any,
    path: str,
    rows: list[dict[str, Any]],
    source_file: str,
    depth: int = 0,
    max_depth: int = 7,
    max_dict_items: int = 200,
    max_list_items: int = 4,
) -> None:
    row: dict[str, Any] = {
        "source_file": source_file,
        "json_path": path or "$",
        "value_type": type(value).__name__,
        "preview": compact_value(value),
    }
    if isinstance(value, (dict, list)):
        row["container_length"] = len(value)
    else:
        row["container_length"] = ""
    rows.append(row)

    if depth >= max_depth:
        return
    if isinstance(value, dict):
        for index, (key, child) in enumerate(value.items()):
            if index >= max_dict_items:
                rows.append(
                    {
                        "source_file": source_file,
                        "json_path": f"{path or '$'}.__TRUNCATED__",
                        "value_type": "marker",
                        "container_length": len(value) - max_dict_items,
                        "preview": "remaining dictionary items not expanded",
                    }
                )
                break
            child_path = f"{path}.{key}" if path else f"$.{key}"
            recursive_inventory(
                child,
                child_path,
                rows,
                source_file,
                depth + 1,
                max_depth,
                max_dict_items,
                max_list_items,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value[:max_list_items]):
            child_path = f"{path}[{index}]" if path else f"$[{index}]"
            recursive_inventory(
                child,
                child_path,
                rows,
                source_file,
                depth + 1,
                max_depth,
                max_dict_items,
                max_list_items,
            )
        if len(value) > max_list_items:
            rows.append(
                {
                    "source_file": source_file,
                    "json_path": f"{path or '$'}[...TRUNCATED...]",
                    "value_type": "marker",
                    "container_length": len(value) - max_list_items,
                    "preview": "remaining list elements not expanded",
                }
            )


def top_level_summary(value: Any) -> tuple[str, int | str, str]:
    value_type = type(value).__name__
    if isinstance(value, dict):
        return value_type, len(value), "|".join(map(str, list(value.keys())[:80]))
    if isinstance(value, list):
        return value_type, len(value), ""
    return value_type, "", ""


def enumerate_experiment_sets(experiment_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not experiment_root.is_dir():
        return rows
    for set_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        language = infer_language(set_dir.name)
        all_files = [p for p in set_dir.rglob("*") if p.is_file()]
        suffix_counts = Counter((p.suffix.lower() or "[no_suffix]") for p in all_files)
        rows.append(
            {
                "set_name": set_dir.name,
                "language": language,
                "n_files_recursive": len(all_files),
                "n_csv": suffix_counts.get(".csv", 0),
                "n_json": suffix_counts.get(".json", 0),
                "n_html": suffix_counts.get(".html", 0),
                "n_conf": suffix_counts.get(".conf", 0),
                "n_png": suffix_counts.get(".png", 0),
                "n_jpg_jpeg": suffix_counts.get(".jpg", 0) + suffix_counts.get(".jpeg", 0),
                "has_trialdata_csv": any(p.name.lower() == "trialdata.csv" for p in all_files),
                "has_webgazer_sample_csv": any(
                    p.name.lower() == "webgazer-sample-data.csv" for p in all_files
                ),
            }
        )
    return rows


def repository_inventory(root: Path) -> list[dict[str, Any]]:
    important = [
        "README.md",
        "LICENCE",
        "requirements.txt",
        "scripts/experiment_data",
        "scripts/pre_processed_data",
        "scripts/pre_processed_data/fixation_data_per_part",
        "scripts/target_dataframes",
        "scripts/all_resolution.txt",
        "scripts/Loading Eye-Tracking Data + Word Boundaries.ipynb",
    ]
    rows: list[dict[str, Any]] = []
    for relative in important:
        path = root / relative
        if path.is_file():
            kind = "file"
            size = path.stat().st_size
        elif path.is_dir():
            kind = "directory"
            size = ""
        else:
            kind = "missing"
            size = ""
        rows.append(
            {
                "relative_path": relative,
                "status": "present" if path.exists() else "missing",
                "kind": kind,
                "size_bytes": size,
            }
        )
    return rows


def aggregate_schema(schema_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for row in schema_rows:
        key = (str(row["json_path"]), str(row["value_type"]))
        if key not in aggregate:
            aggregate[key] = {
                "json_path": key[0],
                "value_type": key[1],
                "n_sample_files": 0,
                "example_source_file": row["source_file"],
                "example_preview": row["preview"],
            }
        aggregate[key]["n_sample_files"] += 1
    return sorted(aggregate.values(), key=lambda item: (item["json_path"], item["value_type"]))


def candidate_fields(schema_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for row in schema_rows:
        path = str(row["json_path"])
        if CANDIDATE_FIELD_PATTERN.search(path):
            current = candidates.setdefault(
                path,
                {
                    "json_path": path,
                    "observed_types": set(),
                    "n_occurrences": 0,
                    "example_source_file": row["source_file"],
                    "example_preview": row["preview"],
                },
            )
            current["observed_types"].add(str(row["value_type"]))
            current["n_occurrences"] += 1
    output: list[dict[str, Any]] = []
    for path, item in sorted(candidates.items()):
        output.append(
            {
                "json_path": path,
                "observed_types": "|".join(sorted(item["observed_types"])),
                "n_occurrences": item["n_occurrences"],
                "example_source_file": item["example_source_file"],
                "example_preview": item["example_preview"],
            }
        )
    return output


def select_schema_samples(paths: Iterable[Path], n: int) -> list[Path]:
    english = [p for p in paths if infer_language(p.name) == "EN"]
    other = [p for p in paths if infer_language(p.name) != "EN"]
    return (english + other)[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/webqamgaze_repository_audit"),
    )
    parser.add_argument(
        "--schema-sample-files",
        type=int,
        default=12,
        help="Number of participant JSON files to parse recursively for schema discovery.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"WebQAmGaze root not found: {root}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_root = root / "scripts" / "pre_processed_data"
    experiment_root = root / "scripts" / "experiment_data"
    participant_paths = sorted(preprocessed_root.glob("*.json"))

    participant_rows: list[dict[str, Any]] = []
    language_counts: Counter[str] = Counter()
    set_counts: Counter[str] = Counter()

    for path in participant_paths:
        participant_id, set_name = split_participant_filename(path)
        language = infer_language(path.name)
        language_counts[language] += 1
        set_counts[set_name] += 1
        participant_rows.append(
            {
                "participant_id": participant_id,
                "set_name": set_name,
                "language": language,
                "filename": path.name,
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )

    sample_paths = select_schema_samples(participant_paths, args.schema_sample_files)
    schema_rows: list[dict[str, Any]] = []
    sample_parse_rows: list[dict[str, Any]] = []

    for path in sample_paths:
        value, error = safe_json_load(path)
        if error is not None:
            sample_parse_rows.append(
                {
                    "filename": path.name,
                    "language": infer_language(path.name),
                    "parse_status": "error",
                    "top_level_type": "",
                    "top_level_length": "",
                    "top_level_keys": "",
                    "error": error,
                }
            )
            continue
        top_type, top_length, top_keys = top_level_summary(value)
        sample_parse_rows.append(
            {
                "filename": path.name,
                "language": infer_language(path.name),
                "parse_status": "ok",
                "top_level_type": top_type,
                "top_level_length": top_length,
                "top_level_keys": top_keys,
                "error": "",
            }
        )
        recursive_inventory(value, "", schema_rows, path.name)

    experiment_rows = enumerate_experiment_sets(experiment_root)
    repo_rows = repository_inventory(root)
    aggregated_schema_rows = aggregate_schema(schema_rows)
    candidate_rows = candidate_fields(schema_rows)

    language_rows = [
        {"language": language, "n_participant_json_files": count}
        for language, count in sorted(language_counts.items())
    ]
    set_rows = [
        {"set_name": set_name, "language": infer_language(set_name), "n_participant_json_files": count}
        for set_name, count in sorted(set_counts.items())
    ]

    write_csv(output_dir / "repository_inventory.csv", repo_rows)
    write_csv(output_dir / "participant_files.csv", participant_rows)
    write_csv(output_dir / "participant_language_counts.csv", language_rows)
    write_csv(output_dir / "participant_set_counts.csv", set_rows)
    write_csv(output_dir / "experiment_set_inventory.csv", experiment_rows)
    write_csv(output_dir / "sample_json_parse_summary.csv", sample_parse_rows)
    write_csv(output_dir / "sample_json_schema_inventory.csv", aggregated_schema_rows)
    write_csv(output_dir / "candidate_external_validation_fields.csv", candidate_rows)

    manifest = {
        "audit_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "repository_root": str(root),
        "git_commit": git_commit(root),
        "purpose": "Pre-modeling audit for no-retraining ZuCo-to-WebQAmGaze external validation",
        "n_participant_json_files": len(participant_paths),
        "participant_json_by_language": dict(sorted(language_counts.items())),
        "n_experiment_set_directories": len(experiment_rows),
        "n_schema_sample_files_requested": int(args.schema_sample_files),
        "n_schema_sample_files_selected": len(sample_paths),
        "n_schema_sample_files_parsed": sum(row["parse_status"] == "ok" for row in sample_parse_rows),
        "constraints": [
            "No WebQAmGaze labels are used for training or hyperparameter tuning.",
            "Frozen ZuCo model artifacts must not be modified.",
            "Primary comparison uses only the matched Common16 five-condition bundle.",
            "The all18 Compact model is secondary and reported separately.",
            "Official sent_gaze_sacc transfer requires an explicitly documented event extractor and identical feature definitions/order.",
        ],
        "outputs": {
            "repository_inventory": "repository_inventory.csv",
            "participant_files": "participant_files.csv",
            "participant_language_counts": "participant_language_counts.csv",
            "participant_set_counts": "participant_set_counts.csv",
            "experiment_set_inventory": "experiment_set_inventory.csv",
            "sample_json_parse_summary": "sample_json_parse_summary.csv",
            "sample_json_schema_inventory": "sample_json_schema_inventory.csv",
            "candidate_external_validation_fields": "candidate_external_validation_fields.csv",
        },
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE REPOSITORY AUDIT ===")
    print(f"Root: {root}")
    print(f"Git commit: {manifest['git_commit']}")
    print(f"Participant JSON files: {len(participant_paths)}")
    print("By language:")
    for language, count in sorted(language_counts.items()):
        print(f"  {language}: {count}")
    print(f"Experiment set directories: {len(experiment_rows)}")
    print(
        "Schema samples parsed: "
        f"{manifest['n_schema_sample_files_parsed']}/"
        f"{manifest['n_schema_sample_files_selected']}"
    )
    print(f"Candidate field paths: {len(candidate_rows)}")
    print("Written:")
    for name in manifest["outputs"].values():
        print(f"  {output_dir / name}")
    print(f"  {output_dir / 'audit_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
