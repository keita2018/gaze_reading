from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


LANGUAGE_ALIASES = {
    "EN": {"en", "eng", "english"},
    "TR": {"tr", "tur", "turkish", "türkçe", "turkce"},
    "ES": {"es", "spa", "spanish", "español", "espanol"},
    "DE": {"de", "deu", "ger", "german", "deutsch"},
    "GR": {"gr", "el", "ell", "greek", "ελληνικά"},
    "RU": {"ru", "rus", "russian", "русский"},
}

LANGUAGE_KEYS = {
    "language", "lang", "language_code", "locale", "experiment_language",
}
PARTICIPANT_KEYS = {
    "participant_id", "participant", "subject_id", "subject",
    "worker_id", "workerid", "worker", "prolific_pid",
    "assignment_id", "assignmentid", "user_id", "userid",
}
TASK_KEYS = {
    "task", "condition", "reading_task", "reading_goal",
    "experiment", "trial_type", "task_name", "label",
}
NR_TOKENS = {
    "nr", "normal reading", "natural reading", "ordinary",
    "ordinary reading", "normal_reading", "natural_reading",
}
IS_TOKENS = {
    "is", "information seeking", "information-seeking",
    "information_seeking", "task specific reading", "task-specific reading",
    "tsr",
}
QUALITY_KEY_TOKENS = {
    "approved", "approval", "reload", "fixation_error", "target_error",
    "calibration", "validation", "accuracy", "precision",
}
GAZE_KEY_TOKENS = {
    "gaze", "webgazer", "x", "y", "fixation", "timestamp",
}
AOI_KEY_TOKENS = {
    "target", "aoi", "boundary", "boundaries", "rectangle",
    "bbox", "word_box", "paragraph",
}

DEFAULT_SCAN_ROOTS = [
    "data/raw/webqamgaze",
    "external/WebQAmGaze/scripts/experiment_data",
    "external/WebQAmGaze/scripts/pre_processed_data",
    "external/WebQAmGaze",
]


@dataclass
class RecordAudit:
    path: str
    bytes: int
    sha256: str
    parse_ok: bool
    detected_language: str
    language_evidence: str
    participant_id: str
    participant_evidence: str
    has_nr_token: bool
    has_is_token: bool
    has_both_tasks: bool
    gaze_like_array_count: int
    gaze_like_object_count: int
    aoi_like_key_count: int
    quality_key_count: int
    top_level_type: str
    top_level_keys: str
    likely_participant_record: bool
    parse_error: str = ""


def norm(value: Any) -> str:
    return str(value).strip().lower()


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flatten(
    obj: Any,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 12,
) -> Iterable[tuple[str, Any]]:
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield child, value
            yield from flatten(value, child, depth + 1, max_depth)
    elif isinstance(obj, list):
        # Scan all small lists, and samples from large lists to avoid huge audits.
        if len(obj) <= 100:
            indices = range(len(obj))
        else:
            indices = list(range(min(25, len(obj))))
            tail_start = max(25, len(obj) - 5)
            indices = list(indices) + list(range(tail_start, len(obj)))
        for index in indices:
            child = f"{prefix}[{index}]"
            value = obj[index]
            yield child, value
            yield from flatten(value, child, depth + 1, max_depth)


def token_to_language(value: Any) -> str:
    token = norm(value)
    token = token.replace("_", " ").replace("-", " ")
    for code, aliases in LANGUAGE_ALIASES.items():
        if token in aliases:
            return code
    return ""


def detect_language(data: Any, path: Path) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []

    for key_path, value in flatten(data):
        last_key = norm(key_path.rsplit(".", 1)[-1].split("[", 1)[0])
        if last_key in LANGUAGE_KEYS and isinstance(value, (str, int)):
            code = token_to_language(value)
            if code:
                candidates.append((code, f"{key_path}={value!r}"))

    # Strong path token evidence.
    path_tokens = [
        token for token in re.split(r"[^A-Za-zА-Яа-яΑ-Ωα-ω]+", str(path))
        if token
    ]
    for token in reversed(path_tokens):
        code = token_to_language(token)
        if code:
            candidates.append((code, f"path_token={token!r}"))
            break

    # Filename prefix/suffix such as EN_123, participant_TR, etc.
    stem = path.stem
    for code in LANGUAGE_ALIASES:
        if re.search(rf"(^|[_\-.]){re.escape(code)}([_\-.]|$)", stem, re.I):
            candidates.append((code, f"filename_code={code}"))
            break

    if not candidates:
        return "", ""

    counts: dict[str, int] = {}
    for code, _ in candidates:
        counts[code] = counts.get(code, 0) + 1
    selected = sorted(counts, key=lambda code: (-counts[code], code))[0]
    evidence = " | ".join(
        evidence for code, evidence in candidates if code == selected
    )
    return selected, evidence[:1000]


def detect_participant(data: Any, path: Path) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for key_path, value in flatten(data):
        last_key = norm(key_path.rsplit(".", 1)[-1].split("[", 1)[0])
        if last_key in PARTICIPANT_KEYS and isinstance(value, (str, int, float)):
            token = str(value).strip()
            if token and token.lower() not in {"nan", "none", "null"}:
                candidates.append((token, f"{key_path}={value!r}"))

    if candidates:
        # Prefer the most frequently occurring candidate, then shortest non-empty ID.
        counts: dict[str, int] = {}
        for value, _ in candidates:
            counts[value] = counts.get(value, 0) + 1
        selected = sorted(
            counts, key=lambda value: (-counts[value], len(value), value)
        )[0]
        evidence = " | ".join(
            evidence for value, evidence in candidates if value == selected
        )
        return selected, evidence[:1000]

    return path.stem, "fallback=file_stem"


def detect_tasks(data: Any) -> tuple[bool, bool]:
    has_nr = False
    has_is = False

    def inspect_scalar(value: Any) -> None:
        nonlocal has_nr, has_is
        if not isinstance(value, (str, int, float, bool)):
            return
        token = norm(value).replace("_", " ")
        if token in NR_TOKENS or any(
            phrase in token for phrase in ["normal reading", "natural reading"]
        ):
            has_nr = True
        if token in IS_TOKENS or any(
            phrase in token for phrase in [
                "information seeking", "task specific reading"
            ]
        ):
            has_is = True

    for key_path, value in flatten(data):
        last_key = norm(key_path.rsplit(".", 1)[-1].split("[", 1)[0])
        if last_key in TASK_KEYS:
            inspect_scalar(value)
        elif isinstance(value, str) and len(value) <= 100:
            inspect_scalar(value)

    return has_nr, has_is


def structure_counts(data: Any) -> tuple[int, int, int, int]:
    gaze_arrays = 0
    gaze_objects = 0
    aoi_keys = 0
    quality_keys = 0

    for key_path, value in flatten(data):
        last_key = norm(key_path.rsplit(".", 1)[-1].split("[", 1)[0])
        key_lower = norm(key_path)

        if any(token in last_key for token in AOI_KEY_TOKENS):
            aoi_keys += 1
        if any(token in key_lower for token in QUALITY_KEY_TOKENS):
            quality_keys += 1

        if isinstance(value, list) and len(value) >= 5:
            if any(token in last_key for token in ["gaze", "webgazer", "fixation"]):
                gaze_arrays += 1

        if isinstance(value, dict):
            keys = {norm(k) for k in value}
            has_x = any(k in keys for k in {"x", "xpos", "gaze_x", "clientx"})
            has_y = any(k in keys for k in {"y", "ypos", "gaze_y", "clienty"})
            has_time = any(
                k in keys for k in {"t", "time", "timestamp", "elapsed_time"}
            )
            if has_x and has_y and (has_time or any("gaze" in k for k in keys)):
                gaze_objects += 1

    return gaze_arrays, gaze_objects, aoi_keys, quality_keys


def inspect_json(path: Path) -> RecordAudit:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        language, language_evidence = detect_language(data, path)
        participant, participant_evidence = detect_participant(data, path)
        has_nr, has_is = detect_tasks(data)
        gaze_arrays, gaze_objects, aoi_keys, quality_keys = structure_counts(data)

        if isinstance(data, dict):
            top_keys = "|".join(map(str, list(data.keys())[:100]))
        else:
            top_keys = ""

        likely = bool(
            language
            and (
                has_nr
                or has_is
                or gaze_arrays > 0
                or gaze_objects > 0
                or quality_keys > 0
            )
        )

        return RecordAudit(
            path=str(path),
            bytes=path.stat().st_size,
            sha256=sha256(path),
            parse_ok=True,
            detected_language=language,
            language_evidence=language_evidence,
            participant_id=participant,
            participant_evidence=participant_evidence,
            has_nr_token=has_nr,
            has_is_token=has_is,
            has_both_tasks=has_nr and has_is,
            gaze_like_array_count=gaze_arrays,
            gaze_like_object_count=gaze_objects,
            aoi_like_key_count=aoi_keys,
            quality_key_count=quality_keys,
            top_level_type=type(data).__name__,
            top_level_keys=top_keys,
            likely_participant_record=likely,
        )
    except Exception as exc:
        return RecordAudit(
            path=str(path),
            bytes=path.stat().st_size,
            sha256=sha256(path),
            parse_ok=False,
            detected_language="",
            language_evidence="",
            participant_id=path.stem,
            participant_evidence="fallback=file_stem",
            has_nr_token=False,
            has_is_token=False,
            has_both_tasks=False,
            gaze_like_array_count=0,
            gaze_like_object_count=0,
            aoi_like_key_count=0,
            quality_key_count=0,
            top_level_type="",
            top_level_keys="",
            likely_participant_record=False,
            parse_error=f"{type(exc).__name__}: {exc}",
        )


def discover_json(project_root: Path, explicit_roots: list[str]) -> list[Path]:
    found: dict[str, Path] = {}
    for rel in explicit_roots:
        root = project_root / rel
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if not path.is_file():
                continue
            lower_parts = {p.lower() for p in path.parts}
            if ".git" in lower_parts or "__pycache__" in lower_parts:
                continue
            found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda p: str(p).lower())


def audit_stage2_script(project_root: Path) -> dict[str, Any]:
    path = project_root / "src/data/prepare_webqamgaze_external_validation_stage2.py"
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "argparse_options": [],
        "language_literals": [],
        "interesting_string_literals": [],
        "sha256": "",
    }
    if not path.exists():
        return result

    result["sha256"] = sha256(path)
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
        options: set[str] = set()
        literals: set[str] = set()
        language_literals: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_argument":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if arg.value.startswith("-"):
                                options.add(arg.value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                lower = value.lower()
                if value.upper() in LANGUAGE_ALIASES:
                    language_literals.add(value)
                if any(token in lower for token in [
                    "language", "approved", "reload", "fixation",
                    "target_error", "participant", "eligible",
                ]):
                    literals.add(value[:300])
        result["argparse_options"] = sorted(options)
        result["language_literals"] = sorted(language_literals)
        result["interesting_string_literals"] = sorted(literals)[:200]
    except Exception as exc:
        result["ast_error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[RecordAudit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language in sorted(LANGUAGE_ALIASES):
        subset = [r for r in records if r.detected_language == language]
        likely = [r for r in subset if r.likely_participant_record]
        participant_ids = {
            r.participant_id for r in likely if r.participant_id
        }
        parse_errors = sum(not r.parse_ok for r in subset)
        both = sum(r.has_both_tasks for r in likely)
        gaze = sum(
            (r.gaze_like_array_count > 0 or r.gaze_like_object_count > 0)
            for r in likely
        )
        aoi = sum(r.aoi_like_key_count > 0 for r in likely)
        quality = sum(r.quality_key_count > 0 for r in likely)

        # This is a repository-structure gate, not final feature eligibility.
        structure_pass = (
            len(participant_ids) >= 40
            and len(likely) >= 40
            and parse_errors == 0
            and (both >= 1 or gaze >= 40)
        )

        rows.append({
            "language": language,
            "json_files_detected": len(subset),
            "likely_participant_records": len(likely),
            "unique_participant_ids": len(participant_ids),
            "records_with_both_task_tokens": both,
            "records_with_gaze_structure": gaze,
            "records_with_aoi_structure": aoi,
            "records_with_quality_keys": quality,
            "parse_errors": parse_errors,
            "minimum_40_participants": len(participant_ids) >= 40,
            "repository_structure_pass": structure_pass,
            "final_feature_eligibility_status": "NOT_YET_EVALUATED",
        })
    return rows


def select_language(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row for row in summary_rows
        if row["language"] != "EN" and row["repository_structure_pass"]
    ]
    candidates.sort(
        key=lambda row: (
            -int(row["unique_participant_ids"]),
            -int(row["likely_participant_records"]),
            row["language"],
        )
    )
    selected = candidates[0]["language"] if candidates else ""
    return {
        "selection_stage": "repository_structure_only",
        "selected_language": selected,
        "selection_rule": [
            "exclude EN",
            "repository_structure_pass == true",
            "largest unique_participant_ids",
            "largest likely_participant_records",
            "alphabetical language code",
        ],
        "important_limitation": (
            "Final confirmatory selection remains conditional on applying the "
            "existing English eligibility and Full-41 feature-extraction audits. "
            "No model performance has been inspected."
        ),
        "candidate_order": [row["language"] for row in candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit unused WebQAmGaze languages without loading a model or "
            "calculating target performance."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/iui_p0_unused_language_audit_v1",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        dest="scan_roots",
        help="Relative repository/data root. Repeatable.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_roots = args.scan_roots or DEFAULT_SCAN_ROOTS
    paths = discover_json(project_root, scan_roots)
    print(f"JSON files discovered: {len(paths)}")

    records: list[RecordAudit] = []
    for index, path in enumerate(paths, start=1):
        records.append(inspect_json(path))
        if index % 100 == 0:
            print(f"Audited {index}/{len(paths)} JSON files...")

    record_rows = [asdict(record) for record in records]
    write_csv(output_dir / "participant_record_audit.csv", record_rows)

    summary_rows = summarize(records)
    write_csv(output_dir / "candidate_language_audit.csv", summary_rows)

    selected = select_language(summary_rows)
    (output_dir / "selected_language.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stage2 = audit_stage2_script(project_root)
    (output_dir / "stage2_script_audit.json").write_text(
        json.dumps(stage2, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    examples: dict[str, list[dict[str, Any]]] = {}
    for language in LANGUAGE_ALIASES:
        subset = [
            record for record in records
            if record.detected_language == language
            and record.likely_participant_record
        ]
        subset.sort(
            key=lambda record: (
                -record.gaze_like_array_count,
                -record.gaze_like_object_count,
                record.path,
            )
        )
        examples[language] = [
            {
                "path": record.path,
                "participant_id": record.participant_id,
                "language_evidence": record.language_evidence,
                "has_both_tasks": record.has_both_tasks,
                "gaze_like_array_count": record.gaze_like_array_count,
                "gaze_like_object_count": record.gaze_like_object_count,
                "top_level_keys": record.top_level_keys,
            }
            for record in subset[:5]
        ]
    (output_dir / "schema_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    release = {
        "artifact_version": "iui_p0_unused_language_audit_v1",
        "project_root": str(project_root),
        "scan_roots": scan_roots,
        "n_json_files": len(paths),
        "selected_language_stage1": selected["selected_language"],
        "model_loaded": False,
        "performance_calculated": False,
        "label_performance_inspected": False,
        "stage2_script_sha256": stage2.get("sha256", ""),
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    table_lines = [
        "| language | JSON | likely records | unique participants | both tasks | "
        "gaze structure | AOI structure | structure pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        table_lines.append(
            f"| {row['language']} | {row['json_files_detected']} | "
            f"{row['likely_participant_records']} | "
            f"{row['unique_participant_ids']} | "
            f"{row['records_with_both_task_tokens']} | "
            f"{row['records_with_gaze_structure']} | "
            f"{row['records_with_aoi_structure']} | "
            f"{row['repository_structure_pass']} |"
        )

    summary = f"""# IUI P0-3 Unused-Language Repository Audit

This audit did **not** load the frozen model, generate predictions, or calculate any
target metric.

## Language inventory

{chr(10).join(table_lines)}

## Stage-1 selected language

`{selected['selected_language'] or 'NONE'}`

Selection at this stage is based only on repository structure and participant-record
counts. Final selection remains conditional on reproducing the existing English
eligibility filters and extracting the exact frozen Full 41 schema.

## Existing Stage-2 script audit

- Exists: `{stage2.get('exists')}`
- Path: `{stage2.get('path')}`
- CLI options: `{stage2.get('argparse_options')}`
- Language literals: `{stage2.get('language_literals')}`

See `stage2_script_audit.json` for details.

## Required files for the next step

Share:

- `candidate_language_audit.csv`
- `selected_language.json`
- `stage2_script_audit.json`
- `schema_examples.json`
- `P0_UNUSED_LANGUAGE_AUDIT_SUMMARY.md`

No target performance should be opened before the exact language-specific eligibility
and feature-extraction pipeline is frozen and its unlabeled prediction table is hashed.
"""
    (output_dir / "P0_UNUSED_LANGUAGE_AUDIT_SUMMARY.md").write_text(
        summary,
        encoding="utf-8",
    )

    print("")
    print("Unused-language repository audit completed.")
    print(f"Output: {output_dir}")
    print(
        "Stage-1 selected language: "
        f"{selected['selected_language'] or 'NONE'}"
    )
    print("No model was loaded and no target performance was calculated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
