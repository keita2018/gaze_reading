from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


QUALITY_FEATURES = [
    "valid_rate",
    "active_rate",
    "word_box_hit_rate",
    "sentence_envelope_hit_rate",
    "qmax_mean",
    "qmax_median",
    "qmax_q10",
    "qmax_ge_0_5_rate",
    "nearest_distance_mean_word_heights",
    "nearest_distance_median_word_heights",
    "nearest_distance_q90_word_heights",
    "nearest_distance_q95_word_heights",
]

PARTICIPANT_ALIASES = [
    "participant_id", "participant", "subject_id", "subject",
    "worker_id", "worker", "reader_id", "reader",
]
LABEL_ALIASES = [
    "label", "y", "target", "class", "reading_goal",
    "task_label", "goal_label", "condition_label",
]
TRIAL_ALIASES = [
    "trial_id", "trial", "trial_index", "sample_id",
    "item_id", "record_id",
]
SCORE_ALIASES = [
    "score", "decision_score", "decision_function", "raw_score",
    "frozen_score", "qcn_score", "calibrated_score",
]
PREFERRED_ROOTS = [
    "outputs/frozen_onestop_external_validation_v1",
    "outputs/linear_feature_group_factorization_v1",
    "outputs/webqamgaze_external_validation_stage3c",
    "outputs/webqamgaze_external_validation_final_release_v1",
    "outputs/onestop_internal_validation_v1",
    "outputs/onestop_soft_aoi_asc344",
    "outputs/onestop_to_webqamgaze_zero_shot_v1",
    "outputs/webqamgaze_participant_self_calibration_v1",
    "outputs/final_factorized_selective_prediction_v1",
    "data/processed/onestop/soft_aoi_asc344",
    "data/processed/webqamgaze",
]


@dataclass
class TableCandidate:
    path: str
    kind: str
    bytes: int
    n_columns: int
    columns: str
    quality_feature_hits: int
    participant_columns: str
    label_columns: str
    trial_columns: str
    score_columns: str
    likely_domain: str
    candidate_role: str
    rank_score: int
    inspection_error: str = ""


@dataclass
class ModelCandidate:
    path: str
    kind: str
    bytes: int
    object_type: str
    dict_keys: str
    pipeline_steps: str
    feature_names_count: int
    feature_names_preview: str
    coef_shape: str
    classes: str
    threshold_values: str
    rank_score: int
    inspection_error: str = ""


@dataclass
class JsonCandidate:
    path: str
    bytes: int
    interesting_keys: str
    feature_names_count: int
    threshold_values: str
    reference_values: str
    rank_score: int
    inspection_error: str = ""


def normalize(name: str) -> str:
    return str(name).strip().lower()


def find_aliases(columns: Iterable[str], aliases: list[str]) -> list[str]:
    lower_map = {normalize(c): str(c) for c in columns}
    hits: list[str] = []
    for alias in aliases:
        if alias in lower_map:
            hits.append(lower_map[alias])
    return hits


def infer_domain(path: Path) -> str:
    text = str(path).lower()
    if "webqamgaze" in text:
        return "webqamgaze"
    if "onestop" in text:
        return "onestop"
    if "zuco" in text:
        return "zuco"
    return "unknown"


def inspect_parquet(path: Path) -> tuple[list[str], str]:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return list(pf.schema_arrow.names), ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def inspect_csv_header(path: Path) -> tuple[list[str], str]:
    encodings = ["utf-8-sig", "utf-8", "cp932", "latin-1"]
    last = ""
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                return next(reader), ""
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
    return [], last


def inspect_table(path: Path) -> TableCandidate:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        columns, error = inspect_parquet(path)
        kind = "parquet"
    else:
        columns, error = inspect_csv_header(path)
        kind = "csv"

    lower = {normalize(c) for c in columns}
    quality_hits = [f for f in QUALITY_FEATURES if f in lower]
    participant = find_aliases(columns, PARTICIPANT_ALIASES)
    label = find_aliases(columns, LABEL_ALIASES)
    trial = find_aliases(columns, TRIAL_ALIASES)
    score = find_aliases(columns, SCORE_ALIASES)
    domain = infer_domain(path)

    role = []
    rank = 0
    if quality_hits:
        rank += 5 * len(quality_hits)
    if len(columns) >= 41:
        rank += 10
    if participant:
        rank += 15
    if label:
        rank += 15
    if trial:
        rank += 5
    if score:
        rank += 10
    if domain == "onestop" and quality_hits and label and participant:
        role.append("source_feature_table")
        rank += 35
    if domain == "webqamgaze" and quality_hits and participant:
        role.append("target_feature_table")
        rank += 35
    if domain == "webqamgaze" and score and participant:
        role.append("target_score_table")
        rank += 25
    if "prediction" in str(path).lower() and participant:
        role.append("prediction_table")
        rank += 15

    return TableCandidate(
        path=str(path),
        kind=kind,
        bytes=path.stat().st_size,
        n_columns=len(columns),
        columns="|".join(columns),
        quality_feature_hits=len(quality_hits),
        participant_columns="|".join(participant),
        label_columns="|".join(label),
        trial_columns="|".join(trial),
        score_columns="|".join(score),
        likely_domain=domain,
        candidate_role="|".join(role) if role else "other",
        rank_score=rank,
        inspection_error=error,
    )


def flatten_keys(obj: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    if depth > 5:
        return []
    rows: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.append((name, value))
            rows.extend(flatten_keys(value, name, depth + 1))
    elif isinstance(obj, (list, tuple)) and len(obj) <= 100:
        for i, value in enumerate(obj):
            name = f"{prefix}[{i}]"
            rows.extend(flatten_keys(value, name, depth + 1))
    return rows


def compact_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    if hasattr(value, "shape"):
        return f"<{type(value).__name__} shape={getattr(value, 'shape', None)}>"
    return f"<{type(value).__name__}>"


def inspect_model(path: Path) -> ModelCandidate:
    try:
        import joblib
        obj = joblib.load(path)

        object_type = f"{type(obj).__module__}.{type(obj).__name__}"
        dict_keys = ""
        steps = ""
        feature_names: list[str] = []
        coef_shape = ""
        classes = ""
        thresholds: list[str] = []

        if isinstance(obj, dict):
            dict_keys = "|".join(sorted(map(str, obj.keys())))
            flattened = flatten_keys(obj)
            for key, value in flattened:
                k = key.lower()
                if "threshold" in k and isinstance(value, (int, float)):
                    thresholds.append(f"{key}={value}")
                if ("feature" in k and "name" in k) and isinstance(value, (list, tuple)):
                    feature_names = list(map(str, value))

        candidate_objects = [obj]
        if isinstance(obj, dict):
            candidate_objects.extend(v for v in obj.values() if hasattr(v, "__dict__"))

        for candidate in candidate_objects:
            if hasattr(candidate, "steps"):
                try:
                    steps = "|".join(name for name, _ in candidate.steps)
                    candidate_objects.extend(step for _, step in candidate.steps)
                except Exception:
                    pass
            if hasattr(candidate, "feature_names_in_"):
                try:
                    feature_names = list(map(str, candidate.feature_names_in_))
                except Exception:
                    pass
            if hasattr(candidate, "coef_"):
                try:
                    coef_shape = str(tuple(candidate.coef_.shape))
                except Exception:
                    pass
            if hasattr(candidate, "classes_"):
                try:
                    classes = repr(candidate.classes_.tolist())
                except Exception:
                    classes = repr(candidate.classes_)

        rank = 0
        text = str(path).lower()
        if "onestop" in text:
            rank += 20
        if "frozen" in text:
            rank += 20
        if "full" in text or len(feature_names) == 41 or coef_shape.endswith("41)"):
            rank += 30
        if coef_shape:
            rank += 10
        if thresholds:
            rank += 10
        if feature_names:
            rank += 10

        return ModelCandidate(
            path=str(path),
            kind=path.suffix.lower().lstrip("."),
            bytes=path.stat().st_size,
            object_type=object_type,
            dict_keys=dict_keys,
            pipeline_steps=steps,
            feature_names_count=len(feature_names),
            feature_names_preview="|".join(feature_names[:15]),
            coef_shape=coef_shape,
            classes=classes,
            threshold_values="|".join(thresholds[:20]),
            rank_score=rank,
        )
    except Exception as exc:
        return ModelCandidate(
            path=str(path),
            kind=path.suffix.lower().lstrip("."),
            bytes=path.stat().st_size,
            object_type="",
            dict_keys="",
            pipeline_steps="",
            feature_names_count=0,
            feature_names_preview="",
            coef_shape="",
            classes="",
            threshold_values="",
            rank_score=0,
            inspection_error=f"{type(exc).__name__}: {exc}",
        )


def inspect_json(path: Path) -> JsonCandidate:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        flattened = flatten_keys(data)
        interesting: list[str] = []
        thresholds: list[str] = []
        references: list[str] = []
        feature_names_count = 0

        for key, value in flattened:
            k = key.lower()
            if any(token in k for token in [
                "threshold", "feature", "reference", "median", "quality",
                "participant", "class", "label", "schema", "score"
            ]):
                interesting.append(key)
            if "threshold" in k and isinstance(value, (int, float)):
                thresholds.append(f"{key}={value}")
            if any(token in k for token in ["reference", "source_median", "quality_median"]) \
                    and isinstance(value, (int, float)):
                references.append(f"{key}={value}")
            if ("feature" in k and "name" in k) and isinstance(value, list):
                feature_names_count = max(feature_names_count, len(value))

        rank = min(len(interesting), 30)
        if thresholds:
            rank += 20
        if references:
            rank += 20
        if feature_names_count in (27, 41):
            rank += 25
        if "onestop" in str(path).lower():
            rank += 10

        return JsonCandidate(
            path=str(path),
            bytes=path.stat().st_size,
            interesting_keys="|".join(interesting[:50]),
            feature_names_count=feature_names_count,
            threshold_values="|".join(thresholds[:20]),
            reference_values="|".join(references[:20]),
            rank_score=rank,
        )
    except Exception as exc:
        return JsonCandidate(
            path=str(path),
            bytes=path.stat().st_size,
            interesting_keys="",
            feature_names_count=0,
            threshold_values="",
            reference_values="",
            rank_score=0,
            inspection_error=f"{type(exc).__name__}: {exc}",
        )


def discover_files(project_root: Path) -> list[Path]:
    found: dict[str, Path] = {}
    roots: list[Path] = []

    for rel in PREFERRED_ROOTS:
        candidate = project_root / rel
        if candidate.exists():
            roots.append(candidate)

    # Conservative fallback: only scan outputs/data processed roots, not raw data.
    for rel in ["outputs", "data/processed"]:
        candidate = project_root / rel
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)

    allowed = {".parquet", ".csv", ".joblib", ".pkl", ".pickle", ".json"}
    skip_parts = {
        ".git", ".venv", "__pycache__", "external",
        "raw", "logs",
    }

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            rel_parts = {p.lower() for p in path.relative_to(project_root).parts}
            if rel_parts & skip_parts:
                continue
            found[str(path.resolve())] = path

    return sorted(found.values(), key=lambda p: str(p).lower())


def write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def md_table(rows: list[Any], fields: list[str], limit: int = 10) -> str:
    if not rows:
        return "_No candidates found._\n"
    header = "| " + " | ".join(fields) + " |\n"
    divider = "| " + " | ".join(["---"] * len(fields)) + " |\n"
    lines = []
    for row in rows[:limit]:
        data = asdict(row)
        values = []
        for field in fields:
            value = str(data.get(field, "")).replace("|", ", ").replace("\n", " ")
            if len(value) > 140:
                value = value[:137] + "..."
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return header + divider + "\n".join(lines) + "\n"


def choose_top(table_rows: list[TableCandidate], role: str) -> str:
    candidates = [r for r in table_rows if role in r.candidate_role]
    candidates.sort(key=lambda r: (-r.rank_score, r.path))
    return candidates[0].path if candidates else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit inputs required for IUI P0 standard-control comparisons."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/iui_p0_control_input_audit_v1",
    )
    parser.add_argument(
        "--max-model-bytes",
        type=int,
        default=500_000_000,
        help="Do not joblib-load model files larger than this value.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(project_root)
    tables: list[TableCandidate] = []
    models: list[ModelCandidate] = []
    jsons: list[JsonCandidate] = []

    print(f"Discovered candidate files: {len(files)}")
    for index, path in enumerate(files, start=1):
        suffix = path.suffix.lower()
        try:
            if suffix in {".csv", ".parquet"}:
                tables.append(inspect_table(path))
            elif suffix in {".joblib", ".pkl", ".pickle"}:
                if path.stat().st_size <= args.max_model_bytes:
                    models.append(inspect_model(path))
                else:
                    models.append(ModelCandidate(
                        path=str(path),
                        kind=suffix.lstrip("."),
                        bytes=path.stat().st_size,
                        object_type="",
                        dict_keys="",
                        pipeline_steps="",
                        feature_names_count=0,
                        feature_names_preview="",
                        coef_shape="",
                        classes="",
                        threshold_values="",
                        rank_score=0,
                        inspection_error="Skipped: exceeds --max-model-bytes",
                    ))
            elif suffix == ".json":
                jsons.append(inspect_json(path))
        except Exception as exc:
            print(f"WARNING: failed to inspect {path}: {exc}", file=sys.stderr)

        if index % 250 == 0:
            print(f"Inspected {index}/{len(files)} files...")

    tables.sort(key=lambda r: (-r.rank_score, r.path))
    models.sort(key=lambda r: (-r.rank_score, r.path))
    jsons.sort(key=lambda r: (-r.rank_score, r.path))

    write_csv(output_dir / "candidate_tables.csv", tables)
    write_csv(output_dir / "candidate_models.csv", models)
    write_csv(output_dir / "candidate_json_artifacts.csv", jsons)

    selected = {
        "generated_by": "audit_iui_p0_control_inputs.py",
        "project_root": str(project_root),
        "top_source_feature_table": choose_top(tables, "source_feature_table"),
        "top_target_feature_table": choose_top(tables, "target_feature_table"),
        "top_target_score_table": choose_top(tables, "target_score_table"),
        "top_prediction_table": choose_top(tables, "prediction_table"),
        "top_model_artifact": models[0].path if models else "",
        "top_json_artifact": jsons[0].path if jsons else "",
        "quality_features_expected": QUALITY_FEATURES,
        "note": (
            "These are audit suggestions only. Do not begin confirmatory target "
            "performance evaluation from this file."
        ),
    }
    (output_dir / "selected_candidates.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = f"""# IUI P0 Control-Input Audit

Project root: `{project_root}`

Discovered:

- Table candidates: **{len(tables)}**
- Model candidates: **{len(models)}**
- JSON candidates: **{len(jsons)}**

## Suggested paths

```json
{json.dumps(selected, ensure_ascii=False, indent=2)}
```

## Top table candidates

{md_table(tables, ["rank_score", "candidate_role", "likely_domain", "quality_feature_hits", "n_columns", "participant_columns", "label_columns", "score_columns", "path"], 15)}

## Top model candidates

{md_table(models, ["rank_score", "object_type", "feature_names_count", "coef_shape", "classes", "threshold_values", "pipeline_steps", "path"], 15)}

## Top JSON candidates

{md_table(jsons, ["rank_score", "feature_names_count", "threshold_values", "reference_values", "interesting_keys", "path"], 15)}

## Next step

Provide the contents of `selected_candidates.json` and the top rows of the three CSV
files. The exact P0 control-comparison runner can then be bound to the existing
artifact schemas without guessing.

This audit does not calculate target performance and does not alter the frozen
confirmatory protocol.
"""
    (output_dir / "AUDIT_SUMMARY.md").write_text(summary, encoding="utf-8")

    print("")
    print("P0 control-input audit completed.")
    print(f"Output: {output_dir}")
    print(f"Suggested source table: {selected['top_source_feature_table'] or 'NOT FOUND'}")
    print(f"Suggested target table: {selected['top_target_feature_table'] or 'NOT FOUND'}")
    print(f"Suggested model: {selected['top_model_artifact'] or 'NOT FOUND'}")
    print("")
    print("Share:")
    print(f"  {output_dir / 'selected_candidates.json'}")
    print(f"  {output_dir / 'AUDIT_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
