from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_REL = Path("outputs/frozen_onestop_external_validation_v1/models/soft_full_all.joblib")

SEARCH_ROOTS = [
    "outputs/frozen_onestop_external_validation_v1",
    "outputs/linear_feature_group_factorization_v1",
    "outputs/webqamgaze_participant_self_calibration_v1",
    "outputs/final_factorized_selective_prediction_v1",
    "outputs/webqamgaze_selective_prediction_v1",
    "outputs/webqamgaze_external_validation_stage3c",
    "outputs/webqamgaze_external_validation_stage8_final",
    "outputs/webqamgaze_external_validation_final_release_v1",
    "outputs/onestop_to_webqamgaze_zero_shot_v1",
    "data/processed/onestop/quality_views_asc344_v1",
    "data/processed/onestop/soft_aoi_asc344",
]

META_ALIASES = {
    "participant": [
        "participant_id", "participant", "subject_id", "subject",
        "worker_id", "reader_id",
    ],
    "trial": [
        "trial_id", "trial", "trial_index", "sample_id", "item_id",
        "record_id", "trial_uid", "trial_key",
    ],
    "label": [
        "label", "y", "target", "class", "label_nr", "y_true",
        "true_label", "reading_goal", "goal_label", "condition_label",
    ],
    "view": [
        "view", "quality_view", "condition", "degradation",
        "quality_condition", "degradation_condition",
    ],
    "score": [
        "score", "decision_score", "decision_function", "raw_score",
        "frozen_score", "calibrated_score", "qcn_score", "qcn_pc_score",
    ],
}

REFERENCE_TOKENS = (
    "reference", "source_median", "quality_median", "participant_median",
    "r_quality", "r_participant", "r_qcn", "center", "threshold",
)


@dataclass
class MatchRow:
    path: str
    kind: str
    bytes: int
    rows: int
    columns: int
    exact_feature_matches: int
    canonical_feature_matches: int
    missing_model_features: str
    extra_feature_like_columns: str
    participant_columns: str
    trial_columns: str
    label_columns: str
    view_columns: str
    score_columns: str
    likely_domain: str
    likely_role: str
    rank_score: int
    inspection_error: str = ""


def norm(value: Any) -> str:
    return str(value).strip().lower()


def canonical_feature(name: str) -> str:
    value = norm(name)
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"__+", "_", value)

    prefixes = [
        "soft_aoi_", "event_free_aoi_", "eventfree_aoi_", "aoi_",
        "feature_", "feat_", "full41_", "full_",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]
                changed = True

    suffixes = ["_feature", "_value"]
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[:-len(suffix)]

    aliases = {
        "soft_valid_rate": "valid_rate",
        "soft_active_rate": "active_rate",
        "box_hit_rate": "word_box_hit_rate",
        "sentence_hit_rate": "sentence_envelope_hit_rate",
        "q_max_mean": "qmax_mean",
        "q_max_median": "qmax_median",
        "q_max_q10": "qmax_q10",
        "q_max_ge_0_5_rate": "qmax_ge_0_5_rate",
    }
    return aliases.get(value, value)


def first_aliases(columns: Iterable[str], aliases: list[str]) -> list[str]:
    lower = {norm(c): str(c) for c in columns}
    return [lower[a] for a in aliases if a in lower]


def get_columns_and_rows(path: Path) -> tuple[list[str], int, str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            return list(pf.schema_arrow.names), pf.metadata.num_rows, ""
        except Exception as exc:
            return [], -1, f"{type(exc).__name__}: {exc}"

    encodings = ["utf-8-sig", "utf-8", "cp932", "latin-1"]
    last_error = ""
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                # Avoid full scan for very large CSVs.
                rows = sum(1 for _ in reader) if path.stat().st_size < 100_000_000 else -1
            return header, rows, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return [], -1, last_error


def load_model_structure(model_path: Path) -> tuple[Any, list[str], dict[str, Any]]:
    import joblib

    obj = joblib.load(model_path)
    info: dict[str, Any] = {
        "object_type": f"{type(obj).__module__}.{type(obj).__name__}",
        "top_level_keys": sorted(map(str, obj.keys())) if isinstance(obj, dict) else [],
    }

    feature_names: list[str] = []

    def inspect(candidate: Any, path: str) -> None:
        nonlocal feature_names
        if hasattr(candidate, "feature_names_in_"):
            try:
                names = list(map(str, candidate.feature_names_in_))
                if len(names) > len(feature_names):
                    feature_names = names
                    info[f"{path}.feature_names_in_count"] = len(names)
            except Exception:
                pass
        if hasattr(candidate, "coef_"):
            try:
                info[f"{path}.coef_shape"] = list(candidate.coef_.shape)
            except Exception:
                pass
        if hasattr(candidate, "classes_"):
            try:
                info[f"{path}.classes"] = candidate.classes_.tolist()
            except Exception:
                info[f"{path}.classes"] = repr(candidate.classes_)
        if hasattr(candidate, "steps"):
            try:
                info[f"{path}.steps"] = [name for name, _ in candidate.steps]
                for name, step in candidate.steps:
                    inspect(step, f"{path}.steps.{name}")
            except Exception:
                pass

    inspect(obj, "root")
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                info[f"dict.{key}"] = value
            elif isinstance(value, (list, tuple)) and len(value) <= 100:
                if all(isinstance(v, (str, int, float, bool)) for v in value):
                    info[f"dict.{key}"] = list(value)
            inspect(value, f"dict.{key}")

            key_lower = norm(key)
            if "feature" in key_lower and "name" in key_lower and isinstance(value, (list, tuple)):
                candidate_names = list(map(str, value))
                if len(candidate_names) > len(feature_names):
                    feature_names = candidate_names

    if len(feature_names) != 41:
        raise RuntimeError(
            f"Expected 41 frozen Full features, found {len(feature_names)} in {model_path}"
        )

    return obj, feature_names, info


def infer_domain(path: Path) -> str:
    text = str(path).lower()
    if "webqamgaze" in text:
        return "webqamgaze"
    if "onestop" in text:
        return "onestop"
    if "zuco" in text:
        return "zuco"
    return "unknown"


def inspect_table(path: Path, model_features: list[str]) -> MatchRow:
    columns, rows, error = get_columns_and_rows(path)
    col_set = set(columns)
    exact = [f for f in model_features if f in col_set]

    canonical_to_actual: dict[str, list[str]] = {}
    for col in columns:
        canonical_to_actual.setdefault(canonical_feature(col), []).append(col)

    canonical_matches = [
        f for f in model_features if canonical_feature(f) in canonical_to_actual
    ]
    missing = [
        f for f in model_features if canonical_feature(f) not in canonical_to_actual
    ]

    model_canon = {canonical_feature(f) for f in model_features}
    extras = [
        c for c in columns
        if canonical_feature(c) not in model_canon
        and any(token in norm(c) for token in (
            "entropy", "progress", "coverage", "qmax", "distance",
            "valid", "active", "transition", "regression", "occupancy",
        ))
    ]

    participant = first_aliases(columns, META_ALIASES["participant"])
    trial = first_aliases(columns, META_ALIASES["trial"])
    label = first_aliases(columns, META_ALIASES["label"])
    view = first_aliases(columns, META_ALIASES["view"])
    score = first_aliases(columns, META_ALIASES["score"])
    domain = infer_domain(path)

    role: list[str] = []
    rank = 4 * len(canonical_matches) + 2 * len(exact)
    if participant:
        rank += 15
    if trial:
        rank += 8
    if label:
        rank += 12
    if view:
        rank += 8
    if score:
        rank += 8

    if len(canonical_matches) >= 39:
        role.append("feature_table")
        rank += 40
        if domain == "onestop":
            role.append("source_feature_table")
            rank += 30
        if domain == "webqamgaze":
            role.append("target_feature_table")
            rank += 30
    if score and participant:
        role.append("score_table")
    if domain == "onestop" and view and label and len(canonical_matches) >= 39:
        role.append("source_quality_view_table")
        rank += 25
    if domain == "webqamgaze" and label and len(canonical_matches) >= 39:
        role.append("labeled_target_feature_table")
        rank += 20

    return MatchRow(
        path=str(path),
        kind=path.suffix.lower().lstrip("."),
        bytes=path.stat().st_size,
        rows=rows,
        columns=len(columns),
        exact_feature_matches=len(exact),
        canonical_feature_matches=len(canonical_matches),
        missing_model_features="|".join(missing),
        extra_feature_like_columns="|".join(extras[:40]),
        participant_columns="|".join(participant),
        trial_columns="|".join(trial),
        label_columns="|".join(label),
        view_columns="|".join(view),
        score_columns="|".join(score),
        likely_domain=domain,
        likely_role="|".join(role) if role else "other",
        rank_score=rank,
        inspection_error=error,
    )


def discover(project_root: Path) -> list[Path]:
    paths: dict[str, Path] = {}
    for rel in SEARCH_ROOTS:
        root = project_root / rel
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".json"}:
                paths[str(path.resolve())] = path
    return sorted(paths.values(), key=lambda p: str(p).lower())


def safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value
    return None


def scan_reference_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(obj: Any, prefix: str = "", depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                lower = child.lower()
                scalar = safe_scalar(value)
                if scalar is not None and any(token in lower for token in REFERENCE_TOKENS):
                    rows.append({"key": child, "value": scalar})
                walk(value, child, depth + 1)
        elif isinstance(obj, list) and len(obj) <= 1000:
            for index, value in enumerate(obj):
                walk(value, f"{prefix}[{index}]", depth + 1)

    output: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() != ".json":
            continue
        if not any(token in str(path).lower() for token in (
            "factor", "calibr", "selective", "frozen_onestop", "zero_shot",
        )):
            continue
        rows.clear()
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            walk(data)
            for row in rows:
                output.append({
                    "path": str(path),
                    "key": row["key"],
                    "value": row["value"],
                })
        except Exception:
            continue
    return output


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


def top(rows: list[MatchRow], token: str) -> str:
    candidates = [r for r in rows if token in r.likely_role]
    candidates.sort(key=lambda r: (-r.rank_score, r.path))
    return candidates[0].path if candidates else ""


def table_markdown(rows: list[MatchRow], limit: int = 20) -> str:
    selected = rows[:limit]
    if not selected:
        return "_No tables found._"
    fields = [
        "rank_score", "likely_role", "likely_domain",
        "canonical_feature_matches", "exact_feature_matches",
        "participant_columns", "trial_columns", "label_columns",
        "view_columns", "score_columns", "rows", "path",
    ]
    out = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in selected:
        data = asdict(row)
        vals = []
        for field in fields:
            value = str(data[field]).replace("|", ", ").replace("\n", " ")
            if len(value) > 130:
                value = value[:127] + "..."
            vals.append(value)
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind exact Full-41 source/target schemas and source-reference artifacts."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/iui_p0_schema_binding_v1",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = project_root / MODEL_REL
    if not model_path.exists():
        raise FileNotFoundError(f"Frozen model not found: {model_path}")

    _, model_features, model_info = load_model_structure(model_path)
    (output_dir / "model_feature_order.json").write_text(
        json.dumps(model_features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "frozen_model_structure.json").write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    paths = discover(project_root)
    rows: list[MatchRow] = []
    for i, path in enumerate(paths, start=1):
        if path.suffix.lower() in {".csv", ".parquet"}:
            rows.append(inspect_table(path, model_features))
        if i % 500 == 0:
            print(f"Inspected {i}/{len(paths)} artifacts...")

    rows.sort(key=lambda r: (-r.rank_score, r.path))
    write_csv(output_dir / "schema_match_candidates.csv", [asdict(r) for r in rows])

    reference_rows = scan_reference_artifacts(paths)
    write_csv(output_dir / "reference_value_candidates.csv", reference_rows)

    binding = {
        "frozen_model": str(model_path),
        "model_feature_count": len(model_features),
        "model_feature_order_file": str(output_dir / "model_feature_order.json"),
        "source_quality_view_table": top(rows, "source_quality_view_table"),
        "source_feature_table": top(rows, "source_feature_table"),
        "target_labeled_feature_table": top(rows, "labeled_target_feature_table"),
        "target_feature_table": top(rows, "target_feature_table"),
        "top_score_table": top(rows, "score_table"),
        "reference_candidate_count": len(reference_rows),
        "status": "AUDIT_ONLY_NOT_YET_EXECUTABLE",
    }
    (output_dir / "binding_manifest.json").write_text(
        json.dumps(binding, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = f"""# IUI P0 Exact Schema Binding

## Frozen model

- Path: `{model_path}`
- Full feature count: **{len(model_features)}**
- Decision threshold and pipeline structure:
  `frozen_model_structure.json`
- Exact feature order:
  `model_feature_order.json`

## Suggested binding

```json
{json.dumps(binding, ensure_ascii=False, indent=2)}
```

## Top schema matches

{table_markdown(rows, 25)}

## Reference-value candidates

Found **{len(reference_rows)}** scalar reference/threshold candidates in relevant JSON
artifacts. See `reference_value_candidates.csv`.

## Required return files

Share:

1. `binding_manifest.json`
2. `frozen_model_structure.json`
3. `model_feature_order.json`
4. The first 30 rows of `reference_value_candidates.csv`
5. The first 15 rows of `schema_match_candidates.csv`

This step does not evaluate an unused language and does not change the frozen
confirmatory protocol.
"""
    (output_dir / "SCHEMA_BINDING_SUMMARY.md").write_text(summary, encoding="utf-8")

    print("")
    print("Exact schema binding audit completed.")
    print(f"Output: {output_dir}")
    print(f"Source quality-view table: {binding['source_quality_view_table'] or 'NOT FOUND'}")
    print(f"Target feature table: {binding['target_feature_table'] or 'NOT FOUND'}")
    print(f"Reference candidates: {len(reference_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
