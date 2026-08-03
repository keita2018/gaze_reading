from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PARTICIPANT_ALIASES = [
    "participant_id", "participant", "subject_id", "subject",
    "worker_id", "worker",
]
TRIAL_ALIASES = [
    "external_trial_id", "trial_id", "trial", "trial_index",
    "sample_id", "item_id",
]
LABEL_ALIASES = [
    "label_nr", "label", "y", "target", "class",
]
TASK_ALIASES = [
    "task", "condition", "reading_goal", "label_name",
]
EXCLUSION_ALIASES = [
    "participant_exclusion_reasons", "exclusion_reason",
    "failure_reason", "filter_failure_reason",
]


def norm(value: Any) -> str:
    return str(value).strip().lower()


def first_existing(columns: list[str], aliases: list[str]) -> str:
    lower = {norm(c): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table_schema(path: Path) -> tuple[list[str], int]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return list(pf.schema_arrow.names), int(pf.metadata.num_rows)
    frame = pd.read_csv(path, nrows=5)
    return list(frame.columns), sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1


def load_small_or_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns or None)
    return pd.read_csv(path, usecols=columns or None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Turkish WebQAmGaze Stage-2 outputs without loading a model "
            "or calculating predictive performance."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--stage2-output",
        default="outputs/webqamgaze_tr_confirmatory_stage2_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/iui_p0_tr_stage2_audit_v1",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    stage2 = (root / args.stage2_output).resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not stage2.exists():
        raise FileNotFoundError(stage2)

    files = sorted(
        [
            p for p in stage2.rglob("*")
            if p.is_file()
        ],
        key=lambda p: str(p).lower(),
    )

    hash_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    participant_candidates: list[Path] = []
    trial_candidates: list[Path] = []

    for path in files:
        hash_rows.append({
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "path": str(path),
        })
        if path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        try:
            columns, rows = table_schema(path)
            participant_col = first_existing(columns, PARTICIPANT_ALIASES)
            trial_col = first_existing(columns, TRIAL_ALIASES)
            label_col = first_existing(columns, LABEL_ALIASES)
            task_col = first_existing(columns, TASK_ALIASES)
            exclusion_col = first_existing(columns, EXCLUSION_ALIASES)

            role: list[str] = []
            if participant_col and exclusion_col:
                role.append("participant_audit")
                participant_candidates.append(path)
            if participant_col and (trial_col or label_col or task_col):
                role.append("trial_manifest")
                trial_candidates.append(path)
            if "sample" in path.name.lower():
                role.append("samples")
            if "fixation" in path.name.lower():
                role.append("fixations")

            table_rows.append({
                "path": str(path),
                "rows": rows,
                "columns": len(columns),
                "participant_column": participant_col,
                "trial_column": trial_col,
                "label_column": label_col,
                "task_column": task_col,
                "exclusion_column": exclusion_col,
                "role": "|".join(role) if role else "other",
                "column_names": "|".join(columns),
            })
        except Exception as exc:
            table_rows.append({
                "path": str(path),
                "rows": -1,
                "columns": 0,
                "participant_column": "",
                "trial_column": "",
                "label_column": "",
                "task_column": "",
                "exclusion_column": "",
                "role": "inspection_error",
                "column_names": f"{type(exc).__name__}: {exc}",
            })

    pd.DataFrame(hash_rows).to_csv(
        output / "stage2_file_hashes.csv", index=False
    )
    pd.DataFrame(table_rows).to_csv(
        output / "stage2_table_inventory.csv", index=False
    )

    participant_summary: dict[str, Any] = {}
    if participant_candidates:
        # Prefer the official filename, then the smallest participant-level table.
        participant_candidates.sort(
            key=lambda p: (
                "participant_quality_audit" not in p.name.lower(),
                p.stat().st_size,
                str(p),
            )
        )
        path = participant_candidates[0]
        columns, _ = table_schema(path)
        participant_col = first_existing(columns, PARTICIPANT_ALIASES)
        exclusion_col = first_existing(columns, EXCLUSION_ALIASES)
        selected_columns = [c for c in [participant_col, exclusion_col] if c]
        frame = load_small_or_selected(path, selected_columns)

        participants = frame[participant_col].astype(str)
        exclusions = (
            frame[exclusion_col].fillna("").astype(str)
            if exclusion_col
            else pd.Series([""] * len(frame))
        )
        reason_counter: Counter[str] = Counter()
        for value in exclusions:
            token = value.strip()
            if not token or token.lower() in {"nan", "none", "[]"}:
                continue
            for part in re_split_reasons(token):
                if part:
                    reason_counter[part] += 1

        participant_summary = {
            "path": str(path),
            "rows": int(len(frame)),
            "unique_participant_ids": int(participants.nunique()),
            "duplicate_participant_rows": int(
                len(frame) - participants.nunique()
            ),
            "exclusion_reason_counts": dict(reason_counter),
        }

    trial_summary: dict[str, Any] = {}
    if trial_candidates:
        trial_candidates.sort(
            key=lambda p: (
                "manifest" not in p.name.lower(),
                -p.stat().st_size,
                str(p),
            )
        )
        path = trial_candidates[0]
        columns, _ = table_schema(path)
        participant_col = first_existing(columns, PARTICIPANT_ALIASES)
        trial_col = first_existing(columns, TRIAL_ALIASES)
        label_col = first_existing(columns, LABEL_ALIASES)
        task_col = first_existing(columns, TASK_ALIASES)
        selected_columns = [
            c for c in [participant_col, trial_col, label_col, task_col] if c
        ]
        frame = load_small_or_selected(path, selected_columns)

        trial_summary = {
            "path": str(path),
            "rows": int(len(frame)),
            "unique_participant_ids": (
                int(frame[participant_col].astype(str).nunique())
                if participant_col else None
            ),
            "unique_trials": (
                int(frame[trial_col].astype(str).nunique())
                if trial_col else int(len(frame))
            ),
            "duplicate_trial_rows": (
                int(len(frame) - frame[trial_col].astype(str).nunique())
                if trial_col else None
            ),
            "class_counts": (
                {
                    str(k): int(v)
                    for k, v in frame[label_col].value_counts(
                        dropna=False
                    ).to_dict().items()
                }
                if label_col else {}
            ),
            "task_counts": (
                {
                    str(k): int(v)
                    for k, v in frame[task_col].value_counts(
                        dropna=False
                    ).to_dict().items()
                }
                if task_col else {}
            ),
        }

    # Basic safeguards.
    failures: list[str] = []
    if not participant_summary:
        failures.append("participant audit table not identified")
    if not trial_summary:
        failures.append("trial manifest not identified")
    if trial_summary and (trial_summary.get("unique_participant_ids") or 0) < 40:
        failures.append("fewer than 40 eligible trial participants")
    if trial_summary and trial_summary.get("duplicate_trial_rows") not in {None, 0}:
        failures.append("duplicate trial IDs detected")
    class_counts = trial_summary.get("class_counts", {}) if trial_summary else {}
    task_counts = trial_summary.get("task_counts", {}) if trial_summary else {}
    if class_counts and len(class_counts) < 2:
        failures.append("fewer than two label classes in trial manifest")
    if not class_counts and task_counts and len(task_counts) < 2:
        failures.append("fewer than two task classes in trial manifest")

    audit = {
        "artifact_version": "iui_p0_tr_stage2_audit_v1",
        "stage2_output": str(stage2),
        "participant_summary": participant_summary,
        "trial_summary": trial_summary,
        "file_count": len(files),
        "table_count": len(table_rows),
        "model_loaded": False,
        "performance_calculated": False,
        "predictive_metrics_present": False,
        "audit_failures": failures,
        "audit_status": "PASS" if not failures else "FAIL",
    }
    (output / "tr_stage2_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = f"""# Turkish WebQAmGaze Stage-2 Audit

- Status: **{audit['audit_status']}**
- Model loaded: **False**
- Predictive performance calculated: **False**
- Stage-2 file count: **{len(files)}**

## Participant audit

```json
{json.dumps(participant_summary, ensure_ascii=False, indent=2)}
```

## Trial manifest

```json
{json.dumps(trial_summary, ensure_ascii=False, indent=2)}
```

## Failures

```json
{json.dumps(failures, ensure_ascii=False, indent=2)}
```

The next step must reproduce Stage 3A–3C for the selected Turkish target, remove
labels from the feature/prediction artifact used for freezing, and hash that unlabeled
artifact before any metric is calculated.
"""
    (output / "TR_STAGE2_AUDIT_SUMMARY.md").write_text(
        summary, encoding="utf-8"
    )

    print("")
    print("Turkish Stage-2 audit completed.")
    print(f"Status: {audit['audit_status']}")
    print(f"Output: {output}")
    print(f"Participant summary: {participant_summary}")
    print(f"Trial summary: {trial_summary}")
    if failures:
        print(f"Failures: {failures}")
        return 1
    return 0


def re_split_reasons(value: str) -> list[str]:
    import re
    cleaned = value.strip("[](){}")
    return [
        part.strip(" '\"")
        for part in re.split(r"[|,;]+", cleaned)
        if part.strip(" '\"")
    ]


if __name__ == "__main__":
    raise SystemExit(main())
