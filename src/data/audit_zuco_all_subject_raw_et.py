#!/usr/bin/env python
"""Audit ZuCo 2.0 raw ET availability for all benchmark subjects.

This script scans the public OSF project and identifies session-level raw
eye-tracking files matching:

    task1 - NR/Raw data/<SUBJECT>/<SUBJECT>_NR<BLOCK>_ET.mat
    task2 - TSR/Raw data/<SUBJECT>/<SUBJECT>_TSR<BLOCK>_ET.mat

It also scans a local official-feature directory for files named:

    <SUBJECT>_sent_gaze_sacc.npy

Outputs:
- zuco2_raw_et_file_inventory.csv
- zuco2_subject_coverage_summary.csv
- zuco2_benchmark_subjects_ready.csv

No files are downloaded.

Example:
    python src/data/audit_zuco_all_subject_raw_et.py \
        --project-id 2urht \
        --feature-dir external/zuco-benchmark/src/features \
        --output-dir outputs/tables/zuco2_subject_audit
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd
from osfclient.api import OSF


RAW_ET_RE = re.compile(
    r"^task(?P<task_number>[12]) - (?P<task>NR|TSR)/"
    r"Raw data/(?P<subject>[^/]+)/"
    r"(?P=subject)_(?P=task)(?P<block>[1-7])_ET\.mat$",
    re.IGNORECASE,
)

FEATURE_RE = re.compile(
    r"^(?P<subject>.+)_sent_gaze_sacc\.npy$",
    re.IGNORECASE,
)


def normalize_path(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("/")


def parse_size(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def get_remote_size(remote_file: Any) -> int | None:
    size = parse_size(getattr(remote_file, "size", None))
    if size is not None:
        return size

    raw_json = getattr(remote_file, "_json", {})
    if isinstance(raw_json, dict):
        attributes = raw_json.get("attributes", {})
        if isinstance(attributes, dict):
            return parse_size(attributes.get("size"))
    return None


def scan_official_subjects(feature_dir: Path) -> set[str]:
    subjects: set[str] = set()
    if not feature_dir.is_dir():
        return subjects

    for path in feature_dir.glob("*_sent_gaze_sacc.npy"):
        match = FEATURE_RE.match(path.name)
        if match:
            subjects.add(match.group("subject").upper())
    return subjects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="2urht")
    parser.add_argument(
        "--feature-dir",
        default="external/zuco-benchmark/src/features",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tables/zuco2_subject_audit",
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    official_subjects = scan_official_subjects(feature_dir)
    print(
        f"Official sent_gaze_sacc subjects found locally: "
        f"{len(official_subjects)}"
    )

    project = OSF().project(args.project_id)
    storage = project.storage("osfstorage")

    file_rows: list[dict[str, Any]] = []

    print(f"Scanning OSF project {args.project_id} ...")
    for remote_file in storage.files:
        remote_path = normalize_path(remote_file.path)
        match = RAW_ET_RE.match(remote_path)
        if match is None:
            continue

        subject = match.group("subject").upper()
        task = match.group("task").upper()
        block = int(match.group("block"))
        size_bytes = get_remote_size(remote_file)

        file_rows.append(
            {
                "subject_id": subject,
                "task": task,
                "block_id": block,
                "remote_path": "/" + remote_path,
                "size_bytes": size_bytes if size_bytes is not None else "",
                "size_mb": (
                    size_bytes / (1024 * 1024)
                    if size_bytes is not None
                    else ""
                ),
                "has_local_official_sent_gaze_sacc": int(
                    subject in official_subjects
                ),
            }
        )

    if not file_rows:
        raise RuntimeError("No raw ET files matched the expected OSF paths.")

    files = pd.DataFrame(file_rows).sort_values(
        ["subject_id", "task", "block_id"]
    )

    duplicated = files.duplicated(
        ["subject_id", "task", "block_id"],
        keep=False,
    )
    if duplicated.any():
        raise ValueError(
            "Duplicate raw ET task/block entries:\n"
            + files.loc[duplicated].to_string(index=False)
        )

    osf_subjects = set(files["subject_id"])
    all_subjects = sorted(osf_subjects | official_subjects)

    summary_rows: list[dict[str, Any]] = []
    ready_rows: list[dict[str, Any]] = []

    for subject in all_subjects:
        subset = files[files["subject_id"] == subject]
        nr_blocks = sorted(
            subset.loc[subset["task"] == "NR", "block_id"]
            .astype(int)
            .tolist()
        )
        tsr_blocks = sorted(
            subset.loc[subset["task"] == "TSR", "block_id"]
            .astype(int)
            .tolist()
        )

        missing_nr = sorted(set(range(1, 8)) - set(nr_blocks))
        missing_tsr = sorted(set(range(1, 8)) - set(tsr_blocks))
        has_official = subject in official_subjects
        complete_raw = not missing_nr and not missing_tsr
        ready = has_official and complete_raw

        total_size_mb = pd.to_numeric(
            subset["size_mb"],
            errors="coerce",
        ).sum(min_count=1)

        summary_rows.append(
            {
                "subject_id": subject,
                "has_local_official_sent_gaze_sacc": int(has_official),
                "n_nr_raw_blocks": len(nr_blocks),
                "n_tsr_raw_blocks": len(tsr_blocks),
                "n_total_raw_blocks": len(subset),
                "missing_nr_blocks": ",".join(map(str, missing_nr)),
                "missing_tsr_blocks": ",".join(map(str, missing_tsr)),
                "complete_14_raw_blocks": int(complete_raw),
                "ready_for_raw_loso": int(ready),
                "total_raw_et_size_mb": (
                    float(total_size_mb)
                    if pd.notna(total_size_mb)
                    else ""
                ),
            }
        )

        if ready:
            ready_rows.append(
                {
                    "subject_id": subject,
                    "n_raw_blocks": 14,
                    "total_raw_et_size_mb": float(total_size_mb),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values("subject_id")
    ready = pd.DataFrame(ready_rows).sort_values("subject_id")

    file_path = output_dir / "zuco2_raw_et_file_inventory.csv"
    summary_path = output_dir / "zuco2_subject_coverage_summary.csv"
    ready_path = output_dir / "zuco2_benchmark_subjects_ready.csv"

    files.to_csv(file_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    ready.to_csv(ready_path, index=False, encoding="utf-8-sig")

    print("\n=== SUBJECT COVERAGE SUMMARY ===")
    print(
        summary[
            [
                "subject_id",
                "has_local_official_sent_gaze_sacc",
                "n_nr_raw_blocks",
                "n_tsr_raw_blocks",
                "complete_14_raw_blocks",
                "ready_for_raw_loso",
                "total_raw_et_size_mb",
            ]
        ].to_string(index=False)
    )

    print("\n=== AUDIT TOTALS ===")
    print(f"OSF raw-ET subjects: {len(osf_subjects)}")
    print(f"Local official-feature subjects: {len(official_subjects)}")
    print(f"Subjects ready for raw LOSO: {len(ready)}")
    print(f"Raw ET files matched: {len(files)}")
    print(
        "Total size for ready subjects: "
        f"{ready['total_raw_et_size_mb'].sum():.3f} MB"
        if not ready.empty
        else "Total size for ready subjects: 0 MB"
    )

    only_official = sorted(official_subjects - osf_subjects)
    only_osf = sorted(osf_subjects - official_subjects)
    print(f"Official subjects without OSF raw ET: {only_official}")
    print(f"OSF raw-ET subjects without official features: {only_osf}")

    print("\nWritten:")
    print(f"  {file_path}")
    print(f"  {summary_path}")
    print(f"  {ready_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
