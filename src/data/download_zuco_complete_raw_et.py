#!/usr/bin/env python
"""Download complete ZuCo 2.0 raw ET files for multiple subjects.

The subject list is selected from the audit CSV:
- complete_raw: all subjects with 14 raw ET blocks
- official_overlap: subjects with 14 raw ET blocks and a local official
  sent_gaze_sacc feature file
- explicit: comma-separated subjects passed with --subjects

Files are placed at:
    <output-root>/<SUBJECT>/session_et/raw/<SUBJECT>_<TASK><BLOCK>_ET.mat

Existing files are skipped unless --overwrite is specified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from osfclient.api import OSF


def normalize_path(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("/")


def select_subjects(
    coverage: pd.DataFrame,
    mode: str,
    explicit: str | None,
) -> list[str]:
    coverage = coverage.copy()
    coverage["subject_id"] = (
        coverage["subject_id"].astype(str).str.upper()
    )

    if explicit:
        subjects = sorted(
            {
                item.strip().upper()
                for item in explicit.split(",")
                if item.strip()
            }
        )
        if not subjects:
            raise ValueError("--subjects did not contain any subject IDs.")
        return subjects

    complete = pd.to_numeric(
        coverage["complete_14_raw_blocks"],
        errors="coerce",
    ).fillna(0).astype(int).eq(1)

    if mode == "complete_raw":
        selected = coverage[complete]
    elif mode == "official_overlap":
        official = pd.to_numeric(
            coverage["has_local_official_sent_gaze_sacc"],
            errors="coerce",
        ).fillna(0).astype(int).eq(1)
        selected = coverage[complete & official]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sorted(selected["subject_id"].tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="2urht")
    parser.add_argument(
        "--inventory",
        default=(
            "outputs/tables/zuco2_subject_audit/"
            "zuco2_raw_et_file_inventory.csv"
        ),
    )
    parser.add_argument(
        "--coverage",
        default=(
            "outputs/tables/zuco2_subject_audit/"
            "zuco2_subject_coverage_summary.csv"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["complete_raw", "official_overlap"],
        default="complete_raw",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject IDs; overrides --mode.",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw/zuco2",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    coverage_path = Path(args.coverage)
    output_root = Path(args.output_root)

    inventory = pd.read_csv(inventory_path)
    coverage = pd.read_csv(coverage_path)
    inventory["subject_id"] = (
        inventory["subject_id"].astype(str).str.upper()
    )
    inventory["task"] = inventory["task"].astype(str).str.upper()
    inventory["block_id"] = inventory["block_id"].astype(int)
    inventory["remote_norm"] = (
        inventory["remote_path"].map(normalize_path)
    )

    subjects = select_subjects(
        coverage,
        mode=args.mode,
        explicit=args.subjects,
    )
    if not subjects:
        raise ValueError("No subjects selected.")

    selected = inventory[
        inventory["subject_id"].isin(subjects)
    ].copy()

    expected_files = len(subjects) * 14
    if len(selected) != expected_files:
        counts = (
            selected.groupby("subject_id")
            .size()
            .reindex(subjects, fill_value=0)
        )
        raise ValueError(
            f"Selected inventory has {len(selected)} files, "
            f"expected {expected_files}.\n"
            f"Per-subject counts:\n{counts.to_string()}"
        )

    duplicate_mask = selected.duplicated(
        ["subject_id", "task", "block_id"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate task/block rows:\n"
            + selected.loc[duplicate_mask].to_string(index=False)
        )

    planned_size_mb = pd.to_numeric(
        selected["size_mb"],
        errors="coerce",
    ).sum()

    target_by_remote: dict[str, Path] = {}
    metadata_by_remote: dict[str, dict[str, Any]] = {}

    for row in selected.itertuples(index=False):
        filename = (
            f"{row.subject_id}_{row.task}{int(row.block_id)}_ET.mat"
        )
        destination = (
            output_root
            / row.subject_id
            / "session_et"
            / "raw"
            / filename
        )
        remote = str(row.remote_norm)
        target_by_remote[remote] = destination
        metadata_by_remote[remote] = {
            "subject_id": row.subject_id,
            "task": row.task,
            "block_id": int(row.block_id),
            "expected_size_bytes": (
                int(row.size_bytes)
                if pd.notna(row.size_bytes)
                and str(row.size_bytes) != ""
                else None
            ),
        }

    print("=== DOWNLOAD PLAN ===")
    print(f"mode: {args.mode}")
    print(f"subjects ({len(subjects)}): {', '.join(subjects)}")
    print(f"files: {len(target_by_remote)}")
    print(f"planned raw size: {planned_size_mb:.3f} MB")
    print(f"output root: {output_root}")

    storage = OSF().project(args.project_id).storage("osfstorage")
    found: dict[str, Any] = {}

    print(f"\nScanning OSF project {args.project_id} once ...")
    for remote_file in storage.files:
        remote = normalize_path(remote_file.path)
        if remote in target_by_remote:
            found[remote] = remote_file
            if len(found) == len(target_by_remote):
                break

    missing_remote = sorted(set(target_by_remote) - set(found))
    if missing_remote:
        print("Remote files not found:", file=sys.stderr)
        for remote in missing_remote:
            print(f"  /{remote}", file=sys.stderr)
        return 2

    downloaded = 0
    skipped = 0
    repaired = 0
    downloaded_bytes = 0

    for index, remote in enumerate(sorted(target_by_remote), start=1):
        destination = target_by_remote[remote]
        metadata = metadata_by_remote[remote]
        destination.parent.mkdir(parents=True, exist_ok=True)

        expected_size = metadata["expected_size_bytes"]
        existing_valid = (
            destination.is_file()
            and (
                expected_size is None
                or destination.stat().st_size == expected_size
            )
        )

        if existing_valid and not args.overwrite:
            print(
                f"[{index}/{len(target_by_remote)}] "
                f"Skip existing: {destination}"
            )
            skipped += 1
            continue

        if destination.exists() and not existing_valid:
            print(
                f"[{index}/{len(target_by_remote)}] "
                f"Replace size-mismatched file: {destination}"
            )
            repaired += 1
        else:
            print(
                f"[{index}/{len(target_by_remote)}] "
                f"Download: /{remote}"
            )

        temporary = destination.with_suffix(
            destination.suffix + ".part"
        )
        if temporary.exists():
            temporary.unlink()

        with temporary.open("wb") as fp:
            found[remote].write_to(fp)

        actual_size = temporary.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            temporary.unlink(missing_ok=True)
            raise IOError(
                f"Downloaded size mismatch for {remote}: "
                f"actual={actual_size}, expected={expected_size}"
            )

        temporary.replace(destination)
        downloaded += 1
        downloaded_bytes += actual_size

    print("\n=== DOWNLOAD SUMMARY ===")
    print(f"subjects: {len(subjects)}")
    print(f"available files: {len(target_by_remote)}")
    print(f"downloaded: {downloaded}")
    print(f"skipped existing: {skipped}")
    print(f"repaired/replaced: {repaired}")
    print(
        f"bytes downloaded this run: "
        f"{downloaded_bytes / (1024 * 1024):.3f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
