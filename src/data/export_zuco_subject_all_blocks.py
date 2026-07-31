#!/usr/bin/env python
"""Export all ZuCo raw ET blocks for one subject and build compact manifests.

This driver calls export_zuco_session_trials.py for NR1..NR7 and TSR1..TSR7.
It retains one Parquet file per block to avoid loading all samples into RAM.

Outputs:
- 14 block-level sample Parquet files
- 14 block-level trial CSV files
- <SUBJECT>_all_trials.csv
- <SUBJECT>_all_non_trial_events.csv
- <SUBJECT>_block_summary.csv
- <SUBJECT>_sample_manifest.csv

Example:
    python src/data/export_zuco_subject_all_blocks.py \
        --subject YAC \
        --raw-dir data/raw/zuco2/YAC/session_et/raw \
        --output-dir data/interim/zuco2/YAC \
        --exporter src/data/export_zuco_session_trials.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_TRIALS = {"NR": 349, "TSR": 390}


def run_export(
    exporter: Path,
    raw_file: Path,
    subject: str,
    task: str,
    block: int,
    output_dir: Path,
) -> None:
    command = [
        sys.executable,
        str(exporter),
        str(raw_file),
        "--subject",
        subject,
        "--task",
        task,
        "--block",
        str(block),
        "--output-dir",
        str(output_dir),
    ]
    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="YAC")
    parser.add_argument(
        "--raw-dir",
        default="data/raw/zuco2/YAC/session_et/raw",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/zuco2/YAC",
    )
    parser.add_argument(
        "--exporter",
        default="src/data/export_zuco_session_trials.py",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rerun blocks whose Parquet and trial CSV already exist.",
    )
    args = parser.parse_args()

    subject = args.subject.upper()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    exporter = Path(args.exporter)

    if not exporter.is_file():
        raise FileNotFoundError(exporter)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_raw = []
    for task in ("NR", "TSR"):
        for block in range(1, 8):
            raw_file = raw_dir / f"{subject}_{task}{block}_ET.mat"
            if not raw_file.is_file():
                missing_raw.append(raw_file)

    if missing_raw:
        print("Missing raw ET files:", file=sys.stderr)
        for path in missing_raw:
            print(f"  {path}", file=sys.stderr)
        return 2

    for task in ("NR", "TSR"):
        for block in range(1, 8):
            stem = f"{subject}_{task}{block}"
            raw_file = raw_dir / f"{stem}_ET.mat"
            sample_file = output_dir / f"{stem}_samples.parquet"
            trial_file = output_dir / f"{stem}_trials.csv"

            if (
                args.skip_existing
                and sample_file.is_file()
                and trial_file.is_file()
            ):
                print(f"Skipping existing block: {stem}")
                continue

            run_export(
                exporter=exporter,
                raw_file=raw_file,
                subject=subject,
                task=task,
                block=block,
                output_dir=output_dir,
            )

    trial_frames = []
    event_frames = []
    manifest_rows: list[dict[str, Any]] = []

    for task in ("NR", "TSR"):
        for block in range(1, 8):
            stem = f"{subject}_{task}{block}"
            sample_path = output_dir / f"{stem}_samples.parquet"
            trial_path = output_dir / f"{stem}_trials.csv"
            event_path = output_dir / f"{stem}_non_trial_events.csv"

            if not sample_path.is_file():
                raise FileNotFoundError(sample_path)
            if not trial_path.is_file():
                raise FileNotFoundError(trial_path)
            if not event_path.is_file():
                raise FileNotFoundError(event_path)

            trials = pd.read_csv(trial_path)
            events = pd.read_csv(event_path)
            trial_frames.append(trials)
            event_frames.append(events)

            manifest_rows.append(
                {
                    "subject_id": subject,
                    "task": task,
                    "block_id": block,
                    "sample_parquet": sample_path.as_posix(),
                    "trial_csv": trial_path.as_posix(),
                    "non_trial_events_csv": event_path.as_posix(),
                    "n_trials": int(len(trials)),
                    "n_samples": int(trials["n_samples"].sum()),
                    "file_size_mb": sample_path.stat().st_size / (1024 * 1024),
                }
            )

    all_trials = pd.concat(trial_frames, ignore_index=True)
    all_events = pd.concat(event_frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)

    if all_trials["trial_id"].duplicated().any():
        duplicated = all_trials.loc[
            all_trials["trial_id"].duplicated(keep=False),
            "trial_id",
        ].tolist()
        raise ValueError(f"Duplicated trial IDs: {duplicated[:20]}")

    block_summary = (
        all_trials.groupby(["subject_id", "task", "block_id"], as_index=False)
        .agg(
            n_trials=("trial_id", "size"),
            n_samples=("n_samples", "sum"),
            median_duration_sec=("duration_sec", "median"),
            median_estimated_hz=("estimated_hz", "median"),
            median_valid_basic_rate=("valid_basic_rate", "median"),
            min_valid_basic_rate=("valid_basic_rate", "min"),
        )
    )

    all_trials_path = output_dir / f"{subject}_all_trials.csv"
    all_events_path = output_dir / f"{subject}_all_non_trial_events.csv"
    block_summary_path = output_dir / f"{subject}_block_summary.csv"
    manifest_path = output_dir / f"{subject}_sample_manifest.csv"

    all_trials.to_csv(all_trials_path, index=False, encoding="utf-8-sig")
    all_events.to_csv(all_events_path, index=False, encoding="utf-8-sig")
    block_summary.to_csv(
        block_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print("\n=== SUBJECT EXPORT SUMMARY ===")
    for task in ("NR", "TSR"):
        subset = all_trials[all_trials["task"] == task]
        actual = len(subset)
        expected = EXPECTED_TRIALS[task]
        status = "OK" if actual == expected else "CHECK"
        print(
            f"{task}: trials={actual}, expected={expected}, "
            f"status={status}, samples={int(subset['n_samples'].sum())}, "
            f"median_valid={subset['valid_basic_rate'].median():.6f}"
        )

    print(f"total trials: {len(all_trials)}")
    print(f"total samples: {int(all_trials['n_samples'].sum())}")
    print(f"sample Parquet files: {len(manifest)}")
    print("\nWritten:")
    print(f"  {all_trials_path}")
    print(f"  {all_events_path}")
    print(f"  {block_summary_path}")
    print(f"  {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
