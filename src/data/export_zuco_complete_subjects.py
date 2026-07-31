#!/usr/bin/env python
"""Export complete ZuCo 2.0 raw ET subjects and build combined manifests.

For each selected subject, this script calls the existing
export_zuco_subject_all_blocks.py driver. It then combines subject-level
trial tables and sample manifests.

Selection modes:
- complete_raw: all subjects with all 14 raw ET files
- official_overlap: subjects with all 14 raw ET files and local official
  sent_gaze_sacc features
- explicit subjects supplied with --subjects

Expected complete raw output per subject:
- NR: 349 trials
- TSR: 390 trials
- total: 739 trials
- 14 sample Parquet files
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_NR = 349
EXPECTED_TSR = 390
EXPECTED_TOTAL = 739


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
            raise ValueError("--subjects did not contain any IDs.")
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
        raise ValueError(mode)

    return sorted(selected["subject_id"].tolist())


def run_subject_export(
    driver: Path,
    exporter: Path,
    subject: str,
    raw_root: Path,
    output_root: Path,
    skip_existing: bool,
) -> None:
    raw_dir = raw_root / subject / "session_et" / "raw"
    output_dir = output_root / subject

    command = [
        sys.executable,
        str(driver),
        "--subject",
        subject,
        "--raw-dir",
        str(raw_dir),
        "--output-dir",
        str(output_dir),
        "--exporter",
        str(exporter),
    ]
    if skip_existing:
        command.append("--skip-existing")

    print("\n" + "=" * 72)
    print(f"Exporting subject {subject}")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
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
        "--raw-root",
        default="data/raw/zuco2",
    )
    parser.add_argument(
        "--output-root",
        default="data/interim/zuco2",
    )
    parser.add_argument(
        "--driver",
        default="src/data/export_zuco_subject_all_blocks.py",
    )
    parser.add_argument(
        "--exporter",
        default="src/data/export_zuco_session_trials.py",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    coverage = pd.read_csv(args.coverage)
    coverage["subject_id"] = (
        coverage["subject_id"].astype(str).str.upper()
    )
    coverage_lookup = coverage.set_index("subject_id")

    subjects = select_subjects(
        coverage,
        mode=args.mode,
        explicit=args.subjects,
    )
    if not subjects:
        raise ValueError("No subjects selected.")

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    driver = Path(args.driver)
    exporter = Path(args.exporter)

    if not driver.is_file():
        raise FileNotFoundError(driver)
    if not exporter.is_file():
        raise FileNotFoundError(exporter)

    print("=== MULTI-SUBJECT EXPORT PLAN ===")
    print(f"mode: {args.mode}")
    print(f"subjects ({len(subjects)}): {', '.join(subjects)}")
    print(f"expected total trials: {len(subjects) * EXPECTED_TOTAL}")
    print(f"expected Parquet files: {len(subjects) * 14}")

    for subject in subjects:
        run_subject_export(
            driver=driver,
            exporter=exporter,
            subject=subject,
            raw_root=raw_root,
            output_root=output_root,
            skip_existing=args.skip_existing,
        )

    trial_frames: list[pd.DataFrame] = []
    manifest_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    subject_rows: list[dict[str, Any]] = []

    for subject in subjects:
        subject_dir = output_root / subject
        trials_path = subject_dir / f"{subject}_all_trials.csv"
        manifest_path = subject_dir / f"{subject}_sample_manifest.csv"
        events_path = (
            subject_dir / f"{subject}_all_non_trial_events.csv"
        )

        for path in (trials_path, manifest_path, events_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        trials = pd.read_csv(trials_path)
        manifest = pd.read_csv(manifest_path)
        events = pd.read_csv(events_path)

        task_counts = trials.groupby("task").size().to_dict()
        nr_count = int(task_counts.get("NR", 0))
        tsr_count = int(task_counts.get("TSR", 0))

        if (
            nr_count != EXPECTED_NR
            or tsr_count != EXPECTED_TSR
            or len(trials) != EXPECTED_TOTAL
        ):
            raise ValueError(
                f"{subject} trial count mismatch: "
                f"NR={nr_count}, TSR={tsr_count}, total={len(trials)}"
            )

        if trials["trial_id"].duplicated().any():
            raise ValueError(f"Duplicate trial IDs within {subject}.")
        if len(manifest) != 14:
            raise ValueError(
                f"{subject} manifest has {len(manifest)} rows, expected 14."
            )

        has_official = int(
            pd.to_numeric(
                pd.Series(
                    [
                        coverage_lookup.loc[
                            subject,
                            "has_local_official_sent_gaze_sacc",
                        ]
                    ]
                ),
                errors="coerce",
            ).fillna(0).iloc[0]
        )

        trials["has_local_official_sent_gaze_sacc"] = has_official
        manifest["has_local_official_sent_gaze_sacc"] = has_official
        events["has_local_official_sent_gaze_sacc"] = has_official

        trial_frames.append(trials)
        manifest_frames.append(manifest)
        event_frames.append(events)

        subject_rows.append(
            {
                "subject_id": subject,
                "has_local_official_sent_gaze_sacc": has_official,
                "n_trials": len(trials),
                "n_nr": nr_count,
                "n_tsr": tsr_count,
                "n_samples": int(trials["n_samples"].sum()),
                "n_parquet_files": len(manifest),
                "median_duration_sec": float(
                    trials["duration_sec"].median()
                ),
                "median_valid_basic_rate": float(
                    trials["valid_basic_rate"].median()
                ),
                "min_valid_basic_rate": float(
                    trials["valid_basic_rate"].min()
                ),
                "total_parquet_size_mb": float(
                    manifest["file_size_mb"].sum()
                ),
            }
        )

    all_trials = pd.concat(trial_frames, ignore_index=True)
    all_manifests = pd.concat(manifest_frames, ignore_index=True)
    all_events = pd.concat(event_frames, ignore_index=True)
    subject_summary = pd.DataFrame(subject_rows)

    if all_trials["trial_id"].duplicated().any():
        duplicate_ids = all_trials.loc[
            all_trials["trial_id"].duplicated(keep=False),
            "trial_id",
        ].tolist()
        raise ValueError(
            f"Duplicate trial IDs across subjects: {duplicate_ids[:20]}"
        )

    expected_all_trials = len(subjects) * EXPECTED_TOTAL
    if len(all_trials) != expected_all_trials:
        raise ValueError(
            f"Combined trial count={len(all_trials)}, "
            f"expected={expected_all_trials}"
        )

    combined_dir = output_root / "all_subjects"
    combined_dir.mkdir(parents=True, exist_ok=True)
    mode_prefix = (
        "complete_raw"
        if args.subjects is None
        else "explicit_subjects"
    )

    trials_out = (
        combined_dir / f"{mode_prefix}_all_trials.csv"
    )
    manifests_out = (
        combined_dir / f"{mode_prefix}_sample_manifest.csv"
    )
    events_out = (
        combined_dir / f"{mode_prefix}_all_non_trial_events.csv"
    )
    summary_out = (
        combined_dir / f"{mode_prefix}_subject_summary.csv"
    )

    all_trials.to_csv(
        trials_out,
        index=False,
        encoding="utf-8-sig",
    )
    all_manifests.to_csv(
        manifests_out,
        index=False,
        encoding="utf-8-sig",
    )
    all_events.to_csv(
        events_out,
        index=False,
        encoding="utf-8-sig",
    )
    subject_summary.to_csv(
        summary_out,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== MULTI-SUBJECT EXPORT SUMMARY ===")
    print(subject_summary.to_string(index=False))
    print()
    print(f"subjects: {len(subjects)}")
    print(f"trials: {len(all_trials)}")
    print(f"unique trial IDs: {all_trials['trial_id'].nunique()}")
    print(f"NR trials: {(all_trials['task'] == 'NR').sum()}")
    print(f"TSR trials: {(all_trials['task'] == 'TSR').sum()}")
    print(f"samples: {int(all_trials['n_samples'].sum())}")
    print(f"Parquet files: {len(all_manifests)}")
    print(
        "subjects with official features: "
        f"{subject_summary['has_local_official_sent_gaze_sacc'].sum()}"
    )
    print("\nWritten:")
    print(f"  {trials_out}")
    print(f"  {manifests_out}")
    print(f"  {events_out}")
    print(f"  {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
