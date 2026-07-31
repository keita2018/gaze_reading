#!/usr/bin/env python
"""Extract and combine ZuCo Soft-AOI features for every complete subject.

For each subject this driver:
1. validates 14 raw-sample Parquet blocks and 739 trials;
2. creates the task-level sentence-index map when needed;
3. invokes extract_zuco_soft_aoi_features.py;
4. validates the per-subject feature CSV;
5. combines all subjects into one 13,302-row feature table.

Existing valid per-subject outputs are skipped by default, so interrupted runs
can be resumed safely. Pass --overwrite to recompute them.
"""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS_PER_SUBJECT = 739
EXPECTED_TASK_COUNTS = {"NR": 349, "TSR": 390}
EXPECTED_PARQUETS = 14


def parse_subjects(text: str | None) -> list[str] | None:
    if text is None:
        return None
    values = sorted(
        {
            item.strip().upper()
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values:
        raise ValueError("--subjects did not contain any subject IDs.")
    return values


def discover_subjects(interim_root: Path) -> list[str]:
    subjects: list[str] = []
    for directory in sorted(interim_root.iterdir()):
        if not directory.is_dir():
            continue
        subject = directory.name.upper()
        parquets = list(directory.glob(f"{subject}_*_samples.parquet"))
        trial_file = directory / f"{subject}_all_trials.csv"
        if parquets or trial_file.is_file():
            subjects.append(subject)
    return subjects


def validate_raw_subject(interim_root: Path, subject: str) -> pd.DataFrame:
    subject_dir = interim_root / subject
    if not subject_dir.is_dir():
        raise FileNotFoundError(subject_dir)

    parquets = sorted(subject_dir.glob(f"{subject}_*_samples.parquet"))
    if len(parquets) != EXPECTED_PARQUETS:
        raise ValueError(
            f"{subject}: found {len(parquets)} Parquet files; "
            f"expected {EXPECTED_PARQUETS}."
        )

    trial_path = subject_dir / f"{subject}_all_trials.csv"
    if not trial_path.is_file():
        raise FileNotFoundError(trial_path)
    trials = pd.read_csv(trial_path)
    if len(trials) != EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(
            f"{subject}: trials={len(trials)}, "
            f"expected={EXPECTED_TRIALS_PER_SUBJECT}."
        )
    if trials["trial_id"].duplicated().any():
        raise ValueError(f"{subject}: duplicate trial IDs in {trial_path}.")

    task_counts = (
        trials["task"].astype(str).str.upper().value_counts().to_dict()
    )
    for task, expected in EXPECTED_TASK_COUNTS.items():
        actual = int(task_counts.get(task, 0))
        if actual != expected:
            raise ValueError(
                f"{subject}: {task} trials={actual}, expected={expected}."
            )
    return trials


def load_mapper(mapper_path: Path):
    if not mapper_path.is_file():
        raise FileNotFoundError(mapper_path)
    namespace = runpy.run_path(str(mapper_path))
    mapper = namespace.get("add_original_sentence_indices")
    if mapper is None:
        raise ValueError(
            f"add_original_sentence_indices was not found in {mapper_path}."
        )
    return mapper


def ensure_trial_map(
    trials: pd.DataFrame,
    subject: str,
    map_path: Path,
    mapper: Any,
    overwrite: bool,
) -> pd.DataFrame:
    if map_path.is_file() and not overwrite:
        mapped = pd.read_csv(map_path)
    else:
        mapped = mapper(trials)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        mapped.to_csv(map_path, index=False, encoding="utf-8-sig")
        print(f"Created trial map: {map_path}")

    required = {
        "subject_id",
        "task",
        "trial_id",
        "task_sentence_index",
    }
    missing = sorted(required - set(mapped.columns))
    if missing:
        raise ValueError(f"{subject}: trial map missing columns: {missing}")
    if len(mapped) != EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(
            f"{subject}: mapped trials={len(mapped)}, "
            f"expected={EXPECTED_TRIALS_PER_SUBJECT}."
        )
    if mapped["trial_id"].duplicated().any():
        raise ValueError(f"{subject}: duplicate trial IDs in trial map.")
    mapped_subjects = set(
        mapped["subject_id"].astype(str).str.upper().unique()
    )
    if mapped_subjects != {subject}:
        raise ValueError(
            f"{subject}: unexpected subject IDs in trial map: "
            f"{sorted(mapped_subjects)}"
        )
    key_columns = ["task", "task_sentence_index"]
    if mapped.duplicated(key_columns).any():
        raise ValueError(f"{subject}: duplicate sentence keys in trial map.")
    return mapped


def validate_feature_file(
    feature_path: Path,
    subject: str,
    expected_columns: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    features = pd.read_csv(feature_path)
    if len(features) != EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(
            f"{subject}: feature rows={len(features)}, "
            f"expected={EXPECTED_TRIALS_PER_SUBJECT}."
        )
    if features["trial_id"].duplicated().any():
        raise ValueError(f"{subject}: duplicate feature trial IDs.")
    feature_subjects = set(
        features["subject_id"].astype(str).str.upper().unique()
    )
    if feature_subjects != {subject}:
        raise ValueError(
            f"{subject}: unexpected feature subject IDs: "
            f"{sorted(feature_subjects)}"
        )
    task_counts = (
        features["task"].astype(str).str.upper().value_counts().to_dict()
    )
    for task, expected in EXPECTED_TASK_COUNTS.items():
        if int(task_counts.get(task, 0)) != expected:
            raise ValueError(
                f"{subject}: feature task count mismatch for {task}."
            )

    columns = list(features.columns)
    if expected_columns is not None:
        missing_columns = [
            column for column in expected_columns if column not in columns
        ]
        extra_columns = [
            column for column in columns if column not in expected_columns
        ]
        if missing_columns or extra_columns:
            raise ValueError(
                f"{subject}: feature column sets differ from the first "
                f"subject; missing={missing_columns}, extra={extra_columns}."
            )

        # Some trials can have no valid or no active Soft-AOI samples. In that
        # case the extractor inserts otherwise identical feature keys in a
        # different dictionary order, which can change CSV column order across
        # subjects. Treat order-only differences as valid and align every
        # subject to the first subject before concatenation.
        if columns != expected_columns:
            features = features.reindex(columns=expected_columns)
            columns = list(expected_columns)
            print(
                f"Aligned {subject} feature column order to the first "
                "subject."
            )
    return features, columns


def invoke_extractor(
    extractor: Path,
    subject: str,
    interim_dir: Path,
    map_path: Path,
    word_template: Path,
    output_dir: Path,
    rate_hz: int,
    sigma_ratio: float,
    active_threshold: float,
) -> None:
    command = [
        sys.executable,
        str(extractor),
        "--subject",
        subject,
        "--interim-dir",
        str(interim_dir),
        "--trial-map",
        str(map_path),
        "--word-template",
        str(word_template),
        "--output-dir",
        str(output_dir),
        "--rate-hz",
        str(rate_hz),
        "--sigma-word-height-ratio",
        str(sigma_ratio),
        "--active-threshold",
        str(active_threshold),
        "--expected-trials",
        str(EXPECTED_TRIALS_PER_SUBJECT),
    ]
    print("Running: " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interim-root",
        default="data/interim/zuco2",
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed/zuco2",
    )
    parser.add_argument(
        "--word-template",
        default=(
            "data/processed/zuco2/soft_aoi_template/"
            "zuco_soft_aoi_word_template.csv"
        ),
    )
    parser.add_argument(
        "--extractor",
        default="src/data/extract_zuco_soft_aoi_features.py",
    )
    parser.add_argument(
        "--mapper",
        default="src/data/build_zuco_raw_official_trial_map.py",
    )
    parser.add_argument(
        "--combined-output-dir",
        default="data/processed/zuco2/soft_aoi_all_subjects",
    )
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--rate-hz", type=int, default=30)
    parser.add_argument(
        "--sigma-word-height-ratio",
        type=float,
        default=2.0 / 3.0,
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=EXPECTED_SUBJECTS,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-maps", action="store_true")
    args = parser.parse_args()

    interim_root = Path(args.interim_root)
    processed_root = Path(args.processed_root)
    word_template = Path(args.word_template)
    extractor = Path(args.extractor)
    mapper_path = Path(args.mapper)
    combined_output_dir = Path(args.combined_output_dir)

    if not interim_root.is_dir():
        raise FileNotFoundError(interim_root)
    if not word_template.is_file():
        raise FileNotFoundError(word_template)
    if not extractor.is_file():
        raise FileNotFoundError(extractor)

    requested_subjects = parse_subjects(args.subjects)
    subjects = requested_subjects or discover_subjects(interim_root)
    subjects = sorted(subjects)
    if len(subjects) != args.expected_subjects:
        raise ValueError(
            f"subjects={len(subjects)}, expected={args.expected_subjects}: "
            f"{subjects}"
        )

    mapper = load_mapper(mapper_path)
    combined_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    expected_columns: list[str] | None = None

    for index, subject in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(f"[{index}/{len(subjects)}] Subject {subject}")
        trials = validate_raw_subject(interim_root, subject)

        subject_processed = processed_root / subject
        map_path = (
            subject_processed
            / f"{subject}_all_trials_with_sentence_index.csv"
        )
        ensure_trial_map(
            trials=trials,
            subject=subject,
            map_path=map_path,
            mapper=mapper,
            overwrite=args.overwrite_maps,
        )

        output_dir = subject_processed / "soft_aoi"
        feature_path = output_dir / (
            f"{subject}_soft_aoi_features_{args.rate_hz}hz.csv"
        )

        can_skip = False
        if feature_path.is_file() and not args.overwrite:
            try:
                features, columns = validate_feature_file(
                    feature_path,
                    subject,
                    expected_columns,
                )
                can_skip = True
                print(f"Skipping valid existing output: {feature_path}")
            except Exception as exc:
                print(f"Existing output is invalid and will be rebuilt: {exc}")

        if not can_skip:
            output_dir.mkdir(parents=True, exist_ok=True)
            invoke_extractor(
                extractor=extractor,
                subject=subject,
                interim_dir=interim_root / subject,
                map_path=map_path,
                word_template=word_template,
                output_dir=output_dir,
                rate_hz=args.rate_hz,
                sigma_ratio=args.sigma_word_height_ratio,
                active_threshold=args.active_threshold,
            )
            features, columns = validate_feature_file(
                feature_path,
                subject,
                expected_columns,
            )

        if expected_columns is None:
            expected_columns = columns

        numeric = features.select_dtypes(include=[np.number])
        inf_cells = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        nan_cells = int(numeric.isna().sum().sum())
        combined_frames.append(features)
        summary_rows.append(
            {
                "subject_id": subject,
                "trials": len(features),
                "nr_trials": int(features["task"].eq("NR").sum()),
                "tsr_trials": int(features["task"].eq("TSR").sum()),
                "nan_numeric_cells": nan_cells,
                "inf_numeric_cells": inf_cells,
                "median_valid_rate": float(
                    features["soft_aoi_valid_rate"].median()
                ),
                "median_active_rate": float(
                    features["soft_aoi_active_rate"].median()
                ),
                "median_qmax_mean": float(
                    features["soft_aoi_qmax_mean"].median()
                ),
            }
        )
        print(
            f"Validated {subject}: rows={len(features)}, "
            f"nan_numeric_cells={nan_cells}, inf_numeric_cells={inf_cells}"
        )

    combined = pd.concat(combined_frames, ignore_index=True)
    expected_rows = len(subjects) * EXPECTED_TRIALS_PER_SUBJECT
    if len(combined) != expected_rows:
        raise ValueError(
            f"Combined rows={len(combined)}, expected={expected_rows}."
        )
    if combined["trial_id"].duplicated().any():
        raise ValueError("Combined output contains duplicate trial IDs.")
    if combined["subject_id"].nunique() != len(subjects):
        raise ValueError("Combined subject count mismatch.")

    combined_output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_output_dir / (
        f"zuco2_soft_aoi_features_{args.rate_hz}hz_all_subjects.csv"
    )
    subject_summary_path = combined_output_dir / (
        f"zuco2_soft_aoi_subject_summary_{args.rate_hz}hz.csv"
    )
    task_summary_path = combined_output_dir / (
        f"zuco2_soft_aoi_task_summary_{args.rate_hz}hz.csv"
    )
    manifest_path = combined_output_dir / (
        f"zuco2_soft_aoi_combined_manifest_{args.rate_hz}hz.json"
    )

    combined.sort_values(
        ["subject_id", "task", "task_sentence_index"]
    ).to_csv(combined_path, index=False, encoding="utf-8-sig")

    subject_summary = pd.DataFrame(summary_rows).sort_values("subject_id")
    subject_summary.to_csv(
        subject_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    task_summary = combined.groupby("task").agg(
        trials=("trial_id", "size"),
        subjects=("subject_id", "nunique"),
        median_valid_rate=("soft_aoi_valid_rate", "median"),
        median_active_rate=("soft_aoi_active_rate", "median"),
        median_box_hit_rate=("soft_aoi_word_box_hit_rate", "median"),
        median_qmax_mean=("soft_aoi_qmax_mean", "median"),
        median_occupancy_entropy=(
            "soft_aoi_occupancy_entropy_norm",
            "median",
        ),
        median_word_coverage=(
            "soft_aoi_dominant_word_coverage",
            "median",
        ),
        median_progress_slope=(
            "soft_aoi_progress_slope_per_sec",
            "median",
        ),
        median_backward_fraction=(
            "soft_aoi_backward_transition_fraction",
            "median",
        ),
    ).reset_index()
    task_summary.to_csv(
        task_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "subjects": subjects,
        "n_subjects": len(subjects),
        "trials_per_subject": EXPECTED_TRIALS_PER_SUBJECT,
        "combined_trials": len(combined),
        "sampling_rate_hz": args.rate_hz,
        "sigma_word_height_ratio": args.sigma_word_height_ratio,
        "active_threshold": args.active_threshold,
        "feature_columns": expected_columns,
        "combined_feature_file": str(combined_path),
        "subject_summary_file": str(subject_summary_path),
        "task_summary_file": str(task_summary_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    numeric = combined.select_dtypes(include=[np.number])
    total_nan = int(numeric.isna().sum().sum())
    total_inf = int(np.isinf(numeric.to_numpy(dtype=float)).sum())

    print("\n" + "=" * 78)
    print("=== ALL-SUBJECT SOFT-AOI SUMMARY ===")
    print(f"subjects: {combined['subject_id'].nunique()}")
    print(f"trials: {len(combined)}")
    print(f"unique trial IDs: {combined['trial_id'].nunique()}")
    print(f"columns: {len(combined.columns)}")
    print(f"numeric NaN cells: {total_nan}")
    print(f"numeric infinite cells: {total_inf}")
    print("\n=== TASK SUMMARY ===")
    print(task_summary.to_string(index=False))
    print("\nWritten:")
    print(f"  {combined_path}")
    print(f"  {subject_summary_path}")
    print(f"  {task_summary_path}")
    print(f"  {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
