#!/usr/bin/env python
"""Prepare only the Soft-AOI caches needed for five-repeat final comparisons.

This script intentionally bypasses the RawStat evaluation wrappers.  The
Soft-vs-Hard comparison scripts require Soft-AOI caches to exist in advance,
but they do not need RawStat caches.  We therefore call the existing
deterministic Soft-AOI extraction functions directly for:

- coordinate noise: sigma=50 px;
- missingness: random/burst 40%;
- combined degradation: high_random/high_burst.

Existing repeats are validated and reused.  Only missing repeats are computed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import run_zuco_soft_aoi_combined_degradation_30hz as combined_module
from run_zuco_soft_aoi_combined_degradation_30hz import (
    extract_or_load_soft_conditions as extract_combined_soft,
)
from run_zuco_soft_aoi_coordinate_noise_30hz import (
    extract_or_load_soft_noise,
    load_word_template,
)
from run_zuco_soft_aoi_missingness_30hz import (
    extract_or_load_soft_conditions as extract_missingness_soft,
)


EXPECTED_SUBJECTS = 18
TARGET_COMBINED = {"high_random", "high_burst"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "data/interim/zuco2/all_subjects/"
            "complete_raw_sample_manifest.csv"
        ),
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
        "--noise-cache-root",
        default=(
            "outputs/soft_aoi_coordinate_noise_30hz/"
            "subject_soft_aoi_noise_features"
        ),
    )
    parser.add_argument(
        "--missingness-cache-root",
        default=(
            "outputs/soft_aoi_missingness_30hz/"
            "subject_soft_aoi_missingness_features"
        ),
    )
    parser.add_argument(
        "--combined-cache-root",
        default=(
            "outputs/soft_aoi_combined_degradation_30hz/"
            "subject_soft_aoi_combined_features"
        ),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--burst-seconds", type=float, default=1.0)
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
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.burst_seconds <= 0:
        raise ValueError("--burst-seconds must be positive.")
    if args.sigma_word_height_ratio <= 0:
        raise ValueError(
            "--sigma-word-height-ratio must be positive."
        )
    if not (0 < args.active_threshold <= 1):
        raise ValueError(
            "--active-threshold must be within (0, 1]."
        )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)
    required = {
        "subject_id",
        "task",
        "block_id",
        "sample_parquet",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"Manifest lacks columns: {missing}"
        )
    manifest["subject_id"] = (
        manifest["subject_id"].astype(str).str.upper()
    )
    manifest["task"] = (
        manifest["task"].astype(str).str.upper()
    )
    subjects = sorted(manifest["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Manifest subjects={len(subjects)}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    processed_root = Path(args.processed_root)
    aoi_lookup = load_word_template(
        Path(args.word_template)
    )

    print(
        "=== PREPARE TARGET SOFT-AOI FIVE-REPEAT CACHES ==="
    )
    print(f"subjects: {len(subjects)}")
    print(f"repeats: {args.repeats}")
    print("noise target: sigma=50 px")
    print("missingness target: random/burst 40%")
    print(
        "combined targets: high_random, high_burst"
    )

    print("\n" + "=" * 78)
    print("1/3 Coordinate-noise Soft-AOI caches")
    extract_or_load_soft_noise(
        manifest=manifest,
        subjects=subjects,
        processed_root=processed_root,
        cache_root=Path(args.noise_cache_root),
        aoi_lookup=aoi_lookup,
        sigmas=[50.0],
        repeats=args.repeats,
        seed=args.seed,
        sigma_word_height_ratio=(
            args.sigma_word_height_ratio
        ),
        active_threshold=args.active_threshold,
        overwrite=args.overwrite,
    )

    print("\n" + "=" * 78)
    print("2/3 Missingness Soft-AOI caches")
    extract_missingness_soft(
        manifest=manifest,
        subjects=subjects,
        processed_root=processed_root,
        cache_root=Path(args.missingness_cache_root),
        aoi_lookup=aoi_lookup,
        percentages=[40.0],
        repeats=args.repeats,
        seed=args.seed,
        burst_seconds=args.burst_seconds,
        sigma_word_height_ratio=(
            args.sigma_word_height_ratio
        ),
        active_threshold=args.active_threshold,
        overwrite=args.overwrite,
    )

    print("\n" + "=" * 78)
    print("3/3 High-combined Soft-AOI caches")
    original_conditions = list(
        combined_module.CONDITIONS
    )
    target_conditions = [
        condition
        for condition in original_conditions
        if condition.name in TARGET_COMBINED
    ]
    names = {
        condition.name
        for condition in target_conditions
    }
    if names != TARGET_COMBINED:
        raise ValueError(
            "Combined target conditions missing: "
            f"{sorted(TARGET_COMBINED - names)}"
        )

    # The existing extractor reads the module-level CONDITIONS list.
    # Narrow it only for this cache-preparation process.
    combined_module.CONDITIONS = target_conditions
    try:
        extract_combined_soft(
            manifest=manifest,
            subjects=subjects,
            processed_root=processed_root,
            cache_root=Path(args.combined_cache_root),
            aoi_lookup=aoi_lookup,
            repeats=args.repeats,
            seed=args.seed,
            burst_seconds=args.burst_seconds,
            sigma_word_height_ratio=(
                args.sigma_word_height_ratio
            ),
            active_threshold=args.active_threshold,
            overwrite=args.overwrite,
        )
    finally:
        combined_module.CONDITIONS = original_conditions

    print(
        "\n=== TARGET SOFT-AOI CACHE PREPARATION COMPLETE ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
