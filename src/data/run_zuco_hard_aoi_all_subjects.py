#!/usr/bin/env python
"""Extract all-subject confidence-weighted Hard-AOI features from ZuCo raw gaze.

Primary purpose
---------------
Create the Hard-AOI side of the direct Soft-AOI vs Hard-AOI comparison.

The comparison intentionally keeps the following components identical to the
existing Soft-AOI extractor:

- raw gaze samples;
- temporal downsampling;
- word rectangles;
- Gaussian distance membership q_j(t);
- active criterion max_j q_j(t) >= threshold;
- q_max confidence weighting;
- membership/quality measurements.

Only the allocation rule changes:

Soft-AOI
    A sample contributes continuously to multiple word AOIs.

Hard-AOI
    A sample is assigned to the single word with argmax_j q_j(t). Its total
    contribution is still weighted by q_max, so the experiment isolates the
    effect of soft multi-AOI allocation from distance confidence.

The two Soft-AOI sample-entropy features are not defined as informative Hard-AOI
features because one-hot assignment has zero entropy. The primary comparison
therefore uses 13 shared distribution summaries plus 12 identical
membership/quality summaries: 25 features in total.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extract_zuco_soft_aoi_features import (
    downsample_trial,
    normalized_entropy,
    rectangle_distances,
    safe_fraction,
    to_bool_array,
    weighted_mean,
    weighted_quantile,
    weighted_std,
)


EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_PER_SUBJECT = 739
EXPECTED_TASK_COUNTS_PER_SUBJECT = {"NR": 349, "TSR": 390}

SOFT_DISTRIBUTION_COMMON = [
    "soft_aoi_occupancy_entropy_norm",
    "soft_aoi_occupancy_effective_word_fraction",
    "soft_aoi_occupancy_top1_share",
    "soft_aoi_occupancy_top3_share",
    "soft_aoi_dominant_word_coverage",
    "soft_aoi_dominant_line_coverage",
    "soft_aoi_progress_mean",
    "soft_aoi_progress_std",
    "soft_aoi_progress_q25",
    "soft_aoi_progress_median",
    "soft_aoi_progress_q75",
    "soft_aoi_line_progress_mean",
    "soft_aoi_line_position_mean",
]

SOFT_QUALITY = [
    "soft_aoi_valid_rate",
    "soft_aoi_active_rate",
    "soft_aoi_word_box_hit_rate",
    "soft_aoi_sentence_envelope_hit_rate",
    "soft_aoi_qmax_mean",
    "soft_aoi_qmax_median",
    "soft_aoi_qmax_q10",
    "soft_aoi_qmax_ge_0_5_rate",
    "soft_aoi_nearest_distance_mean_word_heights",
    "soft_aoi_nearest_distance_median_word_heights",
    "soft_aoi_nearest_distance_q90_word_heights",
    "soft_aoi_nearest_distance_q95_word_heights",
]


def hard_name(soft_name: str) -> str:
    if not soft_name.startswith("soft_aoi_"):
        raise ValueError(f"Unexpected Soft-AOI feature name: {soft_name}")
    return "hard_aoi_" + soft_name.removeprefix("soft_aoi_")


HARD_DISTRIBUTION_COMMON = [hard_name(name) for name in SOFT_DISTRIBUTION_COMMON]
HARD_QUALITY = [hard_name(name) for name in SOFT_QUALITY]
HARD_COMMON25 = HARD_DISTRIBUTION_COMMON + HARD_QUALITY


def load_word_template(path: Path) -> dict[tuple[str, int], pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {
        "task",
        "task_sentence_index",
        "word_index",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "height",
        "word_progress",
        "line_index",
        "line_progress",
        "line_index_norm",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Word template lacks columns: {missing}")

    frame["task"] = frame["task"].astype(str).str.upper()
    frame["task_sentence_index"] = pd.to_numeric(
        frame["task_sentence_index"], errors="raise"
    ).astype(int)

    lookup = {
        (str(task), int(sentence_index)): group.sort_values("word_index")
        for (task, sentence_index), group in frame.groupby(
            ["task", "task_sentence_index"], sort=False
        )
    }
    if len(lookup) != 739:
        raise ValueError(
            f"Word-template sentence count={len(lookup)}, expected=739."
        )
    return lookup


def load_trial_map(path: Path, subject: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {
        "subject_id",
        "task",
        "block_id",
        "trial_index_in_block",
        "trial_id",
        "task_sentence_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{subject} trial map lacks columns: {missing}")

    frame["trial_id"] = frame["trial_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str).str.upper()
    frame["task"] = frame["task"].astype(str).str.upper()
    frame["task_sentence_index"] = pd.to_numeric(
        frame["task_sentence_index"], errors="raise"
    ).astype(int)

    if len(frame) != EXPECTED_PER_SUBJECT:
        raise ValueError(
            f"{subject} trial-map rows={len(frame)}, "
            f"expected={EXPECTED_PER_SUBJECT}."
        )
    if frame["trial_id"].duplicated().any():
        raise ValueError(f"{subject} trial map contains duplicate trial IDs.")
    if set(frame["subject_id"]) != {subject}:
        raise ValueError(f"{subject} trial map has unexpected subject IDs.")

    counts = frame.groupby("task").size().to_dict()
    if counts != EXPECTED_TASK_COUNTS_PER_SUBJECT:
        raise ValueError(
            f"{subject} trial-map task counts={counts}, "
            f"expected={EXPECTED_TASK_COUNTS_PER_SUBJECT}."
        )
    return frame


def extract_hard_trial_features(
    samples: pd.DataFrame,
    trial_row: pd.Series,
    aoi: pd.DataFrame,
    rate_hz: int,
    sigma_word_height_ratio: float,
    active_threshold: float,
) -> dict[str, Any]:
    """Extract 25 Hard-AOI-comparable trial-level features."""
    downsampled = downsample_trial(samples, rate_hz)
    n_samples = int(len(downsampled))

    x_all = downsampled["x_px"].to_numpy(dtype=float)
    y_all = downsampled["y_px"].to_numpy(dtype=float)
    valid = (
        to_bool_array(downsampled["valid_basic"])
        & np.isfinite(x_all)
        & np.isfinite(y_all)
        & (x_all > 0)
        & (y_all > 0)
    )
    valid_samples = downsampled.loc[valid].copy()
    n_valid = int(len(valid_samples))

    n_words = int(len(aoi))
    n_lines = int(aoi["line_index"].max()) + 1
    median_word_height = float(aoi["height"].median())
    if not math.isfinite(median_word_height) or median_word_height <= 0:
        raise ValueError(
            f"Invalid median word height for {trial_row['trial_id']}: "
            f"{median_word_height}"
        )
    sigma_px = float(sigma_word_height_ratio * median_word_height)
    if sigma_px <= 0:
        raise ValueError("Derived sigma must be positive.")

    base: dict[str, Any] = {
        "subject_id": str(trial_row["subject_id"]).upper(),
        "task": str(trial_row["task"]).upper(),
        "block_id": int(trial_row["block_id"]),
        "trial_index_in_block": int(trial_row["trial_index_in_block"]),
        "trial_id": str(trial_row["trial_id"]),
        "task_sentence_index": int(trial_row["task_sentence_index"]),
        "hard_aoi_rate_hz": int(rate_hz),
        "hard_aoi_sigma_word_height_ratio": float(sigma_word_height_ratio),
        "hard_aoi_sigma_px": sigma_px,
        "hard_aoi_active_threshold": float(active_threshold),
        "hard_aoi_assignment_rule": "argmax_q_with_qmax_mass",
        "hard_aoi_n_words": n_words,
        "hard_aoi_n_lines": n_lines,
        "hard_aoi_n_samples": n_samples,
        "hard_aoi_n_valid_samples": n_valid,
        "hard_aoi_valid_rate": safe_fraction(n_valid, n_samples),
    }

    if n_valid == 0:
        for name in HARD_COMMON25:
            base.setdefault(name, math.nan)
        return base

    x = valid_samples["x_px"].to_numpy(dtype=float)
    y = valid_samples["y_px"].to_numpy(dtype=float)
    distances = rectangle_distances(x, y, aoi)
    nearest = distances.min(axis=1)

    q = np.exp(-(distances**2) / (2.0 * sigma_px**2))
    q_max = q.max(axis=1)
    dominant = q.argmax(axis=1)
    active = q_max >= active_threshold
    n_active = int(np.sum(active))

    sentence_x_min = float(aoi["x_min"].min())
    sentence_x_max = float(aoi["x_max"].max())
    sentence_y_min = float(aoi["y_min"].min())
    sentence_y_max = float(aoi["y_max"].max())
    envelope_hit = (
        (x >= sentence_x_min)
        & (x <= sentence_x_max)
        & (y >= sentence_y_min)
        & (y <= sentence_y_max)
    )

    base.update(
        {
            "hard_aoi_active_rate": safe_fraction(n_active, n_valid),
            "hard_aoi_word_box_hit_rate": float(np.mean(nearest == 0)),
            "hard_aoi_sentence_envelope_hit_rate": float(
                np.mean(envelope_hit)
            ),
            "hard_aoi_qmax_mean": float(np.mean(q_max)),
            "hard_aoi_qmax_median": float(np.median(q_max)),
            "hard_aoi_qmax_q10": float(np.quantile(q_max, 0.10)),
            "hard_aoi_qmax_ge_0_5_rate": float(np.mean(q_max >= 0.5)),
            "hard_aoi_nearest_distance_mean_word_heights": float(
                np.mean(nearest / median_word_height)
            ),
            "hard_aoi_nearest_distance_median_word_heights": float(
                np.median(nearest / median_word_height)
            ),
            "hard_aoi_nearest_distance_q90_word_heights": float(
                np.quantile(nearest / median_word_height, 0.90)
            ),
            "hard_aoi_nearest_distance_q95_word_heights": float(
                np.quantile(nearest / median_word_height, 0.95)
            ),
        }
    )

    if n_active == 0:
        for name in HARD_DISTRIBUTION_COMMON:
            base[name] = math.nan
        return base

    dominant_active = dominant[active]
    qmax_active = q_max[active]

    word_progress = aoi["word_progress"].to_numpy(dtype=float)
    line_progress = aoi["line_progress"].to_numpy(dtype=float)
    line_position = aoi["line_index_norm"].to_numpy(dtype=float)
    line_index = aoi["line_index"].to_numpy(dtype=int)

    hard_progress = word_progress[dominant_active]
    hard_line_progress = line_progress[dominant_active]
    hard_line_position = line_position[dominant_active]
    hard_lines = line_index[dominant_active]

    # Preserve q_max as confidence mass while forcing a single AOI assignment.
    occupancy_mass = np.bincount(
        dominant_active,
        weights=qmax_active,
        minlength=n_words,
    ).astype(float)
    total_mass = float(occupancy_mass.sum())
    if total_mass <= 0:
        raise RuntimeError("Active Hard-AOI mass is non-positive.")
    occupancy = occupancy_mass / total_mass
    occupancy_sorted = np.sort(occupancy)[::-1]
    occupancy_entropy = normalized_entropy(occupancy)
    effective_fraction = (
        math.exp(occupancy_entropy * math.log(n_words)) / n_words
        if n_words > 1
        else 1.0
    )

    base.update(
        {
            "hard_aoi_occupancy_entropy_norm": float(occupancy_entropy),
            "hard_aoi_occupancy_effective_word_fraction": float(
                effective_fraction
            ),
            "hard_aoi_occupancy_top1_share": float(occupancy_sorted[0]),
            "hard_aoi_occupancy_top3_share": float(
                occupancy_sorted[: min(3, n_words)].sum()
            ),
            "hard_aoi_dominant_word_coverage": float(
                len(np.unique(dominant_active)) / n_words
            ),
            "hard_aoi_dominant_line_coverage": float(
                len(np.unique(hard_lines)) / n_lines
            ),
            "hard_aoi_progress_mean": weighted_mean(
                hard_progress, qmax_active
            ),
            "hard_aoi_progress_std": weighted_std(
                hard_progress, qmax_active
            ),
            "hard_aoi_progress_q25": weighted_quantile(
                hard_progress, qmax_active, 0.25
            ),
            "hard_aoi_progress_median": weighted_quantile(
                hard_progress, qmax_active, 0.50
            ),
            "hard_aoi_progress_q75": weighted_quantile(
                hard_progress, qmax_active, 0.75
            ),
            "hard_aoi_line_progress_mean": weighted_mean(
                hard_line_progress, qmax_active
            ),
            "hard_aoi_line_position_mean": weighted_mean(
                hard_line_position, qmax_active
            ),
        }
    )
    return base


def validate_subject_features(
    frame: pd.DataFrame,
    subject: str,
    expected_trial_ids: set[str],
) -> pd.DataFrame:
    required = {
        "subject_id",
        "task",
        "trial_id",
        "task_sentence_index",
        *HARD_COMMON25,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{subject} Hard-AOI output lacks columns: {missing}")

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = result["subject_id"].astype(str).str.upper()
    result["task"] = result["task"].astype(str).str.upper()

    if len(result) != EXPECTED_PER_SUBJECT:
        raise ValueError(
            f"{subject} Hard-AOI rows={len(result)}, "
            f"expected={EXPECTED_PER_SUBJECT}."
        )
    if result["trial_id"].duplicated().any():
        raise ValueError(f"{subject} Hard-AOI has duplicate trial IDs.")
    if set(result["trial_id"]) != expected_trial_ids:
        raise ValueError(f"{subject} Hard-AOI trial IDs do not match.")
    if set(result["subject_id"]) != {subject}:
        raise ValueError(f"{subject} Hard-AOI subject IDs are invalid.")

    counts = result.groupby("task").size().to_dict()
    if counts != EXPECTED_TASK_COUNTS_PER_SUBJECT:
        raise ValueError(
            f"{subject} Hard-AOI task counts={counts}, "
            f"expected={EXPECTED_TASK_COUNTS_PER_SUBJECT}."
        )

    numeric = result[HARD_COMMON25].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError(f"{subject} Hard-AOI contains infinity.")
    return result


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
        "--output-dir",
        default="data/processed/zuco2/hard_aoi_all_subjects",
    )
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.rate_hz <= 0 or args.rate_hz > 500:
        raise ValueError("--rate-hz must be within 1..500.")
    if args.sigma_word_height_ratio <= 0:
        raise ValueError("--sigma-word-height-ratio must be positive.")
    if not (0 < args.active_threshold <= 1):
        raise ValueError("--active-threshold must be within (0, 1].")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)

    required_manifest = {"subject_id", "sample_parquet"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(
            f"Raw sample manifest lacks columns: {missing_manifest}"
        )
    manifest["subject_id"] = (
        manifest["subject_id"].astype(str).str.upper()
    )
    subjects = sorted(manifest["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Manifest subjects={len(subjects)}, "
            f"expected={EXPECTED_SUBJECTS}: {subjects}"
        )

    processed_root = Path(args.processed_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aoi_lookup = load_word_template(Path(args.word_template))

    combined_frames: list[pd.DataFrame] = []
    for subject_index, subject in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(f"[{subject_index:02d}/{len(subjects)}] {subject}")

        subject_output = (
            output_dir
            / f"{subject}_hard_aoi_features_{args.rate_hz}hz.csv"
        )
        map_path = (
            processed_root
            / subject
            / f"{subject}_all_trials_with_sentence_index.csv"
        )
        trial_map = load_trial_map(map_path, subject)
        expected_trial_ids = set(trial_map["trial_id"])
        trial_lookup = trial_map.set_index(
            "trial_id", drop=False, verify_integrity=True
        )

        if subject_output.is_file() and not args.overwrite:
            print(f"Loading existing: {subject_output}")
            existing = validate_subject_features(
                pd.read_csv(subject_output),
                subject,
                expected_trial_ids,
            )
            combined_frames.append(existing)
            continue

        subject_manifest = manifest[
            manifest["subject_id"].eq(subject)
        ].copy()
        if len(subject_manifest) != 14:
            raise ValueError(
                f"{subject} manifest rows={len(subject_manifest)}, expected=14."
            )

        rows: list[dict[str, Any]] = []
        processed_ids: set[str] = set()
        for block_index, manifest_row in enumerate(
            subject_manifest.itertuples(index=False),
            start=1,
        ):
            sample_path = Path(str(manifest_row.sample_parquet))
            if not sample_path.is_file():
                raise FileNotFoundError(sample_path)
            print(
                f"  [{block_index:02d}/14] "
                f"{sample_path.name}"
            )

            samples = pd.read_parquet(sample_path)
            required_samples = {
                "trial_id",
                "sample_index_in_trial",
                "timestamp_ms_abs",
                "x_px",
                "y_px",
                "valid_basic",
            }
            missing_samples = sorted(
                required_samples - set(samples.columns)
            )
            if missing_samples:
                raise ValueError(
                    f"{sample_path} lacks columns: {missing_samples}"
                )

            samples["trial_id"] = samples["trial_id"].astype(str)
            for trial_id, trial_samples in samples.groupby(
                "trial_id", sort=False
            ):
                trial_id = str(trial_id)
                if trial_id in processed_ids:
                    raise ValueError(
                        f"Duplicate trial across sample files: {trial_id}"
                    )
                if trial_id not in trial_lookup.index:
                    raise ValueError(
                        f"Trial missing from map: {trial_id}"
                    )

                trial_row = trial_lookup.loc[trial_id]
                key = (
                    str(trial_row["task"]).upper(),
                    int(trial_row["task_sentence_index"]),
                )
                if key not in aoi_lookup:
                    raise ValueError(f"AOI template missing for key={key}")

                rows.append(
                    extract_hard_trial_features(
                        samples=trial_samples,
                        trial_row=trial_row,
                        aoi=aoi_lookup[key],
                        rate_hz=args.rate_hz,
                        sigma_word_height_ratio=(
                            args.sigma_word_height_ratio
                        ),
                        active_threshold=args.active_threshold,
                    )
                )
                processed_ids.add(trial_id)

        missing_ids = sorted(expected_trial_ids - processed_ids)
        extra_ids = sorted(processed_ids - expected_trial_ids)
        if missing_ids or extra_ids:
            raise ValueError(
                f"{subject} processed trial mismatch. "
                f"missing={missing_ids[:10]}, extra={extra_ids[:10]}"
            )

        subject_features = validate_subject_features(
            pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True),
            subject,
            expected_trial_ids,
        )
        subject_features.to_csv(
            subject_output,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Written: {subject_output}")
        combined_frames.append(subject_features)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined = combined.sort_values(
        ["subject_id", "task", "task_sentence_index", "trial_id"]
    ).reset_index(drop=True)

    if len(combined) != EXPECTED_TRIALS:
        raise ValueError(
            f"Combined rows={len(combined)}, expected={EXPECTED_TRIALS}."
        )
    if combined["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError("Combined subject count is invalid.")
    if combined["trial_id"].duplicated().any():
        raise ValueError("Combined Hard-AOI contains duplicate trial IDs.")
    if np.isinf(combined[HARD_COMMON25].to_numpy(dtype=float)).any():
        raise ValueError("Combined Hard-AOI contains infinity.")

    combined_path = (
        output_dir
        / f"zuco2_hard_aoi_features_{args.rate_hz}hz_all_subjects.csv"
    )
    manifest_output = (
        output_dir
        / f"zuco2_hard_aoi_feature_manifest_{args.rate_hz}hz.json"
    )
    combined.to_csv(
        combined_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest_payload = {
        "experiment": "soft_vs_hard_aoi_direct_comparison",
        "representation": "confidence_weighted_argmax_hard_aoi",
        "assignment_rule": (
            "Each active sample is assigned to argmax_j q_j(t); "
            "its aggregate mass remains q_max."
        ),
        "sampling_rate_hz": int(args.rate_hz),
        "sigma_word_height_ratio": float(
            args.sigma_word_height_ratio
        ),
        "active_threshold": float(args.active_threshold),
        "n_subjects": int(combined["subject_id"].nunique()),
        "n_trials": int(len(combined)),
        "distribution_feature_columns": HARD_DISTRIBUTION_COMMON,
        "membership_quality_feature_columns": HARD_QUALITY,
        "primary_feature_columns": HARD_COMMON25,
        "excluded_soft_features_for_fairness": [
            "soft_aoi_sample_entropy_mean",
            "soft_aoi_sample_entropy_median",
        ],
        "inputs": {
            "raw_sample_manifest": str(manifest_path),
            "processed_root": str(processed_root),
            "word_template": str(args.word_template),
        },
    }
    manifest_output.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== HARD-AOI EXTRACTION COMPLETE ===")
    print(f"subjects: {combined['subject_id'].nunique()}")
    print(f"trials: {len(combined)}")
    print(f"features: {len(HARD_COMMON25)}")
    print(f"combined: {combined_path}")
    print(f"manifest: {manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
