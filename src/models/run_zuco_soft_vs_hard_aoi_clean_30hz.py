#!/usr/bin/env python
"""Primary clean-condition direct comparison of Soft-AOI and Hard-AOI.

Primary estimand
----------------
Does probabilistic multi-word allocation improve reading-goal classification
relative to winner-take-all AOI allocation when every other component is held
constant?

Feature sets
------------
soft_common25
    13 Soft-AOI distribution features plus 12 membership/quality features.

hard_argmax_common25
    The corresponding 13 confidence-weighted argmax Hard-AOI distribution
    features plus the same 12 membership/quality features.

The two Soft-AOI sample-entropy features are excluded from both sets because
one-hot Hard-AOI sample entropy is identically zero.

Evaluation
----------
- 18 subjects and 13,302 trials;
- same pre-existing five sentence folds;
- simultaneous unseen-subject and unseen-sentence holdout;
- LinearSVC C fixed before this comparison at 0.01;
- training-only median imputation, missingness indicators, and scaling through
  the existing evaluation utilities;
- paired subject-level Wilcoxon tests.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_zuco_soft_aoi_ablation import (
    evaluate_dual_ablation,
    holm_adjust,
    load_sentence_folds,
)
from run_zuco_soft_aoi_confound_controls import (
    METRICS,
    load_merged_data,
    subject_metrics_from_predictions,
    summarize_subject_metrics,
)


EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302

SOFT_COMMON_DISTRIBUTION = [
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
    return "hard_aoi_" + soft_name.removeprefix("soft_aoi_")


HARD_COMMON_DISTRIBUTION = [
    hard_name(name) for name in SOFT_COMMON_DISTRIBUTION
]
HARD_QUALITY = [hard_name(name) for name in SOFT_QUALITY]

SOFT_NAME = "soft_common25"
HARD_NAME = "hard_argmax_common25"


def normalize_hard_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {
        "trial_id",
        "subject_id",
        "task",
        *HARD_COMMON_DISTRIBUTION,
        *HARD_QUALITY,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Hard-AOI table lacks columns: {missing}")

    frame["trial_id"] = frame["trial_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str).str.upper()
    frame["task"] = frame["task"].astype(str).str.upper()

    if len(frame) != EXPECTED_TRIALS:
        raise ValueError(
            f"Hard-AOI rows={len(frame)}, expected={EXPECTED_TRIALS}."
        )
    if frame["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Hard-AOI subjects={frame['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )
    if frame["trial_id"].duplicated().any():
        raise ValueError("Hard-AOI table contains duplicate trial IDs.")

    numeric_columns = HARD_COMMON_DISTRIBUTION + HARD_QUALITY
    if np.isinf(frame[numeric_columns].to_numpy(dtype=float)).any():
        raise ValueError("Hard-AOI table contains infinity.")
    return frame


def add_hard_features(
    data: pd.DataFrame,
    hard: pd.DataFrame,
) -> pd.DataFrame:
    hard_columns = HARD_COMMON_DISTRIBUTION + HARD_QUALITY
    renamed = hard[
        ["trial_id", "subject_id", "task"] + hard_columns
    ].rename(
        columns={
            name: f"hard__{name}"
            for name in hard_columns
        }
    )

    merged = data.merge(
        renamed,
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(data):
        raise ValueError(
            f"Soft/Hard merge rows={len(merged)}, expected={len(data)}."
        )
    return merged


def audit_shared_quality(data: pd.DataFrame) -> pd.DataFrame:
    """Confirm that Soft and Hard use identical quality inputs."""
    rows: list[dict[str, Any]] = []
    for soft_name, hard_name_value in zip(
        SOFT_QUALITY,
        HARD_QUALITY,
    ):
        soft_column = f"soft__{soft_name}"
        hard_column = f"hard__{hard_name_value}"
        soft = data[soft_column].to_numpy(dtype=float)
        hard = data[hard_column].to_numpy(dtype=float)

        same_missing = np.array_equal(
            np.isnan(soft),
            np.isnan(hard),
        )
        finite = np.isfinite(soft) & np.isfinite(hard)
        absolute_difference = (
            np.abs(soft[finite] - hard[finite])
            if np.any(finite)
            else np.asarray([], dtype=float)
        )
        max_difference = (
            float(np.max(absolute_difference))
            if absolute_difference.size
            else 0.0
        )
        all_close = bool(
            same_missing
            and np.allclose(
                soft,
                hard,
                rtol=0.0,
                atol=1e-9,
                equal_nan=True,
            )
        )
        rows.append(
            {
                "soft_feature": soft_name,
                "hard_feature": hard_name_value,
                "same_missing_pattern": same_missing,
                "max_absolute_difference": max_difference,
                "all_close_at_1e_9": all_close,
            }
        )

    audit = pd.DataFrame(rows)
    if not audit["all_close_at_1e_9"].all():
        failures = audit[
            ~audit["all_close_at_1e_9"]
        ]
        raise ValueError(
            "Soft/Hard shared-quality audit failed:\n"
            + failures.to_string(index=False)
        )
    return audit


def build_feature_sets() -> dict[str, list[str]]:
    soft = [
        f"soft__{name}"
        for name in (
            SOFT_COMMON_DISTRIBUTION
            + SOFT_QUALITY
        )
    ]
    hard = [
        f"hard__{name}"
        for name in (
            HARD_COMMON_DISTRIBUTION
            + HARD_QUALITY
        )
    ]
    if len(soft) != 25 or len(hard) != 25:
        raise ValueError(
            f"Unexpected feature counts: soft={len(soft)}, hard={len(hard)}"
        )
    return {
        SOFT_NAME: soft,
        HARD_NAME: hard,
    }


def build_pairwise_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in METRICS:
        soft = (
            subject_metrics[
                subject_metrics["feature_set"].eq(SOFT_NAME)
            ]
            .set_index("held_out_subject")[metric]
            .sort_index()
        )
        hard = (
            subject_metrics[
                subject_metrics["feature_set"].eq(HARD_NAME)
            ]
            .set_index("held_out_subject")[metric]
            .sort_index()
        )
        paired = pd.concat(
            [
                soft.rename("soft"),
                hard.rename("hard"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        differences = paired["soft"] - paired["hard"]

        if len(paired) == 0:
            statistic = math.nan
            p_value = math.nan
        elif np.allclose(
            differences.to_numpy(dtype=float),
            0.0,
        ):
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                paired["soft"],
                paired["hard"],
                alternative="two-sided",
                zero_method="wilcox",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append(
            {
                "metric": metric,
                "comparison": "soft_common25_vs_hard_argmax_common25",
                "reference_feature_set": HARD_NAME,
                "candidate_feature_set": SOFT_NAME,
                "n_subjects": int(len(paired)),
                "hard_mean": float(paired["hard"].mean()),
                "soft_mean": float(paired["soft"].mean()),
                "mean_difference_soft_minus_hard": float(
                    differences.mean()
                ),
                "soft_wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(np.isclose(differences, 0.0))),
                "soft_losses": int(np.sum(differences < 0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "p_holm_across_metrics": math.nan,
            }
        )

    adjusted = holm_adjust(
        [float(row["p_value"]) for row in rows]
    )
    for row, adjusted_value in zip(rows, adjusted):
        row["p_holm_across_metrics"] = adjusted_value
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-features",
        default=(
            "outputs/downsampling_robustness/"
            "complete_raw_18_0030hz_features.csv"
        ),
    )
    parser.add_argument(
        "--raw-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--soft-features",
        default=(
            "data/processed/zuco2/soft_aoi_all_subjects/"
            "zuco2_soft_aoi_features_30hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--soft-feature-manifest",
        default=(
            "data/processed/zuco2/YAC/soft_aoi/"
            "YAC_soft_aoi_feature_manifest_30hz.json"
        ),
    )
    parser.add_argument(
        "--hard-features",
        default=(
            "data/processed/zuco2/hard_aoi_all_subjects/"
            "zuco2_hard_aoi_features_30hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--word-template",
        default=(
            "data/processed/zuco2/soft_aoi_template/"
            "zuco_soft_aoi_word_template.csv"
        ),
    )
    parser.add_argument(
        "--sentence-folds-file",
        default=(
            "outputs/soft_aoi_confound_controls_30hz/"
            "dual_holdout_sentence_folds.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/soft_vs_hard_aoi_clean_30hz",
    )
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.c_value <= 0:
        raise ValueError("--c-value must be positive.")

    data, _, _, metadata = load_merged_data(
        raw_path=Path(args.raw_features),
        raw_sets_path=Path(args.raw_feature_sets),
        soft_path=Path(args.soft_features),
        soft_manifest_path=Path(args.soft_feature_manifest),
        word_template_path=Path(args.word_template),
    )
    if len(data) != EXPECTED_TRIALS:
        raise ValueError(
            f"Merged Soft-AOI rows={len(data)}, expected={EXPECTED_TRIALS}."
        )
    if data["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError("Merged Soft-AOI subject count is invalid.")

    hard = normalize_hard_table(Path(args.hard_features))
    data = add_hard_features(data, hard)
    quality_audit = audit_shared_quality(data)
    feature_sets = build_feature_sets()
    sentence_folds = load_sentence_folds(
        Path(args.sentence_folds_file)
    )

    print("=== SOFT-AOI VS HARD-AOI CLEAN INPUT ===")
    print(f"subjects: {data['subject_id'].nunique()}")
    print(f"trials: {len(data)}")
    print(
        f"sentence folds: "
        f"{sentence_folds['sentence_fold'].nunique()}"
    )
    print(f"fixed C: {args.c_value}")
    print(f"seed: {args.seed}")
    for name, columns in feature_sets.items():
        print(f"  {name}: {len(columns)} features")
    print(
        "shared quality audit (atol=1e-9): "
        f"{quality_audit['all_close_at_1e_9'].sum()}/"
        f"{len(quality_audit)} PASS"
    )

    folds, predictions = evaluate_dual_ablation(
        data=data,
        feature_sets=feature_sets,
        sentence_folds=sentence_folds,
        c_value=args.c_value,
        seed=args.seed,
    )
    subject_metrics = subject_metrics_from_predictions(
        predictions
    )
    feature_counts = {
        name: len(columns)
        for name, columns in feature_sets.items()
    }
    summary = summarize_subject_metrics(
        subject_metrics=subject_metrics,
        predictions=predictions,
        feature_counts=feature_counts,
    )
    tests = build_pairwise_tests(subject_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "folds": output_dir / "soft_vs_hard_outer_folds.csv",
        "predictions": output_dir / "soft_vs_hard_predictions.csv",
        "subject_metrics": (
            output_dir / "soft_vs_hard_subject_metrics.csv"
        ),
        "summary": output_dir / "soft_vs_hard_summary.csv",
        "tests": output_dir / "soft_vs_hard_pairwise_tests.csv",
        "quality_audit": (
            output_dir / "soft_vs_hard_shared_quality_audit.csv"
        ),
        "feature_sets": output_dir / "soft_vs_hard_feature_sets.json",
        "manifest": output_dir / "soft_vs_hard_manifest.json",
    }

    folds.to_csv(
        paths["folds"],
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    subject_metrics.to_csv(
        paths["subject_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        paths["summary"],
        index=False,
        encoding="utf-8-sig",
    )
    tests.to_csv(
        paths["tests"],
        index=False,
        encoding="utf-8-sig",
    )
    quality_audit.to_csv(
        paths["quality_audit"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["feature_sets"].write_text(
        json.dumps(
            feature_sets,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "protocol": "dual_unseen_subject_unseen_sentence",
        "analysis_status": "primary_hard_aoi_direct_comparison",
        "n_subjects": int(data["subject_id"].nunique()),
        "n_trials": int(len(data)),
        "n_sentence_folds": int(
            sentence_folds["sentence_fold"].nunique()
        ),
        "c_value": float(args.c_value),
        "seed": int(args.seed),
        "soft_feature_set": SOFT_NAME,
        "hard_feature_set": HARD_NAME,
        "feature_count_each": 25,
        "hard_assignment": "argmax_q_with_qmax_mass",
        "shared_quality_audit": "PASS",
        "excluded_for_fairness": [
            "soft_aoi_sample_entropy_mean",
            "soft_aoi_sample_entropy_median",
        ],
        "holm_family": list(METRICS),
        "inputs": {
            "raw_features": str(args.raw_features),
            "raw_feature_sets": str(args.raw_feature_sets),
            "soft_features": str(args.soft_features),
            "soft_feature_manifest": str(
                args.soft_feature_manifest
            ),
            "hard_features": str(args.hard_features),
            "word_template": str(args.word_template),
            "sentence_folds_file": str(args.sentence_folds_file),
        },
        "soft_metadata": {
            "content_features": len(
                metadata.get("soft_content_columns", [])
            ),
            "quality_features": len(
                metadata.get("soft_quality_columns", [])
            ),
        },
    }
    paths["manifest"].write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    display_columns = [
        "feature_set",
        "n_input_features",
        "macro_f1_subject_mean",
        "balanced_accuracy_subject_mean",
        "roc_auc_subject_mean",
        "pr_auc_nr_subject_mean",
    ]
    print("\n=== SOFT-AOI VS HARD-AOI SUMMARY ===")
    print(summary[display_columns].to_string(index=False))

    print("\n=== PAIRED SUBJECT TESTS ===")
    print(
        tests[
            [
                "metric",
                "hard_mean",
                "soft_mean",
                "mean_difference_soft_minus_hard",
                "soft_wins",
                "ties",
                "soft_losses",
                "p_value",
                "p_holm_across_metrics",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
