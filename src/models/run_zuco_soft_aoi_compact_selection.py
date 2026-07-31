#!/usr/bin/env python
"""Select a compact Soft-AOI representation under dual holdout.

This script evaluates the compact Soft-AOI candidate consisting of:

- 15 distribution features
- 12 Soft-AOI membership / quality features

Temporal and adjacent-order features are omitted because the preceding
ablation found no unique gain from either group when added to the full model.

Evaluation uses exactly the same unseen-subject + unseen-sentence folds as the
confound-control and ablation experiments. LinearSVC C is fixed in advance.
The script also evaluates RawStat + compact Soft-AOI and compares the new
candidates with selected results from the previous ablation.
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

from run_zuco_soft_aoi_confound_controls import (
    METRICS,
    load_merged_data,
    subject_metrics_from_predictions,
    summarize_subject_metrics,
)
from run_zuco_soft_aoi_ablation import (
    DISTRIBUTION_BASE,
    evaluate_dual_ablation,
    holm_adjust,
    load_sentence_folds,
    prefixed,
)


COMPACT_NAME = "soft_compact_distribution_membership"
COMBINED_NAME = "combined_compact_distribution_membership"

PRIOR_FEATURES = [
    "rawstat_all",
    "soft_full_all",
    "soft_without_temporal",
    "soft_without_order",
    "distribution_only",
    "membership_quality_only",
]


def build_feature_sets(
    metadata: dict[str, Any],
    existing_sets: dict[str, list[str]],
) -> dict[str, list[str]]:
    quality = prefixed(list(metadata["soft_quality_columns"]))
    distribution = prefixed(DISTRIBUTION_BASE)
    compact = distribution + quality
    rawstat = list(existing_sets["rawstat_all"])
    combined = list(dict.fromkeys(rawstat + compact))

    feature_sets = {
        COMPACT_NAME: compact,
        COMBINED_NAME: combined,
    }
    for name, columns in feature_sets.items():
        if not columns:
            raise ValueError(f"{name} has no features.")
        if len(columns) != len(set(columns)):
            raise ValueError(f"{name} contains duplicate columns.")
    if len(distribution) != 15:
        raise ValueError(
            f"Distribution feature count={len(distribution)}, expected=15."
        )
    if len(quality) != 12:
        raise ValueError(
            f"Membership/quality feature count={len(quality)}, expected=12."
        )
    return feature_sets


def load_prior_subject_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "feature_set",
        "held_out_subject",
        "n_predictions",
        "n_input_features",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prior subject metrics lack columns: {missing}")

    selected = frame[frame["feature_set"].isin(PRIOR_FEATURES)].copy()
    found = set(selected["feature_set"])
    missing_sets = sorted(set(PRIOR_FEATURES) - found)
    if missing_sets:
        raise ValueError(
            f"Prior subject metrics lack feature sets: {missing_sets}"
        )
    counts = selected.groupby("feature_set")["held_out_subject"].nunique()
    if not counts.eq(18).all():
        raise ValueError(
            f"Prior feature-set subject counts are invalid: {counts.to_dict()}"
        )
    return selected


def pairwise_tests(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("rawstat_all", COMPACT_NAME, "compact_vs_rawstat"),
        (
            "distribution_only",
            COMPACT_NAME,
            "membership_added_to_distribution",
        ),
        (
            "membership_quality_only",
            COMPACT_NAME,
            "distribution_added_to_membership",
        ),
        (
            COMPACT_NAME,
            "soft_without_order",
            "temporal_added_to_compact",
        ),
        (
            COMPACT_NAME,
            "soft_without_temporal",
            "order_added_to_compact",
        ),
        (
            COMPACT_NAME,
            "soft_full_all",
            "temporal_and_order_added_to_compact",
        ),
        (
            COMPACT_NAME,
            COMBINED_NAME,
            "rawstat_added_to_compact",
        ),
        (
            "soft_full_all",
            COMBINED_NAME,
            "combined_compact_vs_soft_full",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for metric in ["macro_f1", "balanced_accuracy", "roc_auc", "pr_auc_nr"]:
        row_indices: list[int] = []
        p_values: list[float] = []

        for reference, candidate, comparison in comparisons:
            ref = (
                subject_metrics[
                    subject_metrics["feature_set"].eq(reference)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            cand = (
                subject_metrics[
                    subject_metrics["feature_set"].eq(candidate)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            paired = pd.concat(
                [ref.rename("reference"), cand.rename("candidate")],
                axis=1,
                join="inner",
            ).dropna()
            differences = paired["candidate"] - paired["reference"]

            if len(paired) == 0:
                statistic = math.nan
                p_value = math.nan
            elif np.allclose(differences.to_numpy(dtype=float), 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    paired["candidate"],
                    paired["reference"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "reference_feature_set": reference,
                    "candidate_feature_set": candidate,
                    "n_subjects": int(len(paired)),
                    "reference_mean": float(paired["reference"].mean()),
                    "candidate_mean": float(paired["candidate"].mean()),
                    "mean_difference_candidate_minus_reference": float(
                        differences.mean()
                    ),
                    "wins": int(np.sum(differences > 0)),
                    "ties": int(np.sum(np.isclose(differences, 0.0))),
                    "losses": int(np.sum(differences < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_within_metric": math.nan,
                }
            )
            row_indices.append(len(rows) - 1)
            p_values.append(p_value)

        adjusted = holm_adjust(p_values)
        for row_index, adjusted_value in zip(row_indices, adjusted):
            rows[row_index]["p_holm_within_metric"] = adjusted_value

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
        "--prior-subject-metrics",
        default=(
            "outputs/soft_aoi_ablation_30hz/"
            "dual_holdout_ablation_subject_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/soft_aoi_compact_selection_30hz",
    )
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.c_value <= 0:
        raise ValueError("--c-value must be positive.")

    data, existing_sets, _, metadata = load_merged_data(
        raw_path=Path(args.raw_features),
        raw_sets_path=Path(args.raw_feature_sets),
        soft_path=Path(args.soft_features),
        soft_manifest_path=Path(args.soft_feature_manifest),
        word_template_path=Path(args.word_template),
    )
    sentence_folds = load_sentence_folds(Path(args.sentence_folds_file))
    feature_sets = build_feature_sets(metadata, existing_sets)

    print("=== COMPACT SOFT-AOI SELECTION INPUT ===")
    print(f"subjects: {data['subject_id'].nunique()}")
    print(f"trials: {len(data)}")
    print(f"sentence folds: {sentence_folds['sentence_fold'].nunique()}")
    print(f"fixed C: {args.c_value}")
    for name, columns in feature_sets.items():
        print(f"  {name}: {len(columns)} features")

    folds, predictions = evaluate_dual_ablation(
        data=data,
        feature_sets=feature_sets,
        sentence_folds=sentence_folds,
        c_value=args.c_value,
        seed=args.seed,
    )
    new_subject_metrics = subject_metrics_from_predictions(predictions)
    prior_subject_metrics = load_prior_subject_metrics(
        Path(args.prior_subject_metrics)
    )
    all_subject_metrics = pd.concat(
        [prior_subject_metrics, new_subject_metrics],
        ignore_index=True,
    )

    duplicate = all_subject_metrics[
        ["feature_set", "held_out_subject"]
    ].duplicated()
    if duplicate.any():
        examples = all_subject_metrics.loc[
            duplicate, ["feature_set", "held_out_subject"]
        ].head(20)
        raise ValueError(
            "Duplicate feature-set/subject rows after combining results:\n"
            + examples.to_string(index=False)
        )

    feature_counts = (
        all_subject_metrics.groupby("feature_set")["n_input_features"]
        .first()
        .astype(int)
        .to_dict()
    )
    summary = summarize_subject_metrics(
        subject_metrics=all_subject_metrics,
        predictions=None,
        feature_counts=feature_counts,
    )
    tests = pairwise_tests(all_subject_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "folds": output_dir / "compact_dual_holdout_fold_metrics.csv",
        "predictions": output_dir / "compact_dual_holdout_predictions.csv",
        "new_subject_metrics": (
            output_dir / "compact_dual_holdout_subject_metrics.csv"
        ),
        "all_subject_metrics": (
            output_dir / "compact_selection_all_subject_metrics.csv"
        ),
        "summary": output_dir / "compact_selection_summary.csv",
        "tests": output_dir / "compact_selection_pairwise_tests.csv",
        "feature_sets": output_dir / "compact_feature_sets.json",
        "manifest": output_dir / "compact_selection_manifest.json",
    }

    folds.to_csv(paths["folds"], index=False, encoding="utf-8-sig")
    predictions.to_csv(
        paths["predictions"], index=False, encoding="utf-8-sig"
    )
    new_subject_metrics.to_csv(
        paths["new_subject_metrics"], index=False, encoding="utf-8-sig"
    )
    all_subject_metrics.to_csv(
        paths["all_subject_metrics"], index=False, encoding="utf-8-sig"
    )
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    tests.to_csv(paths["tests"], index=False, encoding="utf-8-sig")
    paths["feature_sets"].write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "protocol": "dual_unseen_subject_unseen_sentence",
        "selection_status": (
            "exploratory_compact_selection_after_component_ablation"
        ),
        "n_subjects": int(data["subject_id"].nunique()),
        "n_trials": int(len(data)),
        "n_sentence_folds": int(
            sentence_folds["sentence_fold"].nunique()
        ),
        "c_value": float(args.c_value),
        "seed": int(args.seed),
        "compact_definition": {
            "distribution_features": list(DISTRIBUTION_BASE),
            "membership_quality_features": list(
                metadata["soft_quality_columns"]
            ),
            "excluded_groups": ["temporal", "order"],
        },
        "feature_counts": feature_counts,
        "inputs": {
            "raw_features": str(args.raw_features),
            "raw_feature_sets": str(args.raw_feature_sets),
            "soft_features": str(args.soft_features),
            "soft_feature_manifest": str(args.soft_feature_manifest),
            "word_template": str(args.word_template),
            "sentence_folds_file": str(args.sentence_folds_file),
            "prior_subject_metrics": str(args.prior_subject_metrics),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_sets = [
        COMBINED_NAME,
        COMPACT_NAME,
        "soft_without_temporal",
        "soft_without_order",
        "soft_full_all",
        "distribution_only",
        "rawstat_all",
        "membership_quality_only",
    ]
    display = summary[summary["feature_set"].isin(display_sets)][
        [
            "feature_set",
            "n_input_features",
            "macro_f1_subject_mean",
            "balanced_accuracy_subject_mean",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
        ]
    ]

    print("\n=== COMPACT SOFT-AOI SELECTION SUMMARY ===")
    print(display.to_string(index=False))

    key_tests = tests[tests["metric"].eq("macro_f1")][
        [
            "comparison",
            "reference_feature_set",
            "candidate_feature_set",
            "reference_mean",
            "candidate_mean",
            "mean_difference_candidate_minus_reference",
            "wins",
            "ties",
            "losses",
            "p_value",
            "p_holm_within_metric",
        ]
    ]
    print("\n=== COMPACT SELECTION MACRO-F1 TESTS ===")
    print(key_tests.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
