#!/usr/bin/env python
"""Soft-AOI component ablation under unseen-subject + unseen-sentence evaluation.

This script reuses the merged-data and evaluation utilities from
run_zuco_soft_aoi_confound_controls.py. It partitions the 29 Soft-AOI content
features into mutually exclusive groups:

- distribution: order-invariant AOI occupancy and progress distribution
- temporal: time trend, first/second-half change, and event rates
- order: adjacent dominant-word/line transition features

It also derives a collapsed-path representation that removes consecutive
stationary samples algebraically from the existing transition summaries.
All models are evaluated on exactly the sentence folds used by the previous
confound-control experiment. The held-out subject and every instance of the
held-out sentences are excluded from training.
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
from sklearn.dummy import DummyClassifier

from run_zuco_soft_aoi_confound_controls import (
    EXPECTED_SUBJECTS,
    EXPECTED_TRIALS,
    EXPECTED_TRIALS_PER_SUBJECT,
    METRICS,
    decision_scores,
    load_merged_data,
    make_svm,
    score_predictions,
    subject_metrics_from_predictions,
    summarize_subject_metrics,
)


DISTRIBUTION_BASE = [
    "soft_aoi_sample_entropy_mean",
    "soft_aoi_sample_entropy_median",
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

TEMPORAL_BASE = [
    "soft_aoi_progress_slope_per_sec",
    "soft_aoi_progress_first_half",
    "soft_aoi_progress_second_half",
    "soft_aoi_progress_half_delta",
    "soft_aoi_transition_rate_per_sec",
    "soft_aoi_regression_events_per_sec",
]

ORDER_BASE = [
    "soft_aoi_forward_transition_fraction",
    "soft_aoi_backward_transition_fraction",
    "soft_aoi_stationary_transition_fraction",
    "soft_aoi_mean_abs_word_jump_norm",
    "soft_aoi_backward_jump_mean_norm",
    "soft_aoi_line_forward_fraction",
    "soft_aoi_line_backward_fraction",
    "soft_aoi_line_return_fraction",
]

COLLAPSED_COLUMNS = [
    "abl__collapsed_changed_fraction",
    "abl__collapsed_forward_given_change",
    "abl__collapsed_backward_given_change",
    "abl__collapsed_net_direction_given_change",
    "abl__collapsed_abs_jump_given_change_norm",
    "abl__collapsed_backward_jump_mean_norm",
    "abl__collapsed_line_forward_given_line_change",
    "abl__collapsed_line_return_given_line_forward",
]


def prefixed(names: list[str]) -> list[str]:
    return [f"soft__{name}" for name in names]


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(num), np.nan, dtype=float)
    keep = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    output[keep] = num[keep] / den[keep]
    return pd.Series(output, index=numerator.index)


def add_collapsed_path_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    forward = pd.to_numeric(
        result["soft__soft_aoi_forward_transition_fraction"],
        errors="coerce",
    )
    backward = pd.to_numeric(
        result["soft__soft_aoi_backward_transition_fraction"],
        errors="coerce",
    )
    stationary = pd.to_numeric(
        result["soft__soft_aoi_stationary_transition_fraction"],
        errors="coerce",
    )
    mean_abs_jump = pd.to_numeric(
        result["soft__soft_aoi_mean_abs_word_jump_norm"],
        errors="coerce",
    )
    backward_jump = pd.to_numeric(
        result["soft__soft_aoi_backward_jump_mean_norm"],
        errors="coerce",
    )
    line_forward = pd.to_numeric(
        result["soft__soft_aoi_line_forward_fraction"],
        errors="coerce",
    )
    line_backward = pd.to_numeric(
        result["soft__soft_aoi_line_backward_fraction"],
        errors="coerce",
    )
    line_return = pd.to_numeric(
        result["soft__soft_aoi_line_return_fraction"],
        errors="coerce",
    )

    # Because forward + backward + stationary = 1 for valid adjacent pairs,
    # forward + backward is the fraction remaining after collapsing repeats.
    changed = forward + backward
    line_changed = line_forward + line_backward

    result["abl__collapsed_changed_fraction"] = changed
    result["abl__collapsed_forward_given_change"] = safe_divide(
        forward, changed
    )
    result["abl__collapsed_backward_given_change"] = safe_divide(
        backward, changed
    )
    result["abl__collapsed_net_direction_given_change"] = (
        result["abl__collapsed_forward_given_change"]
        - result["abl__collapsed_backward_given_change"]
    )
    # The original mean includes zero jumps. Dividing by the changed fraction
    # gives the mean jump among non-stationary transitions.
    result["abl__collapsed_abs_jump_given_change_norm"] = safe_divide(
        mean_abs_jump, changed
    )
    result["abl__collapsed_backward_jump_mean_norm"] = backward_jump
    result["abl__collapsed_line_forward_given_line_change"] = safe_divide(
        line_forward, line_changed
    )
    result["abl__collapsed_line_return_given_line_forward"] = safe_divide(
        line_return, line_forward
    )

    values = result[COLLAPSED_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Collapsed-path features contain infinity.")
    return result


def build_ablation_sets(
    metadata: dict[str, Any],
    existing_sets: dict[str, list[str]],
) -> dict[str, list[str]]:
    soft_content_base = list(metadata["soft_content_columns"])
    soft_quality_base = list(metadata["soft_quality_columns"])

    expected_partition = set(DISTRIBUTION_BASE + TEMPORAL_BASE + ORDER_BASE)
    actual_content = set(soft_content_base)
    if expected_partition != actual_content:
        raise ValueError(
            "Soft content partition does not match the manifest. "
            f"missing_from_partition={sorted(actual_content - expected_partition)}, "
            f"unexpected_in_partition={sorted(expected_partition - actual_content)}"
        )

    distribution = prefixed(DISTRIBUTION_BASE)
    temporal = prefixed(TEMPORAL_BASE)
    order = prefixed(ORDER_BASE)
    quality = prefixed(soft_quality_base)
    content = distribution + temporal + order
    full = content + quality

    feature_sets = {
        "dummy_reference": [],
        "rawstat_all": list(existing_sets["rawstat_all"]),
        "soft_full_all": full,
        "soft_content_full": content,
        "membership_quality_only": quality,
        "distribution_only": distribution,
        "temporal_only": temporal,
        "order_only": order,
        "collapsed_path_only": list(COLLAPSED_COLUMNS),
        "soft_without_distribution": temporal + order + quality,
        "soft_without_temporal": distribution + order + quality,
        "soft_without_order": distribution + temporal + quality,
    }

    for name, columns in feature_sets.items():
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate columns in {name}.")
    return feature_sets


def load_sentence_folds(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"sentence_key", "sentence_fold", "task", "task_sentence_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sentence-fold file lacks columns: {missing}")
    frame = frame.copy()
    frame["sentence_fold"] = pd.to_numeric(
        frame["sentence_fold"], errors="raise"
    ).astype(int)
    if frame["sentence_key"].duplicated().any():
        raise ValueError("Sentence-fold file contains duplicate sentence keys.")
    if frame["sentence_key"].nunique() != 739:
        raise ValueError(
            f"Sentence-fold keys={frame['sentence_key'].nunique()}, expected=739."
        )
    return frame


def evaluate_dual_ablation(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    sentence_folds: pd.DataFrame,
    c_value: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = data.merge(
        sentence_folds[["sentence_key", "sentence_fold"]],
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if work["sentence_fold"].isna().any():
        raise ValueError("Some trials lack sentence folds.")
    work["sentence_fold"] = work["sentence_fold"].astype(int)

    for name, columns in feature_sets.items():
        missing = sorted(set(columns) - set(work.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if columns and np.isinf(work[columns].to_numpy(dtype=float)).any():
            raise ValueError(f"{name} contains infinity.")

    subjects = sorted(work["subject_id"].unique())
    folds = sorted(work["sentence_fold"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(f"Subjects={len(subjects)}, expected={EXPECTED_SUBJECTS}.")

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for subject_index, held_out in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"Ablation dual holdout subject {subject_index:02d}/{len(subjects)}: "
            f"{held_out}"
        )
        for fold_index in folds:
            train = work[
                (work["subject_id"] != held_out)
                & (work["sentence_fold"] != fold_index)
            ].copy()
            test = work[
                (work["subject_id"] == held_out)
                & (work["sentence_fold"] == fold_index)
            ].copy()
            if test.empty:
                raise ValueError(
                    f"Empty test split: subject={held_out}, fold={fold_index}."
                )
            if set(train["sentence_key"]) & set(test["sentence_key"]):
                raise ValueError("Sentence leakage detected.")
            if held_out in set(train["subject_id"]):
                raise ValueError("Subject leakage detected.")

            y_train = train["label_nr"].to_numpy(dtype=int)
            y_test = test["label_nr"].to_numpy(dtype=int)
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError("Training split lacks both classes.")
            if set(np.unique(y_test)) != {0, 1}:
                raise ValueError("Test split lacks both classes.")

            print(
                f"  Sentence fold {fold_index + 1}/{len(folds)}: "
                f"train={len(train)}, test={len(test)}"
            )
            for feature_set, columns in feature_sets.items():
                if feature_set == "dummy_reference":
                    model = DummyClassifier(strategy="most_frequent")
                    x_train = np.zeros((len(train), 1), dtype=float)
                    x_test = np.zeros((len(test), 1), dtype=float)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test).astype(int)
                    scores = decision_scores(model, x_test)
                    used_c = math.nan
                else:
                    model = make_svm(c_value, seed)
                    model.fit(train[columns], y_train)
                    y_pred = model.predict(test[columns]).astype(int)
                    scores = decision_scores(model, test[columns])
                    used_c = float(c_value)

                metrics = score_predictions(y_test, y_pred, scores)
                fold_rows.append(
                    {
                        "protocol": "dual_unseen_subject_unseen_sentence",
                        "feature_set": feature_set,
                        "held_out_subject": held_out,
                        "sentence_fold": int(fold_index),
                        "n_input_features": int(len(columns)),
                        "c_value": used_c,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        **metrics,
                    }
                )

                ordered = test.reset_index(drop=True)
                for row_index, row in ordered.iterrows():
                    prediction_rows.append(
                        {
                            "protocol": "dual_unseen_subject_unseen_sentence",
                            "feature_set": feature_set,
                            "held_out_subject": held_out,
                            "sentence_fold": int(fold_index),
                            "trial_id": row["trial_id"],
                            "subject_id": row["subject_id"],
                            "task": row["task"],
                            "task_sentence_index": int(row["task_sentence_index"]),
                            "sentence_key": row["sentence_key"],
                            "label_nr": int(row["label_nr"]),
                            "prediction_nr": int(y_pred[row_index]),
                            "decision_score_nr": float(scores[row_index]),
                            "n_input_features": int(len(columns)),
                            "c_value": used_c,
                        }
                    )

    folds_frame = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    expected = EXPECTED_TRIALS * len(feature_sets)
    if len(predictions) != expected:
        raise ValueError(
            f"Predictions={len(predictions)}, expected={expected}."
        )
    if predictions[["feature_set", "trial_id"]].duplicated().any():
        raise ValueError("Duplicate feature-set/trial predictions.")
    counts = predictions.groupby("feature_set").size()
    if not counts.eq(EXPECTED_TRIALS).all():
        raise ValueError(f"Incomplete prediction counts: {counts.to_dict()}")
    return folds_frame, predictions


def holm_adjust(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    result = np.full(len(array), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    order = finite[np.argsort(array[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (m - rank) * array[index])
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def targeted_tests(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("rawstat_all", "soft_full_all", "proposal_vs_rawstat"),
        ("soft_content_full", "soft_full_all", "membership_added"),
        ("membership_quality_only", "soft_full_all", "content_added"),
        ("distribution_only", "soft_full_all", "full_vs_distribution"),
        ("temporal_only", "soft_full_all", "full_vs_temporal"),
        ("order_only", "soft_full_all", "full_vs_order"),
        ("collapsed_path_only", "soft_full_all", "full_vs_collapsed"),
        ("soft_without_distribution", "soft_full_all", "distribution_unique"),
        ("soft_without_temporal", "soft_full_all", "temporal_unique"),
        ("soft_without_order", "soft_full_all", "order_unique"),
        ("collapsed_path_only", "order_only", "collapsed_vs_sample_order"),
    ]
    rows: list[dict[str, Any]] = []
    for metric in ["macro_f1", "balanced_accuracy", "roc_auc", "pr_auc_nr"]:
        metric_indices: list[int] = []
        p_values: list[float] = []
        for reference, candidate, comparison in comparisons:
            a = subject_metrics[
                subject_metrics["feature_set"].eq(reference)
            ].set_index("held_out_subject")[metric]
            b = subject_metrics[
                subject_metrics["feature_set"].eq(candidate)
            ].set_index("held_out_subject")[metric]
            paired = pd.concat(
                [a.rename("reference"), b.rename("candidate")],
                axis=1,
                join="inner",
            ).dropna()
            diff = paired["candidate"] - paired["reference"]
            if len(paired) == 0:
                statistic = math.nan
                p_value = math.nan
            elif np.allclose(diff.to_numpy(dtype=float), 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    paired["candidate"],
                    paired["reference"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
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
                        diff.mean()
                    ),
                    "wins": int(np.sum(diff > 0)),
                    "ties": int(np.sum(np.isclose(diff, 0.0))),
                    "losses": int(np.sum(diff < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_within_metric": math.nan,
                }
            )
            metric_indices.append(len(rows) - 1)
            p_values.append(p_value)
        adjusted = holm_adjust(p_values)
        for index, value in zip(metric_indices, adjusted):
            rows[index]["p_holm_within_metric"] = value
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
        "--output-dir",
        default="outputs/soft_aoi_ablation_30hz",
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
    data = add_collapsed_path_features(data)
    feature_sets = build_ablation_sets(metadata, existing_sets)
    sentence_folds = load_sentence_folds(Path(args.sentence_folds_file))

    print("=== SOFT-AOI ABLATION INPUT ===")
    print(f"subjects: {data['subject_id'].nunique()}")
    print(f"trials: {len(data)}")
    print(f"sentence folds: {sentence_folds['sentence_fold'].nunique()}")
    print(f"fixed C: {args.c_value}")
    print("feature sets:")
    for name, columns in feature_sets.items():
        print(f"  {name}: {len(columns)}")

    folds, predictions = evaluate_dual_ablation(
        data=data,
        feature_sets=feature_sets,
        sentence_folds=sentence_folds,
        c_value=args.c_value,
        seed=args.seed,
    )
    subject_metrics = subject_metrics_from_predictions(predictions)
    feature_counts = {name: len(columns) for name, columns in feature_sets.items()}
    summary = summarize_subject_metrics(
        subject_metrics=subject_metrics,
        predictions=predictions,
        feature_counts=feature_counts,
    )
    tests = targeted_tests(subject_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "folds": output_dir / "dual_holdout_ablation_fold_metrics.csv",
        "predictions": output_dir / "dual_holdout_ablation_predictions.csv",
        "subject_metrics": output_dir / "dual_holdout_ablation_subject_metrics.csv",
        "summary": output_dir / "dual_holdout_ablation_summary.csv",
        "tests": output_dir / "dual_holdout_ablation_key_tests.csv",
        "feature_sets": output_dir / "soft_aoi_ablation_feature_sets.json",
        "manifest": output_dir / "soft_aoi_ablation_manifest.json",
    }
    folds.to_csv(paths["folds"], index=False, encoding="utf-8-sig")
    predictions.to_csv(
        paths["predictions"], index=False, encoding="utf-8-sig"
    )
    subject_metrics.to_csv(
        paths["subject_metrics"], index=False, encoding="utf-8-sig"
    )
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    tests.to_csv(paths["tests"], index=False, encoding="utf-8-sig")
    paths["feature_sets"].write_text(
        json.dumps(feature_sets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "protocol": "dual_unseen_subject_unseen_sentence",
        "n_subjects": int(data["subject_id"].nunique()),
        "n_trials": int(len(data)),
        "n_sentence_folds": int(sentence_folds["sentence_fold"].nunique()),
        "c_value": float(args.c_value),
        "seed": int(args.seed),
        "distribution_columns": DISTRIBUTION_BASE,
        "temporal_columns": TEMPORAL_BASE,
        "order_columns": ORDER_BASE,
        "collapsed_columns": COLLAPSED_COLUMNS,
        "feature_counts": feature_counts,
        "inputs": {
            "raw_features": str(args.raw_features),
            "raw_feature_sets": str(args.raw_feature_sets),
            "soft_features": str(args.soft_features),
            "soft_feature_manifest": str(args.soft_feature_manifest),
            "word_template": str(args.word_template),
            "sentence_folds_file": str(args.sentence_folds_file),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
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
    print("\n=== DUAL HOLDOUT SOFT-AOI ABLATION SUMMARY ===")
    print(summary[display_columns].to_string(index=False))

    key = tests[tests["metric"].eq("macro_f1")][
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
    print("\n=== KEY MACRO-F1 ABLATION TESTS ===")
    print(key.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
