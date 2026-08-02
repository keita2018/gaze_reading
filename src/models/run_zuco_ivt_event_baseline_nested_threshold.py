#!/usr/bin/env python
"""Nested quality-specific I-VT threshold selection for ZuCo.

For every outer unseen-subject + unseen-sentence test split and every
measurement condition, this script selects an I-VT velocity threshold using
only the corresponding outer-training subjects and outer-training sentences.

Threshold-selection estimand
----------------------------
For a candidate velocity threshold:

1. detect events from clean gaze for classifier training;
2. detect events from the target-quality gaze for inner validation;
3. evaluate clean-trained -> target-quality validation performance;
4. select the threshold with the highest inner subject-mean Macro F1;
5. refit on the full outer-training clean data;
6. evaluate once on the untouched outer test subject and sentence fold.

The outer test labels never participate in threshold selection.

The fixed 30 deg/s baseline remains the primary standard event baseline. This
nested analysis tests whether quality-specific event-detector tuning can rescue
the event representation without using outer-test information.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score

from run_zuco_ivt_event_baseline_fixed import (
    EXPECTED_SUBJECTS,
    EXPECTED_TRIALS,
    IVT_COLUMNS,
    build_metadata,
    condition_repeat_count,
    condition_table,
    extract_conditions,
    load_common_official_maps,
    load_word_template,
    parse_conditions,
)
from run_zuco_soft_aoi_ablation import (
    holm_adjust,
    load_sentence_folds,
)
from run_zuco_soft_aoi_confound_controls import (
    METRICS,
    decision_scores,
    make_svm,
    score_predictions,
)


def parse_thresholds(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError(
            "--velocity-thresholds must contain positive values."
        )
    return values


def subject_mean_inner_metrics(
    predictions: pd.DataFrame,
) -> tuple[float, float]:
    """Return subject-mean Macro F1 and balanced accuracy.

    Metrics are first calculated for every validation subject and corruption
    repeat and then averaged, preventing conditions with more trials from
    dominating threshold selection.
    """
    rows: list[dict[str, float]] = []
    for (_, _), group in predictions.groupby(
        ["subject_id", "corruption_repeat"],
        sort=True,
    ):
        y_true = group["label_nr"].to_numpy(dtype=int)
        y_pred = group["prediction_nr"].to_numpy(dtype=int)
        rows.append(
            {
                "macro_f1": float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        labels=[0, 1],
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, y_pred)
                ),
            }
        )

    if not rows:
        raise ValueError("No inner validation predictions.")
    metrics = pd.DataFrame(rows)
    return (
        float(metrics["macro_f1"].mean()),
        float(metrics["balanced_accuracy"].mean()),
    )


def make_inner_splits(
    outer_subjects: list[str],
    outer_sentence_folds: list[int],
    n_requested: int,
) -> list[tuple[list[str], int]]:
    """Create deterministic paired subject/sentence validation splits."""
    n_splits = min(
        n_requested,
        len(outer_sentence_folds),
        len(outer_subjects),
    )
    if n_splits < 2:
        raise ValueError("At least two inner splits are required.")

    subject_groups = [[] for _ in range(n_splits)]
    for index, subject in enumerate(sorted(outer_subjects)):
        subject_groups[index % n_splits].append(subject)

    sentence_folds = sorted(outer_sentence_folds)
    return [
        (subject_groups[index], sentence_folds[index])
        for index in range(n_splits)
    ]


def candidate_sort_key(
    row: dict[str, Any],
    standard_threshold: float,
) -> tuple[float, float, float, float]:
    """Maximize F1, then BA, then prefer a threshold close to standard."""
    threshold = float(row["velocity_threshold_deg_s"])
    distance_from_standard = abs(
        math.log2(threshold / standard_threshold)
    )
    return (
        float(row["inner_macro_f1"]),
        float(row["inner_balanced_accuracy"]),
        -distance_from_standard,
        -threshold,
    )


def select_threshold_for_outer_split(
    tables: dict[
        float,
        dict[tuple[str, int], pd.DataFrame],
    ],
    condition: str,
    candidate_thresholds: list[float],
    outer_held_out_subject: str,
    outer_held_out_sentence_fold: int,
    all_subjects: list[str],
    all_sentence_folds: list[int],
    repeats: int,
    c_value: float,
    seed: int,
    n_inner_splits: int,
    standard_threshold: float,
) -> tuple[float, pd.DataFrame]:
    outer_subjects = [
        subject
        for subject in all_subjects
        if subject != outer_held_out_subject
    ]
    outer_sentence_folds = [
        fold
        for fold in all_sentence_folds
        if fold != outer_held_out_sentence_fold
    ]
    inner_splits = make_inner_splits(
        outer_subjects,
        outer_sentence_folds,
        n_inner_splits,
    )

    candidate_rows: list[dict[str, Any]] = []
    repeat_count = condition_repeat_count(
        condition,
        repeats,
    )

    for threshold in candidate_thresholds:
        clean_table = tables[threshold][("clean", 1)]
        inner_prediction_rows: list[dict[str, Any]] = []

        for inner_index, (
            inner_validation_subjects,
            inner_validation_sentence_fold,
        ) in enumerate(inner_splits):
            inner_train = clean_table[
                clean_table["subject_id"].isin(outer_subjects)
                & ~clean_table["subject_id"].isin(
                    inner_validation_subjects
                )
                & clean_table["sentence_fold"].isin(
                    outer_sentence_folds
                )
                & (
                    clean_table["sentence_fold"]
                    != inner_validation_sentence_fold
                )
            ].copy()

            if inner_train.empty:
                raise ValueError("Empty inner training split.")
            if outer_held_out_subject in set(
                inner_train["subject_id"]
            ):
                raise ValueError("Outer subject leakage in inner training.")
            if outer_held_out_sentence_fold in set(
                inner_train["sentence_fold"]
            ):
                raise ValueError(
                    "Outer sentence-fold leakage in inner training."
                )

            model = make_svm(c_value, seed)
            model.fit(
                inner_train[IVT_COLUMNS],
                inner_train["label_nr"].to_numpy(dtype=int),
            )

            for repeat in range(1, repeat_count + 1):
                validation_table = tables[threshold][
                    (condition, repeat)
                ]
                validation = validation_table[
                    validation_table["subject_id"].isin(
                        inner_validation_subjects
                    )
                    & (
                        validation_table["sentence_fold"]
                        == inner_validation_sentence_fold
                    )
                ].copy().reset_index(drop=True)

                if validation.empty:
                    raise ValueError(
                        "Empty inner validation split: "
                        f"condition={condition}, threshold={threshold}, "
                        f"inner={inner_index}."
                    )
                if set(inner_train["sentence_key"]) & set(
                    validation["sentence_key"]
                ):
                    raise ValueError(
                        "Sentence leakage in inner threshold selection."
                    )

                y_pred = model.predict(
                    validation[IVT_COLUMNS]
                ).astype(int)

                for row_index, row in validation.iterrows():
                    inner_prediction_rows.append(
                        {
                            "subject_id": str(row["subject_id"]),
                            "corruption_repeat": int(repeat),
                            "label_nr": int(row["label_nr"]),
                            "prediction_nr": int(y_pred[row_index]),
                        }
                    )

        inner_predictions = pd.DataFrame(
            inner_prediction_rows
        )
        inner_macro_f1, inner_ba = subject_mean_inner_metrics(
            inner_predictions
        )
        candidate_rows.append(
            {
                "outer_held_out_subject": (
                    outer_held_out_subject
                ),
                "outer_held_out_sentence_fold": int(
                    outer_held_out_sentence_fold
                ),
                "condition": condition,
                "velocity_threshold_deg_s": float(threshold),
                "n_inner_splits": int(len(inner_splits)),
                "n_inner_validation_predictions": int(
                    len(inner_predictions)
                ),
                "inner_macro_f1": inner_macro_f1,
                "inner_balanced_accuracy": inner_ba,
                "selected": False,
            }
        )

    selected_row = max(
        candidate_rows,
        key=lambda row: candidate_sort_key(
            row,
            standard_threshold,
        ),
    )
    selected_threshold = float(
        selected_row["velocity_threshold_deg_s"]
    )
    for row in candidate_rows:
        row["selected"] = bool(
            float(row["velocity_threshold_deg_s"])
            == selected_threshold
        )

    return selected_threshold, pd.DataFrame(candidate_rows)


def nested_evaluation(
    tables: dict[
        float,
        dict[tuple[str, int], pd.DataFrame],
    ],
    conditions: list[str],
    candidate_thresholds: list[float],
    repeats: int,
    c_value: float,
    seed: int,
    n_inner_splits: int,
    standard_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    any_clean = tables[candidate_thresholds[0]][("clean", 1)]
    all_subjects = sorted(any_clean["subject_id"].unique())
    all_sentence_folds = sorted(
        any_clean["sentence_fold"].unique()
    )

    selection_frames: list[pd.DataFrame] = []
    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for subject_index, held_out_subject in enumerate(
        all_subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Nested I-VT outer subject "
            f"{subject_index:02d}/{len(all_subjects)}: "
            f"{held_out_subject}"
        )

        for held_out_sentence_fold in all_sentence_folds:
            outer_train_subjects = [
                subject
                for subject in all_subjects
                if subject != held_out_subject
            ]
            outer_train_sentence_folds = [
                fold
                for fold in all_sentence_folds
                if fold != held_out_sentence_fold
            ]

            for condition in conditions:
                (
                    selected_threshold,
                    candidate_frame,
                ) = select_threshold_for_outer_split(
                    tables=tables,
                    condition=condition,
                    candidate_thresholds=candidate_thresholds,
                    outer_held_out_subject=held_out_subject,
                    outer_held_out_sentence_fold=(
                        held_out_sentence_fold
                    ),
                    all_subjects=all_subjects,
                    all_sentence_folds=all_sentence_folds,
                    repeats=repeats,
                    c_value=c_value,
                    seed=seed,
                    n_inner_splits=n_inner_splits,
                    standard_threshold=standard_threshold,
                )
                selection_frames.append(candidate_frame)

                clean_table = tables[selected_threshold][
                    ("clean", 1)
                ]
                outer_train = clean_table[
                    clean_table["subject_id"].isin(
                        outer_train_subjects
                    )
                    & clean_table["sentence_fold"].isin(
                        outer_train_sentence_folds
                    )
                ].copy()

                model = make_svm(c_value, seed)
                model.fit(
                    outer_train[IVT_COLUMNS],
                    outer_train["label_nr"].to_numpy(dtype=int),
                )

                repeat_count = condition_repeat_count(
                    condition,
                    repeats,
                )
                for repeat in range(1, repeat_count + 1):
                    test_table = tables[selected_threshold][
                        (condition, repeat)
                    ]
                    outer_test = test_table[
                        (
                            test_table["subject_id"]
                            == held_out_subject
                        )
                        & (
                            test_table["sentence_fold"]
                            == held_out_sentence_fold
                        )
                    ].copy().reset_index(drop=True)

                    if outer_test.empty:
                        raise ValueError(
                            "Empty outer test split: "
                            f"subject={held_out_subject}, "
                            f"sentence_fold={held_out_sentence_fold}, "
                            f"condition={condition}."
                        )
                    if set(outer_train["sentence_key"]) & set(
                        outer_test["sentence_key"]
                    ):
                        raise ValueError(
                            "Sentence leakage in outer evaluation."
                        )

                    y_true = outer_test[
                        "label_nr"
                    ].to_numpy(dtype=int)
                    y_pred = model.predict(
                        outer_test[IVT_COLUMNS]
                    ).astype(int)
                    scores = decision_scores(
                        model,
                        outer_test[IVT_COLUMNS],
                    )
                    metrics = score_predictions(
                        y_true,
                        y_pred,
                        scores,
                    )

                    outer_rows.append(
                        {
                            "condition": condition,
                            "corruption_repeat": int(repeat),
                            "held_out_subject": held_out_subject,
                            "held_out_sentence_fold": int(
                                held_out_sentence_fold
                            ),
                            "selected_velocity_threshold_deg_s": float(
                                selected_threshold
                            ),
                            "n_train": int(len(outer_train)),
                            "n_test": int(len(outer_test)),
                            **metrics,
                        }
                    )

                    for row_index, row in outer_test.iterrows():
                        prediction_rows.append(
                            {
                                "condition": condition,
                                "corruption_repeat": int(repeat),
                                "held_out_subject": held_out_subject,
                                "held_out_sentence_fold": int(
                                    held_out_sentence_fold
                                ),
                                "selected_velocity_threshold_deg_s": float(
                                    selected_threshold
                                ),
                                "trial_id": str(row["trial_id"]),
                                "subject_id": str(row["subject_id"]),
                                "task": str(row["task"]),
                                "label_nr": int(row["label_nr"]),
                                "prediction_nr": int(y_pred[row_index]),
                                "decision_score_nr": float(
                                    scores[row_index]
                                ),
                            }
                        )

    selections = pd.concat(
        selection_frames,
        ignore_index=True,
    )
    outer_folds = pd.DataFrame(outer_rows)
    predictions = pd.DataFrame(prediction_rows)
    return selections, outer_folds, predictions


def aggregate_nested_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    for values, group in predictions.groupby(
        [
            "condition",
            "corruption_repeat",
            "held_out_subject",
        ],
        sort=True,
    ):
        condition, repeat, subject = values
        repeat_rows.append(
            {
                "condition": condition,
                "corruption_repeat": int(repeat),
                "held_out_subject": subject,
                "n_predictions": int(len(group)),
                **score_predictions(
                    group["label_nr"].to_numpy(dtype=int),
                    group["prediction_nr"].to_numpy(dtype=int),
                    group["decision_score_nr"].to_numpy(dtype=float),
                ),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)

    subject_metrics = (
        repeat_metrics.groupby(
            ["condition", "held_out_subject"],
            as_index=False,
        )[METRICS]
        .mean()
    )

    summary_rows: list[dict[str, Any]] = []
    for condition, group in subject_metrics.groupby(
        "condition",
        sort=True,
    ):
        row: dict[str, Any] = {
            "condition": condition,
            "n_subjects": int(
                group["held_out_subject"].nunique()
            ),
            "n_input_features": len(IVT_COLUMNS),
        }
        for metric in METRICS:
            row[f"{metric}_subject_mean"] = float(
                group[metric].mean()
            )
        summary_rows.append(row)

    return (
        repeat_metrics,
        subject_metrics,
        pd.DataFrame(summary_rows),
    )


def threshold_selection_summary(
    selections: pd.DataFrame,
) -> pd.DataFrame:
    selected = selections[
        selections["selected"]
    ].copy()
    rows: list[dict[str, Any]] = []

    for condition, group in selected.groupby(
        "condition",
        sort=True,
    ):
        values = group[
            "velocity_threshold_deg_s"
        ].astype(float)
        counts = Counter(values.tolist())
        mode_threshold, mode_count = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]

        row: dict[str, Any] = {
            "condition": condition,
            "n_outer_splits": int(len(group)),
            "selected_threshold_median": float(
                values.median()
            ),
            "selected_threshold_mean": float(
                values.mean()
            ),
            "selected_threshold_mode": float(
                mode_threshold
            ),
            "mode_count": int(mode_count),
        }
        for threshold in sorted(
            selections["velocity_threshold_deg_s"].unique()
        ):
            row[
                f"selected_count_{threshold:g}_deg_s"
            ] = int(np.sum(values == threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_vs_nested_tests(
    nested_subject_metrics: pd.DataFrame,
    fixed_subject_metrics_path: Path,
) -> pd.DataFrame:
    if not fixed_subject_metrics_path.is_file():
        raise FileNotFoundError(
            fixed_subject_metrics_path
        )
    fixed = pd.read_csv(
        fixed_subject_metrics_path
    )
    required = {
        "condition",
        "held_out_subject",
        *METRICS,
    }
    missing = sorted(required - set(fixed.columns))
    if missing:
        raise ValueError(
            f"Fixed subject metrics lack columns: {missing}"
        )

    rows: list[dict[str, Any]] = []
    shared_conditions = sorted(
        set(nested_subject_metrics["condition"])
        & set(fixed["condition"])
    )

    for condition in shared_conditions:
        condition_rows: list[dict[str, Any]] = []
        nested_condition = nested_subject_metrics[
            nested_subject_metrics["condition"].eq(condition)
        ].set_index("held_out_subject")
        fixed_condition = fixed[
            fixed["condition"].eq(condition)
        ].set_index("held_out_subject")

        for metric in METRICS:
            paired = pd.concat(
                [
                    fixed_condition[metric].rename("fixed"),
                    nested_condition[metric].rename("nested"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            difference = paired["nested"] - paired["fixed"]

            if np.allclose(
                difference.to_numpy(dtype=float),
                0.0,
            ):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    paired["nested"],
                    paired["fixed"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            condition_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "fixed_mean": float(
                        paired["fixed"].mean()
                    ),
                    "nested_mean": float(
                        paired["nested"].mean()
                    ),
                    "mean_difference_nested_minus_fixed": float(
                        difference.mean()
                    ),
                    "nested_wins": int(np.sum(difference > 0)),
                    "ties": int(
                        np.sum(np.isclose(difference, 0.0))
                    ),
                    "nested_losses": int(np.sum(difference < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_across_metrics": math.nan,
                }
            )

        adjusted = holm_adjust(
            [float(row["p_value"]) for row in condition_rows]
        )
        for row, adjusted_value in zip(
            condition_rows,
            adjusted,
        ):
            row["p_holm_across_metrics"] = adjusted_value
        rows.extend(condition_rows)

    return pd.DataFrame(rows)


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
        "--common-map",
        default=(
            "outputs/common16_comparison/"
            "common16_trial_map.csv"
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
        default=(
            "outputs/ivt_event_baseline_nested_threshold"
        ),
    )
    parser.add_argument(
        "--fixed-feature-root",
        default=(
            "outputs/ivt_event_baseline_fixed_benchmark_exact/"
            "subject_ivt_features"
        ),
    )
    parser.add_argument(
        "--fixed-subject-metrics",
        default=(
            "outputs/ivt_event_baseline_fixed_benchmark_exact/"
            "ivt_fixed_subject_metrics.csv"
        ),
    )
    parser.add_argument(
        "--conditions",
        default=(
            "clean,rate10,noise50,random40,high_random"
        ),
    )
    parser.add_argument(
        "--velocity-thresholds",
        default="30,60,120,240,480",
    )
    parser.add_argument(
        "--standard-threshold",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-fixation-ms",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--burst-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
    )
    args = parser.parse_args()

    conditions = parse_conditions(
        args.conditions
    )
    if "clean" not in conditions:
        conditions = ["clean"] + conditions

    thresholds = parse_thresholds(
        args.velocity_thresholds
    )
    if args.standard_threshold not in thresholds:
        raise ValueError(
            "--standard-threshold must be included in "
            "--velocity-thresholds."
        )
    if args.inner_splits < 2:
        raise ValueError(
            "--inner-splits must be at least 2."
        )
    if args.repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )

    manifest = pd.read_csv(args.manifest)
    required_manifest = {
        "subject_id",
        "task",
        "block_id",
        "sample_parquet",
    }
    missing_manifest = sorted(
        required_manifest - set(manifest.columns)
    )
    if missing_manifest:
        raise ValueError(
            f"Manifest lacks columns: {missing_manifest}"
        )
    manifest["subject_id"] = (
        manifest["subject_id"].astype(str).str.upper()
    )
    manifest["task"] = (
        manifest["task"].astype(str).str.upper()
    )
    subjects = sorted(
        manifest["subject_id"].unique()
    )
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Manifest subjects={len(subjects)}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    processed_root = Path(
        args.processed_root
    )
    # Validate the frozen official map and exact nine-feature order.
    load_common_official_maps(
        Path(args.common_map)
    )
    aoi_lookup = load_word_template(
        Path(args.word_template)
    )
    sentence_folds = load_sentence_folds(
        Path(args.sentence_folds_file)
    )
    metadata = build_metadata(
        processed_root,
        subjects,
        sentence_folds,
    )
    if len(metadata) != EXPECTED_TRIALS:
        raise ValueError(
            f"Metadata rows={len(metadata)}, "
            f"expected={EXPECTED_TRIALS}."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    feature_root = (
        output_dir / "subject_ivt_features"
    )

    tables: dict[
        float,
        dict[tuple[str, int], pd.DataFrame],
    ] = {}

    for threshold_index, threshold in enumerate(
        thresholds,
        start=1,
    ):
        print("\n" + "#" * 78)
        print(
            f"Threshold feature extraction "
            f"{threshold_index}/{len(thresholds)}: "
            f"{threshold:g} deg/s"
        )
        threshold_output_root = (
            Path(args.fixed_feature_root)
            if (
                float(threshold)
                == float(args.standard_threshold)
                and not args.overwrite_features
            )
            else feature_root
        )
        extracted = extract_conditions(
            manifest=manifest,
            subjects=subjects,
            processed_root=processed_root,
            aoi_lookup=aoi_lookup,
            output_root=threshold_output_root,
            conditions=conditions,
            repeats=args.repeats,
            seed=args.seed,
            burst_seconds=args.burst_seconds,
            velocity_threshold=threshold,
            min_fixation_ms=args.min_fixation_ms,
            overwrite=args.overwrite_features,
        )
        tables[threshold] = {
            key: condition_table(
                metadata,
                frame,
            )
            for key, frame in extracted.items()
        }

    print(
        "\n=== NESTED I-VT THRESHOLD INPUT ==="
    )
    print(f"subjects: {len(subjects)}")
    print(f"trials: {len(metadata)}")
    print(f"conditions: {conditions}")
    print(f"thresholds: {thresholds}")
    print(f"inner splits: {args.inner_splits}")
    print(
        "selection metric: "
        "inner subject-mean Macro F1"
    )

    (
        selections,
        outer_folds,
        predictions,
    ) = nested_evaluation(
        tables=tables,
        conditions=conditions,
        candidate_thresholds=thresholds,
        repeats=args.repeats,
        c_value=args.c_value,
        seed=args.seed,
        n_inner_splits=args.inner_splits,
        standard_threshold=args.standard_threshold,
    )
    (
        repeat_metrics,
        subject_metrics,
        summary,
    ) = aggregate_nested_metrics(
        predictions
    )
    selection_summary = threshold_selection_summary(
        selections
    )
    comparison_tests = fixed_vs_nested_tests(
        subject_metrics,
        Path(args.fixed_subject_metrics),
    )

    paths = {
        "selections": (
            output_dir
            / "ivt_nested_threshold_candidates.csv"
        ),
        "selection_summary": (
            output_dir
            / "ivt_nested_threshold_selection_summary.csv"
        ),
        "outer_folds": (
            output_dir
            / "ivt_nested_outer_fold_metrics.csv"
        ),
        "predictions": (
            output_dir
            / "ivt_nested_predictions.csv"
        ),
        "repeat_metrics": (
            output_dir
            / "ivt_nested_subject_repeat_metrics.csv"
        ),
        "subject_metrics": (
            output_dir
            / "ivt_nested_subject_metrics.csv"
        ),
        "summary": (
            output_dir
            / "ivt_nested_summary.csv"
        ),
        "fixed_comparison": (
            output_dir
            / "ivt_nested_vs_fixed_tests.csv"
        ),
        "manifest": (
            output_dir
            / "ivt_nested_manifest.json"
        ),
    }

    selections.to_csv(
        paths["selections"],
        index=False,
        encoding="utf-8-sig",
    )
    selection_summary.to_csv(
        paths["selection_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    outer_folds.to_csv(
        paths["outer_folds"],
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    repeat_metrics.to_csv(
        paths["repeat_metrics"],
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
    comparison_tests.to_csv(
        paths["fixed_comparison"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "analysis": (
                    "nested_quality_specific_ivt_threshold"
                ),
                "conditions": conditions,
                "velocity_threshold_candidates_deg_s": (
                    thresholds
                ),
                "standard_threshold_deg_s": float(
                    args.standard_threshold
                ),
                "selection_metric": (
                    "inner_subject_mean_macro_f1"
                ),
                "tie_breaking": [
                    "inner_balanced_accuracy",
                    "closest_to_standard_threshold",
                    "lower_threshold",
                ],
                "inner_splits": int(
                    args.inner_splits
                ),
                "outer_protocol": (
                    "unseen_subject_plus_unseen_sentence"
                ),
                "classifier": (
                    "LinearSVC_C_0.01"
                ),
                "min_fixation_ms": float(
                    args.min_fixation_ms
                ),
                "repeats": int(args.repeats),
                "seed": int(args.seed),
                "fixed_feature_root": str(
                    Path(args.fixed_feature_root)
                ),
                "outer_test_used_for_selection": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== NESTED I-VT THRESHOLD SUMMARY ==="
    )
    print(
        summary[
            [
                "condition",
                "n_input_features",
                "macro_f1_subject_mean",
                "balanced_accuracy_subject_mean",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== NESTED I-VT THRESHOLD SELECTIONS ==="
    )
    print(
        selection_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== NESTED I-VT VS FIXED 30 DEG/S ==="
    )
    print(
        comparison_tests[
            comparison_tests["metric"].isin(
                [
                    "macro_f1",
                    "roc_auc",
                    "pr_auc_nr",
                ]
            )
        ][
            [
                "condition",
                "metric",
                "fixed_mean",
                "nested_mean",
                "mean_difference_nested_minus_fixed",
                "nested_wins",
                "ties",
                "nested_losses",
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
