#!/usr/bin/env python
"""Nested threshold-calibration sensitivity analysis for the common16 comparison.

This script uses exactly the same 16-subject, 11,324-trial matched dataset and
the same unseen-subject + unseen-sentence outer evaluation as
run_zuco_soft_aoi_official_common16_nested_dual.py.

For each learned representation and each outer split, the inner data are used
to jointly select:
- LinearSVC C;
- a decision-function threshold.

The threshold is never selected using the outer test subject or outer test
sentences. Inner out-of-fold scores are generated with simultaneous
subject-disjoint and sentence-disjoint validation. Candidate thresholds are
quantiles of the inner out-of-fold decision scores plus the default threshold
zero.

The goal is a sensitivity analysis: determine whether the Macro-F1 ordering of
Compact Soft-AOI, Full Soft-AOI, RawStat, and official sent_gaze_sacc is caused
mainly by use of LinearSVC's default zero threshold.
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
from sklearn.model_selection import GroupKFold

import run_zuco_soft_aoi_official_common16_nested_dual as base


LEARNED_FEATURE_SETS = [
    "rawstat_all",
    "soft_full_all",
    base.COMPACT_NAME,
    "official_sent_gaze_sacc",
]
ALL_FEATURE_SETS = ["dummy_reference"] + LEARNED_FEATURE_SETS


def candidate_thresholds(
    scores: np.ndarray,
    n_quantiles: int,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("No finite inner decision scores.")

    quantiles = np.linspace(
        0.02,
        0.98,
        int(n_quantiles),
        dtype=float,
    )
    values = np.quantile(scores, quantiles)
    thresholds = np.unique(
        np.concatenate(
            [
                values,
                np.asarray([0.0], dtype=float),
            ]
        )
    )
    return thresholds.astype(float)


def subject_mean_macro_f1_by_threshold(
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    if len(frame) != len(scores):
        raise ValueError("Frame and score lengths differ.")

    y_true = frame["label_nr"].to_numpy(dtype=int)
    predictions = scores[:, None] >= thresholds[None, :]
    subject_values: list[np.ndarray] = []

    for _, subject_frame in frame.reset_index(drop=True).groupby(
        "subject_id",
        sort=False,
    ):
        indices = subject_frame.index.to_numpy(dtype=int)
        y = y_true[indices]
        pred = predictions[indices]

        positive = y == 1
        negative = ~positive

        tp = np.sum(pred & positive[:, None], axis=0)
        fp = np.sum(pred & negative[:, None], axis=0)
        fn = np.sum((~pred) & positive[:, None], axis=0)
        tn = np.sum((~pred) & negative[:, None], axis=0)

        denominator_positive = 2 * tp + fp + fn
        denominator_negative = 2 * tn + fp + fn

        f1_positive = np.divide(
            2 * tp,
            denominator_positive,
            out=np.zeros_like(tp, dtype=float),
            where=denominator_positive > 0,
        )
        f1_negative = np.divide(
            2 * tn,
            denominator_negative,
            out=np.zeros_like(tn, dtype=float),
            where=denominator_negative > 0,
        )
        subject_values.append(
            (f1_positive + f1_negative) / 2.0
        )

    if not subject_values:
        return np.full(len(thresholds), np.nan)
    return np.mean(
        np.vstack(subject_values),
        axis=0,
    )


def choose_threshold(
    frame: pd.DataFrame,
    scores: np.ndarray,
    n_quantiles: int,
) -> tuple[float, float, float, int]:
    thresholds = candidate_thresholds(
        scores,
        n_quantiles=n_quantiles,
    )
    values = subject_mean_macro_f1_by_threshold(
        frame=frame,
        scores=scores,
        thresholds=thresholds,
    )
    if not np.isfinite(values).any():
        raise ValueError("All threshold scores are non-finite.")

    best_score = float(np.nanmax(values))
    best_positions = np.flatnonzero(
        np.isclose(values, best_score)
    )
    # Prefer the threshold closest to zero, then the smaller threshold.
    best_position = min(
        best_positions.tolist(),
        key=lambda index: (
            abs(float(thresholds[index])),
            float(thresholds[index]),
        ),
    )
    selected_threshold = float(thresholds[best_position])

    zero_position = int(
        np.argmin(np.abs(thresholds))
    )
    zero_score = float(values[zero_position])
    return (
        selected_threshold,
        best_score,
        zero_score,
        int(len(thresholds)),
    )


def inner_oof_for_c(
    outer_train: pd.DataFrame,
    feature_columns: list[str],
    c_value: float,
    inner_subject_splits: int,
    seed: int,
) -> pd.DataFrame:
    subjects = np.asarray(
        sorted(outer_train["subject_id"].unique()),
        dtype=object,
    )
    n_splits = min(
        int(inner_subject_splits),
        int(len(subjects)),
    )
    if n_splits < 2:
        raise ValueError(
            "At least two inner subject folds are required."
        )

    splitter = GroupKFold(n_splits=n_splits)
    subject_positions = np.arange(len(subjects))
    sentence_folds = sorted(
        outer_train["sentence_fold"].unique()
    )
    rows: list[dict[str, Any]] = []

    for subject_fold, (
        train_positions,
        validation_positions,
    ) in enumerate(
        splitter.split(
            subject_positions,
            groups=subjects,
        ),
        start=1,
    ):
        train_subjects = set(subjects[train_positions])
        validation_subjects = set(
            subjects[validation_positions]
        )

        for validation_sentence_fold in sentence_folds:
            inner_train = outer_train[
                outer_train["subject_id"].isin(
                    train_subjects
                )
                & outer_train["sentence_fold"].ne(
                    validation_sentence_fold
                )
            ].copy()
            inner_validation = outer_train[
                outer_train["subject_id"].isin(
                    validation_subjects
                )
                & outer_train["sentence_fold"].eq(
                    validation_sentence_fold
                )
            ].copy()

            if inner_train.empty or inner_validation.empty:
                continue
            if set(inner_train["subject_id"]) & set(
                inner_validation["subject_id"]
            ):
                raise ValueError(
                    "Inner subject leakage detected."
                )
            if set(inner_train["sentence_key"]) & set(
                inner_validation["sentence_key"]
            ):
                raise ValueError(
                    "Inner sentence leakage detected."
                )

            y_train = inner_train[
                "label_nr"
            ].to_numpy(dtype=int)
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError(
                    "Inner training split lacks both classes."
                )

            x_train, x_validation, _, _ = (
                base.fit_preprocessor(
                    inner_train[feature_columns],
                    inner_validation[feature_columns],
                )
            )
            model = base.fit_linear_svc(
                x_train=x_train,
                y_train=y_train,
                c_value=c_value,
                seed=seed,
            )
            scores = np.asarray(
                model.decision_function(x_validation),
                dtype=float,
            ).reshape(-1)

            ordered = inner_validation.reset_index(
                drop=True
            )
            for index, row in ordered.iterrows():
                rows.append(
                    {
                        "trial_id": row["trial_id"],
                        "subject_id": row["subject_id"],
                        "sentence_key": row[
                            "sentence_key"
                        ],
                        "label_nr": int(
                            row["label_nr"]
                        ),
                        "decision_score_nr": float(
                            scores[index]
                        ),
                        "inner_subject_fold": int(
                            subject_fold
                        ),
                        "inner_validation_sentence_fold": int(
                            validation_sentence_fold
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No inner out-of-fold predictions.")
    if result["trial_id"].duplicated().any():
        duplicates = result.loc[
            result["trial_id"].duplicated(
                keep=False
            ),
            ["trial_id", "subject_id", "sentence_key"],
        ]
        raise ValueError(
            "Inner OOF trials are duplicated:\n"
            + duplicates.head(20).to_string(index=False)
        )
    if set(result["trial_id"]) != set(
        outer_train["trial_id"]
    ):
        missing = sorted(
            set(outer_train["trial_id"])
            - set(result["trial_id"])
        )
        extra = sorted(
            set(result["trial_id"])
            - set(outer_train["trial_id"])
        )
        raise ValueError(
            "Inner OOF coverage mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return result


def select_c_and_threshold(
    outer_train: pd.DataFrame,
    feature_columns: list[str],
    c_grid: list[float],
    inner_subject_splits: int,
    threshold_quantiles: int,
    seed: int,
    outer_subject: str,
    outer_sentence_fold: int,
    feature_set: str,
) -> tuple[
    float,
    float,
    pd.DataFrame,
    pd.DataFrame,
]:
    candidate_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []

    for c_value in c_grid:
        oof = inner_oof_for_c(
            outer_train=outer_train,
            feature_columns=feature_columns,
            c_value=float(c_value),
            inner_subject_splits=inner_subject_splits,
            seed=seed,
        )
        (
            threshold,
            tuned_score,
            approximately_zero_score,
            n_thresholds,
        ) = choose_threshold(
            frame=oof,
            scores=oof["decision_score_nr"].to_numpy(
                dtype=float
            ),
            n_quantiles=threshold_quantiles,
        )
        candidate_rows.append(
            {
                "outer_subject": outer_subject,
                "outer_sentence_fold": int(
                    outer_sentence_fold
                ),
                "feature_set": feature_set,
                "c_value": float(c_value),
                "selected_threshold_for_c": float(
                    threshold
                ),
                "inner_tuned_subject_mean_macro_f1": (
                    float(tuned_score)
                ),
                "inner_nearest_zero_subject_mean_macro_f1": (
                    float(approximately_zero_score)
                ),
                "n_candidate_thresholds": int(
                    n_thresholds
                ),
            }
        )
        oof = oof.copy()
        oof["outer_subject"] = outer_subject
        oof["outer_sentence_fold"] = int(
            outer_sentence_fold
        )
        oof["feature_set"] = feature_set
        oof["c_value"] = float(c_value)
        oof_rows.append(oof)

    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.sort_values(
        [
            "inner_tuned_subject_mean_macro_f1",
            "c_value",
            "selected_threshold_for_c",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best = candidates.iloc[0]
    return (
        float(best["c_value"]),
        float(best["selected_threshold_for_c"]),
        candidates,
        pd.concat(oof_rows, ignore_index=True),
    )


def dummy_scores(
    model: DummyClassifier,
    x_test: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(x_test),
        dtype=float,
    )
    classes = np.asarray(model.classes_)
    if 1 in classes:
        position = int(
            np.flatnonzero(classes == 1)[0]
        )
        return probabilities[:, position]
    return np.zeros(len(x_test), dtype=float)


def evaluate(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    c_grid: list[float],
    inner_subject_splits: int,
    threshold_quantiles: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    subjects = sorted(data["subject_id"].unique())
    sentence_folds = sorted(
        data["sentence_fold"].unique()
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    inner_candidate_rows: list[pd.DataFrame] = []

    total_outer = len(subjects) * len(sentence_folds)
    outer_counter = 0

    for held_out_subject in subjects:
        for held_out_sentence_fold in sentence_folds:
            outer_counter += 1
            print(
                f"[Outer {outer_counter:03d}/{total_outer}] "
                f"subject={held_out_subject}, "
                f"sentence_fold={held_out_sentence_fold}"
            )

            outer_train = data[
                data["subject_id"].ne(
                    held_out_subject
                )
                & data["sentence_fold"].ne(
                    held_out_sentence_fold
                )
            ].copy()
            outer_test = data[
                data["subject_id"].eq(
                    held_out_subject
                )
                & data["sentence_fold"].eq(
                    held_out_sentence_fold
                )
            ].copy()

            if outer_test.empty:
                raise ValueError(
                    "Empty outer test split."
                )
            if held_out_subject in set(
                outer_train["subject_id"]
            ):
                raise ValueError(
                    "Outer subject leakage detected."
                )
            if set(outer_train["sentence_key"]) & set(
                outer_test["sentence_key"]
            ):
                raise ValueError(
                    "Outer sentence leakage detected."
                )

            y_train = outer_train[
                "label_nr"
            ].to_numpy(dtype=int)
            y_test = outer_test[
                "label_nr"
            ].to_numpy(dtype=int)
            ordered_test = outer_test.reset_index(
                drop=True
            )

            dummy = DummyClassifier(
                strategy="most_frequent"
            )
            x_dummy_train = np.zeros(
                (len(outer_train), 1),
                dtype=float,
            )
            x_dummy_test = np.zeros(
                (len(outer_test), 1),
                dtype=float,
            )
            dummy.fit(x_dummy_train, y_train)
            dummy_predictions = dummy.predict(
                x_dummy_test
            ).astype(int)
            dummy_decision = dummy_scores(
                dummy,
                x_dummy_test,
            )
            dummy_metrics = base.score_predictions(
                y_test,
                dummy_predictions,
                dummy_decision,
            )
            fold_rows.append(
                {
                    "feature_set": "dummy_reference",
                    "held_out_subject": held_out_subject,
                    "held_out_sentence_fold": int(
                        held_out_sentence_fold
                    ),
                    "n_input_features": 0,
                    "selected_c": math.nan,
                    "selected_threshold": math.nan,
                    "n_train": int(len(outer_train)),
                    "n_test": int(len(outer_test)),
                    **dummy_metrics,
                }
            )
            for index, row in ordered_test.iterrows():
                prediction_rows.append(
                    {
                        "feature_set": "dummy_reference",
                        "held_out_subject": held_out_subject,
                        "held_out_sentence_fold": int(
                            held_out_sentence_fold
                        ),
                        "trial_id": row["trial_id"],
                        "subject_id": row["subject_id"],
                        "task": row["task"],
                        "sentence_key": row[
                            "sentence_key"
                        ],
                        "true_label_nr": int(
                            row["label_nr"]
                        ),
                        "predicted_label_nr": int(
                            dummy_predictions[index]
                        ),
                        "score_nr": float(
                            dummy_decision[index]
                        ),
                    }
                )

            for feature_set in LEARNED_FEATURE_SETS:
                columns = feature_sets[feature_set]
                (
                    selected_c,
                    selected_threshold,
                    candidates,
                    _,
                ) = select_c_and_threshold(
                    outer_train=outer_train,
                    feature_columns=columns,
                    c_grid=c_grid,
                    inner_subject_splits=(
                        inner_subject_splits
                    ),
                    threshold_quantiles=(
                        threshold_quantiles
                    ),
                    seed=seed,
                    outer_subject=held_out_subject,
                    outer_sentence_fold=(
                        held_out_sentence_fold
                    ),
                    feature_set=feature_set,
                )
                inner_candidate_rows.append(
                    candidates
                )
                best_candidate = candidates.iloc[0]
                selection_rows.append(
                    {
                        "feature_set": feature_set,
                        "held_out_subject": held_out_subject,
                        "held_out_sentence_fold": int(
                            held_out_sentence_fold
                        ),
                        "selected_c": float(
                            selected_c
                        ),
                        "selected_threshold": float(
                            selected_threshold
                        ),
                        "inner_tuned_subject_mean_macro_f1": float(
                            best_candidate[
                                "inner_tuned_subject_mean_macro_f1"
                            ]
                        ),
                        "inner_nearest_zero_subject_mean_macro_f1": float(
                            best_candidate[
                                "inner_nearest_zero_subject_mean_macro_f1"
                            ]
                        ),
                        "n_input_features": int(
                            len(columns)
                        ),
                    }
                )

                x_train, x_test, _, _ = (
                    base.fit_preprocessor(
                        outer_train[columns],
                        outer_test[columns],
                    )
                )
                model = base.fit_linear_svc(
                    x_train=x_train,
                    y_train=y_train,
                    c_value=selected_c,
                    seed=seed,
                )
                scores = np.asarray(
                    model.decision_function(x_test),
                    dtype=float,
                ).reshape(-1)
                predictions = (
                    scores >= selected_threshold
                ).astype(int)
                metrics = base.score_predictions(
                    y_test,
                    predictions,
                    scores,
                )
                fold_rows.append(
                    {
                        "feature_set": feature_set,
                        "held_out_subject": held_out_subject,
                        "held_out_sentence_fold": int(
                            held_out_sentence_fold
                        ),
                        "n_input_features": int(
                            len(columns)
                        ),
                        "selected_c": float(
                            selected_c
                        ),
                        "selected_threshold": float(
                            selected_threshold
                        ),
                        "n_train": int(
                            len(outer_train)
                        ),
                        "n_test": int(
                            len(outer_test)
                        ),
                        **metrics,
                    }
                )
                for index, row in ordered_test.iterrows():
                    prediction_rows.append(
                        {
                            "feature_set": feature_set,
                            "held_out_subject": (
                                held_out_subject
                            ),
                            "held_out_sentence_fold": int(
                                held_out_sentence_fold
                            ),
                            "trial_id": row["trial_id"],
                            "subject_id": row["subject_id"],
                            "task": row["task"],
                            "sentence_key": row[
                                "sentence_key"
                            ],
                            "true_label_nr": int(
                                row["label_nr"]
                            ),
                            "predicted_label_nr": int(
                                predictions[index]
                            ),
                            "score_nr": float(
                                scores[index]
                            ),
                        }
                    )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    candidates = pd.concat(
        inner_candidate_rows,
        ignore_index=True,
    )

    expected_rows = (
        len(ALL_FEATURE_SETS)
        * base.EXPECTED_COMMON_TRIALS
    )
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Prediction rows={len(predictions)}, "
            f"expected={expected_rows}."
        )
    if predictions[
        ["feature_set", "trial_id"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate feature-set/trial predictions."
        )
    return folds, predictions, selections, candidates


def build_subject_metrics(
    predictions: pd.DataFrame,
    feature_counts: dict[str, int],
) -> pd.DataFrame:
    return base.build_subject_metrics(
        predictions=predictions,
        feature_counts=feature_counts,
    )


def summarize(
    subject_metrics: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in subject_metrics.groupby(
        "feature_set",
        sort=False,
    ):
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "n_subjects": int(len(group)),
            "n_input_features": int(
                group["n_input_features"].iloc[0]
            ),
        }
        for metric in base.METRICS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )
            row[f"{metric}_subject_mean"] = float(
                values.mean()
            )
            row[f"{metric}_subject_std"] = float(
                values.std(ddof=1)
            )

        selected = selections[
            selections["feature_set"].eq(
                feature_set
            )
        ]
        row["selected_c_median"] = (
            float(selected["selected_c"].median())
            if len(selected)
            else math.nan
        )
        row["selected_threshold_median"] = (
            float(
                selected[
                    "selected_threshold"
                ].median()
            )
            if len(selected)
            else math.nan
        )
        row["inner_threshold_gain_mean"] = (
            float(
                (
                    selected[
                        "inner_tuned_subject_mean_macro_f1"
                    ]
                    - selected[
                        "inner_nearest_zero_subject_mean_macro_f1"
                    ]
                ).mean()
            )
            if len(selected)
            else math.nan
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "macro_f1_subject_mean",
        ascending=False,
    )


def pairwise_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    return base.build_pairwise_tests(
        subject_metrics
    )


def compare_with_previous(
    tuned: pd.DataFrame,
    previous_path: Path,
) -> pd.DataFrame:
    if not previous_path.is_file():
        raise FileNotFoundError(previous_path)
    previous = pd.read_csv(previous_path)

    required = {
        "feature_set",
        "held_out_subject",
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    }
    missing = sorted(required - set(previous.columns))
    if missing:
        raise ValueError(
            f"Previous subject metrics lack: {missing}"
        )

    rows: list[dict[str, Any]] = []
    for feature_set in LEARNED_FEATURE_SETS:
        for metric in [
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
        ]:
            reference = (
                previous[
                    previous["feature_set"].eq(
                        feature_set
                    )
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            candidate = (
                tuned[
                    tuned["feature_set"].eq(
                        feature_set
                    )
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            paired = pd.concat(
                [
                    reference.rename("zero_threshold"),
                    candidate.rename("threshold_tuned"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            differences = (
                paired["threshold_tuned"]
                - paired["zero_threshold"]
            )

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
                test = wilcoxon(
                    paired["threshold_tuned"],
                    paired["zero_threshold"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(
                    test.statistic
                )
                p_value = float(
                    test.pvalue
                )

            rows.append(
                {
                    "feature_set": feature_set,
                    "metric": metric,
                    "n_subjects": int(len(paired)),
                    "zero_threshold_mean": float(
                        paired["zero_threshold"].mean()
                    ),
                    "threshold_tuned_mean": float(
                        paired["threshold_tuned"].mean()
                    ),
                    "mean_difference_tuned_minus_zero": float(
                        differences.mean()
                    ),
                    "wins": int(
                        np.sum(differences > 0)
                    ),
                    "ties": int(
                        np.sum(
                            np.isclose(
                                differences,
                                0.0,
                            )
                        )
                    ),
                    "losses": int(
                        np.sum(differences < 0)
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_within_metric": math.nan,
                }
            )

    result = pd.DataFrame(rows)
    for metric, indices in result.groupby(
        "metric"
    ).groups.items():
        positions = list(indices)
        adjusted = base.holm_adjust(
            result.loc[
                positions,
                "p_value",
            ].tolist()
        )
        result.loc[
            positions,
            "p_holm_within_metric",
        ] = adjusted
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common-map",
        default=(
            "outputs/common16_comparison/"
            "common16_trial_map.csv"
        ),
    )
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
        "--sentence-folds-file",
        default=(
            "outputs/soft_aoi_confound_controls_30hz/"
            "dual_holdout_sentence_folds.csv"
        ),
    )
    parser.add_argument(
        "--previous-subject-metrics",
        default=(
            "outputs/"
            "soft_aoi_official_common16_nested_dual/"
            "common16_nested_dual_subject_metrics.csv"
        ),
    )
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1,10",
    )
    parser.add_argument(
        "--inner-subject-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--threshold-quantiles",
        type=int,
        default=41,
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "soft_aoi_official_common16_threshold_tuned"
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    c_grid = base.parse_c_grid(args.c_grid)
    if args.inner_subject_splits < 2:
        raise ValueError(
            "--inner-subject-splits must be at least 2."
        )
    if args.threshold_quantiles < 5:
        raise ValueError(
            "--threshold-quantiles must be at least 5."
        )

    common, official_columns = base.load_common_map(
        Path(args.common_map)
    )
    data, feature_sets, metadata = (
        base.load_matched_table(
            common=common,
            official_columns=official_columns,
            raw_path=Path(args.raw_features),
            raw_sets_path=Path(
                args.raw_feature_sets
            ),
            soft_path=Path(args.soft_features),
            soft_manifest_path=Path(
                args.soft_feature_manifest
            ),
            sentence_folds_path=Path(
                args.sentence_folds_file
            ),
        )
    )

    print(
        "=== COMMON16 THRESHOLD-TUNED INPUT ==="
    )
    print(
        f"subjects: {data['subject_id'].nunique()}"
    )
    print(f"trials: {len(data)}")
    print(
        f"unique sentences: "
        f"{data['sentence_key'].nunique()}"
    )
    print(
        f"sentence folds: "
        f"{data['sentence_fold'].nunique()}"
    )
    print(f"C grid: {c_grid}")
    print(
        f"threshold quantiles: "
        f"{args.threshold_quantiles}"
    )
    for feature_set in ALL_FEATURE_SETS:
        count = (
            0
            if feature_set == "dummy_reference"
            else len(feature_sets[feature_set])
        )
        print(
            f"  {feature_set}: {count} features"
        )

    (
        fold_metrics,
        predictions,
        selections,
        inner_candidates,
    ) = evaluate(
        data=data,
        feature_sets=feature_sets,
        c_grid=c_grid,
        inner_subject_splits=(
            args.inner_subject_splits
        ),
        threshold_quantiles=(
            args.threshold_quantiles
        ),
        seed=args.seed,
    )

    feature_counts = {
        "dummy_reference": 0,
        **{
            name: len(columns)
            for name, columns in feature_sets.items()
        },
    }
    subject_metrics = build_subject_metrics(
        predictions=predictions,
        feature_counts=feature_counts,
    )
    summary = summarize(
        subject_metrics=subject_metrics,
        selections=selections,
    )
    tests = pairwise_tests(
        subject_metrics
    )
    zero_comparison = compare_with_previous(
        tuned=subject_metrics,
        previous_path=Path(
            args.previous_subject_metrics
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths = {
        "fold_metrics": (
            output_dir
            / "threshold_tuned_outer_folds.csv"
        ),
        "predictions": (
            output_dir
            / "threshold_tuned_predictions.csv"
        ),
        "selections": (
            output_dir
            / "threshold_tuned_selected_parameters.csv"
        ),
        "inner_candidates": (
            output_dir
            / "threshold_tuned_inner_candidates.csv"
        ),
        "subject_metrics": (
            output_dir
            / "threshold_tuned_subject_metrics.csv"
        ),
        "summary": (
            output_dir
            / "threshold_tuned_summary.csv"
        ),
        "pairwise": (
            output_dir
            / "threshold_tuned_pairwise_tests.csv"
        ),
        "zero_comparison": (
            output_dir
            / "threshold_tuned_vs_zero_tests.csv"
        ),
        "manifest": (
            output_dir
            / "threshold_tuned_manifest.json"
        ),
    }

    fold_metrics.to_csv(
        paths["fold_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    selections.to_csv(
        paths["selections"],
        index=False,
        encoding="utf-8-sig",
    )
    inner_candidates.to_csv(
        paths["inner_candidates"],
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
        paths["pairwise"],
        index=False,
        encoding="utf-8-sig",
    )
    zero_comparison.to_csv(
        paths["zero_comparison"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "analysis_status": (
                    "threshold_calibration_sensitivity"
                ),
                "protocol": (
                    "nested_unseen_subject_and_"
                    "unseen_sentence"
                ),
                "n_subjects": base.EXPECTED_SUBJECTS,
                "n_trials": (
                    base.EXPECTED_COMMON_TRIALS
                ),
                "n_unique_sentences": metadata[
                    "n_unique_sentences"
                ],
                "n_sentence_folds": metadata[
                    "n_sentence_folds"
                ],
                "c_grid": c_grid,
                "inner_subject_splits": int(
                    args.inner_subject_splits
                ),
                "threshold_quantiles": int(
                    args.threshold_quantiles
                ),
                "threshold_selection_metric": (
                    "inner_oof_subject_mean_macro_f1"
                ),
                "seed": int(args.seed),
                "previous_zero_threshold_subject_metrics": (
                    args.previous_subject_metrics
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== COMMON16 THRESHOLD-TUNED SUMMARY ==="
    )
    print(
        summary[
            [
                "feature_set",
                "n_input_features",
                "macro_f1_subject_mean",
                "balanced_accuracy_subject_mean",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
                "selected_c_median",
                "selected_threshold_median",
                "inner_threshold_gain_mean",
            ]
        ].to_string(index=False)
    )

    key_pairs = {
        ("rawstat_all", base.COMPACT_NAME),
        ("soft_full_all", base.COMPACT_NAME),
        (
            base.COMPACT_NAME,
            "official_sent_gaze_sacc",
        ),
        (
            "soft_full_all",
            "official_sent_gaze_sacc",
        ),
        (
            "rawstat_all",
            "official_sent_gaze_sacc",
        ),
    }
    key_tests = tests[
        tests["metric"].eq("macro_f1")
        & tests.apply(
            lambda row: (
                row["reference_feature_set"],
                row["candidate_feature_set"],
            )
            in key_pairs,
            axis=1,
        )
    ][
        [
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
    print(
        "\n=== KEY THRESHOLD-TUNED MACRO-F1 TESTS ==="
    )
    print(key_tests.to_string(index=False))

    zero_macro = zero_comparison[
        zero_comparison["metric"].eq(
            "macro_f1"
        )
    ][
        [
            "feature_set",
            "zero_threshold_mean",
            "threshold_tuned_mean",
            "mean_difference_tuned_minus_zero",
            "wins",
            "ties",
            "losses",
            "p_value",
            "p_holm_within_metric",
        ]
    ]
    print(
        "\n=== THRESHOLD TUNING VS ZERO: MACRO-F1 ==="
    )
    print(zero_macro.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
