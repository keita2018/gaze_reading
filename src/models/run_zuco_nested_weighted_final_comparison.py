#!/usr/bin/env python
"""Nested, weighted final comparison on matched ZuCo 2.0 trials.

This is the confirmatory clean-condition comparison for the final proposal.

Why this rerun is needed
------------------------
The mixed augmentation set contains 18 degraded copies plus one clean copy of
each original trial. Fitting LinearSVC with a fixed C on all copies changes the
effective regularization strength merely because the number of rows increases.

This script corrects that issue by:

1. assigning every clean/augmented copy weight 1/19, so the total weight of
   one original trial remains exactly 1;
2. applying the same weights to median imputation, standardization, and
   LinearSVC fitting;
3. selecting C independently for each representation using inner
   subject-grouped cross-validation inside every outer LOSO fold.

Compared conditions
-------------------
- dummy_reference
- proposed_30hz_content_clean_nested
- proposed_30hz_content_mixed_weighted_nested
- official_sent_gaze_sacc_nested

Outer test data are always the same clean 16-subject, 11,324-trial subset.
Inner validation is also clean and subject-disjoint. No copy of an outer or
inner validation subject enters augmented training.

Outputs
-------
- nested_final_fold_metrics.csv
- nested_final_predictions.csv
- nested_final_selected_hyperparameters.csv
- nested_final_inner_scores.csv
- nested_final_summary.csv
- nested_final_paired_comparisons.csv
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
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from run_zuco_raw_gaze_loso import score_predictions


EXPECTED_SUBJECTS = 16
EXPECTED_COMMON_TRIALS = 11324
COMBINED_CONDITIONS = [
    "moderate_random",
    "moderate_burst",
    "high_random",
    "high_burst",
    "extreme_random",
    "extreme_burst",
]
QUALITY_COLUMNS = [
    "valid_xy_rate",
    "valid_pupil_rate",
    "valid_basic_rate",
    "longest_invalid_run_ratio",
]


def parse_c_grid(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("All C values must be positive.")
    return values


def prepare_labels(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()
    result["label_nr"] = result["task"].map(
        {"NR": 1, "TSR": 0}
    )
    if result["label_nr"].isna().any():
        bad = sorted(
            result.loc[result["label_nr"].isna(), "task"].unique()
        )
        raise ValueError(f"Unexpected task labels: {bad}")
    result["label_nr"] = result["label_nr"].astype(int)
    return result


def load_common_map(
    path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = prepare_labels(pd.read_csv(path))
    official_columns = sorted(
        [
            column
            for column in data.columns
            if column.startswith("official_feature_")
            and column.rsplit("_", 1)[-1].isdigit()
        ],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )

    if len(data) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Common rows={len(data)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if data["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Common subjects={data['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )
    if data["trial_id"].duplicated().any():
        raise ValueError("Common map has duplicate trial IDs.")
    if not official_columns:
        raise ValueError("Official feature columns were not found.")

    return data, official_columns


def load_clean_30hz(
    path: Path,
    common_trial_ids: set[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = prepare_labels(pd.read_csv(path))
    data = data[data["trial_id"].isin(common_trial_ids)].copy()

    if len(data) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Matched clean rows={len(data)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if data["trial_id"].duplicated().any():
        raise ValueError("Matched clean data has duplicate trial IDs.")
    if set(data["trial_id"]) != common_trial_ids:
        raise ValueError("Matched clean trial IDs differ from common map.")

    data["_variant"] = "clean"
    return data


def load_augmented(
    cache_root: Path,
    subjects: list[str],
    repeats: int,
    common_trial_ids: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for condition in COMBINED_CONDITIONS:
        for repeat in range(1, repeats + 1):
            subject_frames: list[pd.DataFrame] = []

            for subject in subjects:
                path = (
                    cache_root
                    / condition
                    / f"repeat_{repeat:02d}"
                    / (
                        f"{subject}_{condition}"
                        f"_repeat_{repeat:02d}.csv"
                    )
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                subject_frames.append(pd.read_csv(path))

            table = prepare_labels(
                pd.concat(subject_frames, ignore_index=True)
            )
            table = table[
                table["trial_id"].isin(common_trial_ids)
            ].copy()

            if len(table) != EXPECTED_COMMON_TRIALS:
                raise ValueError(
                    f"{condition}, repeat={repeat}: "
                    f"rows={len(table)}, "
                    f"expected={EXPECTED_COMMON_TRIALS}."
                )
            if table["trial_id"].duplicated().any():
                raise ValueError(
                    f"{condition}, repeat={repeat} "
                    "has duplicate matched trial IDs."
                )

            table["_variant"] = f"{condition}_r{repeat:02d}"
            frames.append(table)

    augmented = pd.concat(frames, ignore_index=True)
    expected_variants = len(COMBINED_CONDITIONS) * repeats
    expected_rows = EXPECTED_COMMON_TRIALS * expected_variants

    if len(augmented) != expected_rows:
        raise ValueError(
            f"Augmented rows={len(augmented)}, "
            f"expected={expected_rows}."
        )

    counts = augmented.groupby("trial_id").size()
    if not counts.eq(expected_variants).all():
        raise ValueError(
            "Not every matched trial has the same number "
            "of augmented variants."
        )

    return augmented


def weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        raise ValueError("A feature is entirely missing in training.")

    x = values[finite]
    w = weights[finite]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cutoff = 0.5 * float(np.sum(w))
    index = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    index = min(index, len(x) - 1)
    return float(x[index])


class WeightedPreprocessor:
    def __init__(self) -> None:
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "WeightedPreprocessor":
        x = np.asarray(x, dtype=float)
        weights = np.asarray(sample_weight, dtype=float)

        if x.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if len(x) != len(weights):
            raise ValueError("X and sample weights differ in length.")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Sample weights must be finite and positive.")

        medians = np.asarray(
            [
                weighted_median(x[:, column], weights)
                for column in range(x.shape[1])
            ],
            dtype=float,
        )
        filled = np.where(np.isfinite(x), x, medians)

        means = np.average(filled, axis=0, weights=weights)
        variances = np.average(
            (filled - means) ** 2,
            axis=0,
            weights=weights,
        )
        scales = np.sqrt(np.maximum(variances, 0.0))
        scales[~np.isfinite(scales) | (scales == 0)] = 1.0

        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if (
            self.medians_ is None
            or self.means_ is None
            or self.scales_ is None
        ):
            raise RuntimeError("Preprocessor has not been fitted.")

        x = np.asarray(x, dtype=float)
        filled = np.where(np.isfinite(x), x, self.medians_)
        return (filled - self.means_) / self.scales_


def fit_linear_svc(
    train: pd.DataFrame,
    feature_columns: list[str],
    c_value: float,
    sample_weight: np.ndarray,
    seed: int,
) -> tuple[WeightedPreprocessor, LinearSVC]:
    x = train[feature_columns].to_numpy(dtype=float)
    y = train["label_nr"].astype(int).to_numpy()

    preprocessor = WeightedPreprocessor().fit(
        x=x,
        sample_weight=sample_weight,
    )
    x_scaled = preprocessor.transform(x)

    model = LinearSVC(
        C=c_value,
        dual=False,
        max_iter=30000,
        random_state=seed,
    )
    model.fit(
        x_scaled,
        y,
        sample_weight=sample_weight,
    )
    return preprocessor, model


def predict_linear_svc(
    preprocessor: WeightedPreprocessor,
    model: LinearSVC,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x = test[feature_columns].to_numpy(dtype=float)
    x_scaled = preprocessor.transform(x)
    predictions = model.predict(x_scaled).astype(int)
    scores = np.asarray(
        model.decision_function(x_scaled),
        dtype=float,
    )
    return predictions, scores


def subject_macro_f1(
    validation: pd.DataFrame,
    predictions: np.ndarray,
) -> list[float]:
    result: list[float] = []
    validation = validation.reset_index(drop=True)

    for subject in sorted(validation["subject_id"].unique()):
        mask = validation["subject_id"].eq(subject).to_numpy()
        result.append(
            float(
                f1_score(
                    validation.loc[mask, "label_nr"].astype(int),
                    predictions[mask],
                    average="macro",
                    zero_division=0,
                )
            )
        )
    return result


def training_weights(
    train: pd.DataFrame,
    weighted_augmented: bool,
    variants_per_trial: int,
) -> np.ndarray:
    if not weighted_augmented:
        return np.ones(len(train), dtype=float)

    weights = np.full(
        len(train),
        1.0 / float(variants_per_trial),
        dtype=float,
    )

    # Every original trial must contribute total weight one.
    check = (
        pd.DataFrame(
            {
                "trial_id": train["trial_id"].to_numpy(),
                "weight": weights,
            }
        )
        .groupby("trial_id")["weight"]
        .sum()
    )
    if not np.allclose(check.to_numpy(), 1.0):
        raise ValueError(
            "Augmented training weights do not sum to one per trial."
        )
    return weights


def select_c_nested(
    outer_train_subjects: list[str],
    training_source: pd.DataFrame,
    clean_validation_source: pd.DataFrame,
    feature_columns: list[str],
    c_grid: list[float],
    inner_splits: int,
    weighted_augmented: bool,
    variants_per_trial: int,
    seed: int,
    outer_subject: str,
    model_name: str,
) -> tuple[float, list[dict[str, Any]]]:
    unique_subjects = np.asarray(
        sorted(outer_train_subjects),
        dtype=object,
    )
    n_splits = min(inner_splits, len(unique_subjects))
    if n_splits < 2:
        raise ValueError("At least two inner folds are required.")

    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(unique_subjects), 1), dtype=float)

    rows: list[dict[str, Any]] = []

    for c_value in c_grid:
        per_subject_scores: list[float] = []

        for inner_fold, (train_indices, validation_indices) in enumerate(
            splitter.split(
                dummy_x,
                groups=unique_subjects,
            ),
            start=1,
        ):
            inner_train_subjects = set(unique_subjects[train_indices])
            inner_validation_subjects = set(
                unique_subjects[validation_indices]
            )

            inner_train = training_source[
                training_source["subject_id"].isin(
                    inner_train_subjects
                )
            ]
            inner_validation = clean_validation_source[
                clean_validation_source["subject_id"].isin(
                    inner_validation_subjects
                )
            ].copy()

            weights = training_weights(
                train=inner_train,
                weighted_augmented=weighted_augmented,
                variants_per_trial=variants_per_trial,
            )
            preprocessor, model = fit_linear_svc(
                train=inner_train,
                feature_columns=feature_columns,
                c_value=c_value,
                sample_weight=weights,
                seed=seed,
            )
            predictions, _ = predict_linear_svc(
                preprocessor=preprocessor,
                model=model,
                test=inner_validation,
                feature_columns=feature_columns,
            )
            fold_subject_scores = subject_macro_f1(
                validation=inner_validation,
                predictions=predictions,
            )
            per_subject_scores.extend(fold_subject_scores)

            rows.append(
                {
                    "outer_held_out_subject": outer_subject,
                    "model_name": model_name,
                    "c_value": c_value,
                    "inner_fold": inner_fold,
                    "n_inner_train_subjects": len(
                        inner_train_subjects
                    ),
                    "n_inner_validation_subjects": len(
                        inner_validation_subjects
                    ),
                    "n_inner_train_rows": len(inner_train),
                    "effective_train_weight_sum": float(
                        np.sum(weights)
                    ),
                    "inner_subject_mean_macro_f1": float(
                        np.mean(fold_subject_scores)
                    ),
                }
            )

        overall = float(np.mean(per_subject_scores))
        rows.append(
            {
                "outer_held_out_subject": outer_subject,
                "model_name": model_name,
                "c_value": c_value,
                "inner_fold": 0,
                "n_inner_train_subjects": math.nan,
                "n_inner_validation_subjects": len(
                    per_subject_scores
                ),
                "n_inner_train_rows": math.nan,
                "effective_train_weight_sum": math.nan,
                "inner_subject_mean_macro_f1": overall,
            }
        )

    aggregate_rows = [
        row for row in rows if row["inner_fold"] == 0
    ]
    aggregate_rows.sort(
        key=lambda row: (
            -row["inner_subject_mean_macro_f1"],
            row["c_value"],
        )
    )
    best_c = float(aggregate_rows[0]["c_value"])
    return best_c, rows


def evaluate_outer_loso(
    common: pd.DataFrame,
    clean: pd.DataFrame,
    augmented_all: pd.DataFrame,
    content_columns: list[str],
    official_columns: list[str],
    c_grid: list[float],
    inner_splits: int,
    variants_per_trial: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    subjects = sorted(common["subject_id"].unique())

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []

    model_specs = [
        {
            "name": "proposed_30hz_content_clean_nested",
            "training_source": clean,
            "validation_source": clean,
            "test_source": clean,
            "features": content_columns,
            "weighted_augmented": False,
        },
        {
            "name": (
                "proposed_30hz_content_mixed_weighted_nested"
            ),
            "training_source": augmented_all,
            "validation_source": clean,
            "test_source": clean,
            "features": content_columns,
            "weighted_augmented": True,
        },
        {
            "name": "official_sent_gaze_sacc_nested",
            "training_source": common,
            "validation_source": common,
            "test_source": common,
            "features": official_columns,
            "weighted_augmented": False,
        },
    ]

    for fold_index, held_out_subject in enumerate(
        subjects,
        start=1,
    ):
        outer_train_subjects = [
            subject
            for subject in subjects
            if subject != held_out_subject
        ]

        clean_test_sorted = (
            clean[clean["subject_id"].eq(held_out_subject)]
            .sort_values("trial_id")
            .reset_index(drop=True)
        )
        official_test_sorted = (
            common[common["subject_id"].eq(held_out_subject)]
            .sort_values("trial_id")
            .reset_index(drop=True)
        )
        if not clean_test_sorted["trial_id"].equals(
            official_test_sorted["trial_id"]
        ):
            raise ValueError(
                f"{held_out_subject}: test trial order differs."
            )
        if not clean_test_sorted["label_nr"].equals(
            official_test_sorted["label_nr"]
        ):
            raise ValueError(
                f"{held_out_subject}: test labels differ."
            )

        # Dummy reference on the same proposal test rows.
        dummy_train = clean[
            clean["subject_id"].isin(outer_train_subjects)
        ]
        dummy_test = clean_test_sorted
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(
            dummy_train[[content_columns[0]]],
            dummy_train["label_nr"],
        )
        dummy_pred = dummy.predict(
            dummy_test[[content_columns[0]]]
        ).astype(int)
        dummy_scores = np.asarray(
            dummy.predict_proba(
                dummy_test[[content_columns[0]]]
            )[:, 1],
            dtype=float,
        )
        dummy_metrics = score_predictions(
            dummy_test["label_nr"].to_numpy(),
            dummy_pred,
            dummy_scores,
        )
        fold_rows.append(
            {
                "feature_set": "dummy_reference",
                "model": "dummy_most_frequent",
                "fold_index": fold_index,
                "held_out_subject": held_out_subject,
                "selected_c": math.nan,
                "n_train_rows": len(dummy_train),
                "effective_train_weight_sum": float(
                    len(dummy_train)
                ),
                "n_test": len(dummy_test),
                **dummy_metrics,
            }
        )
        for index, row in dummy_test.iterrows():
            prediction_rows.append(
                {
                    "feature_set": "dummy_reference",
                    "model": "dummy_most_frequent",
                    "fold_index": fold_index,
                    "held_out_subject": held_out_subject,
                    "trial_id": row["trial_id"],
                    "true_label_nr": int(row["label_nr"]),
                    "predicted_label_nr": int(dummy_pred[index]),
                    "score_nr": float(dummy_scores[index]),
                }
            )

        for spec in model_specs:
            best_c, model_inner_rows = select_c_nested(
                outer_train_subjects=outer_train_subjects,
                training_source=spec["training_source"],
                clean_validation_source=spec[
                    "validation_source"
                ],
                feature_columns=spec["features"],
                c_grid=c_grid,
                inner_splits=inner_splits,
                weighted_augmented=spec[
                    "weighted_augmented"
                ],
                variants_per_trial=variants_per_trial,
                seed=seed,
                outer_subject=held_out_subject,
                model_name=spec["name"],
            )
            inner_rows.extend(model_inner_rows)

            outer_train = spec["training_source"][
                spec["training_source"]["subject_id"].isin(
                    outer_train_subjects
                )
            ]
            outer_test = (
                spec["test_source"][
                    spec["test_source"]["subject_id"].eq(
                        held_out_subject
                    )
                ]
                .sort_values("trial_id")
                .reset_index(drop=True)
            )

            weights = training_weights(
                train=outer_train,
                weighted_augmented=spec[
                    "weighted_augmented"
                ],
                variants_per_trial=variants_per_trial,
            )
            preprocessor, model = fit_linear_svc(
                train=outer_train,
                feature_columns=spec["features"],
                c_value=best_c,
                sample_weight=weights,
                seed=seed,
            )
            predictions, scores = predict_linear_svc(
                preprocessor=preprocessor,
                model=model,
                test=outer_test,
                feature_columns=spec["features"],
            )
            metrics = score_predictions(
                outer_test["label_nr"].to_numpy(),
                predictions,
                scores,
            )

            fold_rows.append(
                {
                    "feature_set": spec["name"],
                    "model": "weighted_linear_svm_nested",
                    "fold_index": fold_index,
                    "held_out_subject": held_out_subject,
                    "selected_c": best_c,
                    "n_train_rows": len(outer_train),
                    "effective_train_weight_sum": float(
                        np.sum(weights)
                    ),
                    "n_test": len(outer_test),
                    **metrics,
                }
            )
            selected_rows.append(
                {
                    "outer_held_out_subject": held_out_subject,
                    "feature_set": spec["name"],
                    "selected_c": best_c,
                    "n_train_rows": len(outer_train),
                    "effective_train_weight_sum": float(
                        np.sum(weights)
                    ),
                }
            )

            for index, row in outer_test.iterrows():
                prediction_rows.append(
                    {
                        "feature_set": spec["name"],
                        "model": "weighted_linear_svm_nested",
                        "fold_index": fold_index,
                        "held_out_subject": held_out_subject,
                        "trial_id": row["trial_id"],
                        "true_label_nr": int(row["label_nr"]),
                        "predicted_label_nr": int(
                            predictions[index]
                        ),
                        "score_nr": float(scores[index]),
                    }
                )

        print(
            f"[{fold_index}/{len(subjects)}] "
            f"Completed nested LOSO for {held_out_subject}."
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(inner_rows),
    )


def summarize_results(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "roc_auc",
        "pr_auc_nr",
    ]
    rows: list[dict[str, Any]] = []

    for (feature_set, model), group in folds.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        pred = predictions[
            predictions["feature_set"].eq(feature_set)
            & predictions["model"].eq(model)
        ]
        pooled = score_predictions(
            pred["true_label_nr"].astype(int).to_numpy(),
            pred["predicted_label_nr"].astype(int).to_numpy(),
            pred["score_nr"].astype(float).to_numpy(),
        )

        row: dict[str, Any] = {
            "feature_set": feature_set,
            "model": model,
            "n_folds": len(group),
            "n_oof_predictions": len(pred),
        }

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_sd"] = float(values.std(ddof=1))
            row[f"{metric}_pooled_oof"] = float(pooled[metric])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "macro_f1_subject_mean",
        ascending=False,
    )


def bootstrap_mean_difference_ci(
    differences: np.ndarray,
    seed: int,
    n_resamples: int = 20000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    indices = rng.integers(
        0,
        n,
        size=(n_resamples, n),
    )
    means = differences[indices].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0

    for rank, original_index in enumerate(order):
        value = min(
            1.0,
            (n - rank) * p_values[original_index],
        )
        running = max(running, value)
        adjusted[original_index] = running

    return adjusted.tolist()


def paired_comparisons(
    folds: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        (
            "proposed_30hz_content_clean_nested",
            "official_sent_gaze_sacc_nested",
        ),
        (
            "proposed_30hz_content_mixed_weighted_nested",
            "official_sent_gaze_sacc_nested",
        ),
        (
            "proposed_30hz_content_mixed_weighted_nested",
            "proposed_30hz_content_clean_nested",
        ),
    ]
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    rows: list[dict[str, Any]] = []

    for metric_index, metric in enumerate(metrics):
        row_indices: list[int] = []
        p_values: list[float] = []

        for comparison_index, (candidate_name, reference_name) in enumerate(
            comparisons
        ):
            candidate = (
                folds[folds["feature_set"].eq(candidate_name)]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            reference = (
                folds[folds["feature_set"].eq(reference_name)]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )

            paired = pd.concat(
                [
                    candidate.rename("candidate"),
                    reference.rename("reference"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            differences = (
                paired["candidate"] - paired["reference"]
            ).to_numpy(dtype=float)

            if np.allclose(differences, 0.0):
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

            ci_low, ci_high = bootstrap_mean_difference_ci(
                differences=differences,
                seed=seed + 100 * metric_index + comparison_index,
            )

            rows.append(
                {
                    "metric": metric,
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "n_subjects": len(paired),
                    "candidate_mean": float(
                        paired["candidate"].mean()
                    ),
                    "reference_mean": float(
                        paired["reference"].mean()
                    ),
                    "mean_difference": float(
                        np.mean(differences)
                    ),
                    "mean_difference_bootstrap_ci95_low": ci_low,
                    "mean_difference_bootstrap_ci95_high": ci_high,
                    "median_difference": float(
                        np.median(differences)
                    ),
                    "wins": int(np.sum(differences > 0)),
                    "ties": int(np.sum(np.isclose(differences, 0))),
                    "losses": int(np.sum(differences < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_within_metric": math.nan,
                }
            )
            row_indices.append(len(rows) - 1)
            p_values.append(p_value)

        adjusted = holm_adjust(p_values)
        for row_index, adjusted_value in zip(
            row_indices,
            adjusted,
        ):
            rows[row_index]["p_holm_within_metric"] = adjusted_value

    return pd.DataFrame(rows)


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
        "--clean-30hz-features",
        default=(
            "outputs/downsampling_robustness/"
            "complete_raw_18_0030hz_features.csv"
        ),
    )
    parser.add_argument(
        "--combined-cache-root",
        default=(
            "outputs/combined_degradation_30hz/"
            "subject_combined_features"
        ),
    )
    parser.add_argument(
        "--rate-invariant-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1,10",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/nested_final_common16",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    c_grid = parse_c_grid(args.c_grid)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.inner_splits < 2:
        raise ValueError("--inner-splits must be at least 2.")

    common, official_columns = load_common_map(
        Path(args.common_map)
    )
    common_trial_ids = set(common["trial_id"])
    subjects = sorted(common["subject_id"].unique())

    clean = load_clean_30hz(
        Path(args.clean_30hz_features),
        common_trial_ids,
    )
    augmented = load_augmented(
        cache_root=Path(args.combined_cache_root),
        subjects=subjects,
        repeats=args.repeats,
        common_trial_ids=common_trial_ids,
    )

    feature_sets = json.loads(
        Path(args.rate_invariant_feature_sets).read_text(
            encoding="utf-8"
        )
    )
    content_columns = [
        column
        for column in feature_sets["strict_rate_invariant"]
        if column not in QUALITY_COLUMNS
    ]

    missing = sorted(set(content_columns) - set(clean.columns))
    if missing:
        raise ValueError(
            f"Content feature columns missing: {missing}"
        )

    variants_per_trial = 1 + len(COMBINED_CONDITIONS) * args.repeats
    clean_weighted = clean.copy()
    augmented_all = pd.concat(
        [clean_weighted, augmented],
        ignore_index=True,
    )

    counts = augmented_all.groupby("trial_id").size()
    if not counts.eq(variants_per_trial).all():
        raise ValueError(
            "The mixed augmentation table does not contain "
            f"{variants_per_trial} variants per trial."
        )

    (
        folds,
        predictions,
        selected,
        inner_scores,
    ) = evaluate_outer_loso(
        common=common,
        clean=clean,
        augmented_all=augmented_all,
        content_columns=content_columns,
        official_columns=official_columns,
        c_grid=c_grid,
        inner_splits=args.inner_splits,
        variants_per_trial=variants_per_trial,
        seed=args.seed,
    )

    summary = summarize_results(folds, predictions)
    paired = paired_comparisons(folds, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_path = output_dir / "nested_final_fold_metrics.csv"
    predictions_path = output_dir / "nested_final_predictions.csv"
    selected_path = (
        output_dir / "nested_final_selected_hyperparameters.csv"
    )
    inner_path = output_dir / "nested_final_inner_scores.csv"
    summary_path = output_dir / "nested_final_summary.csv"
    paired_path = output_dir / "nested_final_paired_comparisons.csv"

    folds.to_csv(
        folds_path,
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(
        selected_path,
        index=False,
        encoding="utf-8-sig",
    )
    inner_scores.to_csv(
        inner_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        paired_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== NESTED WEIGHTED FINAL SUMMARY ===")
    print(
        summary[
            [
                "feature_set",
                "model",
                "n_folds",
                "accuracy_subject_mean",
                "macro_f1_subject_mean",
                "macro_f1_subject_sd",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
                "macro_f1_pooled_oof",
            ]
        ].to_string(index=False)
    )

    print("\n=== SELECTED C COUNTS ===")
    print(
        selected.groupby(
            ["feature_set", "selected_c"]
        ).size().rename("n_outer_folds").to_string()
    )

    print("\n=== PAIRED MACRO F1 COMPARISONS ===")
    print(
        paired[paired["metric"].eq("macro_f1")][
            [
                "candidate",
                "reference",
                "candidate_mean",
                "reference_mean",
                "mean_difference",
                "mean_difference_bootstrap_ci95_low",
                "mean_difference_bootstrap_ci95_high",
                "wins",
                "ties",
                "losses",
                "p_value",
                "p_holm_within_metric",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    print(f"  {folds_path}")
    print(f"  {predictions_path}")
    print(f"  {selected_path}")
    print(f"  {inner_path}")
    print(f"  {summary_path}")
    print(f"  {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
