#!/usr/bin/env python
"""Layer 2b: source-only quality-conditioned decision-score canonicalization.

This experiment complements feature-level canonicalization.  Instead of
reconstructing all 27 Compact features, it learns to reconstruct the clean
classifier decision score from the score observed under degraded gaze and the
12 Soft-AOI quality features.

For a clean-trained source classifier f,

    s_clean = f(x_clean)
    s_obs   = f(x_observed)

and a source-only Ridge model learns

    delta_s = s_clean - s_obs
             ~ h(s_obs, quality_features, quality_missing_flags).

The corrected decision score is

    s_hat = s_obs + h(...).

No WebQAmGaze label, target statistic, target scaling parameter, or target
threshold is used during fitting.  The design is evaluated with the same
Common16 unseen-participant + unseen-sentence protocol and then frozen for
WebQAmGaze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_zuco_quality_conditioned_canonicalization import (  # noqa: E402
    COMPACT_COLUMNS,
    CONDITION_ORDER,
    QUALITY_COLUMNS,
    load_common_metadata,
    load_condition_tables,
    load_external_compact,
    metric_table,
)
from run_zuco_soft_aoi_ablation import holm_adjust  # noqa: E402
from run_zuco_soft_aoi_confound_controls import (  # noqa: E402
    METRICS,
    decision_scores,
    make_svm,
    score_predictions,
)


EXPECTED_COMMON_SUBJECTS = 16
EXPECTED_COMMON_TRIALS = 11324

METHOD_ORDER = [
    "identity_compact27",
    "quality_score_canonical_compact27",
]


@dataclass
class ScoreCanonicalizer:
    feature_columns: list[str]
    quality_columns: list[str]
    quality_imputer: SimpleImputer
    input_scaler: StandardScaler
    ridge: Ridge
    alpha: float
    clean_anchor_multiplier: float
    max_absolute_correction: float

    def build_input(
        self,
        frame: pd.DataFrame,
        observed_scores: np.ndarray,
    ) -> np.ndarray:
        quality = frame[self.quality_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        quality_values = quality.to_numpy(dtype=float)
        quality_missing = ~np.isfinite(quality_values)
        quality_imputed = self.quality_imputer.transform(
            quality
        )
        raw_input = np.hstack(
            [
                np.asarray(
                    observed_scores,
                    dtype=float,
                ).reshape(-1, 1),
                quality_imputed,
                quality_missing.astype(float),
            ]
        )
        return self.input_scaler.transform(raw_input)

    def correct(
        self,
        frame: pd.DataFrame,
        observed_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_scaled = self.build_input(
            frame,
            observed_scores,
        )
        predicted_delta = self.ridge.predict(
            x_scaled
        ).reshape(-1)
        clipped_delta = np.clip(
            predicted_delta,
            -self.max_absolute_correction,
            self.max_absolute_correction,
        )
        corrected = (
            np.asarray(
                observed_scores,
                dtype=float,
            ).reshape(-1)
            + clipped_delta
        )
        return corrected, clipped_delta


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def environment_weights(
    labels: list[str],
    clean_anchor_multiplier: float,
) -> np.ndarray:
    series = pd.Series(labels)
    counts = series.value_counts().to_dict()
    base = np.asarray(
        [
            len(series)
            / (
                len(counts)
                * counts[label]
            )
            for label in labels
        ],
        dtype=float,
    )
    clean_mask = np.asarray(
        [label == "clean" for label in labels],
        dtype=bool,
    )
    base[clean_mask] *= float(
        clean_anchor_multiplier
    )
    return base


def fit_score_canonicalizer(
    clean_train: pd.DataFrame,
    train_mask: np.ndarray,
    condition_tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
    base_model: Any,
    alpha: float,
    clean_anchor_multiplier: float,
    max_absolute_correction: float,
) -> ScoreCanonicalizer:
    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )
    if clean_anchor_multiplier < 1:
        raise ValueError(
            "clean_anchor_multiplier must be >= 1."
        )
    if max_absolute_correction <= 0:
        raise ValueError(
            "max_absolute_correction must be positive."
        )

    quality_imputer = SimpleImputer(
        strategy="median",
    )
    quality_imputer.fit(
        clean_train[QUALITY_COLUMNS]
    )

    clean_scores = decision_scores(
        base_model,
        clean_train[COMPACT_COLUMNS],
    )

    inputs = []
    targets = []
    environment_labels: list[str] = []

    for (
        condition,
        _repeat,
    ), table in condition_tables.items():
        observed = table.loc[
            train_mask
        ].reset_index(drop=True)
        observed_scores = decision_scores(
            base_model,
            observed[COMPACT_COLUMNS],
        )
        quality = observed[
            QUALITY_COLUMNS
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )
        quality_values = quality.to_numpy(
            dtype=float
        )
        quality_missing = ~np.isfinite(
            quality_values
        )
        quality_imputed = (
            quality_imputer.transform(quality)
        )
        raw_input = np.hstack(
            [
                observed_scores.reshape(
                    -1,
                    1,
                ),
                quality_imputed,
                quality_missing.astype(
                    float
                ),
            ]
        )
        target_delta = (
            clean_scores - observed_scores
        )

        inputs.append(raw_input)
        targets.append(target_delta)
        environment_labels.extend(
            [condition] * len(observed)
        )

    x = np.vstack(inputs)
    y = np.concatenate(targets)
    weights = environment_weights(
        environment_labels,
        clean_anchor_multiplier=(
            clean_anchor_multiplier
        ),
    )

    input_scaler = StandardScaler()
    x_scaled = input_scaler.fit_transform(
        x,
        sample_weight=weights,
    )
    ridge = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
    )
    ridge.fit(
        x_scaled,
        y,
        sample_weight=weights,
    )
    return ScoreCanonicalizer(
        feature_columns=list(
            COMPACT_COLUMNS
        ),
        quality_columns=list(
            QUALITY_COLUMNS
        ),
        quality_imputer=quality_imputer,
        input_scaler=input_scaler,
        ridge=ridge,
        alpha=float(alpha),
        clean_anchor_multiplier=float(
            clean_anchor_multiplier
        ),
        max_absolute_correction=float(
            max_absolute_correction
        ),
    )


def evaluate_internal(
    tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
    alpha: float,
    clean_anchor_multiplier: float,
    max_absolute_correction: float,
    c_value: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = tables[("clean", 0)]
    subjects = sorted(
        clean["subject_id"].unique()
    )
    sentence_folds = sorted(
        clean["sentence_fold"].unique()
    )
    prediction_rows: list[
        dict[str, Any]
    ] = []
    score_reconstruction_rows: list[
        dict[str, Any]
    ] = []

    for subject_index, held_out in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Layer-2b outer subject "
            f"{subject_index:02d}/{len(subjects)}: "
            f"{held_out}"
        )

        for sentence_fold in sentence_folds:
            train_mask = (
                clean["subject_id"].ne(
                    held_out
                )
                & clean["sentence_fold"].ne(
                    sentence_fold
                )
            ).to_numpy()
            test_mask = (
                clean["subject_id"].eq(
                    held_out
                )
                & clean["sentence_fold"].eq(
                    sentence_fold
                )
            ).to_numpy()

            clean_train = clean.loc[
                train_mask
            ].reset_index(drop=True)
            clean_test = clean.loc[
                test_mask
            ].reset_index(drop=True)

            base_model = make_svm(
                c_value,
                seed,
            )
            base_model.fit(
                clean_train[
                    COMPACT_COLUMNS
                ],
                clean_train[
                    "label_nr"
                ].to_numpy(dtype=int),
            )
            canonicalizer = (
                fit_score_canonicalizer(
                    clean_train=clean_train,
                    train_mask=train_mask,
                    condition_tables=tables,
                    base_model=base_model,
                    alpha=alpha,
                    clean_anchor_multiplier=(
                        clean_anchor_multiplier
                    ),
                    max_absolute_correction=(
                        max_absolute_correction
                    ),
                )
            )

            clean_test_scores = decision_scores(
                base_model,
                clean_test[
                    COMPACT_COLUMNS
                ],
            )

            for (
                condition,
                repeat,
            ), table in tables.items():
                test = table.loc[
                    test_mask
                ].reset_index(drop=True)
                if test[
                    "trial_id"
                ].tolist() != clean_test[
                    "trial_id"
                ].tolist():
                    raise ValueError(
                        "Layer-2b test trial "
                        "alignment failed."
                    )

                observed_scores = (
                    decision_scores(
                        base_model,
                        test[
                            COMPACT_COLUMNS
                        ],
                    )
                )
                (
                    corrected_scores,
                    predicted_delta,
                ) = canonicalizer.correct(
                    test,
                    observed_scores,
                )
                y_true = test[
                    "label_nr"
                ].to_numpy(dtype=int)

                for (
                    method,
                    scores,
                ) in (
                    (
                        METHOD_ORDER[0],
                        observed_scores,
                    ),
                    (
                        METHOD_ORDER[1],
                        corrected_scores,
                    ),
                ):
                    predictions = (
                        scores >= 0.0
                    ).astype(int)
                    for row_index, row in (
                        test.iterrows()
                    ):
                        prediction_rows.append(
                            {
                                "condition": condition,
                                "corruption_repeat": int(
                                    repeat
                                ),
                                "method": method,
                                "held_out_subject": (
                                    held_out
                                ),
                                "sentence_fold": int(
                                    sentence_fold
                                ),
                                "trial_id": str(
                                    row["trial_id"]
                                ),
                                "subject_id": str(
                                    row["subject_id"]
                                ),
                                "task": str(
                                    row["task"]
                                ),
                                "label_nr": int(
                                    y_true[
                                        row_index
                                    ]
                                ),
                                "prediction_nr": int(
                                    predictions[
                                        row_index
                                    ]
                                ),
                                "decision_score_nr": float(
                                    scores[
                                        row_index
                                    ]
                                ),
                            }
                        )

                for row_index, row in (
                    test.iterrows()
                ):
                    raw_error = float(
                        observed_scores[
                            row_index
                        ]
                        - clean_test_scores[
                            row_index
                        ]
                    )
                    corrected_error = float(
                        corrected_scores[
                            row_index
                        ]
                        - clean_test_scores[
                            row_index
                        ]
                    )
                    score_reconstruction_rows.append(
                        {
                            "condition": condition,
                            "corruption_repeat": int(
                                repeat
                            ),
                            "held_out_subject": (
                                held_out
                            ),
                            "sentence_fold": int(
                                sentence_fold
                            ),
                            "trial_id": str(
                                row["trial_id"]
                            ),
                            "clean_score": float(
                                clean_test_scores[
                                    row_index
                                ]
                            ),
                            "observed_score": float(
                                observed_scores[
                                    row_index
                                ]
                            ),
                            "corrected_score": float(
                                corrected_scores[
                                    row_index
                                ]
                            ),
                            "predicted_delta": float(
                                predicted_delta[
                                    row_index
                                ]
                            ),
                            "raw_absolute_error": abs(
                                raw_error
                            ),
                            "corrected_absolute_error": abs(
                                corrected_error
                            ),
                            "raw_squared_error": (
                                raw_error**2
                            ),
                            "corrected_squared_error": (
                                corrected_error**2
                            ),
                        }
                    )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(
            score_reconstruction_rows
        ),
    )


def aggregate_internal_metrics(
    predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    repeat_rows = []
    for values, group in predictions.groupby(
        [
            "condition",
            "corruption_repeat",
            "method",
            "held_out_subject",
        ],
        sort=True,
    ):
        (
            condition,
            repeat,
            method,
            subject,
        ) = values
        repeat_rows.append(
            {
                "condition": condition,
                "corruption_repeat": int(
                    repeat
                ),
                "method": method,
                "held_out_subject": subject,
                "n_predictions": int(
                    len(group)
                ),
                **score_predictions(
                    group[
                        "label_nr"
                    ].to_numpy(dtype=int),
                    group[
                        "prediction_nr"
                    ].to_numpy(dtype=int),
                    group[
                        "decision_score_nr"
                    ].to_numpy(
                        dtype=float
                    ),
                ),
            }
        )
    repeat_metrics = pd.DataFrame(
        repeat_rows
    )
    subject_metrics = (
        repeat_metrics.groupby(
            [
                "condition",
                "method",
                "held_out_subject",
            ],
            as_index=False,
        )[METRICS]
        .mean()
    )
    summary_rows = []
    for (
        condition,
        method,
    ), group in subject_metrics.groupby(
        ["condition", "method"],
        sort=False,
    ):
        row: dict[str, Any] = {
            "condition": condition,
            "method": method,
            "n_subjects": int(
                group[
                    "held_out_subject"
                ].nunique()
            ),
        }
        for metric in METRICS:
            row[
                f"{metric}_subject_mean"
            ] = float(
                group[metric].mean()
            )
        summary_rows.append(row)
    return (
        repeat_metrics,
        subject_metrics,
        pd.DataFrame(summary_rows),
    )


def paired_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        family = []
        for condition in CONDITION_ORDER:
            subset = subject_metrics[
                subject_metrics[
                    "condition"
                ].eq(condition)
            ]
            identity = (
                subset[
                    subset["method"].eq(
                        METHOD_ORDER[0]
                    )
                ]
                .set_index(
                    "held_out_subject"
                )[metric]
                .sort_index()
            )
            canonical = (
                subset[
                    subset["method"].eq(
                        METHOD_ORDER[1]
                    )
                ]
                .set_index(
                    "held_out_subject"
                )[metric]
                .sort_index()
            )
            paired = pd.concat(
                [
                    identity.rename(
                        "identity"
                    ),
                    canonical.rename(
                        "canonical"
                    ),
                ],
                axis=1,
                join="inner",
            ).dropna()
            difference = (
                paired["canonical"]
                - paired["identity"]
            )
            if np.allclose(
                difference.to_numpy(
                    dtype=float
                ),
                0.0,
            ):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    paired["canonical"],
                    paired["identity"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(
                    result.statistic
                )
                p_value = float(
                    result.pvalue
                )
            family.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "identity_mean": float(
                        paired[
                            "identity"
                        ].mean()
                    ),
                    "canonical_mean": float(
                        paired[
                            "canonical"
                        ].mean()
                    ),
                    "mean_difference_canonical_minus_identity": float(
                        difference.mean()
                    ),
                    "canonical_wins": int(
                        np.sum(
                            difference > 0
                        )
                    ),
                    "ties": int(
                        np.sum(
                            np.isclose(
                                difference,
                                0.0,
                            )
                        )
                    ),
                    "canonical_losses": int(
                        np.sum(
                            difference < 0
                        )
                    ),
                    "wilcoxon_statistic": (
                        statistic
                    ),
                    "p_value": p_value,
                    "p_holm_seven_conditions_within_metric": (
                        math.nan
                    ),
                }
            )
        adjusted = holm_adjust(
            [
                float(row["p_value"])
                for row in family
            ]
        )
        for row, adjusted_value in zip(
            family,
            adjusted,
        ):
            row[
                "p_holm_seven_conditions_within_metric"
            ] = adjusted_value
        rows.extend(family)
    return pd.DataFrame(rows)


def score_reconstruction_summary(
    reconstruction: pd.DataFrame,
) -> pd.DataFrame:
    repeat_subject = (
        reconstruction.groupby(
            [
                "condition",
                "corruption_repeat",
                "held_out_subject",
            ],
            as_index=False,
        )
        .agg(
            raw_mae=(
                "raw_absolute_error",
                "mean",
            ),
            corrected_mae=(
                "corrected_absolute_error",
                "mean",
            ),
            raw_rmse=(
                "raw_squared_error",
                lambda value: float(
                    np.sqrt(
                        np.mean(value)
                    )
                ),
            ),
            corrected_rmse=(
                "corrected_squared_error",
                lambda value: float(
                    np.sqrt(
                        np.mean(value)
                    )
                ),
            ),
            mean_absolute_delta=(
                "predicted_delta",
                lambda value: float(
                    np.mean(
                        np.abs(value)
                    )
                ),
            ),
        )
    )
    subject = (
        repeat_subject.groupby(
            [
                "condition",
                "held_out_subject",
            ],
            as_index=False,
        )[
            [
                "raw_mae",
                "corrected_mae",
                "raw_rmse",
                "corrected_rmse",
                "mean_absolute_delta",
            ]
        ]
        .mean()
    )
    return (
        subject.groupby(
            "condition",
            as_index=False,
        )
        .agg(
            n_subjects=(
                "held_out_subject",
                "nunique",
            ),
            raw_mae_subject_mean=(
                "raw_mae",
                "mean",
            ),
            corrected_mae_subject_mean=(
                "corrected_mae",
                "mean",
            ),
            mae_improvement=(
                "raw_mae",
                "mean",
            ),
            raw_rmse_subject_mean=(
                "raw_rmse",
                "mean",
            ),
            corrected_rmse_subject_mean=(
                "corrected_rmse",
                "mean",
            ),
            mean_absolute_delta_subject_mean=(
                "mean_absolute_delta",
                "mean",
            ),
        )
        .assign(
            mae_improvement=lambda frame: (
                frame[
                    "raw_mae_subject_mean"
                ]
                - frame[
                    "corrected_mae_subject_mean"
                ]
            ),
            rmse_improvement=lambda frame: (
                frame[
                    "raw_rmse_subject_mean"
                ]
                - frame[
                    "corrected_rmse_subject_mean"
                ]
            ),
        )
    )



def cluster_bootstrap_score_methods(
    predictions: pd.DataFrame,
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = sorted(
        predictions["participant_id"].unique()
    )
    rng = np.random.default_rng(seed)
    rows = []

    for bootstrap_index in range(
        n_bootstraps
    ):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )
        parts = []
        for draw_index, participant in enumerate(
            sampled
        ):
            part = predictions[
                predictions[
                    "participant_id"
                ].eq(participant)
            ].copy()
            part["bootstrap_cluster"] = (
                f"{draw_index}:{participant}"
            )
            parts.append(part)
        bootstrap = pd.concat(
            parts,
            ignore_index=True,
        )
        metrics = metric_table(
            bootstrap
        )
        lookup = {
            row.method: row
            for row in metrics.itertuples(
                index=False
            )
        }
        identity = lookup[
            METHOD_ORDER[0]
        ]
        canonical = lookup[
            METHOD_ORDER[1]
        ]
        for metric in (
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
        ):
            identity_value = float(
                getattr(identity, metric)
            )
            canonical_value = float(
                getattr(canonical, metric)
            )
            rows.append(
                {
                    "bootstrap_index": int(
                        bootstrap_index
                    ),
                    "metric": metric,
                    "identity": identity_value,
                    "canonical": canonical_value,
                    "difference_canonical_minus_identity": (
                        canonical_value
                        - identity_value
                    ),
                }
            )

    replicates = pd.DataFrame(rows)
    summary_rows = []
    for metric, group in replicates.groupby(
        "metric",
        sort=True,
    ):
        difference = group[
            "difference_canonical_minus_identity"
        ].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "n_bootstraps": int(
                    len(group)
                ),
                "bootstrap_mean_difference": float(
                    np.mean(difference)
                ),
                "ci95_lower": float(
                    np.quantile(
                        difference,
                        0.025,
                    )
                ),
                "ci95_upper": float(
                    np.quantile(
                        difference,
                        0.975,
                    )
                ),
                "probability_canonical_gt_identity": float(
                    np.mean(
                        difference > 0
                    )
                ),
            }
        )
    return (
        replicates,
        pd.DataFrame(
            summary_rows
        ),
    )


def fit_external(
    clean: pd.DataFrame,
    tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
    external: pd.DataFrame,
    alpha: float,
    clean_anchor_multiplier: float,
    max_absolute_correction: float,
    c_value: float,
    seed: int,
    n_bootstraps: int,
) -> tuple[
    Any,
    ScoreCanonicalizer,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    base_model = make_svm(
        c_value,
        seed,
    )
    base_model.fit(
        clean[COMPACT_COLUMNS],
        clean["label_nr"].to_numpy(
            dtype=int
        ),
    )
    train_mask = np.ones(
        len(clean),
        dtype=bool,
    )
    canonicalizer = (
        fit_score_canonicalizer(
            clean_train=clean,
            train_mask=train_mask,
            condition_tables=tables,
            base_model=base_model,
            alpha=alpha,
            clean_anchor_multiplier=(
                clean_anchor_multiplier
            ),
            max_absolute_correction=(
                max_absolute_correction
            ),
        )
    )

    identity_scores = decision_scores(
        base_model,
        external[COMPACT_COLUMNS],
    )
    (
        canonical_scores,
        predicted_delta,
    ) = canonicalizer.correct(
        external,
        identity_scores,
    )

    parts = []
    for method, scores in (
        (
            METHOD_ORDER[0],
            identity_scores,
        ),
        (
            METHOD_ORDER[1],
            canonical_scores,
        ),
    ):
        part = external[
            [
                "external_trial_id",
                "participant_id",
                "task",
                "label_nr",
            ]
        ].copy()
        part["method"] = method
        part["prediction_nr"] = (
            scores >= 0.0
        ).astype(int)
        part[
            "decision_score_nr"
        ] = scores
        if method == METHOD_ORDER[1]:
            part[
                "predicted_score_delta"
            ] = predicted_delta
        else:
            part[
                "predicted_score_delta"
            ] = 0.0
        parts.append(part)
    predictions = pd.concat(
        parts,
        ignore_index=True,
    )

    pooled = metric_table(predictions)
    participant_metrics = metric_table(
        predictions,
        group_column="participant_id",
    )
    participant_summary = (
        participant_metrics.groupby(
            "method",
            as_index=False,
        )[METRICS]
        .mean()
    )
    (
        bootstrap_replicates,
        bootstrap_summary,
    ) = cluster_bootstrap_score_methods(
        predictions,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    return (
        base_model,
        canonicalizer,
        predictions,
        pooled,
        participant_metrics,
        participant_summary,
        bootstrap_replicates,
        bootstrap_summary,
    )


def coefficient_table(
    canonicalizer: ScoreCanonicalizer,
) -> pd.DataFrame:
    input_names = [
        "observed_decision_score",
        *QUALITY_COLUMNS,
        *[
            f"{column}__missing"
            for column in QUALITY_COLUMNS
        ],
    ]
    coefficients = np.asarray(
        canonicalizer.ridge.coef_,
        dtype=float,
    ).reshape(-1)
    return pd.DataFrame(
        {
            "input_name": input_names,
            "coefficient_for_score_delta": (
                coefficients
            ),
            "absolute_coefficient": np.abs(
                coefficients
            ),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    )


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
        "--sentence-folds-file",
        default=(
            "outputs/soft_aoi_confound_controls_30hz/"
            "dual_holdout_sentence_folds.csv"
        ),
    )
    parser.add_argument(
        "--clean-soft-features",
        default=(
            "data/processed/zuco2/"
            "soft_aoi_all_subjects/"
            "zuco2_soft_aoi_features_30hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--rate10-soft-features",
        default=(
            "data/processed/zuco2/"
            "soft_aoi_all_subjects/"
            "zuco2_soft_aoi_features_10hz_all_subjects.csv"
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
    parser.add_argument(
        "--external-compact",
        default=(
            "outputs/webqamgaze_external_validation_stage3c/"
            "soft_compact_features_primary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "quality_conditioned_score_canonicalization_v1"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--canonicalizer-alpha",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--clean-anchor-multiplier",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--max-absolute-correction",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    metadata = load_common_metadata(
        Path(args.common_map),
        Path(
            args.sentence_folds_file
        ),
    )
    tables = load_condition_tables(
        metadata=metadata,
        clean_path=Path(
            args.clean_soft_features
        ),
        rate10_path=Path(
            args.rate10_soft_features
        ),
        noise_root=Path(
            args.noise_cache_root
        ),
        missingness_root=Path(
            args.missingness_cache_root
        ),
        combined_root=Path(
            args.combined_cache_root
        ),
        repeats=args.repeats,
    )
    clean = tables[("clean", 0)]
    external = load_external_compact(
        Path(args.external_compact)
    )

    print(
        "=== LAYER-2B QUALITY-CONDITIONED SCORE CANONICALIZATION INPUT ==="
    )
    print(
        f"Common16 subjects: "
        f"{clean['subject_id'].nunique()}"
    )
    print(
        f"Common16 trials: {len(clean)}"
    )
    print(
        f"External participant IDs: "
        f"{external['participant_id'].nunique()}"
    )
    print(
        f"External trials: {len(external)}"
    )
    print(
        f"Ridge alpha: "
        f"{args.canonicalizer_alpha}"
    )
    print(
        "Clean anchor multiplier: "
        f"{args.clean_anchor_multiplier}"
    )
    print(
        "Maximum absolute score correction: "
        f"{args.max_absolute_correction}"
    )
    print(
        "No WebQAmGaze label is used before prediction."
    )

    (
        internal_predictions,
        score_reconstruction,
    ) = evaluate_internal(
        tables=tables,
        alpha=args.canonicalizer_alpha,
        clean_anchor_multiplier=(
            args.clean_anchor_multiplier
        ),
        max_absolute_correction=(
            args.max_absolute_correction
        ),
        c_value=args.c_value,
        seed=args.seed,
    )
    (
        internal_repeat_metrics,
        internal_subject_metrics,
        internal_summary,
    ) = aggregate_internal_metrics(
        internal_predictions
    )
    test_table = paired_tests(
        internal_subject_metrics
    )
    reconstruction_summary = (
        score_reconstruction_summary(
            score_reconstruction
        )
    )

    (
        base_model,
        final_canonicalizer,
        external_predictions,
        external_pooled,
        external_participant_metrics,
        external_participant_summary,
        bootstrap_replicates,
        bootstrap_summary,
    ) = fit_external(
        clean=clean,
        tables=tables,
        external=external,
        alpha=args.canonicalizer_alpha,
        clean_anchor_multiplier=(
            args.clean_anchor_multiplier
        ),
        max_absolute_correction=(
            args.max_absolute_correction
        ),
        c_value=args.c_value,
        seed=args.seed,
        n_bootstraps=args.n_bootstraps,
    )
    coefficients = coefficient_table(
        final_canonicalizer
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths = {
        "internal_predictions": (
            output_dir
            / "layer2b_internal_predictions.csv"
        ),
        "internal_repeat_metrics": (
            output_dir
            / "layer2b_internal_subject_repeat_metrics.csv"
        ),
        "internal_subject_metrics": (
            output_dir
            / "layer2b_internal_subject_metrics.csv"
        ),
        "internal_summary": (
            output_dir
            / "layer2b_internal_summary.csv"
        ),
        "internal_tests": (
            output_dir
            / "layer2b_internal_identity_vs_canonical_tests.csv"
        ),
        "score_reconstruction_trials": (
            output_dir
            / "layer2b_score_reconstruction_trials.csv"
        ),
        "score_reconstruction_summary": (
            output_dir
            / "layer2b_score_reconstruction_summary.csv"
        ),
        "external_predictions": (
            output_dir
            / "layer2b_external_predictions.csv"
        ),
        "external_pooled": (
            output_dir
            / "layer2b_external_pooled_metrics.csv"
        ),
        "external_participant_metrics": (
            output_dir
            / "layer2b_external_participant_metrics.csv"
        ),
        "external_participant_summary": (
            output_dir
            / "layer2b_external_participant_summary.csv"
        ),
        "bootstrap_replicates": (
            output_dir
            / "layer2b_external_bootstrap_replicates.csv"
        ),
        "bootstrap_summary": (
            output_dir
            / "layer2b_external_bootstrap_summary.csv"
        ),
        "coefficients": (
            output_dir
            / "layer2b_score_canonicalizer_coefficients.csv"
        ),
        "artifact": (
            output_dir
            / "layer2b_source_only_score_canonicalizer.joblib"
        ),
        "manifest": (
            output_dir
            / "layer2b_manifest.json"
        ),
    }

    internal_predictions.to_csv(
        paths["internal_predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    internal_repeat_metrics.to_csv(
        paths[
            "internal_repeat_metrics"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    internal_subject_metrics.to_csv(
        paths[
            "internal_subject_metrics"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    internal_summary.to_csv(
        paths["internal_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    test_table.to_csv(
        paths["internal_tests"],
        index=False,
        encoding="utf-8-sig",
    )
    score_reconstruction.to_csv(
        paths[
            "score_reconstruction_trials"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    reconstruction_summary.to_csv(
        paths[
            "score_reconstruction_summary"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    external_predictions.to_csv(
        paths[
            "external_predictions"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    external_pooled.to_csv(
        paths["external_pooled"],
        index=False,
        encoding="utf-8-sig",
    )
    external_participant_metrics.to_csv(
        paths[
            "external_participant_metrics"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    external_participant_summary.to_csv(
        paths[
            "external_participant_summary"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_replicates.to_csv(
        paths[
            "bootstrap_replicates"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        paths[
            "bootstrap_summary"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    coefficients.to_csv(
        paths["coefficients"],
        index=False,
        encoding="utf-8-sig",
    )

    artifact = {
        "artifact_version": "1.0",
        "created_at_utc": (
            utc_now_iso()
        ),
        "base_model": base_model,
        "score_canonicalizer": (
            final_canonicalizer
        ),
        "feature_columns": (
            COMPACT_COLUMNS
        ),
        "quality_columns": (
            QUALITY_COLUMNS
        ),
        "decision_threshold": 0.0,
        "source_scope": (
            "ZuCo Common16 only"
        ),
        "external_labels_used_for_fit": False,
    }
    joblib.dump(
        artifact,
        paths["artifact"],
        compress=3,
    )

    key_inputs = {
        "common_map": Path(
            args.common_map
        ),
        "sentence_folds": Path(
            args.sentence_folds_file
        ),
        "clean_soft_features": Path(
            args.clean_soft_features
        ),
        "rate10_soft_features": Path(
            args.rate10_soft_features
        ),
        "external_compact": Path(
            args.external_compact
        ),
    }
    manifest = {
        "stage": (
            "layer2b_source_only_quality_conditioned_score_canonicalization"
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "method": {
            "base_classifier": (
                "LinearSVC C=0.01 on source clean Compact27"
            ),
            "score_corrector": (
                "Ridge predicting clean-minus-observed decision score"
            ),
            "ridge_alpha": float(
                args.canonicalizer_alpha
            ),
            "clean_anchor_multiplier": float(
                args.clean_anchor_multiplier
            ),
            "max_absolute_correction": float(
                args.max_absolute_correction
            ),
            "corrector_inputs": [
                "observed_decision_score",
                *QUALITY_COLUMNS,
                *[
                    f"{column}__missing"
                    for column in QUALITY_COLUMNS
                ],
            ],
        },
        "source_conditions": {
            condition: int(
                sum(
                    key[0] == condition
                    for key in tables
                )
            )
            for condition in CONDITION_ORDER
        },
        "internal_protocol": (
            "Common16 unseen participant + unseen sentence fold"
        ),
        "external_protocol": {
            "target_labels_used_for_fit": False,
            "target_labels_used_for_tuning": False,
            "target_features_used_for_fit": False,
            "decision_threshold": 0.0,
            "bootstrap_unit": (
                "participant_id"
            ),
            "n_bootstraps": int(
                args.n_bootstraps
            ),
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": (
                    sha256_file(path)
                    if path.is_file()
                    else None
                ),
            }
            for name, path in key_inputs.items()
        },
        "software": {
            "python": sys.version,
            "platform": (
                platform.platform()
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
            "joblib": joblib.__version__,
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

    print(
        "\n=== LAYER-2B INTERNAL SUMMARY ==="
    )
    display = internal_summary[
        [
            "condition",
            "method",
            "macro_f1_subject_mean",
            "balanced_accuracy_subject_mean",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
        ]
    ].copy()
    display["condition"] = pd.Categorical(
        display["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    display["method"] = pd.Categorical(
        display["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    print(
        display.sort_values(
            ["condition", "method"]
        ).to_string(index=False)
    )

    print(
        "\n=== LAYER-2B INTERNAL SCORE-CANONICAL VS IDENTITY TESTS ==="
    )
    print(
        test_table[
            test_table["metric"].isin(
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
                "identity_mean",
                "canonical_mean",
                "mean_difference_canonical_minus_identity",
                "canonical_wins",
                "ties",
                "canonical_losses",
                "p_value",
                "p_holm_seven_conditions_within_metric",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== LAYER-2B SCORE RECONSTRUCTION SUMMARY ==="
    )
    print(
        reconstruction_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-2B WEBQAMGAZE POOLED METRICS ==="
    )
    print(
        external_pooled[
            [
                "method",
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc_nr",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== LAYER-2B WEBQAMGAZE PARTICIPANT-MEAN METRICS ==="
    )
    print(
        external_participant_summary[
            [
                "method",
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc_nr",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== LAYER-2B WEBQAMGAZE BOOTSTRAP DIFFERENCE ==="
    )
    print(
        bootstrap_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-2B TOP SCORE-CORRECTION COEFFICIENTS ==="
    )
    print(
        coefficients.head(15).to_string(
            index=False
        )
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
