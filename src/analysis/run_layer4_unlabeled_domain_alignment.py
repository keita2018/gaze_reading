#!/usr/bin/env python
"""Layer 4: unlabeled target-domain alignment for Compact Soft-AOI.

This experiment evaluates batch unsupervised domain adaptation without using
WebQAmGaze labels during fitting, scaling, alignment, clipping, model
selection, score orientation, or threshold selection.

Methods
-------
1. identity_compact27
2. quality_canonical_compact27
3. mean_target_to_source_compact27
4. coral_target_to_source_compact27
5. quality_mean_target_to_source_compact27
6. quality_coral_target_to_source_compact27

The domain adapters transform an unlabeled target batch into the source feature
space.  Source-only median imputation, clipping, and scaling are used before
alignment.  Missing entries are restored after inverse transformation so the
existing classifier pipeline retains its missingness indicators.

Mean alignment:
    z_t' = z_t - mean(z_t) + mean(z_s)

CORAL target-to-source alignment:
    z_t' = (z_t - mean(z_t)) C_t^{-1/2} C_s^{1/2} + mean(z_s)

Covariances are estimated with Ledoit-Wolf shrinkage.

Internal evaluation uses the same unseen-participant + unseen-sentence
dual-holdout protocol.  For each outer test batch, target features are visible
without labels, matching transductive batch UDA.  External evaluation fits the
adapter on all unlabeled WebQAmGaze features and evaluates labels only after
all predictions have been frozen.
"""

from __future__ import annotations

import argparse
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
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from run_zuco_quality_conditioned_canonicalization import (  # noqa: E402
    COMPACT_COLUMNS,
    CONDITION_ORDER,
    fit_canonicalizer,
    load_common_metadata,
    load_condition_tables,
    load_external_compact,
)
from run_zuco_soft_aoi_ablation import holm_adjust  # noqa: E402
from run_zuco_soft_aoi_confound_controls import (  # noqa: E402
    METRICS,
    decision_scores,
    make_svm,
    score_predictions,
)


METHOD_ORDER = [
    "identity_compact27",
    "quality_canonical_compact27",
    "mean_target_to_source_compact27",
    "coral_target_to_source_compact27",
    "quality_mean_target_to_source_compact27",
    "quality_coral_target_to_source_compact27",
]

PRIMARY_COMPARISONS = [
    (
        "quality_canonical_compact27",
        "identity_compact27",
    ),
    (
        "mean_target_to_source_compact27",
        "identity_compact27",
    ),
    (
        "coral_target_to_source_compact27",
        "identity_compact27",
    ),
    (
        "quality_mean_target_to_source_compact27",
        "quality_canonical_compact27",
    ),
    (
        "quality_coral_target_to_source_compact27",
        "quality_canonical_compact27",
    ),
    (
        "quality_coral_target_to_source_compact27",
        "identity_compact27",
    ),
]


@dataclass
class TargetToSourceAdapter:
    feature_columns: list[str]
    method: str
    source_imputer: SimpleImputer
    source_scaler: StandardScaler
    clip_low: np.ndarray
    clip_high: np.ndarray
    source_mean_z: np.ndarray
    target_mean_z: np.ndarray
    transform_matrix: np.ndarray
    clip_quantile: float
    diagnostics: dict[str, float]

    def transform(
        self,
        target_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        raw = target_frame[
            self.feature_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )
        raw_values = raw.to_numpy(
            dtype=float
        )
        missing = ~np.isfinite(
            raw_values
        )
        imputed = (
            self.source_imputer.transform(
                raw
            )
        )
        clipped = np.clip(
            imputed,
            self.clip_low[None, :],
            self.clip_high[None, :],
        )
        z = self.source_scaler.transform(
            clipped
        )
        centered = (
            z - self.target_mean_z[
                None,
                :
            ]
        )
        aligned_z = (
            centered
            @ self.transform_matrix
            + self.source_mean_z[
                None,
                :
            ]
        )
        aligned_raw = (
            self.source_scaler.inverse_transform(
                aligned_z
            )
        )
        aligned_raw = np.where(
            missing,
            np.nan,
            aligned_raw,
        )
        return pd.DataFrame(
            aligned_raw,
            columns=self.feature_columns,
            index=target_frame.index,
        )


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def matrix_power_psd(
    matrix: np.ndarray,
    power: float,
    floor: float = 1e-8,
) -> np.ndarray:
    symmetric = (
        matrix + matrix.T
    ) / 2.0
    values, vectors = np.linalg.eigh(
        symmetric
    )
    values = np.clip(
        values,
        floor,
        None,
    )
    powered = np.diag(
        values**power
    )
    return (
        vectors
        @ powered
        @ vectors.T
    )


def covariance_frobenius_gap(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    covariance_x = np.cov(
        x,
        rowvar=False,
        ddof=0,
    )
    covariance_y = np.cov(
        y,
        rowvar=False,
        ddof=0,
    )
    numerator = np.linalg.norm(
        covariance_x - covariance_y,
        ord="fro",
    )
    denominator = max(
        np.linalg.norm(
            covariance_x,
            ord="fro",
        ),
        1e-12,
    )
    return float(
        numerator / denominator
    )


def fit_target_to_source_adapter(
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    method: str,
    clip_quantile: float,
) -> TargetToSourceAdapter:
    if method not in {
        "mean",
        "coral",
    }:
        raise ValueError(
            f"Unsupported adapter method: {method}"
        )
    if not (
        0 <= clip_quantile < 0.5
    ):
        raise ValueError(
            "clip_quantile must be within [0, 0.5)."
        )

    source_raw = source_frame[
        COMPACT_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )
    target_raw = target_frame[
        COMPACT_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    imputer = SimpleImputer(
        strategy="median",
    )
    source_imputed = (
        imputer.fit_transform(
            source_raw
        )
    )
    target_imputed = (
        imputer.transform(
            target_raw
        )
    )

    clip_low = np.quantile(
        source_imputed,
        clip_quantile,
        axis=0,
    )
    clip_high = np.quantile(
        source_imputed,
        1.0 - clip_quantile,
        axis=0,
    )
    degenerate = (
        clip_high <= clip_low
    )
    clip_low[
        degenerate
    ] = np.min(
        source_imputed[
            :,
            degenerate,
        ],
        axis=0,
    )
    clip_high[
        degenerate
    ] = np.max(
        source_imputed[
            :,
            degenerate,
        ],
        axis=0,
    )

    source_clipped = np.clip(
        source_imputed,
        clip_low[None, :],
        clip_high[None, :],
    )
    target_clipped = np.clip(
        target_imputed,
        clip_low[None, :],
        clip_high[None, :],
    )

    scaler = StandardScaler()
    source_z = scaler.fit_transform(
        source_clipped
    )
    target_z = scaler.transform(
        target_clipped
    )

    source_mean = np.mean(
        source_z,
        axis=0,
    )
    target_mean = np.mean(
        target_z,
        axis=0,
    )

    if method == "mean":
        transform_matrix = np.eye(
            len(COMPACT_COLUMNS),
            dtype=float,
        )
    else:
        source_covariance = (
            LedoitWolf().fit(
                source_z
            ).covariance_
        )
        target_covariance = (
            LedoitWolf().fit(
                target_z
            ).covariance_
        )
        transform_matrix = (
            matrix_power_psd(
                target_covariance,
                -0.5,
            )
            @ matrix_power_psd(
                source_covariance,
                0.5,
            )
        )

    centered_target = (
        target_z - target_mean[
            None,
            :
        ]
    )
    aligned_target_z = (
        centered_target
        @ transform_matrix
        + source_mean[
            None,
            :
        ]
    )

    diagnostics = {
        "n_source": float(
            len(source_frame)
        ),
        "n_target": float(
            len(target_frame)
        ),
        "mean_gap_before": float(
            np.linalg.norm(
                target_mean
                - source_mean
            )
            / math.sqrt(
                len(COMPACT_COLUMNS)
            )
        ),
        "mean_gap_after": float(
            np.linalg.norm(
                np.mean(
                    aligned_target_z,
                    axis=0,
                )
                - source_mean
            )
            / math.sqrt(
                len(COMPACT_COLUMNS)
            )
        ),
        "covariance_gap_before": (
            covariance_frobenius_gap(
                source_z,
                target_z,
            )
        ),
        "covariance_gap_after": (
            covariance_frobenius_gap(
                source_z,
                aligned_target_z,
            )
        ),
        "target_clip_rate": float(
            np.mean(
                ~np.isclose(
                    target_clipped,
                    target_imputed,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        ),
    }

    return TargetToSourceAdapter(
        feature_columns=list(
            COMPACT_COLUMNS
        ),
        method=method,
        source_imputer=imputer,
        source_scaler=scaler,
        clip_low=clip_low,
        clip_high=clip_high,
        source_mean_z=source_mean,
        target_mean_z=target_mean,
        transform_matrix=(
            transform_matrix
        ),
        clip_quantile=float(
            clip_quantile
        ),
        diagnostics=diagnostics,
    )


def prediction_rows(
    base: pd.DataFrame,
    method: str,
    scores: np.ndarray,
    condition: str,
    repeat: int,
    held_out_subject: str,
    sentence_fold: int,
) -> list[dict[str, Any]]:
    predictions = (
        scores >= 0.0
    ).astype(int)
    rows = []
    for index, row in base.iterrows():
        rows.append(
            {
                "condition": condition,
                "corruption_repeat": int(
                    repeat
                ),
                "method": method,
                "held_out_subject": (
                    held_out_subject
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
                    row["label_nr"]
                ),
                "prediction_nr": int(
                    predictions[index]
                ),
                "decision_score_nr": float(
                    scores[index]
                ),
            }
        )
    return rows


def evaluate_internal(
    tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
    quality_alpha: float,
    quality_clip_quantile: float,
    domain_clip_quantile: float,
    c_value: float,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    clean = tables[("clean", 0)]
    subjects = sorted(
        clean["subject_id"].unique()
    )
    sentence_folds = sorted(
        clean["sentence_fold"].unique()
    )
    rows: list[
        dict[str, Any]
    ] = []
    diagnostics_rows: list[
        dict[str, Any]
    ] = []

    for subject_index, held_out in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Layer-4 outer subject "
            f"{subject_index:02d}/{len(subjects)}: "
            f"{held_out}"
        )

        for sentence_fold in sentence_folds:
            train_mask = (
                clean["subject_id"].ne(
                    held_out
                )
                & clean[
                    "sentence_fold"
                ].ne(
                    sentence_fold
                )
            ).to_numpy()
            test_mask = (
                clean["subject_id"].eq(
                    held_out
                )
                & clean[
                    "sentence_fold"
                ].eq(
                    sentence_fold
                )
            ).to_numpy()

            clean_train = clean.loc[
                train_mask
            ].reset_index(drop=True)

            quality_canonicalizer = (
                fit_canonicalizer(
                    clean_train=(
                        clean_train
                    ),
                    train_mask=(
                        train_mask
                    ),
                    tables=tables,
                    alpha=quality_alpha,
                    clip_quantile=(
                        quality_clip_quantile
                    ),
                )
            )
            quality_clean_train, _ = (
                quality_canonicalizer.transform(
                    clean_train
                )
            )

            raw_model = make_svm(
                c_value,
                seed,
            )
            raw_model.fit(
                clean_train[
                    COMPACT_COLUMNS
                ],
                clean_train[
                    "label_nr"
                ].to_numpy(dtype=int),
            )

            quality_model = make_svm(
                c_value,
                seed,
            )
            quality_model.fit(
                quality_clean_train[
                    COMPACT_COLUMNS
                ],
                clean_train[
                    "label_nr"
                ].to_numpy(dtype=int),
            )

            for (
                condition,
                repeat,
            ), table in tables.items():
                target = table.loc[
                    test_mask
                ].reset_index(drop=True)

                quality_target, _ = (
                    quality_canonicalizer.transform(
                        target
                    )
                )
                quality_target_frame = (
                    target.copy()
                )
                quality_target_frame[
                    COMPACT_COLUMNS
                ] = quality_target[
                    COMPACT_COLUMNS
                ]

                raw_mean_adapter = (
                    fit_target_to_source_adapter(
                        source_frame=(
                            clean_train
                        ),
                        target_frame=target,
                        method="mean",
                        clip_quantile=(
                            domain_clip_quantile
                        ),
                    )
                )
                raw_coral_adapter = (
                    fit_target_to_source_adapter(
                        source_frame=(
                            clean_train
                        ),
                        target_frame=target,
                        method="coral",
                        clip_quantile=(
                            domain_clip_quantile
                        ),
                    )
                )
                quality_mean_adapter = (
                    fit_target_to_source_adapter(
                        source_frame=(
                            quality_clean_train
                        ),
                        target_frame=(
                            quality_target_frame
                        ),
                        method="mean",
                        clip_quantile=(
                            domain_clip_quantile
                        ),
                    )
                )
                quality_coral_adapter = (
                    fit_target_to_source_adapter(
                        source_frame=(
                            quality_clean_train
                        ),
                        target_frame=(
                            quality_target_frame
                        ),
                        method="coral",
                        clip_quantile=(
                            domain_clip_quantile
                        ),
                    )
                )

                matrices = {
                    "identity_compact27": (
                        target[
                            COMPACT_COLUMNS
                        ]
                    ),
                    "quality_canonical_compact27": (
                        quality_target[
                            COMPACT_COLUMNS
                        ]
                    ),
                    "mean_target_to_source_compact27": (
                        raw_mean_adapter.transform(
                            target
                        )
                    ),
                    "coral_target_to_source_compact27": (
                        raw_coral_adapter.transform(
                            target
                        )
                    ),
                    "quality_mean_target_to_source_compact27": (
                        quality_mean_adapter.transform(
                            quality_target_frame
                        )
                    ),
                    "quality_coral_target_to_source_compact27": (
                        quality_coral_adapter.transform(
                            quality_target_frame
                        )
                    ),
                }
                model_lookup = {
                    "identity_compact27": (
                        raw_model
                    ),
                    "quality_canonical_compact27": (
                        quality_model
                    ),
                    "mean_target_to_source_compact27": (
                        raw_model
                    ),
                    "coral_target_to_source_compact27": (
                        raw_model
                    ),
                    "quality_mean_target_to_source_compact27": (
                        quality_model
                    ),
                    "quality_coral_target_to_source_compact27": (
                        quality_model
                    ),
                }

                for method in METHOD_ORDER:
                    scores = decision_scores(
                        model_lookup[
                            method
                        ],
                        matrices[
                            method
                        ],
                    )
                    rows.extend(
                        prediction_rows(
                            base=target,
                            method=method,
                            scores=scores,
                            condition=condition,
                            repeat=repeat,
                            held_out_subject=(
                                held_out
                            ),
                            sentence_fold=(
                                sentence_fold
                            ),
                        )
                    )

                for name, adapter in (
                    (
                        "raw_mean",
                        raw_mean_adapter,
                    ),
                    (
                        "raw_coral",
                        raw_coral_adapter,
                    ),
                    (
                        "quality_mean",
                        quality_mean_adapter,
                    ),
                    (
                        "quality_coral",
                        quality_coral_adapter,
                    ),
                ):
                    diagnostics_rows.append(
                        {
                            "condition": (
                                condition
                            ),
                            "corruption_repeat": int(
                                repeat
                            ),
                            "held_out_subject": (
                                held_out
                            ),
                            "sentence_fold": int(
                                sentence_fold
                            ),
                            "adapter": name,
                            **adapter.diagnostics,
                        }
                    )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            diagnostics_rows
        ),
    )


def aggregate_metrics(
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
                "held_out_subject": (
                    subject
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
                    ].to_numpy(dtype=float),
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
        pd.DataFrame(
            summary_rows
        ),
    )


def paired_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (
        candidate,
        reference,
    ) in PRIMARY_COMPARISONS:
        comparison = (
            f"{candidate}_minus_"
            f"{reference}"
        )
        for metric in METRICS:
            family = []
            for condition in CONDITION_ORDER:
                subset = subject_metrics[
                    subject_metrics[
                        "condition"
                    ].eq(condition)
                ]
                ref = (
                    subset[
                        subset[
                            "method"
                        ].eq(reference)
                    ]
                    .set_index(
                        "held_out_subject"
                    )[metric]
                    .sort_index()
                )
                cand = (
                    subset[
                        subset[
                            "method"
                        ].eq(candidate)
                    ]
                    .set_index(
                        "held_out_subject"
                    )[metric]
                    .sort_index()
                )
                paired = pd.concat(
                    [
                        ref.rename(
                            "reference"
                        ),
                        cand.rename(
                            "candidate"
                        ),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                difference = (
                    paired[
                        "candidate"
                    ]
                    - paired[
                        "reference"
                    ]
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
                        paired[
                            "candidate"
                        ],
                        paired[
                            "reference"
                        ],
                        alternative=(
                            "two-sided"
                        ),
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
                        "comparison": comparison,
                        "candidate": (
                            candidate
                        ),
                        "reference": (
                            reference
                        ),
                        "condition": (
                            condition
                        ),
                        "metric": metric,
                        "reference_mean": float(
                            paired[
                                "reference"
                            ].mean()
                        ),
                        "candidate_mean": float(
                            paired[
                                "candidate"
                            ].mean()
                        ),
                        "mean_difference_candidate_minus_reference": float(
                            difference.mean()
                        ),
                        "candidate_wins": int(
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
                        "candidate_losses": int(
                            np.sum(
                                difference < 0
                            )
                        ),
                        "wilcoxon_statistic": (
                            statistic
                        ),
                        "p_value": (
                            p_value
                        ),
                        "p_holm_seven_conditions_within_comparison_metric": (
                            math.nan
                        ),
                    }
                )
            adjusted = holm_adjust(
                [
                    float(
                        row["p_value"]
                    )
                    for row in family
                ]
            )
            for row, value in zip(
                family,
                adjusted,
            ):
                row[
                    "p_holm_seven_conditions_within_comparison_metric"
                ] = value
            rows.extend(family)
    return pd.DataFrame(rows)


def pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for method, group in predictions.groupby(
        "method",
        sort=False,
    ):
        rows.append(
            {
                "method": method,
                "n_trials": int(
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
                    ].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def participant_metrics(
    predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rows = []
    for (
        method,
        participant,
    ), group in predictions.groupby(
        [
            "method",
            "participant_id",
        ],
        sort=False,
    ):
        rows.append(
            {
                "method": method,
                "participant_id": (
                    participant
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
                    ].to_numpy(dtype=float),
                ),
            }
        )
    detailed = pd.DataFrame(rows)
    summary = (
        detailed.groupby(
            "method",
            as_index=False,
        )[METRICS]
        .mean()
    )
    return detailed, summary


def cluster_bootstrap(
    predictions: pd.DataFrame,
    n_bootstraps: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    participants = sorted(
        predictions[
            "participant_id"
        ].unique()
    )
    rng = np.random.default_rng(
        seed
    )
    rows = []
    for bootstrap_index in range(
        n_bootstraps
    ):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )
        pieces = []
        for draw_index, participant in enumerate(
            sampled
        ):
            piece = predictions[
                predictions[
                    "participant_id"
                ].eq(participant)
            ].copy()
            piece[
                "bootstrap_cluster"
            ] = (
                f"{draw_index}:"
                f"{participant}"
            )
            pieces.append(piece)
        bootstrap = pd.concat(
            pieces,
            ignore_index=True,
        )
        lookup = {
            row.method: row
            for row in pooled_metrics(
                bootstrap
            ).itertuples(index=False)
        }
        for candidate, reference in (
            PRIMARY_COMPARISONS
        ):
            comparison = (
                f"{candidate}_minus_"
                f"{reference}"
            )
            for metric in METRICS:
                candidate_value = float(
                    getattr(
                        lookup[
                            candidate
                        ],
                        metric,
                    )
                )
                reference_value = float(
                    getattr(
                        lookup[
                            reference
                        ],
                        metric,
                    )
                )
                rows.append(
                    {
                        "bootstrap_index": int(
                            bootstrap_index
                        ),
                        "comparison": (
                            comparison
                        ),
                        "candidate": (
                            candidate
                        ),
                        "reference": (
                            reference
                        ),
                        "metric": metric,
                        "difference_candidate_minus_reference": (
                            candidate_value
                            - reference_value
                        ),
                    }
                )

    replicates = pd.DataFrame(rows)
    summary_rows = []
    for (
        comparison,
        metric,
    ), group in replicates.groupby(
        [
            "comparison",
            "metric",
        ],
        sort=True,
    ):
        difference = group[
            "difference_candidate_minus_reference"
        ].to_numpy(dtype=float)
        summary_rows.append(
            {
                "comparison": (
                    comparison
                ),
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
                "probability_candidate_gt_reference": float(
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


def external_fit_and_predict(
    clean: pd.DataFrame,
    tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
    external: pd.DataFrame,
    quality_alpha: float,
    quality_clip_quantile: float,
    domain_clip_quantile: float,
    c_value: float,
    seed: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    all_mask = np.ones(
        len(clean),
        dtype=bool,
    )
    quality_canonicalizer = (
        fit_canonicalizer(
            clean_train=clean,
            train_mask=all_mask,
            tables=tables,
            alpha=quality_alpha,
            clip_quantile=(
                quality_clip_quantile
            ),
        )
    )
    quality_clean, _ = (
        quality_canonicalizer.transform(
            clean
        )
    )
    quality_external, _ = (
        quality_canonicalizer.transform(
            external
        )
    )
    quality_external_frame = (
        external.copy()
    )
    quality_external_frame[
        COMPACT_COLUMNS
    ] = quality_external[
        COMPACT_COLUMNS
    ]

    raw_model = make_svm(
        c_value,
        seed,
    )
    raw_model.fit(
        clean[
            COMPACT_COLUMNS
        ],
        clean[
            "label_nr"
        ].to_numpy(dtype=int),
    )
    quality_model = make_svm(
        c_value,
        seed,
    )
    quality_model.fit(
        quality_clean[
            COMPACT_COLUMNS
        ],
        clean[
            "label_nr"
        ].to_numpy(dtype=int),
    )

    raw_mean_adapter = (
        fit_target_to_source_adapter(
            source_frame=clean,
            target_frame=external,
            method="mean",
            clip_quantile=(
                domain_clip_quantile
            ),
        )
    )
    raw_coral_adapter = (
        fit_target_to_source_adapter(
            source_frame=clean,
            target_frame=external,
            method="coral",
            clip_quantile=(
                domain_clip_quantile
            ),
        )
    )
    quality_mean_adapter = (
        fit_target_to_source_adapter(
            source_frame=(
                quality_clean
            ),
            target_frame=(
                quality_external_frame
            ),
            method="mean",
            clip_quantile=(
                domain_clip_quantile
            ),
        )
    )
    quality_coral_adapter = (
        fit_target_to_source_adapter(
            source_frame=(
                quality_clean
            ),
            target_frame=(
                quality_external_frame
            ),
            method="coral",
            clip_quantile=(
                domain_clip_quantile
            ),
        )
    )

    matrices = {
        "identity_compact27": (
            external[
                COMPACT_COLUMNS
            ]
        ),
        "quality_canonical_compact27": (
            quality_external[
                COMPACT_COLUMNS
            ]
        ),
        "mean_target_to_source_compact27": (
            raw_mean_adapter.transform(
                external
            )
        ),
        "coral_target_to_source_compact27": (
            raw_coral_adapter.transform(
                external
            )
        ),
        "quality_mean_target_to_source_compact27": (
            quality_mean_adapter.transform(
                quality_external_frame
            )
        ),
        "quality_coral_target_to_source_compact27": (
            quality_coral_adapter.transform(
                quality_external_frame
            )
        ),
    }
    model_lookup = {
        "identity_compact27": (
            raw_model
        ),
        "quality_canonical_compact27": (
            quality_model
        ),
        "mean_target_to_source_compact27": (
            raw_model
        ),
        "coral_target_to_source_compact27": (
            raw_model
        ),
        "quality_mean_target_to_source_compact27": (
            quality_model
        ),
        "quality_coral_target_to_source_compact27": (
            quality_model
        ),
    }

    parts = []
    for method in METHOD_ORDER:
        scores = decision_scores(
            model_lookup[
                method
            ],
            matrices[
                method
            ],
        )
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
        parts.append(part)

    diagnostics = []
    for name, adapter in (
        (
            "raw_mean",
            raw_mean_adapter,
        ),
        (
            "raw_coral",
            raw_coral_adapter,
        ),
        (
            "quality_mean",
            quality_mean_adapter,
        ),
        (
            "quality_coral",
            quality_coral_adapter,
        ),
    ):
        diagnostics.append(
            {
                "adapter": name,
                **adapter.diagnostics,
            }
        )

    artifact = {
        "artifact_version": "1.0",
        "quality_canonicalizer": (
            quality_canonicalizer
        ),
        "raw_model": raw_model,
        "quality_model": (
            quality_model
        ),
        "raw_mean_adapter": (
            raw_mean_adapter
        ),
        "raw_coral_adapter": (
            raw_coral_adapter
        ),
        "quality_mean_adapter": (
            quality_mean_adapter
        ),
        "quality_coral_adapter": (
            quality_coral_adapter
        ),
        "feature_columns": (
            COMPACT_COLUMNS
        ),
        "decision_threshold": 0.0,
        "external_labels_used_for_fit": (
            False
        ),
    }
    return (
        artifact,
        pd.concat(
            parts,
            ignore_index=True,
        ),
        pd.DataFrame(
            diagnostics
        ),
    )


def summarize_diagnostics(
    diagnostics: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    numeric_columns = [
        "mean_gap_before",
        "mean_gap_after",
        "covariance_gap_before",
        "covariance_gap_after",
        "target_clip_rate",
    ]
    return (
        diagnostics.groupby(
            group_columns,
            as_index=False,
        )[numeric_columns]
        .mean()
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
            "layer4_unlabeled_domain_alignment_v1"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--quality-alpha",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--quality-clip-quantile",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--domain-clip-quantile",
        type=float,
        default=0.005,
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
        "=== LAYER-4 UNLABELED DOMAIN ALIGNMENT INPUT ==="
    )
    print(
        f"Common16 source trials: "
        f"{len(clean)}"
    )
    print(
        f"WebQAmGaze target trials: "
        f"{len(external)}"
    )
    print(
        f"Compact features: "
        f"{len(COMPACT_COLUMNS)}"
    )
    print(
        "Target labels used before prediction: False"
    )
    print(
        "Target features used for batch adaptation: True"
    )
    print(
        "Adapters: mean alignment and Ledoit-Wolf CORAL, "
        "target-to-source direction"
    )

    (
        internal_predictions,
        internal_diagnostics,
    ) = evaluate_internal(
        tables=tables,
        quality_alpha=(
            args.quality_alpha
        ),
        quality_clip_quantile=(
            args.quality_clip_quantile
        ),
        domain_clip_quantile=(
            args.domain_clip_quantile
        ),
        c_value=args.c_value,
        seed=args.seed,
    )
    (
        internal_repeat_metrics,
        internal_subject_metrics,
        internal_summary,
    ) = aggregate_metrics(
        internal_predictions
    )
    internal_tests = paired_tests(
        internal_subject_metrics
    )
    internal_diagnostics_summary = (
        summarize_diagnostics(
            internal_diagnostics,
            [
                "condition",
                "adapter",
            ],
        )
    )

    (
        artifact,
        external_predictions,
        external_diagnostics,
    ) = external_fit_and_predict(
        clean=clean,
        tables=tables,
        external=external,
        quality_alpha=(
            args.quality_alpha
        ),
        quality_clip_quantile=(
            args.quality_clip_quantile
        ),
        domain_clip_quantile=(
            args.domain_clip_quantile
        ),
        c_value=args.c_value,
        seed=args.seed,
    )
    external_pooled = pooled_metrics(
        external_predictions
    )
    (
        external_participant_metrics,
        external_participant_summary,
    ) = participant_metrics(
        external_predictions
    )
    (
        bootstrap_replicates,
        bootstrap_summary,
    ) = cluster_bootstrap(
        external_predictions,
        n_bootstraps=(
            args.n_bootstraps
        ),
        seed=args.seed,
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths = {
        "internal_predictions": (
            output_dir
            / "layer4_internal_predictions.csv"
        ),
        "internal_repeat_metrics": (
            output_dir
            / "layer4_internal_subject_repeat_metrics.csv"
        ),
        "internal_subject_metrics": (
            output_dir
            / "layer4_internal_subject_metrics.csv"
        ),
        "internal_summary": (
            output_dir
            / "layer4_internal_summary.csv"
        ),
        "internal_tests": (
            output_dir
            / "layer4_internal_planned_tests.csv"
        ),
        "internal_diagnostics": (
            output_dir
            / "layer4_internal_alignment_diagnostics.csv"
        ),
        "internal_diagnostics_summary": (
            output_dir
            / "layer4_internal_alignment_diagnostics_summary.csv"
        ),
        "external_predictions": (
            output_dir
            / "layer4_external_predictions.csv"
        ),
        "external_pooled": (
            output_dir
            / "layer4_external_pooled_metrics.csv"
        ),
        "external_participant_metrics": (
            output_dir
            / "layer4_external_participant_metrics.csv"
        ),
        "external_participant_summary": (
            output_dir
            / "layer4_external_participant_summary.csv"
        ),
        "external_diagnostics": (
            output_dir
            / "layer4_external_alignment_diagnostics.csv"
        ),
        "bootstrap_replicates": (
            output_dir
            / "layer4_external_bootstrap_replicates.csv"
        ),
        "bootstrap_summary": (
            output_dir
            / "layer4_external_bootstrap_summary.csv"
        ),
        "artifact": (
            output_dir
            / "layer4_unlabeled_domain_alignment_artifact.joblib"
        ),
        "manifest": (
            output_dir
            / "layer4_manifest.json"
        ),
    }

    internal_predictions.to_csv(
        paths[
            "internal_predictions"
        ],
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
    internal_tests.to_csv(
        paths["internal_tests"],
        index=False,
        encoding="utf-8-sig",
    )
    internal_diagnostics.to_csv(
        paths[
            "internal_diagnostics"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    internal_diagnostics_summary.to_csv(
        paths[
            "internal_diagnostics_summary"
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
    external_diagnostics.to_csv(
        paths[
            "external_diagnostics"
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
    joblib.dump(
        artifact,
        paths["artifact"],
        compress=3,
    )

    manifest = {
        "stage": (
            "layer4_batch_unsupervised_domain_alignment"
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "methods": METHOD_ORDER,
        "alignment": {
            "direction": (
                "target-to-source"
            ),
            "mean_alignment": True,
            "coral": (
                "Ledoit-Wolf covariance, symmetric eigendecomposition"
            ),
            "source_clip_quantile": float(
                args.domain_clip_quantile
            ),
            "target_labels_used_for_fit": (
                False
            ),
            "target_features_used_for_fit": (
                True
            ),
            "setting": (
                "transductive batch unsupervised domain adaptation"
            ),
        },
        "quality_layer": {
            "type": (
                "Layer-2a source-only feature canonicalization"
            ),
            "alpha": float(
                args.quality_alpha
            ),
            "clip_quantile": float(
                args.quality_clip_quantile
            ),
        },
        "classifier": {
            "type": (
                "existing median-imputation + missing-indicator + "
                "StandardScaler + LinearSVC pipeline"
            ),
            "c_value": float(
                args.c_value
            ),
            "decision_threshold": 0.0,
        },
        "external_protocol": {
            "target_labels_used_for_fit": (
                False
            ),
            "target_labels_used_for_tuning": (
                False
            ),
            "score_reversal": False,
            "threshold_tuning": False,
            "n_bootstraps": int(
                args.n_bootstraps
            ),
            "bootstrap_unit": (
                "participant_id"
            ),
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
        "\n=== LAYER-4 INTERNAL SUMMARY ==="
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
    display[
        "condition"
    ] = pd.Categorical(
        display["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    display[
        "method"
    ] = pd.Categorical(
        display["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    print(
        display.sort_values(
            [
                "condition",
                "method",
            ]
        ).to_string(index=False)
    )

    print(
        "\n=== LAYER-4 INTERNAL PLANNED TESTS ==="
    )
    print(
        internal_tests[
            internal_tests[
                "metric"
            ].isin(
                [
                    "macro_f1",
                    "roc_auc",
                    "pr_auc_nr",
                ]
            )
        ][
            [
                "comparison",
                "condition",
                "metric",
                "reference_mean",
                "candidate_mean",
                "mean_difference_candidate_minus_reference",
                "candidate_wins",
                "ties",
                "candidate_losses",
                "p_value",
                "p_holm_seven_conditions_within_comparison_metric",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== LAYER-4 INTERNAL ALIGNMENT DIAGNOSTICS ==="
    )
    print(
        internal_diagnostics_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-4 WEBQAMGAZE POOLED METRICS ==="
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
        "\n=== LAYER-4 WEBQAMGAZE PARTICIPANT-MEAN METRICS ==="
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
        "\n=== LAYER-4 WEBQAMGAZE BOOTSTRAP DIFFERENCE ==="
    )
    print(
        bootstrap_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-4 WEBQAMGAZE ALIGNMENT DIAGNOSTICS ==="
    )
    print(
        external_diagnostics.to_string(
            index=False
        )
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
