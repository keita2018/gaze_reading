#!/usr/bin/env python
"""Layer 3: source-only layout residualization for Compact Soft-AOI.

This experiment removes the component of Compact Soft-AOI features that is
predictable from source-document geometry. It evaluates:

1. identity Compact27;
2. Layer-2a quality-conditioned feature canonicalization;
3. full source-only layout residualization;
4. support-shrunk source-only layout residualization;
5. Layer-2a followed by support-shrunk layout residualization.

No WebQAmGaze label is used for fitting, scaling, clipping, support estimation,
feature selection, or threshold selection.

External layout audit showed that every WebQAmGaze layout is outside the
source 95% support region. Therefore, the main Layer-3 method uses a bounded
support-aware correction:

    x_res = x - w(c) * (g(c) - mean_source(x))

where g(c) is a multi-output Ridge prediction from layout descriptors and

    w(c) = min(1, q95_source_distance / distance(c)).

Layout inputs used here are geometry-only. Character-length descriptors are
excluded because the external AOI text field produced implausibly long
"word" lengths and is not safely comparable with ZuCo token text.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

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


LAYOUT_COLUMNS = [
    "layout_log1p_n_words",
    "layout_log1p_n_lines",
    "layout_lines_per_word",
    "layout_envelope_aspect_ratio",
    "layout_word_box_density",
    "layout_word_width_rel_mean",
    "layout_word_width_rel_std",
    "layout_word_width_rel_median",
    "layout_word_width_rel_q90",
    "layout_word_height_rel_mean",
    "layout_word_height_rel_std",
    "layout_word_height_rel_median",
    "layout_word_height_cv",
    "layout_line_word_count_mean",
    "layout_line_word_count_std",
    "layout_line_word_count_max_fraction",
    "layout_line_width_rel_mean",
    "layout_line_width_rel_std",
    "layout_line_width_rel_max",
    "layout_line_height_rel_mean",
    "layout_horizontal_gap_wordwidth_mean",
    "layout_horizontal_gap_wordwidth_std",
    "layout_vertical_gap_wordheight_mean",
    "layout_vertical_gap_wordheight_std",
]

METHOD_ORDER = [
    "identity_compact27",
    "quality_canonical_compact27",
    "layout_residual_full_compact27",
    "layout_residual_support_compact27",
    "quality_plus_layout_support_compact27",
]

PRIMARY_COMPARISONS = [
    (
        "quality_canonical_compact27",
        "identity_compact27",
    ),
    (
        "layout_residual_support_compact27",
        "identity_compact27",
    ),
    (
        "quality_plus_layout_support_compact27",
        "quality_canonical_compact27",
    ),
    (
        "quality_plus_layout_support_compact27",
        "identity_compact27",
    ),
]


@dataclass
class LayoutResidualizer:
    feature_columns: list[str]
    layout_columns: list[str]
    layout_imputer: SimpleImputer
    layout_scaler: RobustScaler
    feature_imputer: SimpleImputer
    ridge: Ridge
    feature_mean: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    source_center: np.ndarray
    support_q95: float
    alpha: float
    clip_quantile: float

    def transform(
        self,
        feature_frame: pd.DataFrame,
        layout_frame: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ]:
        layout_raw = layout_frame[
            self.layout_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )
        layout_imputed = (
            self.layout_imputer.transform(
                layout_raw
            )
        )

        layout_scaled_unclipped = (
            self.layout_scaler.transform(
                layout_imputed
            )
        )
        distance = np.sqrt(
            np.mean(
                (
                    layout_scaled_unclipped
                    - self.source_center[
                        None,
                        :
                    ]
                )
                ** 2,
                axis=1,
            )
        )

        safe_distance = np.maximum(
            distance,
            1e-12,
        )
        support_weight = np.minimum(
            1.0,
            self.support_q95
            / safe_distance,
        )

        layout_clipped = np.clip(
            layout_imputed,
            self.clip_low[None, :],
            self.clip_high[None, :],
        )
        layout_scaled = (
            self.layout_scaler.transform(
                layout_clipped
            )
        )
        predicted_layout_component = (
            self.ridge.predict(
                layout_scaled
            )
        )
        centered_component = (
            predicted_layout_component
            - self.feature_mean[
                None,
                :
            ]
        )

        raw_features = feature_frame[
            self.feature_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=float)
        finite_mask = np.isfinite(
            raw_features
        )

        full_values = (
            raw_features
            - centered_component
        )
        shrunk_values = (
            raw_features
            - support_weight[
                :,
                None,
            ]
            * centered_component
        )
        full_values = np.where(
            finite_mask,
            full_values,
            np.nan,
        )
        shrunk_values = np.where(
            finite_mask,
            shrunk_values,
            np.nan,
        )

        full = pd.DataFrame(
            full_values,
            columns=self.feature_columns,
            index=feature_frame.index,
        )
        shrunk = pd.DataFrame(
            shrunk_values,
            columns=self.feature_columns,
            index=feature_frame.index,
        )
        return (
            full,
            shrunk,
            distance,
            support_weight,
        )


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = sorted(
        set(columns) - set(
            frame.columns
        )
    )
    if missing:
        raise ValueError(
            f"{name} lacks columns: {missing}"
        )


def load_layout_tables(
    source_path: Path,
    external_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    source = pd.read_csv(
        source_path
    )
    external = pd.read_csv(
        external_path
    )
    require_columns(
        source,
        [
            "layout_id",
            *LAYOUT_COLUMNS,
        ],
        "source layout descriptors",
    )
    require_columns(
        external,
        [
            "external_trial_id",
            *LAYOUT_COLUMNS,
        ],
        "external layout descriptors",
    )
    source[
        "layout_id"
    ] = source[
        "layout_id"
    ].astype(str)
    external[
        "external_trial_id"
    ] = external[
        "external_trial_id"
    ].astype(str)

    if source[
        "layout_id"
    ].duplicated().any():
        raise ValueError(
            "Source layout IDs are not unique."
        )
    if external[
        "external_trial_id"
    ].duplicated().any():
        raise ValueError(
            "External layout trial IDs are not unique."
        )
    return source, external


def attach_source_layout(
    frame: pd.DataFrame,
    source_layout: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.merge(
        source_layout[
            [
                "layout_id",
                *LAYOUT_COLUMNS,
            ]
        ],
        left_on="sentence_key",
        right_on="layout_id",
        how="left",
        validate="many_to_one",
    )
    if result[
        LAYOUT_COLUMNS
    ].isna().all(axis=1).any():
        missing = result.loc[
            result[
                LAYOUT_COLUMNS
            ].isna().all(axis=1),
            "sentence_key",
        ].drop_duplicates().head(
            20
        ).tolist()
        raise ValueError(
            "Source trials lack layout descriptors: "
            f"{missing}"
        )
    return result.drop(
        columns=["layout_id"]
    )


def attach_external_layout(
    frame: pd.DataFrame,
    external_layout: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.merge(
        external_layout[
            [
                "external_trial_id",
                *LAYOUT_COLUMNS,
            ]
        ],
        on="external_trial_id",
        how="left",
        validate="one_to_one",
    )
    if result[
        LAYOUT_COLUMNS
    ].isna().all(axis=1).any():
        missing = result.loc[
            result[
                LAYOUT_COLUMNS
            ].isna().all(axis=1),
            "external_trial_id",
        ].head(20).tolist()
        raise ValueError(
            "External trials lack layout descriptors: "
            f"{missing}"
        )
    return result


def layout_equalizing_weights(
    sentence_keys: pd.Series,
) -> np.ndarray:
    counts = (
        sentence_keys.astype(str)
        .value_counts()
        .to_dict()
    )
    weights = np.asarray(
        [
            1.0
            / counts[str(key)]
            for key in sentence_keys
        ],
        dtype=float,
    )
    weights *= (
        len(weights)
        / np.sum(weights)
    )
    return weights


def fit_layout_residualizer(
    feature_frame: pd.DataFrame,
    layout_frame: pd.DataFrame,
    sentence_keys: pd.Series,
    alpha: float,
    clip_quantile: float,
) -> LayoutResidualizer:
    if alpha <= 0:
        raise ValueError(
            "Layout Ridge alpha must be positive."
        )
    if not (
        0 <= clip_quantile < 0.5
    ):
        raise ValueError(
            "clip_quantile must be within [0, 0.5)."
        )

    layout_raw = layout_frame[
        LAYOUT_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )
    layout_imputer = SimpleImputer(
        strategy="median",
    )
    layout_imputed = (
        layout_imputer.fit_transform(
            layout_raw
        )
    )
    clip_low = np.quantile(
        layout_imputed,
        clip_quantile,
        axis=0,
    )
    clip_high = np.quantile(
        layout_imputed,
        1.0 - clip_quantile,
        axis=0,
    )
    degenerate = (
        clip_high <= clip_low
    )
    clip_low[
        degenerate
    ] = np.min(
        layout_imputed[
            :,
            degenerate,
        ],
        axis=0,
    )
    clip_high[
        degenerate
    ] = np.max(
        layout_imputed[
            :,
            degenerate,
        ],
        axis=0,
    )

    layout_scaler = RobustScaler(
        quantile_range=(
            25.0,
            75.0,
        )
    )
    layout_scaled = (
        layout_scaler.fit_transform(
            layout_imputed
        )
    )
    source_center = np.median(
        layout_scaled,
        axis=0,
    )
    source_distance = np.sqrt(
        np.mean(
            (
                layout_scaled
                - source_center[
                    None,
                    :
                ]
            )
            ** 2,
            axis=1,
        )
    )
    support_q95 = float(
        np.quantile(
            source_distance,
            0.95,
        )
    )

    feature_raw = feature_frame[
        COMPACT_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )
    feature_imputer = SimpleImputer(
        strategy="median",
    )
    feature_target = (
        feature_imputer.fit_transform(
            feature_raw
        )
    )

    weights = (
        layout_equalizing_weights(
            sentence_keys
        )
    )
    feature_mean = np.average(
        feature_target,
        axis=0,
        weights=weights,
    )

    ridge = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
    )
    ridge.fit(
        layout_scaled,
        feature_target,
        sample_weight=weights,
    )
    return LayoutResidualizer(
        feature_columns=list(
            COMPACT_COLUMNS
        ),
        layout_columns=list(
            LAYOUT_COLUMNS
        ),
        layout_imputer=(
            layout_imputer
        ),
        layout_scaler=(
            layout_scaler
        ),
        feature_imputer=(
            feature_imputer
        ),
        ridge=ridge,
        feature_mean=(
            feature_mean
        ),
        clip_low=clip_low,
        clip_high=clip_high,
        source_center=(
            source_center
        ),
        support_q95=(
            support_q95
        ),
        alpha=float(alpha),
        clip_quantile=float(
            clip_quantile
        ),
    )


def make_prediction_rows(
    base: pd.DataFrame,
    method: str,
    scores: np.ndarray,
    condition: str,
    repeat: int,
    held_out_subject: str,
    sentence_fold: int,
    layout_distance: np.ndarray | None = None,
    support_weight: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    predictions = (
        scores >= 0.0
    ).astype(int)
    rows = []
    for index, row in base.iterrows():
        output = {
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
        if layout_distance is not None:
            output[
                "layout_distance"
            ] = float(
                layout_distance[index]
            )
        if support_weight is not None:
            output[
                "layout_support_weight"
            ] = float(
                support_weight[index]
            )
        rows.append(output)
    return rows


def evaluate_internal(
    tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
    source_layout: pd.DataFrame,
    quality_alpha: float,
    quality_clip_quantile: float,
    layout_alpha: float,
    layout_clip_quantile: float,
    c_value: float,
    seed: int,
) -> pd.DataFrame:
    clean = attach_source_layout(
        tables[("clean", 0)],
        source_layout,
    )
    subjects = sorted(
        clean["subject_id"].unique()
    )
    sentence_folds = sorted(
        clean["sentence_fold"].unique()
    )
    rows: list[
        dict[str, Any]
    ] = []

    for subject_index, held_out in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Layer-3 outer subject "
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
                    alpha=(
                        quality_alpha
                    ),
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

            raw_layout_residualizer = (
                fit_layout_residualizer(
                    feature_frame=(
                        clean_train
                    ),
                    layout_frame=(
                        clean_train
                    ),
                    sentence_keys=(
                        clean_train[
                            "sentence_key"
                        ]
                    ),
                    alpha=layout_alpha,
                    clip_quantile=(
                        layout_clip_quantile
                    ),
                )
            )
            (
                raw_layout_full_train,
                raw_layout_support_train,
                _,
                _,
            ) = raw_layout_residualizer.transform(
                clean_train,
                clean_train,
            )

            quality_train_frame = (
                clean_train.copy()
            )
            quality_train_frame[
                COMPACT_COLUMNS
            ] = quality_clean_train[
                COMPACT_COLUMNS
            ]
            quality_layout_residualizer = (
                fit_layout_residualizer(
                    feature_frame=(
                        quality_train_frame
                    ),
                    layout_frame=(
                        clean_train
                    ),
                    sentence_keys=(
                        clean_train[
                            "sentence_key"
                        ]
                    ),
                    alpha=layout_alpha,
                    clip_quantile=(
                        layout_clip_quantile
                    ),
                )
            )
            (
                _,
                quality_layout_support_train,
                _,
                _,
            ) = (
                quality_layout_residualizer.transform(
                    quality_train_frame,
                    clean_train,
                )
            )

            train_matrices = {
                "identity_compact27": (
                    clean_train[
                        COMPACT_COLUMNS
                    ]
                ),
                "quality_canonical_compact27": (
                    quality_clean_train[
                        COMPACT_COLUMNS
                    ]
                ),
                "layout_residual_full_compact27": (
                    raw_layout_full_train
                ),
                "layout_residual_support_compact27": (
                    raw_layout_support_train
                ),
                "quality_plus_layout_support_compact27": (
                    quality_layout_support_train
                ),
            }
            models = {}
            y_train = clean_train[
                "label_nr"
            ].to_numpy(dtype=int)
            for method in METHOD_ORDER:
                model = make_svm(
                    c_value,
                    seed,
                )
                model.fit(
                    train_matrices[
                        method
                    ],
                    y_train,
                )
                models[method] = model

            for (
                condition,
                repeat,
            ), table in tables.items():
                attached = attach_source_layout(
                    table.loc[
                        test_mask
                    ].reset_index(
                        drop=True
                    ),
                    source_layout,
                )

                quality_test, _ = (
                    quality_canonicalizer.transform(
                        attached
                    )
                )
                (
                    raw_full_test,
                    raw_support_test,
                    raw_distance,
                    raw_weight,
                ) = (
                    raw_layout_residualizer.transform(
                        attached,
                        attached,
                    )
                )
                quality_test_frame = (
                    attached.copy()
                )
                quality_test_frame[
                    COMPACT_COLUMNS
                ] = quality_test[
                    COMPACT_COLUMNS
                ]
                (
                    _,
                    quality_support_test,
                    quality_distance,
                    quality_weight,
                ) = (
                    quality_layout_residualizer.transform(
                        quality_test_frame,
                        attached,
                    )
                )

                test_matrices = {
                    "identity_compact27": (
                        attached[
                            COMPACT_COLUMNS
                        ]
                    ),
                    "quality_canonical_compact27": (
                        quality_test[
                            COMPACT_COLUMNS
                        ]
                    ),
                    "layout_residual_full_compact27": (
                        raw_full_test
                    ),
                    "layout_residual_support_compact27": (
                        raw_support_test
                    ),
                    "quality_plus_layout_support_compact27": (
                        quality_support_test
                    ),
                }

                for method in METHOD_ORDER:
                    scores = decision_scores(
                        models[method],
                        test_matrices[
                            method
                        ],
                    )
                    if method.startswith(
                        "layout_"
                    ):
                        distance = (
                            raw_distance
                        )
                        weight = (
                            raw_weight
                        )
                    elif method.startswith(
                        "quality_plus_layout"
                    ):
                        distance = (
                            quality_distance
                        )
                        weight = (
                            quality_weight
                        )
                    else:
                        distance = None
                        weight = None

                    rows.extend(
                        make_prediction_rows(
                            base=attached,
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
                            layout_distance=(
                                distance
                            ),
                            support_weight=(
                                weight
                            ),
                        )
                    )
    return pd.DataFrame(rows)


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
        row = {
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


def paired_internal_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (
        candidate,
        reference,
    ) in PRIMARY_COMPARISONS:
        comparison_name = (
            f"{candidate}_minus_{reference}"
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
                        "comparison": (
                            comparison_name
                        ),
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


def external_predictions_and_artifacts(
    clean: pd.DataFrame,
    tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
    external: pd.DataFrame,
    source_layout: pd.DataFrame,
    external_layout: pd.DataFrame,
    quality_alpha: float,
    quality_clip_quantile: float,
    layout_alpha: float,
    layout_clip_quantile: float,
    c_value: float,
    seed: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    clean_attached = (
        attach_source_layout(
            clean,
            source_layout,
        )
    )
    external_attached = (
        attach_external_layout(
            external,
            external_layout,
        )
    )
    all_mask = np.ones(
        len(clean_attached),
        dtype=bool,
    )
    quality_canonicalizer = (
        fit_canonicalizer(
            clean_train=(
                clean_attached
            ),
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
            clean_attached
        )
    )

    raw_layout_residualizer = (
        fit_layout_residualizer(
            feature_frame=(
                clean_attached
            ),
            layout_frame=(
                clean_attached
            ),
            sentence_keys=(
                clean_attached[
                    "sentence_key"
                ]
            ),
            alpha=layout_alpha,
            clip_quantile=(
                layout_clip_quantile
            ),
        )
    )
    (
        raw_full_train,
        raw_support_train,
        _,
        _,
    ) = raw_layout_residualizer.transform(
        clean_attached,
        clean_attached,
    )

    quality_train_frame = (
        clean_attached.copy()
    )
    quality_train_frame[
        COMPACT_COLUMNS
    ] = quality_clean[
        COMPACT_COLUMNS
    ]
    quality_layout_residualizer = (
        fit_layout_residualizer(
            feature_frame=(
                quality_train_frame
            ),
            layout_frame=(
                clean_attached
            ),
            sentence_keys=(
                clean_attached[
                    "sentence_key"
                ]
            ),
            alpha=layout_alpha,
            clip_quantile=(
                layout_clip_quantile
            ),
        )
    )
    (
        _,
        quality_support_train,
        _,
        _,
    ) = (
        quality_layout_residualizer.transform(
            quality_train_frame,
            clean_attached,
        )
    )

    quality_external, _ = (
        quality_canonicalizer.transform(
            external_attached
        )
    )
    (
        raw_full_external,
        raw_support_external,
        raw_distance,
        raw_weight,
    ) = raw_layout_residualizer.transform(
        external_attached,
        external_attached,
    )
    quality_external_frame = (
        external_attached.copy()
    )
    quality_external_frame[
        COMPACT_COLUMNS
    ] = quality_external[
        COMPACT_COLUMNS
    ]
    (
        _,
        quality_support_external,
        quality_distance,
        quality_weight,
    ) = (
        quality_layout_residualizer.transform(
            quality_external_frame,
            external_attached,
        )
    )

    train_matrices = {
        "identity_compact27": (
            clean_attached[
                COMPACT_COLUMNS
            ]
        ),
        "quality_canonical_compact27": (
            quality_clean[
                COMPACT_COLUMNS
            ]
        ),
        "layout_residual_full_compact27": (
            raw_full_train
        ),
        "layout_residual_support_compact27": (
            raw_support_train
        ),
        "quality_plus_layout_support_compact27": (
            quality_support_train
        ),
    }
    test_matrices = {
        "identity_compact27": (
            external_attached[
                COMPACT_COLUMNS
            ]
        ),
        "quality_canonical_compact27": (
            quality_external[
                COMPACT_COLUMNS
            ]
        ),
        "layout_residual_full_compact27": (
            raw_full_external
        ),
        "layout_residual_support_compact27": (
            raw_support_external
        ),
        "quality_plus_layout_support_compact27": (
            quality_support_external
        ),
    }

    models = {}
    y_train = clean_attached[
        "label_nr"
    ].to_numpy(dtype=int)
    parts = []
    for method in METHOD_ORDER:
        model = make_svm(
            c_value,
            seed,
        )
        model.fit(
            train_matrices[
                method
            ],
            y_train,
        )
        models[method] = model
        scores = decision_scores(
            model,
            test_matrices[
                method
            ],
        )
        part = external_attached[
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
        if method.startswith(
            "layout_"
        ):
            part[
                "layout_distance"
            ] = raw_distance
            part[
                "layout_support_weight"
            ] = raw_weight
        elif method.startswith(
            "quality_plus_layout"
        ):
            part[
                "layout_distance"
            ] = quality_distance
            part[
                "layout_support_weight"
            ] = quality_weight
        else:
            part[
                "layout_distance"
            ] = np.nan
            part[
                "layout_support_weight"
            ] = np.nan
        parts.append(part)

    artifact = {
        "artifact_version": "1.0",
        "quality_canonicalizer": (
            quality_canonicalizer
        ),
        "raw_layout_residualizer": (
            raw_layout_residualizer
        ),
        "quality_layout_residualizer": (
            quality_layout_residualizer
        ),
        "models": models,
        "feature_columns": (
            COMPACT_COLUMNS
        ),
        "layout_columns": (
            LAYOUT_COLUMNS
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
    )


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
        metric_lookup = {
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
                        metric_lookup[
                            candidate
                        ],
                        metric,
                    )
                )
                reference_value = float(
                    getattr(
                        metric_lookup[
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


def support_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    subset = predictions[
        predictions[
            "layout_support_weight"
        ].notna()
    ]
    return (
        subset.groupby(
            "method",
            as_index=False,
        )
        .agg(
            n_trials=(
                "external_trial_id",
                "size",
            ),
            layout_distance_median=(
                "layout_distance",
                "median",
            ),
            layout_distance_q90=(
                "layout_distance",
                lambda values: float(
                    values.quantile(
                        0.90
                    )
                ),
            ),
            support_weight_mean=(
                "layout_support_weight",
                "mean",
            ),
            support_weight_median=(
                "layout_support_weight",
                "median",
            ),
            support_weight_q10=(
                "layout_support_weight",
                lambda values: float(
                    values.quantile(
                        0.10
                    )
                ),
            ),
        )
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
        "--source-layout",
        default=(
            "outputs/layer3_layout_audit/"
            "source_layout_descriptors.csv"
        ),
    )
    parser.add_argument(
        "--external-layout",
        default=(
            "outputs/layer3_layout_audit/"
            "external_layout_descriptors.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "layer3_layout_residualization_v1"
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
        "--layout-alpha",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--layout-clip-quantile",
        type=float,
        default=0.01,
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
    source_layout, external_layout = (
        load_layout_tables(
            Path(args.source_layout),
            Path(
                args.external_layout
            ),
        )
    )
    external = load_external_compact(
        Path(args.external_compact)
    )

    print(
        "=== LAYER-3 LAYOUT RESIDUALIZATION INPUT ==="
    )
    print(
        f"Common16 trials: "
        f"{len(tables[('clean', 0)])}"
    )
    print(
        f"Source layouts: "
        f"{len(source_layout)}"
    )
    print(
        f"External trials/layout rows: "
        f"{len(external_layout)}"
    )
    print(
        f"Geometry layout descriptors: "
        f"{len(LAYOUT_COLUMNS)}"
    )
    print(
        "Character-length descriptors used: False"
    )
    print(
        "External labels used before prediction: False"
    )
    print(
        f"Layout Ridge alpha: "
        f"{args.layout_alpha}"
    )
    print(
        "Primary correction: "
        "source-support-shrunk residualization"
    )

    internal_predictions = (
        evaluate_internal(
            tables=tables,
            source_layout=source_layout,
            quality_alpha=(
                args.quality_alpha
            ),
            quality_clip_quantile=(
                args.quality_clip_quantile
            ),
            layout_alpha=(
                args.layout_alpha
            ),
            layout_clip_quantile=(
                args.layout_clip_quantile
            ),
            c_value=args.c_value,
            seed=args.seed,
        )
    )
    (
        internal_repeat_metrics,
        internal_subject_metrics,
        internal_summary,
    ) = aggregate_metrics(
        internal_predictions
    )
    internal_tests = (
        paired_internal_tests(
            internal_subject_metrics
        )
    )

    artifact, external_predictions = (
        external_predictions_and_artifacts(
            clean=tables[
                ("clean", 0)
            ],
            tables=tables,
            external=external,
            source_layout=source_layout,
            external_layout=(
                external_layout
            ),
            quality_alpha=(
                args.quality_alpha
            ),
            quality_clip_quantile=(
                args.quality_clip_quantile
            ),
            layout_alpha=(
                args.layout_alpha
            ),
            layout_clip_quantile=(
                args.layout_clip_quantile
            ),
            c_value=args.c_value,
            seed=args.seed,
        )
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
    external_support = (
        support_summary(
            external_predictions
        )
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
            / "layer3_internal_predictions.csv"
        ),
        "internal_repeat_metrics": (
            output_dir
            / "layer3_internal_subject_repeat_metrics.csv"
        ),
        "internal_subject_metrics": (
            output_dir
            / "layer3_internal_subject_metrics.csv"
        ),
        "internal_summary": (
            output_dir
            / "layer3_internal_summary.csv"
        ),
        "internal_tests": (
            output_dir
            / "layer3_internal_planned_tests.csv"
        ),
        "external_predictions": (
            output_dir
            / "layer3_external_predictions.csv"
        ),
        "external_pooled": (
            output_dir
            / "layer3_external_pooled_metrics.csv"
        ),
        "external_participant_metrics": (
            output_dir
            / "layer3_external_participant_metrics.csv"
        ),
        "external_participant_summary": (
            output_dir
            / "layer3_external_participant_summary.csv"
        ),
        "bootstrap_replicates": (
            output_dir
            / "layer3_external_bootstrap_replicates.csv"
        ),
        "bootstrap_summary": (
            output_dir
            / "layer3_external_bootstrap_summary.csv"
        ),
        "external_support": (
            output_dir
            / "layer3_external_support_summary.csv"
        ),
        "artifact": (
            output_dir
            / "layer3_source_only_artifact.joblib"
        ),
        "manifest": (
            output_dir
            / "layer3_manifest.json"
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
    external_support.to_csv(
        paths["external_support"],
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
            "layer3_source_only_layout_residualization"
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "methods": METHOD_ORDER,
        "layout_columns": (
            LAYOUT_COLUMNS
        ),
        "character_length_descriptors_used": (
            False
        ),
        "quality_layer": {
            "type": (
                "Layer-2a feature canonicalization"
            ),
            "alpha": float(
                args.quality_alpha
            ),
            "clip_quantile": float(
                args.quality_clip_quantile
            ),
        },
        "layout_layer": {
            "type": (
                "multi-output Ridge prediction "
                "plus support-aware residualization"
            ),
            "alpha": float(
                args.layout_alpha
            ),
            "clip_quantile": float(
                args.layout_clip_quantile
            ),
            "source_layout_equalizing_weights": (
                True
            ),
            "support_weight": (
                "min(1, source_q95_distance / observed_distance)"
            ),
        },
        "external_protocol": {
            "target_labels_used_for_fit": (
                False
            ),
            "target_labels_used_for_tuning": (
                False
            ),
            "target_layout_used_for_transform": (
                True
            ),
            "target_layout_labels_used": (
                False
            ),
            "decision_threshold": 0.0,
            "bootstrap_unit": (
                "participant_id"
            ),
            "n_bootstraps": int(
                args.n_bootstraps
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
        "\n=== LAYER-3 INTERNAL SUMMARY ==="
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
        "\n=== LAYER-3 INTERNAL PLANNED TESTS ==="
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
        "\n=== LAYER-3 WEBQAMGAZE POOLED METRICS ==="
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
        "\n=== LAYER-3 WEBQAMGAZE PARTICIPANT-MEAN METRICS ==="
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
        "\n=== LAYER-3 WEBQAMGAZE BOOTSTRAP DIFFERENCE ==="
    )
    print(
        bootstrap_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-3 EXTERNAL SUPPORT WEIGHTS ==="
    )
    print(
        external_support.to_string(
            index=False
        )
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
