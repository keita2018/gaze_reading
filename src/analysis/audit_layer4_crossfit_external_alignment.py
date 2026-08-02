#!/usr/bin/env python
"""Layer 4 audit: participant-group cross-fitted unlabeled domain alignment.

This script tests whether the Layer-4 result survives when each evaluated
WebQAmGaze participant ID is excluded from the unlabeled target batch used to
fit domain-alignment parameters.

Protocol
--------
- Source model and Layer-2a quality canonicalizer are fitted on all ZuCo
  Common16 source trials.
- WebQAmGaze participant IDs are split into five deterministic GroupKFold
  folds.
- For each fold, target mean/CORAL adapters are fitted only on unlabeled
  WebQAmGaze trials from the other four folds.
- The fitted adapter is applied once to the held-out participant fold.
- WebQAmGaze labels are used only after all cross-fitted predictions are
  frozen, for evaluation and statistical testing.

The script reports:
- pooled and participant-mean metrics;
- participant-cluster bootstrap CIs for method differences;
- participant-cluster bootstrap absolute CIs;
- within-participant label-permutation tests for pooled ROC-AUC versus chance.

This is an audit of transductive UDA generalization, not a new hyperparameter
search.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_layer4_unlabeled_domain_alignment import (  # noqa: E402
    METHOD_ORDER as ORIGINAL_METHOD_ORDER,
    fit_target_to_source_adapter,
)
from run_zuco_quality_conditioned_canonicalization import (  # noqa: E402
    COMPACT_COLUMNS,
    fit_canonicalizer,
    load_common_metadata,
    load_condition_tables,
    load_external_compact,
)
from run_zuco_soft_aoi_confound_controls import (  # noqa: E402
    METRICS,
    decision_scores,
    make_svm,
    score_predictions,
)


METHOD_ORDER = [
    "identity_compact27",
    "quality_canonical_compact27",
    "crossfit_mean_target_to_source_compact27",
    "crossfit_coral_target_to_source_compact27",
    "crossfit_quality_mean_target_to_source_compact27",
    "crossfit_quality_coral_target_to_source_compact27",
]

COMPARISONS = [
    (
        "quality_canonical_compact27",
        "identity_compact27",
    ),
    (
        "crossfit_coral_target_to_source_compact27",
        "identity_compact27",
    ),
    (
        "crossfit_quality_coral_target_to_source_compact27",
        "quality_canonical_compact27",
    ),
    (
        "crossfit_quality_coral_target_to_source_compact27",
        "identity_compact27",
    ),
]


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                "participant_id": str(
                    participant
                ),
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
    detailed = pd.DataFrame(rows)
    summary = (
        detailed.groupby(
            "method",
            as_index=False,
        )[METRICS]
        .mean()
    )
    return detailed, summary


def make_prediction_frame(
    base: pd.DataFrame,
    method: str,
    scores: np.ndarray,
    target_fold: int,
) -> pd.DataFrame:
    result = base[
        [
            "external_trial_id",
            "participant_id",
            "task",
            "label_nr",
        ]
    ].copy()
    result["method"] = method
    result["target_fold"] = int(
        target_fold
    )
    result["prediction_nr"] = (
        np.asarray(
            scores,
            dtype=float,
        )
        >= 0.0
    ).astype(int)
    result["decision_score_nr"] = np.asarray(
        scores,
        dtype=float,
    )
    return result


def fit_source_models(
    clean: pd.DataFrame,
    tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ],
    quality_alpha: float,
    quality_clip_quantile: float,
    c_value: float,
    seed: int,
) -> tuple[
    Any,
    Any,
    Any,
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
    return (
        quality_canonicalizer,
        raw_model,
        quality_model,
        quality_clean,
    )


def crossfit_predict(
    clean: pd.DataFrame,
    quality_clean: pd.DataFrame,
    external: pd.DataFrame,
    quality_external: pd.DataFrame,
    raw_model: Any,
    quality_model: Any,
    n_splits: int,
    domain_clip_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_splits < 2:
        raise ValueError(
            "n_splits must be >= 2."
        )
    groups = external[
        "participant_id"
    ].astype(str).to_numpy()
    unique_groups = np.unique(
        groups
    )
    if len(unique_groups) < n_splits:
        raise ValueError(
            "Not enough participant IDs for GroupKFold."
        )

    splitter = GroupKFold(
        n_splits=n_splits
    )
    parts: list[pd.DataFrame] = []
    diagnostics: list[
        dict[str, Any]
    ] = []

    for fold_index, (
        adaptation_index,
        heldout_index,
    ) in enumerate(
        splitter.split(
            external,
            groups=groups,
        ),
        start=1,
    ):
        adaptation_raw = external.iloc[
            adaptation_index
        ].reset_index(drop=True)
        heldout_raw = external.iloc[
            heldout_index
        ].reset_index(drop=True)

        adaptation_quality = (
            quality_external.iloc[
                adaptation_index
            ].reset_index(drop=True)
        )
        heldout_quality = (
            quality_external.iloc[
                heldout_index
            ].reset_index(drop=True)
        )

        raw_mean = (
            fit_target_to_source_adapter(
                source_frame=clean,
                target_frame=adaptation_raw,
                method="mean",
                clip_quantile=(
                    domain_clip_quantile
                ),
            )
        )
        raw_coral = (
            fit_target_to_source_adapter(
                source_frame=clean,
                target_frame=adaptation_raw,
                method="coral",
                clip_quantile=(
                    domain_clip_quantile
                ),
            )
        )
        quality_mean = (
            fit_target_to_source_adapter(
                source_frame=quality_clean,
                target_frame=adaptation_quality,
                method="mean",
                clip_quantile=(
                    domain_clip_quantile
                ),
            )
        )
        quality_coral = (
            fit_target_to_source_adapter(
                source_frame=quality_clean,
                target_frame=adaptation_quality,
                method="coral",
                clip_quantile=(
                    domain_clip_quantile
                ),
            )
        )

        matrices = {
            "identity_compact27": (
                heldout_raw[
                    COMPACT_COLUMNS
                ]
            ),
            "quality_canonical_compact27": (
                heldout_quality[
                    COMPACT_COLUMNS
                ]
            ),
            "crossfit_mean_target_to_source_compact27": (
                raw_mean.transform(
                    heldout_raw
                )
            ),
            "crossfit_coral_target_to_source_compact27": (
                raw_coral.transform(
                    heldout_raw
                )
            ),
            "crossfit_quality_mean_target_to_source_compact27": (
                quality_mean.transform(
                    heldout_quality
                )
            ),
            "crossfit_quality_coral_target_to_source_compact27": (
                quality_coral.transform(
                    heldout_quality
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
            "crossfit_mean_target_to_source_compact27": (
                raw_model
            ),
            "crossfit_coral_target_to_source_compact27": (
                raw_model
            ),
            "crossfit_quality_mean_target_to_source_compact27": (
                quality_model
            ),
            "crossfit_quality_coral_target_to_source_compact27": (
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
            parts.append(
                make_prediction_frame(
                    base=heldout_raw,
                    method=method,
                    scores=scores,
                    target_fold=fold_index,
                )
            )

        adaptation_groups = np.unique(
            groups[
                adaptation_index
            ]
        )
        heldout_groups = np.unique(
            groups[
                heldout_index
            ]
        )
        if set(
            adaptation_groups
        ) & set(
            heldout_groups
        ):
            raise RuntimeError(
                "Participant leakage across target folds."
            )

        for name, adapter in (
            (
                "raw_mean",
                raw_mean,
            ),
            (
                "raw_coral",
                raw_coral,
            ),
            (
                "quality_mean",
                quality_mean,
            ),
            (
                "quality_coral",
                quality_coral,
            ),
        ):
            diagnostics.append(
                {
                    "target_fold": int(
                        fold_index
                    ),
                    "adapter": name,
                    "n_adaptation_trials": int(
                        len(
                            adaptation_index
                        )
                    ),
                    "n_heldout_trials": int(
                        len(
                            heldout_index
                        )
                    ),
                    "n_adaptation_participant_ids": int(
                        len(
                            adaptation_groups
                        )
                    ),
                    "n_heldout_participant_ids": int(
                        len(
                            heldout_groups
                        )
                    ),
                    **adapter.diagnostics,
                }
            )

    predictions = pd.concat(
        parts,
        ignore_index=True,
    )

    expected_rows = (
        len(external)
        * len(METHOD_ORDER)
    )
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Cross-fitted prediction rows={len(predictions)}, "
            f"expected={expected_rows}."
        )

    duplicate_key = predictions.duplicated(
        [
            "external_trial_id",
            "method",
        ]
    )
    if duplicate_key.any():
        raise RuntimeError(
            "Duplicate cross-fitted trial/method predictions."
        )
    return (
        predictions,
        pd.DataFrame(
            diagnostics
        ),
    )


def cluster_bootstrap(
    predictions: pd.DataFrame,
    n_bootstraps: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
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
    difference_rows = []
    absolute_rows = []

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
                "_bootstrap_draw"
            ] = (
                f"{draw_index}:"
                f"{participant}"
            )
            pieces.append(piece)
        bootstrap = pd.concat(
            pieces,
            ignore_index=True,
        )
        metrics = pooled_metrics(
            bootstrap
        )
        lookup = {
            row.method: row
            for row in metrics.itertuples(
                index=False
            )
        }

        for method in METHOD_ORDER:
            row = lookup[method]
            for metric in (
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc_nr",
            ):
                absolute_rows.append(
                    {
                        "bootstrap_index": int(
                            bootstrap_index
                        ),
                        "method": method,
                        "metric": metric,
                        "value": float(
                            getattr(
                                row,
                                metric,
                            )
                        ),
                    }
                )

        for candidate, reference in (
            COMPARISONS
        ):
            comparison = (
                f"{candidate}_minus_"
                f"{reference}"
            )
            for metric in (
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc_nr",
            ):
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
                difference_rows.append(
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

    absolute = pd.DataFrame(
        absolute_rows
    )
    differences = pd.DataFrame(
        difference_rows
    )

    absolute_summary_rows = []
    for (
        method,
        metric,
    ), group in absolute.groupby(
        [
            "method",
            "metric",
        ],
        sort=True,
    ):
        values = group[
            "value"
        ].to_numpy(dtype=float)
        absolute_summary_rows.append(
            {
                "method": method,
                "metric": metric,
                "n_bootstraps": int(
                    len(values)
                ),
                "bootstrap_mean": float(
                    np.mean(values)
                ),
                "ci95_lower": float(
                    np.quantile(
                        values,
                        0.025,
                    )
                ),
                "ci95_upper": float(
                    np.quantile(
                        values,
                        0.975,
                    )
                ),
            }
        )

    difference_summary_rows = []
    for (
        comparison,
        metric,
    ), group in differences.groupby(
        [
            "comparison",
            "metric",
        ],
        sort=True,
    ):
        values = group[
            "difference_candidate_minus_reference"
        ].to_numpy(dtype=float)
        difference_summary_rows.append(
            {
                "comparison": (
                    comparison
                ),
                "metric": metric,
                "n_bootstraps": int(
                    len(values)
                ),
                "bootstrap_mean_difference": float(
                    np.mean(values)
                ),
                "ci95_lower": float(
                    np.quantile(
                        values,
                        0.025,
                    )
                ),
                "ci95_upper": float(
                    np.quantile(
                        values,
                        0.975,
                    )
                ),
                "probability_candidate_gt_reference": float(
                    np.mean(
                        values > 0
                    )
                ),
            }
        )

    return (
        absolute,
        pd.DataFrame(
            absolute_summary_rows
        ),
        differences,
        pd.DataFrame(
            difference_summary_rows
        ),
    )


def within_participant_auc_permutation(
    predictions: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    base = (
        predictions[
            [
                "external_trial_id",
                "participant_id",
                "label_nr",
            ]
        ]
        .drop_duplicates(
            "external_trial_id"
        )
        .sort_values(
            "external_trial_id"
        )
        .reset_index(drop=True)
    )
    score_wide = (
        predictions.pivot(
            index="external_trial_id",
            columns="method",
            values=(
                "decision_score_nr"
            ),
        )
        .reindex(
            base[
                "external_trial_id"
            ]
        )
    )
    if score_wide.isna().any().any():
        raise RuntimeError(
            "Missing method scores in permutation table."
        )

    labels = base[
        "label_nr"
    ].to_numpy(dtype=int)
    participants = base[
        "participant_id"
    ].astype(str).to_numpy()
    participant_indices = {
        participant: np.where(
            participants == participant
        )[0]
        for participant in np.unique(
            participants
        )
    }
    observed = {
        method: float(
            roc_auc_score(
                labels,
                score_wide[
                    method
                ].to_numpy(dtype=float),
            )
        )
        for method in METHOD_ORDER
    }

    rng = np.random.default_rng(
        seed
    )
    rows = []
    greater_or_equal = {
        method: 0
        for method in METHOD_ORDER
    }
    two_sided_extreme = {
        method: 0
        for method in METHOD_ORDER
    }

    for permutation_index in range(
        n_permutations
    ):
        permuted = labels.copy()
        for indices in (
            participant_indices.values()
        ):
            permuted[indices] = rng.permutation(
                permuted[indices]
            )
        for method in METHOD_ORDER:
            auc = float(
                roc_auc_score(
                    permuted,
                    score_wide[
                        method
                    ].to_numpy(
                        dtype=float
                    ),
                )
            )
            rows.append(
                {
                    "permutation_index": int(
                        permutation_index
                    ),
                    "method": method,
                    "roc_auc": auc,
                }
            )
            if auc >= observed[method]:
                greater_or_equal[
                    method
                ] += 1
            if abs(
                auc - 0.5
            ) >= abs(
                observed[method] - 0.5
            ):
                two_sided_extreme[
                    method
                ] += 1

    summary_rows = []
    for method in METHOD_ORDER:
        summary_rows.append(
            {
                "method": method,
                "observed_roc_auc": (
                    observed[method]
                ),
                "n_permutations": int(
                    n_permutations
                ),
                "p_one_sided_auc_greater_than_chance": float(
                    (
                        greater_or_equal[
                            method
                        ]
                        + 1
                    )
                    / (
                        n_permutations
                        + 1
                    )
                ),
                "p_two_sided_auc_different_from_chance": float(
                    (
                        two_sided_extreme[
                            method
                        ]
                        + 1
                    )
                    / (
                        n_permutations
                        + 1
                    )
                ),
            }
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            summary_rows
        ),
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
            "layer4_crossfit_external_audit_v1"
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
        "--target-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--n-permutations",
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

    (
        quality_canonicalizer,
        raw_model,
        quality_model,
        quality_clean,
    ) = fit_source_models(
        clean=clean,
        tables=tables,
        quality_alpha=(
            args.quality_alpha
        ),
        quality_clip_quantile=(
            args.quality_clip_quantile
        ),
        c_value=args.c_value,
        seed=args.seed,
    )
    quality_external_features, _ = (
        quality_canonicalizer.transform(
            external
        )
    )
    quality_external = external.copy()
    quality_external[
        COMPACT_COLUMNS
    ] = quality_external_features[
        COMPACT_COLUMNS
    ]

    print(
        "=== LAYER-4 CROSSFIT EXTERNAL AUDIT INPUT ==="
    )
    print(
        f"Source trials: {len(clean)}"
    )
    print(
        f"Target trials: {len(external)}"
    )
    print(
        "Target participant IDs: "
        f"{external['participant_id'].nunique()}"
    )
    print(
        f"Target GroupKFold splits: "
        f"{args.target_folds}"
    )
    print(
        "Held-out target participant features used to fit "
        "their adapter: False"
    )
    print(
        "Target labels used before prediction: False"
    )

    (
        predictions,
        diagnostics,
    ) = crossfit_predict(
        clean=clean,
        quality_clean=quality_clean,
        external=external,
        quality_external=(
            quality_external
        ),
        raw_model=raw_model,
        quality_model=(
            quality_model
        ),
        n_splits=args.target_folds,
        domain_clip_quantile=(
            args.domain_clip_quantile
        ),
    )

    pooled = pooled_metrics(
        predictions
    )
    (
        participant_detailed,
        participant_summary,
    ) = participant_metrics(
        predictions
    )
    (
        bootstrap_absolute,
        bootstrap_absolute_summary,
        bootstrap_differences,
        bootstrap_difference_summary,
    ) = cluster_bootstrap(
        predictions=predictions,
        n_bootstraps=(
            args.n_bootstraps
        ),
        seed=args.seed,
    )
    (
        permutation_replicates,
        permutation_summary,
    ) = within_participant_auc_permutation(
        predictions=predictions,
        n_permutations=(
            args.n_permutations
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
        "predictions": (
            output_dir
            / "layer4_crossfit_predictions.csv"
        ),
        "diagnostics": (
            output_dir
            / "layer4_crossfit_alignment_diagnostics.csv"
        ),
        "pooled": (
            output_dir
            / "layer4_crossfit_pooled_metrics.csv"
        ),
        "participant_detailed": (
            output_dir
            / "layer4_crossfit_participant_metrics.csv"
        ),
        "participant_summary": (
            output_dir
            / "layer4_crossfit_participant_summary.csv"
        ),
        "bootstrap_absolute": (
            output_dir
            / "layer4_crossfit_bootstrap_absolute_replicates.csv"
        ),
        "bootstrap_absolute_summary": (
            output_dir
            / "layer4_crossfit_bootstrap_absolute_summary.csv"
        ),
        "bootstrap_differences": (
            output_dir
            / "layer4_crossfit_bootstrap_difference_replicates.csv"
        ),
        "bootstrap_difference_summary": (
            output_dir
            / "layer4_crossfit_bootstrap_difference_summary.csv"
        ),
        "permutation_replicates": (
            output_dir
            / "layer4_crossfit_auc_permutation_replicates.csv"
        ),
        "permutation_summary": (
            output_dir
            / "layer4_crossfit_auc_permutation_summary.csv"
        ),
        "artifact": (
            output_dir
            / "layer4_crossfit_source_artifact.joblib"
        ),
        "manifest": (
            output_dir
            / "layer4_crossfit_manifest.json"
        ),
    }

    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        paths["diagnostics"],
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        paths["pooled"],
        index=False,
        encoding="utf-8-sig",
    )
    participant_detailed.to_csv(
        paths[
            "participant_detailed"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    participant_summary.to_csv(
        paths[
            "participant_summary"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_absolute.to_csv(
        paths[
            "bootstrap_absolute"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_absolute_summary.to_csv(
        paths[
            "bootstrap_absolute_summary"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_differences.to_csv(
        paths[
            "bootstrap_differences"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_difference_summary.to_csv(
        paths[
            "bootstrap_difference_summary"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    permutation_replicates.to_csv(
        paths[
            "permutation_replicates"
        ],
        index=False,
        encoding="utf-8-sig",
    )
    permutation_summary.to_csv(
        paths[
            "permutation_summary"
        ],
        index=False,
        encoding="utf-8-sig",
    )

    artifact = {
        "artifact_version": "1.0",
        "created_at_utc": (
            utc_now_iso()
        ),
        "quality_canonicalizer": (
            quality_canonicalizer
        ),
        "raw_model": raw_model,
        "quality_model": (
            quality_model
        ),
        "feature_columns": (
            COMPACT_COLUMNS
        ),
        "target_fold_count": int(
            args.target_folds
        ),
        "target_group": (
            "participant_id"
        ),
        "target_labels_used_for_fit": (
            False
        ),
    }
    joblib.dump(
        artifact,
        paths["artifact"],
        compress=3,
    )

    manifest = {
        "stage": (
            "layer4_participant_group_crossfit_external_audit"
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "methods": METHOD_ORDER,
        "source_trials": int(
            len(clean)
        ),
        "target_trials": int(
            len(external)
        ),
        "target_participant_ids": int(
            external[
                "participant_id"
            ].nunique()
        ),
        "target_group_folds": int(
            args.target_folds
        ),
        "adapter_fit_scope": (
            "unlabeled target participant folds excluding "
            "the evaluated participant fold"
        ),
        "external_labels_used_for_fit": (
            False
        ),
        "external_labels_used_for_tuning": (
            False
        ),
        "n_bootstraps": int(
            args.n_bootstraps
        ),
        "n_permutations": int(
            args.n_permutations
        ),
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
        "\n=== LAYER-4 CROSSFIT POOLED METRICS ==="
    )
    print(
        pooled[
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
        "\n=== LAYER-4 CROSSFIT PARTICIPANT-MEAN METRICS ==="
    )
    print(
        participant_summary[
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
        "\n=== LAYER-4 CROSSFIT ABSOLUTE BOOTSTRAP CI ==="
    )
    print(
        bootstrap_absolute_summary[
            bootstrap_absolute_summary[
                "metric"
            ].isin(
                [
                    "macro_f1",
                    "balanced_accuracy",
                    "roc_auc",
                    "pr_auc_nr",
                ]
            )
        ].to_string(index=False)
    )

    print(
        "\n=== LAYER-4 CROSSFIT BOOTSTRAP DIFFERENCE ==="
    )
    print(
        bootstrap_difference_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-4 CROSSFIT AUC PERMUTATION TEST ==="
    )
    print(
        permutation_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-4 CROSSFIT ALIGNMENT DIAGNOSTICS ==="
    )
    print(
        diagnostics.groupby(
            "adapter",
            as_index=False,
        )[
            [
                "mean_gap_before",
                "mean_gap_after",
                "covariance_gap_before",
                "covariance_gap_after",
                "target_clip_rate",
            ]
        ]
        .mean()
        .to_string(index=False)
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
