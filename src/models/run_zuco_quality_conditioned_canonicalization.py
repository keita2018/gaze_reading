#!/usr/bin/env python
"""Layer 2: source-only quality-conditioned canonicalization for Compact Soft-AOI.

Purpose
-------
Learn a source-only transformation that maps degraded Compact Soft-AOI features
toward their paired clean ZuCo features.  The transformation uses only the 12
membership/quality features (plus their missingness flags) to predict a
27-dimensional residual correction.

The experiment has two parts.

1. Strict Common16 internal evaluation
   - unknown participant + unknown sentence fold;
   - canonicalizer fitted only on outer-training ZuCo trials;
   - classifier fitted only on outer-training clean labels;
   - test labels never affect canonicalizer or classifier fitting.

2. Frozen WebQAmGaze application
   - canonicalizer and classifier fitted on all Common16 source trials;
   - no WebQAmGaze label is used for fitting, tuning, scaling, clipping, or
     threshold selection;
   - external labels are used only after predictions for evaluation.

Source degradation pairs
------------------------
clean identity pairs
30 Hz -> 10 Hz
50 px coordinate noise, five repeats
40% random missingness, five repeats
40% burst missingness, five repeats
50 px + 40% random missingness, five repeats
50 px + 40% burst missingness, five repeats

Canonicalizer
-------------
Let x_e be the observed Compact vector under environment e and x_clean its
paired clean vector.  A ridge model predicts the standardized residual

    r = (x_clean - x_e) / source_clean_scale

from the observed 12 quality features and 12 quality-missingness indicators.
The corrected vector is

    x_hat = x_e + r_hat * source_clean_scale.

All fitting statistics are source-only.  Corrected values are clipped to the
0.5--99.5 percentile range of the relevant source clean training split to
avoid uncontrolled extrapolation on a new sensor.

This is intentionally a compact, interpretable Layer-2 baseline.  It does not
perform layout residualization (Layer 3) or target-domain alignment (Layer 4).
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
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from extract_zuco_soft_aoi_features import (  # noqa: E402
    MEMBERSHIP_QUALITY_FEATURES,
)
from run_zuco_soft_aoi_ablation import (  # noqa: E402
    DISTRIBUTION_BASE,
    holm_adjust,
    load_sentence_folds,
)
from run_zuco_soft_aoi_combined_degradation_30hz import (  # noqa: E402
    soft_cache_path as combined_soft_cache_path,
)
from run_zuco_soft_aoi_confound_controls import (  # noqa: E402
    METRICS,
    decision_scores,
    make_svm,
    score_predictions,
)
from run_zuco_soft_aoi_coordinate_noise_30hz import (  # noqa: E402
    soft_noise_cache_path,
)
from run_zuco_soft_aoi_missingness_30hz import (  # noqa: E402
    soft_cache_path as missingness_soft_cache_path,
)


EXPECTED_COMMON_SUBJECTS = 16
EXPECTED_COMMON_TRIALS = 11324
EXPECTED_SENTENCE_FOLDS = 5

COMPACT_COLUMNS = list(DISTRIBUTION_BASE) + list(
    MEMBERSHIP_QUALITY_FEATURES
)
QUALITY_COLUMNS = list(MEMBERSHIP_QUALITY_FEATURES)

if len(COMPACT_COLUMNS) != 27:
    raise RuntimeError(
        f"Compact feature count={len(COMPACT_COLUMNS)}, expected=27."
    )
if len(QUALITY_COLUMNS) != 12:
    raise RuntimeError(
        f"Quality feature count={len(QUALITY_COLUMNS)}, expected=12."
    )

METHOD_ORDER = [
    "identity_compact27",
    "quality_canonical_compact27",
]
CONDITION_ORDER = [
    "clean",
    "rate10",
    "noise50",
    "random40",
    "burst40",
    "high_random",
    "high_burst",
]
STOCHASTIC_CONDITIONS = {
    "noise50",
    "random40",
    "burst40",
    "high_random",
    "high_burst",
}


@dataclass
class Canonicalizer:
    feature_columns: list[str]
    quality_columns: list[str]
    feature_imputer: SimpleImputer
    quality_scaler: StandardScaler
    ridge: Ridge
    clean_scale: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    alpha: float
    clip_quantile: float

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        missing = sorted(
            set(self.feature_columns) - set(frame.columns)
        )
        if missing:
            raise ValueError(
                f"Canonicalizer input lacks columns: {missing}"
            )

        raw = frame[self.feature_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        raw_values = raw.to_numpy(dtype=float)
        missing_flags = ~np.isfinite(raw_values)

        imputed = self.feature_imputer.transform(raw)
        quality_indices = [
            self.feature_columns.index(column)
            for column in self.quality_columns
        ]
        quality_values = imputed[:, quality_indices]
        quality_missing = missing_flags[:, quality_indices].astype(float)
        quality_input = np.hstack(
            [quality_values, quality_missing]
        )
        quality_scaled = self.quality_scaler.transform(
            quality_input
        )
        residual_scaled = self.ridge.predict(
            quality_scaled
        )
        corrected = (
            imputed
            + residual_scaled
            * self.clean_scale[None, :]
        )
        clipped = np.clip(
            corrected,
            self.clip_low[None, :],
            self.clip_high[None, :],
        )
        clip_mask = ~np.isclose(
            clipped,
            corrected,
            rtol=0.0,
            atol=1e-12,
        )
        result = pd.DataFrame(
            clipped,
            columns=self.feature_columns,
            index=frame.index,
        )
        return result, clip_mask


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


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def normalize_feature_table(
    frame: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    required = {
        "trial_id",
        "subject_id",
        "task",
        *COMPACT_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{name} lacks columns: {missing}"
        )

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = (
        result["task"].astype(str).str.upper()
    )
    if result["trial_id"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate trial IDs."
        )
    numeric = result[COMPACT_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(
            f"{name} contains infinite feature values."
        )
    result[COMPACT_COLUMNS] = numeric
    return result


def load_common_metadata(
    common_map_path: Path,
    sentence_folds_path: Path,
) -> pd.DataFrame:
    common = pd.read_csv(
        require_file(common_map_path)
    )
    required = {
        "trial_id",
        "subject_id",
        "task",
        "task_sentence_index",
    }
    missing = sorted(required - set(common.columns))
    if missing:
        raise ValueError(
            f"Common16 map lacks columns: {missing}"
        )

    common = common.copy()
    common["trial_id"] = common["trial_id"].astype(str)
    common["subject_id"] = (
        common["subject_id"].astype(str).str.upper()
    )
    common["task"] = common["task"].astype(str).str.upper()
    common["task_sentence_index"] = pd.to_numeric(
        common["task_sentence_index"],
        errors="raise",
    ).astype(int)
    common["sentence_key"] = common.apply(
        lambda row: (
            f"{row['task']}:"
            f"{int(row['task_sentence_index']):03d}"
        ),
        axis=1,
    )
    common["label_nr"] = common["task"].map(
        {"NR": 1, "TSR": 0}
    )
    if common["label_nr"].isna().any():
        raise ValueError(
            "Common16 contains unexpected task labels."
        )
    common["label_nr"] = common["label_nr"].astype(int)

    folds = load_sentence_folds(
        require_file(sentence_folds_path)
    )
    common = common.merge(
        folds[["sentence_key", "sentence_fold"]],
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if common["sentence_fold"].isna().any():
        raise ValueError(
            "Common16 trials lack sentence-fold assignments."
        )
    common["sentence_fold"] = (
        common["sentence_fold"].astype(int)
    )

    if len(common) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Common16 trials={len(common)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if (
        common["subject_id"].nunique()
        != EXPECTED_COMMON_SUBJECTS
    ):
        raise ValueError(
            f"Common16 subjects="
            f"{common['subject_id'].nunique()}, "
            f"expected={EXPECTED_COMMON_SUBJECTS}."
        )
    if (
        common["sentence_fold"].nunique()
        != EXPECTED_SENTENCE_FOLDS
    ):
        raise ValueError(
            "Common16 sentence-fold count is invalid."
        )
    if common["trial_id"].duplicated().any():
        raise ValueError(
            "Common16 map contains duplicate trial IDs."
        )
    return common[
        [
            "trial_id",
            "subject_id",
            "task",
            "task_sentence_index",
            "sentence_key",
            "sentence_fold",
            "label_nr",
        ]
    ].sort_values("trial_id").reset_index(drop=True)


def align_to_common(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    normalized = normalize_feature_table(
        features,
        name,
    )
    result = metadata.merge(
        normalized[
            [
                "trial_id",
                "subject_id",
                "task",
                *COMPACT_COLUMNS,
            ]
        ],
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    )
    if len(result) != len(metadata):
        raise ValueError(
            f"{name} Common16 rows={len(result)}, "
            f"expected={len(metadata)}."
        )
    if result["trial_id"].tolist() != metadata["trial_id"].tolist():
        result = result.sort_values(
            "trial_id"
        ).reset_index(drop=True)
        if result["trial_id"].tolist() != metadata["trial_id"].tolist():
            raise ValueError(
                f"{name} trial ordering/alignment failed."
            )
    return result


def load_subject_cache_condition(
    metadata: pd.DataFrame,
    subjects: list[str],
    path_builder,
    name: str,
) -> pd.DataFrame:
    frames = []
    expected_ids_by_subject = {
        subject: set(
            metadata.loc[
                metadata["subject_id"].eq(subject),
                "trial_id",
            ]
        )
        for subject in subjects
    }
    for subject in subjects:
        path = require_file(
            path_builder(subject)
        )
        frame = normalize_feature_table(
            pd.read_csv(path),
            f"{name}:{subject}",
        )
        subset = frame[
            frame["trial_id"].isin(
                expected_ids_by_subject[subject]
            )
        ].copy()
        if set(subset["trial_id"]) != expected_ids_by_subject[subject]:
            raise ValueError(
                f"{name}:{subject} does not cover Common16 trials."
            )
        frames.append(subset)
    combined = pd.concat(
        frames,
        ignore_index=True,
    )
    return align_to_common(
        metadata,
        combined,
        name,
    )


def load_condition_tables(
    metadata: pd.DataFrame,
    clean_path: Path,
    rate10_path: Path,
    noise_root: Path,
    missingness_root: Path,
    combined_root: Path,
    repeats: int,
) -> dict[tuple[str, int], pd.DataFrame]:
    subjects = sorted(
        metadata["subject_id"].unique()
    )
    tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ] = {}

    clean = align_to_common(
        metadata,
        pd.read_csv(require_file(clean_path)),
        "clean30",
    )
    tables[("clean", 0)] = clean

    rate10 = align_to_common(
        metadata,
        pd.read_csv(require_file(rate10_path)),
        "rate10",
    )
    tables[("rate10", 1)] = rate10

    for repeat in range(1, repeats + 1):
        tables[("noise50", repeat)] = (
            load_subject_cache_condition(
                metadata=metadata,
                subjects=subjects,
                path_builder=lambda subject, repeat=repeat: (
                    soft_noise_cache_path(
                        noise_root,
                        subject,
                        50.0,
                        repeat,
                    )
                ),
                name=f"noise50_repeat{repeat}",
            )
        )
        for mode, condition_name in (
            ("random", "random40"),
            ("burst", "burst40"),
        ):
            tables[(condition_name, repeat)] = (
                load_subject_cache_condition(
                    metadata=metadata,
                    subjects=subjects,
                    path_builder=(
                        lambda subject,
                        mode=mode,
                        repeat=repeat: (
                            missingness_soft_cache_path(
                                missingness_root,
                                subject,
                                mode,
                                40.0,
                                repeat,
                            )
                        )
                    ),
                    name=(
                        f"{condition_name}_repeat{repeat}"
                    ),
                )
            )
        for condition_name in (
            "high_random",
            "high_burst",
        ):
            tables[(condition_name, repeat)] = (
                load_subject_cache_condition(
                    metadata=metadata,
                    subjects=subjects,
                    path_builder=(
                        lambda subject,
                        condition_name=condition_name,
                        repeat=repeat: (
                            combined_soft_cache_path(
                                combined_root,
                                subject,
                                condition_name,
                                repeat,
                            )
                        )
                    ),
                    name=(
                        f"{condition_name}_repeat{repeat}"
                    ),
                )
            )

    expected_keys = {
        ("clean", 0),
        ("rate10", 1),
        *{
            (condition, repeat)
            for condition in STOCHASTIC_CONDITIONS
            for repeat in range(1, repeats + 1)
        },
    }
    if set(tables) != expected_keys:
        raise RuntimeError(
            "Condition-table keys do not match the planned design."
        )

    reference_ids = clean["trial_id"].tolist()
    for key, table in tables.items():
        if table["trial_id"].tolist() != reference_ids:
            raise ValueError(
                f"Condition {key} is not trial aligned."
            )
    return tables


def environment_weight_vector(
    environment_labels: list[str],
) -> np.ndarray:
    series = pd.Series(environment_labels)
    counts = series.value_counts().to_dict()
    n_rows = len(series)
    n_environments = len(counts)
    weights = np.asarray(
        [
            n_rows
            / (
                n_environments
                * counts[label]
            )
            for label in environment_labels
        ],
        dtype=float,
    )
    return weights


def fit_canonicalizer(
    clean_train: pd.DataFrame,
    train_mask: np.ndarray,
    tables: dict[tuple[str, int], pd.DataFrame],
    alpha: float,
    clip_quantile: float,
) -> Canonicalizer:
    if alpha <= 0:
        raise ValueError(
            "Canonicalizer alpha must be positive."
        )
    if not (0 <= clip_quantile < 0.5):
        raise ValueError(
            "clip_quantile must be within [0, 0.5)."
        )

    feature_imputer = SimpleImputer(
        strategy="median",
    )
    clean_raw = clean_train[
        COMPACT_COLUMNS
    ].apply(pd.to_numeric, errors="coerce")
    feature_imputer.fit(clean_raw)
    clean_imputed = feature_imputer.transform(
        clean_raw
    )

    clean_scale = np.nanstd(
        clean_imputed,
        axis=0,
        ddof=0,
    )
    clean_scale = np.where(
        np.isfinite(clean_scale)
        & (clean_scale > 1e-8),
        clean_scale,
        1.0,
    )
    clip_low = np.quantile(
        clean_imputed,
        clip_quantile,
        axis=0,
    )
    clip_high = np.quantile(
        clean_imputed,
        1.0 - clip_quantile,
        axis=0,
    )
    degenerate = clip_high <= clip_low
    clip_low[degenerate] = (
        np.min(clean_imputed[:, degenerate], axis=0)
    )
    clip_high[degenerate] = (
        np.max(clean_imputed[:, degenerate], axis=0)
    )

    quality_indices = [
        COMPACT_COLUMNS.index(column)
        for column in QUALITY_COLUMNS
    ]
    input_parts = []
    target_parts = []
    environment_labels: list[str] = []

    for (condition, repeat), table in tables.items():
        degraded_raw = table.loc[
            train_mask,
            COMPACT_COLUMNS,
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )
        degraded_values = degraded_raw.to_numpy(
            dtype=float
        )
        degraded_missing = ~np.isfinite(
            degraded_values
        )
        degraded_imputed = feature_imputer.transform(
            degraded_raw
        )

        quality_values = degraded_imputed[
            :,
            quality_indices,
        ]
        quality_missing = degraded_missing[
            :,
            quality_indices,
        ].astype(float)
        quality_input = np.hstack(
            [
                quality_values,
                quality_missing,
            ]
        )
        residual_scaled = (
            clean_imputed
            - degraded_imputed
        ) / clean_scale[None, :]

        input_parts.append(quality_input)
        target_parts.append(residual_scaled)
        environment_labels.extend(
            [condition] * len(clean_imputed)
        )

    x_quality = np.vstack(input_parts)
    y_residual = np.vstack(target_parts)
    weights = environment_weight_vector(
        environment_labels
    )

    quality_scaler = StandardScaler()
    x_scaled = quality_scaler.fit_transform(
        x_quality,
        sample_weight=weights,
    )
    ridge = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
    )
    ridge.fit(
        x_scaled,
        y_residual,
        sample_weight=weights,
    )

    return Canonicalizer(
        feature_columns=list(COMPACT_COLUMNS),
        quality_columns=list(QUALITY_COLUMNS),
        feature_imputer=feature_imputer,
        quality_scaler=quality_scaler,
        ridge=ridge,
        clean_scale=clean_scale,
        clip_low=clip_low,
        clip_high=clip_high,
        alpha=float(alpha),
        clip_quantile=float(clip_quantile),
    )


def normalized_reconstruction_error(
    observed: pd.DataFrame,
    target_clean: pd.DataFrame,
    canonicalizer: Canonicalizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = canonicalizer.feature_imputer.transform(
        target_clean[COMPACT_COLUMNS]
    )
    observed_imputed = (
        canonicalizer.feature_imputer.transform(
            observed[COMPACT_COLUMNS]
        )
    )
    corrected, clip_mask = canonicalizer.transform(
        observed
    )
    corrected_values = corrected[
        COMPACT_COLUMNS
    ].to_numpy(dtype=float)
    scale = canonicalizer.clean_scale[None, :]

    raw_nrmse = np.sqrt(
        np.mean(
            (
                (observed_imputed - target)
                / scale
            )
            ** 2,
            axis=1,
        )
    )
    corrected_nrmse = np.sqrt(
        np.mean(
            (
                (corrected_values - target)
                / scale
            )
            ** 2,
            axis=1,
        )
    )
    clip_rate = np.mean(
        clip_mask,
        axis=1,
    )
    return (
        raw_nrmse,
        corrected_nrmse,
        clip_rate,
    )


def evaluate_internal(
    tables: dict[tuple[str, int], pd.DataFrame],
    alpha: float,
    clip_quantile: float,
    c_value: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = tables[("clean", 0)]
    subjects = sorted(
        clean["subject_id"].unique()
    )
    folds = sorted(
        clean["sentence_fold"].unique()
    )
    prediction_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []

    for subject_index, held_out in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Layer-2 outer subject "
            f"{subject_index:02d}/{len(subjects)}: "
            f"{held_out}"
        )

        for sentence_fold in folds:
            train_mask = (
                clean["subject_id"].ne(held_out)
                & clean["sentence_fold"].ne(
                    sentence_fold
                )
            ).to_numpy()
            test_mask = (
                clean["subject_id"].eq(held_out)
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

            canonicalizer = fit_canonicalizer(
                clean_train=clean_train,
                train_mask=train_mask,
                tables=tables,
                alpha=alpha,
                clip_quantile=clip_quantile,
            )

            identity_model = make_svm(
                c_value,
                seed,
            )
            identity_model.fit(
                clean_train[COMPACT_COLUMNS],
                clean_train["label_nr"].to_numpy(
                    dtype=int
                ),
            )

            canonical_train, _ = (
                canonicalizer.transform(
                    clean_train
                )
            )
            canonical_model = make_svm(
                c_value,
                seed,
            )
            canonical_model.fit(
                canonical_train,
                clean_train["label_nr"].to_numpy(
                    dtype=int
                ),
            )

            for (
                condition,
                repeat,
            ), table in tables.items():
                test = table.loc[
                    test_mask
                ].reset_index(drop=True)
                if test["trial_id"].tolist() != (
                    clean_test["trial_id"].tolist()
                ):
                    raise ValueError(
                        "Internal test alignment failed."
                    )

                identity_scores = decision_scores(
                    identity_model,
                    test[COMPACT_COLUMNS],
                )
                identity_predictions = (
                    identity_scores >= 0.0
                ).astype(int)

                canonical_test, clip_mask = (
                    canonicalizer.transform(test)
                )
                canonical_scores = decision_scores(
                    canonical_model,
                    canonical_test,
                )
                canonical_predictions = (
                    canonical_scores >= 0.0
                ).astype(int)

                raw_nrmse, corrected_nrmse, clip_rate = (
                    normalized_reconstruction_error(
                        observed=test,
                        target_clean=clean_test,
                        canonicalizer=canonicalizer,
                    )
                )
                y_true = test[
                    "label_nr"
                ].to_numpy(dtype=int)

                for method, predictions, scores in (
                    (
                        METHOD_ORDER[0],
                        identity_predictions,
                        identity_scores,
                    ),
                    (
                        METHOD_ORDER[1],
                        canonical_predictions,
                        canonical_scores,
                    ),
                ):
                    for row_index, row in test.iterrows():
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
                                    y_true[row_index]
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

                for row_index, row in test.iterrows():
                    reconstruction_rows.append(
                        {
                            "condition": condition,
                            "corruption_repeat": int(
                                repeat
                            ),
                            "held_out_subject": held_out,
                            "sentence_fold": int(
                                sentence_fold
                            ),
                            "trial_id": str(
                                row["trial_id"]
                            ),
                            "raw_nrmse": float(
                                raw_nrmse[row_index]
                            ),
                            "canonical_nrmse": float(
                                corrected_nrmse[
                                    row_index
                                ]
                            ),
                            "canonical_minus_raw_nrmse": float(
                                corrected_nrmse[
                                    row_index
                                ]
                                - raw_nrmse[
                                    row_index
                                ]
                            ),
                            "canonical_clip_rate": float(
                                clip_rate[row_index]
                            ),
                        }
                    )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(reconstruction_rows),
    )


def internal_metrics(
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
        condition, repeat, method, subject = values
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
                    group["label_nr"].to_numpy(
                        dtype=int
                    ),
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
            "n_input_features": 27,
        }
        for metric in METRICS:
            row[
                f"{metric}_subject_mean"
            ] = float(
                group[metric].mean()
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return (
        repeat_metrics,
        subject_metrics,
        summary,
    )


def internal_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        family = []
        for condition in CONDITION_ORDER:
            condition_frame = subject_metrics[
                subject_metrics[
                    "condition"
                ].eq(condition)
            ]
            identity = (
                condition_frame[
                    condition_frame[
                        "method"
                    ].eq(METHOD_ORDER[0])
                ]
                .set_index(
                    "held_out_subject"
                )[metric]
                .sort_index()
            )
            canonical = (
                condition_frame[
                    condition_frame[
                        "method"
                    ].eq(METHOD_ORDER[1])
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
                        np.sum(difference > 0)
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
                        np.sum(difference < 0)
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


def reconstruction_summary(
    reconstruction: pd.DataFrame,
) -> pd.DataFrame:
    subject_repeat = (
        reconstruction.groupby(
            [
                "condition",
                "corruption_repeat",
                "held_out_subject",
            ],
            as_index=False,
        )[
            [
                "raw_nrmse",
                "canonical_nrmse",
                "canonical_minus_raw_nrmse",
                "canonical_clip_rate",
            ]
        ]
        .mean()
    )
    subject = (
        subject_repeat.groupby(
            [
                "condition",
                "held_out_subject",
            ],
            as_index=False,
        )[
            [
                "raw_nrmse",
                "canonical_nrmse",
                "canonical_minus_raw_nrmse",
                "canonical_clip_rate",
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
            raw_nrmse_subject_mean=(
                "raw_nrmse",
                "mean",
            ),
            canonical_nrmse_subject_mean=(
                "canonical_nrmse",
                "mean",
            ),
            improvement_raw_minus_canonical=(
                "canonical_minus_raw_nrmse",
                lambda values: float(
                    -values.mean()
                ),
            ),
            canonical_clip_rate_subject_mean=(
                "canonical_clip_rate",
                "mean",
            ),
        )
    )


def load_external_compact(
    compact_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        require_file(compact_path)
    )
    required = {
        "external_trial_id",
        "participant_id",
        "task",
        "label_nr",
        *COMPACT_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"External Compact table lacks columns: {missing}"
        )
    result = frame.copy()
    result["external_trial_id"] = result[
        "external_trial_id"
    ].astype(str)
    result["participant_id"] = result[
        "participant_id"
    ].astype(str)
    result["task"] = result[
        "task"
    ].astype(str).str.upper()
    result["label_nr"] = pd.to_numeric(
        result["label_nr"],
        errors="raise",
    ).astype(int)
    if result[
        "external_trial_id"
    ].duplicated().any():
        raise ValueError(
            "External Compact table has duplicate trial IDs."
        )
    numeric = result[COMPACT_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if np.isinf(
        numeric.to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "External Compact table contains infinity."
        )
    result[COMPACT_COLUMNS] = numeric
    return result


def metric_table(
    predictions: pd.DataFrame,
    group_column: str | None = None,
) -> pd.DataFrame:
    rows = []
    group_keys = (
        ["method"]
        if group_column is None
        else ["method", group_column]
    )
    for values, group in predictions.groupby(
        group_keys,
        sort=True,
    ):
        if not isinstance(values, tuple):
            values = (values,)
        row = {
            "method": values[0],
            "n_predictions": int(
                len(group)
            ),
            **score_predictions(
                group["label_nr"].to_numpy(
                    dtype=int
                ),
                group[
                    "prediction_nr"
                ].to_numpy(dtype=int),
                group[
                    "decision_score_nr"
                ].to_numpy(dtype=float),
            ),
        }
        if group_column is not None:
            row[group_column] = values[1]
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_external(
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
        bootstrap_parts = []
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
            bootstrap_parts.append(part)
        bootstrap = pd.concat(
            bootstrap_parts,
            ignore_index=True,
        )
        metrics = metric_table(bootstrap)
        metric_lookup = {
            row.method: row
            for row in metrics.itertuples(
                index=False
            )
        }
        identity = metric_lookup[
            METHOD_ORDER[0]
        ]
        canonical = metric_lookup[
            METHOD_ORDER[1]
        ]
        for metric in (
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
        ):
            rows.append(
                {
                    "bootstrap_index": int(
                        bootstrap_index
                    ),
                    "metric": metric,
                    "identity": float(
                        getattr(
                            identity,
                            metric,
                        )
                    ),
                    "canonical": float(
                        getattr(
                            canonical,
                            metric,
                        )
                    ),
                    "difference_canonical_minus_identity": float(
                        getattr(
                            canonical,
                            metric,
                        )
                        - getattr(
                            identity,
                            metric,
                        )
                    ),
                }
            )

    replicates = pd.DataFrame(rows)
    summary_rows = []
    for metric, group in replicates.groupby(
        "metric",
        sort=True,
    ):
        differences = group[
            "difference_canonical_minus_identity"
        ].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "n_bootstraps": int(
                    len(group)
                ),
                "bootstrap_mean_difference": float(
                    np.mean(differences)
                ),
                "ci95_lower": float(
                    np.quantile(
                        differences,
                        0.025,
                    )
                ),
                "ci95_upper": float(
                    np.quantile(
                        differences,
                        0.975,
                    )
                ),
                "probability_canonical_gt_identity": float(
                    np.mean(
                        differences > 0
                    )
                ),
            }
        )
    return (
        replicates,
        pd.DataFrame(summary_rows),
    )


def fit_and_evaluate_external(
    clean: pd.DataFrame,
    tables: dict[tuple[str, int], pd.DataFrame],
    external: pd.DataFrame,
    alpha: float,
    clip_quantile: float,
    c_value: float,
    seed: int,
    n_bootstraps: int,
) -> tuple[
    Canonicalizer,
    Any,
    Any,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_mask = np.ones(
        len(clean),
        dtype=bool,
    )
    canonicalizer = fit_canonicalizer(
        clean_train=clean,
        train_mask=train_mask,
        tables=tables,
        alpha=alpha,
        clip_quantile=clip_quantile,
    )

    identity_model = make_svm(
        c_value,
        seed,
    )
    identity_model.fit(
        clean[COMPACT_COLUMNS],
        clean["label_nr"].to_numpy(
            dtype=int
        ),
    )

    canonical_clean, _ = (
        canonicalizer.transform(clean)
    )
    canonical_model = make_svm(
        c_value,
        seed,
    )
    canonical_model.fit(
        canonical_clean,
        clean["label_nr"].to_numpy(
            dtype=int
        ),
    )

    canonical_external, clip_mask = (
        canonicalizer.transform(external)
    )
    identity_scores = decision_scores(
        identity_model,
        external[COMPACT_COLUMNS],
    )
    canonical_scores = decision_scores(
        canonical_model,
        canonical_external,
    )

    prediction_parts = []
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
        part["decision_score_nr"] = scores
        prediction_parts.append(part)
    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    pooled = metric_table(predictions)
    subject_metrics = metric_table(
        predictions,
        group_column="participant_id",
    )
    subject_summary = (
        subject_metrics.groupby(
            "method",
            as_index=False,
        )[METRICS]
        .mean()
    )
    (
        bootstrap_replicates,
        bootstrap_summary,
    ) = cluster_bootstrap_external(
        predictions=predictions,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    transformed = external[
        [
            "external_trial_id",
            "participant_id",
            "task",
            "label_nr",
        ]
    ].copy()
    for column in COMPACT_COLUMNS:
        transformed[
            f"original__{column}"
        ] = external[column]
        transformed[
            f"canonical__{column}"
        ] = canonical_external[column]
    transformed[
        "canonical_clip_rate"
    ] = np.mean(
        clip_mask,
        axis=1,
    )

    return (
        canonicalizer,
        identity_model,
        canonical_model,
        predictions,
        pooled,
        subject_metrics,
        subject_summary,
        bootstrap_replicates,
        bootstrap_summary,
        transformed,
    )


def canonicalizer_coefficients(
    canonicalizer: Canonicalizer,
) -> pd.DataFrame:
    input_names = (
        list(QUALITY_COLUMNS)
        + [
            f"{column}__missing"
            for column in QUALITY_COLUMNS
        ]
    )
    coefficients = np.asarray(
        canonicalizer.ridge.coef_,
        dtype=float,
    )
    rows = []
    for output_index, output_name in enumerate(
        COMPACT_COLUMNS
    ):
        for input_index, input_name in enumerate(
            input_names
        ):
            rows.append(
                {
                    "output_feature": output_name,
                    "input_quality_feature": (
                        input_name
                    ),
                    "coefficient_standardized_residual": float(
                        coefficients[
                            output_index,
                            input_index,
                        ]
                    ),
                }
            )
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
            "quality_conditioned_canonicalization_v1"
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
        "--clip-quantile",
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

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )
    if args.canonicalizer_alpha <= 0:
        raise ValueError(
            "--canonicalizer-alpha must be positive."
        )
    if not (
        0 <= args.clip_quantile < 0.5
    ):
        raise ValueError(
            "--clip-quantile must be within [0, 0.5)."
        )
    if args.c_value <= 0:
        raise ValueError(
            "--c-value must be positive."
        )
    if args.n_bootstraps < 100:
        raise ValueError(
            "--n-bootstraps must be at least 100."
        )

    metadata = load_common_metadata(
        Path(args.common_map),
        Path(args.sentence_folds_file),
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

    print(
        "=== LAYER-2 QUALITY-CONDITIONED CANONICALIZATION INPUT ==="
    )
    print(
        f"Common16 subjects: "
        f"{clean['subject_id'].nunique()}"
    )
    print(f"Common16 trials: {len(clean)}")
    print(
        f"Compact features: "
        f"{len(COMPACT_COLUMNS)}"
    )
    print(
        f"Quality conditioning features: "
        f"{len(QUALITY_COLUMNS)} + "
        f"{len(QUALITY_COLUMNS)} missing flags"
    )
    print(
        f"Source condition tables: "
        f"{len(tables)}"
    )
    print(
        f"Canonicalizer alpha: "
        f"{args.canonicalizer_alpha}"
    )
    print(
        f"Source clip quantile: "
        f"{args.clip_quantile}"
    )
    print(
        "No WebQAmGaze label is used before external prediction."
    )

    (
        internal_predictions,
        reconstruction,
    ) = evaluate_internal(
        tables=tables,
        alpha=args.canonicalizer_alpha,
        clip_quantile=args.clip_quantile,
        c_value=args.c_value,
        seed=args.seed,
    )
    (
        internal_repeat_metrics,
        internal_subject_metrics,
        internal_summary,
    ) = internal_metrics(
        internal_predictions
    )
    internal_test_table = internal_tests(
        internal_subject_metrics
    )
    reconstruction_table = (
        reconstruction_summary(
            reconstruction
        )
    )

    external = load_external_compact(
        Path(args.external_compact)
    )
    (
        final_canonicalizer,
        identity_model,
        canonical_model,
        external_predictions,
        external_pooled,
        external_subject_metrics,
        external_subject_summary,
        bootstrap_replicates,
        bootstrap_summary,
        transformed_external,
    ) = fit_and_evaluate_external(
        clean=clean,
        tables=tables,
        external=external,
        alpha=args.canonicalizer_alpha,
        clip_quantile=args.clip_quantile,
        c_value=args.c_value,
        seed=args.seed,
        n_bootstraps=args.n_bootstraps,
    )
    coefficients = canonicalizer_coefficients(
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
            / "layer2_internal_predictions.csv"
        ),
        "internal_repeat_metrics": (
            output_dir
            / "layer2_internal_subject_repeat_metrics.csv"
        ),
        "internal_subject_metrics": (
            output_dir
            / "layer2_internal_subject_metrics.csv"
        ),
        "internal_summary": (
            output_dir
            / "layer2_internal_summary.csv"
        ),
        "internal_tests": (
            output_dir
            / "layer2_internal_identity_vs_canonical_tests.csv"
        ),
        "reconstruction_trials": (
            output_dir
            / "layer2_internal_reconstruction_trials.csv"
        ),
        "reconstruction_summary": (
            output_dir
            / "layer2_internal_reconstruction_summary.csv"
        ),
        "external_predictions": (
            output_dir
            / "layer2_external_predictions.csv"
        ),
        "external_pooled": (
            output_dir
            / "layer2_external_pooled_metrics.csv"
        ),
        "external_subject_metrics": (
            output_dir
            / "layer2_external_subject_metrics.csv"
        ),
        "external_subject_summary": (
            output_dir
            / "layer2_external_subject_summary.csv"
        ),
        "bootstrap_replicates": (
            output_dir
            / "layer2_external_bootstrap_replicates.csv"
        ),
        "bootstrap_summary": (
            output_dir
            / "layer2_external_bootstrap_summary.csv"
        ),
        "transformed_external": (
            output_dir
            / "layer2_external_canonicalized_features.csv"
        ),
        "coefficients": (
            output_dir
            / "layer2_canonicalizer_coefficients.csv"
        ),
        "artifact": (
            output_dir
            / "layer2_source_only_canonicalizer_and_models.joblib"
        ),
        "manifest": (
            output_dir
            / "layer2_manifest.json"
        ),
    }

    internal_predictions.to_csv(
        paths["internal_predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    internal_repeat_metrics.to_csv(
        paths["internal_repeat_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    internal_subject_metrics.to_csv(
        paths["internal_subject_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    internal_summary.to_csv(
        paths["internal_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    internal_test_table.to_csv(
        paths["internal_tests"],
        index=False,
        encoding="utf-8-sig",
    )
    reconstruction.to_csv(
        paths["reconstruction_trials"],
        index=False,
        encoding="utf-8-sig",
    )
    reconstruction_table.to_csv(
        paths["reconstruction_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    external_predictions.to_csv(
        paths["external_predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    external_pooled.to_csv(
        paths["external_pooled"],
        index=False,
        encoding="utf-8-sig",
    )
    external_subject_metrics.to_csv(
        paths["external_subject_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    external_subject_summary.to_csv(
        paths["external_subject_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_replicates.to_csv(
        paths["bootstrap_replicates"],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        paths["bootstrap_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    transformed_external.to_csv(
        paths["transformed_external"],
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
        "created_at_utc": utc_now_iso(),
        "canonicalizer": final_canonicalizer,
        "identity_model": identity_model,
        "canonical_model": canonical_model,
        "feature_columns": COMPACT_COLUMNS,
        "quality_columns": QUALITY_COLUMNS,
        "label_mapping": {
            "TSR_or_IS": 0,
            "NR": 1,
        },
        "decision_threshold": 0.0,
        "training_scope": (
            "ZuCo Common16 source-only"
        ),
        "external_labels_used_for_fit": False,
    }
    joblib.dump(
        artifact,
        paths["artifact"],
        compress=3,
    )

    input_paths = {
        "common_map": Path(
            args.common_map
        ),
        "sentence_folds": Path(
            args.sentence_folds_file
        ),
        "clean_soft": Path(
            args.clean_soft_features
        ),
        "rate10_soft": Path(
            args.rate10_soft_features
        ),
        "external_compact": Path(
            args.external_compact
        ),
    }
    manifest = {
        "stage": (
            "layer2_source_only_quality_conditioned_canonicalization"
        ),
        "created_at_utc": utc_now_iso(),
        "n_common16_subjects": int(
            clean["subject_id"].nunique()
        ),
        "n_common16_trials": int(
            len(clean)
        ),
        "n_external_participant_ids": int(
            external[
                "participant_id"
            ].nunique()
        ),
        "n_external_trials": int(
            len(external)
        ),
        "feature_columns": COMPACT_COLUMNS,
        "quality_columns": QUALITY_COLUMNS,
        "canonicalizer": {
            "type": (
                "quality-conditioned residual multi-output Ridge"
            ),
            "alpha": float(
                args.canonicalizer_alpha
            ),
            "input_dimension": int(
                len(QUALITY_COLUMNS) * 2
            ),
            "output_dimension": int(
                len(COMPACT_COLUMNS)
            ),
            "source_environment_balance": (
                "equal total sample weight per condition name"
            ),
            "clean_identity_pairs_included": True,
            "residual_target_standardization": (
                "source clean feature standard deviation"
            ),
            "source_clip_quantiles": [
                float(
                    args.clip_quantile
                ),
                float(
                    1.0
                    - args.clip_quantile
                ),
            ],
        },
        "source_conditions": {
            "clean": 1,
            "rate10": 1,
            "noise50": int(args.repeats),
            "random40": int(args.repeats),
            "burst40": int(args.repeats),
            "high_random": int(
                args.repeats
            ),
            "high_burst": int(
                args.repeats
            ),
        },
        "internal_protocol": (
            "Common16 dual holdout: unseen participant "
            "and unseen sentence fold"
        ),
        "classifier": {
            "type": (
                "median imputation + missing indicators "
                "+ StandardScaler + LinearSVC"
            ),
            "c_value": float(args.c_value),
            "decision_threshold": 0.0,
        },
        "external_protocol": {
            "fit_scope": (
                "all Common16 source trials"
            ),
            "target_labels_used_for_fit": False,
            "target_labels_used_for_tuning": False,
            "target_features_used_for_fit": False,
            "target_features_used_for_scaling": False,
            "target_features_used_for_clipping": False,
            "evaluation_only_after_prediction": True,
            "cluster_bootstrap_unit": (
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
            for name, path in input_paths.items()
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "outputs": {
            name: str(path)
            for name, path in paths.items()
            if name != "manifest"
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
        "\n=== LAYER-2 INTERNAL SUMMARY ==="
    )
    display_internal = internal_summary[
        internal_summary["condition"].isin(
            CONDITION_ORDER
        )
    ][
        [
            "condition",
            "method",
            "macro_f1_subject_mean",
            "balanced_accuracy_subject_mean",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
        ]
    ].copy()
    display_internal["condition"] = pd.Categorical(
        display_internal["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    display_internal["method"] = pd.Categorical(
        display_internal["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    print(
        display_internal.sort_values(
            ["condition", "method"]
        ).to_string(index=False)
    )

    print(
        "\n=== LAYER-2 INTERNAL CANONICAL VS IDENTITY TESTS ==="
    )
    print(
        internal_test_table[
            internal_test_table["metric"].isin(
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
        "\n=== LAYER-2 RECONSTRUCTION SUMMARY ==="
    )
    print(
        reconstruction_table.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-2 WEBQAMGAZE POOLED METRICS ==="
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
        "\n=== LAYER-2 WEBQAMGAZE PARTICIPANT-MEAN METRICS ==="
    )
    print(
        external_subject_summary[
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
        "\n=== LAYER-2 WEBQAMGAZE BOOTSTRAP DIFFERENCE ==="
    )
    print(
        bootstrap_summary.to_string(
            index=False
        )
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
