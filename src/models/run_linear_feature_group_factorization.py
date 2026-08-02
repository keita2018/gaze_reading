#!/usr/bin/env python
r"""
Exact linear score decomposition of the frozen OneStop Full41 model.

Purpose
-------
The preceding learned measurement-bias model failed source validation. This
script therefore does not learn another target correction. Instead, it uses
the exact linear structure of the frozen pipeline to decompose every decision
score into:

    intercept + content-feature contribution + quality-feature contribution

The decomposition is exact after the frozen imputer and scaler.

Methods evaluated on WebQAmGaze
-------------------------------
raw_frozen
    Original frozen Full41 score.

participant_only_oof
    Existing source-median LOO participant calibration from the previous
    analysis. This is retained as the established primary baseline.

participant_only_final_reference
    Recomputed LOO participant calibration using a source reference derived
    from the final frozen model on OneStop clean features.

quality_removed
    Replace each target trial's quality-feature score contribution by the
    source median quality contribution; no participant correction.

quality_removed_plus_participant
    Apply participant LOO calibration after replacing the target quality
    contribution. The primary diagnostic comparison is against
    participant_only_final_reference, because both use the same final-model
    source-reference construction.

No WebQAmGaze label is used to compute the decomposition, source references,
quality replacement, participant centers, or predictions. Labels are used
only after predictions are fixed.

Example
-------
python -u src\models\run_linear_feature_group_factorization.py ^
  --source-features data\processed\onestop\soft_aoi_asc344\onestop_soft_aoi_all_features.parquet ^
  --frozen-model outputs\frozen_onestop_external_validation_v1\models\soft_full_all.joblib ^
  --target-features outputs\webqamgaze_external_validation_stage3c\soft_full_features_primary.csv ^
  --target-calibrated-predictions outputs\webqamgaze_participant_self_calibration_v1\self_calibrated_trial_predictions.csv ^
  --output-dir outputs\linear_feature_group_factorization_v1 ^
  --bootstrap 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

try:
    import extract_zuco_soft_aoi_features as soft
except ImportError as exc:
    raise ImportError(
        "Could not import src/data/extract_zuco_soft_aoi_features.py."
    ) from exc


MODEL_NAME = "soft_full_all"
CONTENT_FEATURES = list(dict.fromkeys(soft.CONTENT_FEATURES))
QUALITY_FEATURES = list(
    dict.fromkeys(soft.MEMBERSHIP_QUALITY_FEATURES)
)
METHOD_ORDER = (
    "raw_frozen",
    "participant_only_oof",
    "participant_only_final_reference",
    "quality_removed",
    "quality_removed_plus_participant",
)
PRIMARY_REFERENCE = "participant_only_final_reference"
PRIMARY_COMPARISON = "quality_removed_plus_participant"

ID_ALIASES = (
    "external_trial_id",
    "trial_id",
    "primary_trial_id",
)
PARTICIPANT_ALIASES = (
    "participant_id",
    "subject_id",
    "participant",
    "subject",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument(
        "--target-calibrated-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=1e-6,
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def first_existing(
    columns: pd.Index,
    aliases: tuple[str, ...],
    purpose: str,
) -> str:
    mapping = {
        str(column).strip().lower(): str(column)
        for column in columns
    }
    for alias in aliases:
        if alias.lower() in mapping:
            return mapping[alias.lower()]
    raise ValueError(
        f"Could not identify {purpose}; aliases={aliases}, "
        f"columns={list(columns)}"
    )


def load_linear_artifact(
    path: Path,
) -> tuple[
    Any,
    Any,
    Any,
    list[str],
    np.ndarray,
    float,
    float,
]:
    artifact = joblib.load(require_file(path))
    if not isinstance(artifact, dict):
        raise TypeError("Frozen artifact must be a dict.")

    pipeline = artifact.get("pipeline")
    feature_columns = [
        str(value)
        for value in artifact.get("feature_columns", [])
    ]
    threshold = float(artifact["decision_threshold"])

    if len(feature_columns) != 41:
        raise ValueError(
            f"Expected 41 frozen features, got {len(feature_columns)}."
        )
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Frozen feature columns are not unique.")
    expected = set(CONTENT_FEATURES + QUALITY_FEATURES)
    if set(feature_columns) != expected:
        raise ValueError(
            "Frozen Full41 columns do not exactly match "
            "CONTENT_FEATURES + MEMBERSHIP_QUALITY_FEATURES."
        )
    if not math.isfinite(threshold):
        raise ValueError("Frozen threshold is not finite.")

    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected a sklearn Pipeline.")
    required_steps = {"imputer", "scaler", "model"}
    missing_steps = sorted(
        required_steps - set(pipeline.named_steps)
    )
    if missing_steps:
        raise ValueError(
            f"Frozen pipeline lacks steps: {missing_steps}"
        )

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    classes = np.asarray(model.classes_)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(
            f"Expected model classes {{0,1}}, got {classes.tolist()}."
        )

    coefficient = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if coefficient.shape != (1, len(feature_columns)):
        raise ValueError(
            f"Unexpected coefficient shape {coefficient.shape}."
        )
    if intercept.size != 1:
        raise ValueError(
            f"Unexpected intercept shape {intercept.shape}."
        )

    sign = 1.0 if int(classes[1]) == 1 else -1.0
    coefficient = sign * coefficient.reshape(-1)
    intercept_value = sign * float(intercept.reshape(-1)[0])

    return (
        pipeline,
        imputer,
        scaler,
        feature_columns,
        coefficient,
        intercept_value,
        threshold,
    )


def exact_decomposition(
    frame: pd.DataFrame,
    imputer: Any,
    scaler: Any,
    feature_columns: list[str],
    coefficient: np.ndarray,
    intercept: float,
) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature table lacks columns: {missing}")

    numeric = frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    imputed = imputer.transform(numeric)
    transformed = np.asarray(
        scaler.transform(imputed),
        dtype=float,
    )
    contributions = transformed * coefficient.reshape(1, -1)

    index = {
        feature: position
        for position, feature in enumerate(feature_columns)
    }
    content_indices = [
        index[feature] for feature in CONTENT_FEATURES
    ]
    quality_indices = [
        index[feature] for feature in QUALITY_FEATURES
    ]

    content_contribution = contributions[
        :, content_indices
    ].sum(axis=1)
    quality_contribution = contributions[
        :, quality_indices
    ].sum(axis=1)
    reconstructed = (
        intercept
        + content_contribution
        + quality_contribution
    )

    return pd.DataFrame(
        {
            "intercept_component": intercept,
            "content_contribution": content_contribution,
            "quality_contribution": quality_contribution,
            "reconstructed_full_score": reconstructed,
        }
    )


def source_references(
    source: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {
        "participant_id",
        "onestop_trial_id",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(
            f"Source feature table lacks columns: {missing}"
        )

    work = source[
        ["onestop_trial_id", "participant_id"]
    ].copy()
    work["participant_id"] = (
        work["participant_id"].astype(str).str.strip()
    )
    work = pd.concat(
        [
            work.reset_index(drop=True),
            decomposition.reset_index(drop=True),
        ],
        axis=1,
    )

    participant = (
        work.groupby(
            "participant_id",
            observed=True,
        )
        .agg(
            n_trials=("onestop_trial_id", "size"),
            full_score_median=(
                "reconstructed_full_score",
                "median",
            ),
            quality_contribution_median=(
                "quality_contribution",
                "median",
            ),
        )
        .reset_index()
    )

    quality_reference = float(
        participant[
            "quality_contribution_median"
        ].median()
    )
    work["quality_removed_score"] = (
        work["reconstructed_full_score"]
        - work["quality_contribution"]
        + quality_reference
    )
    quality_removed_participant = (
        work.groupby(
            "participant_id",
            observed=True,
        )["quality_removed_score"]
        .median()
    )

    references = {
        "full_final_reference": float(
            participant["full_score_median"].median()
        ),
        "quality_contribution_reference": quality_reference,
        "quality_removed_final_reference": float(
            quality_removed_participant.median()
        ),
    }
    participant["quality_removed_score_median"] = (
        participant["participant_id"].map(
            quality_removed_participant
        )
    )
    return participant, references


def normalize_target_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    trial_column = first_existing(
        frame.columns,
        ID_ALIASES,
        "target trial ID",
    )
    participant_column = first_existing(
        frame.columns,
        PARTICIPANT_ALIASES,
        "target participant ID",
    )

    missing_features = sorted(
        set(feature_columns) - set(frame.columns)
    )
    if missing_features:
        raise ValueError(
            "Target full feature table lacks frozen features: "
            f"{missing_features}"
        )

    result = frame[
        [trial_column, participant_column, *feature_columns]
    ].copy()
    result["external_trial_id"] = (
        result[trial_column].astype(str).str.strip()
    )
    result["participant_id"] = (
        result[participant_column].astype(str).str.strip()
    )

    # Keep only identifiers and the frozen model features. This prevents
    # metadata columns such as label_nr/task from colliding with the
    # independently loaded evaluation metadata during the later merge.
    result = result[
        ["external_trial_id", "participant_id", *feature_columns]
    ].copy()

    if result["external_trial_id"].duplicated().any():
        raise ValueError(
            "Target full feature table has duplicate trial IDs."
        )
    return result


def target_metadata(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "external_trial_id",
        "participant_id",
        "label_nr",
        "model_name",
        "calibration_method",
        "raw_score",
        "calibrated_score",
        "frozen_threshold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Calibrated target table lacks columns: {missing}"
        )

    result = frame[
        frame["model_name"].astype(str).eq(MODEL_NAME)
    ].copy()
    raw = result[
        result["calibration_method"].astype(str).eq(
            "raw_frozen"
        )
    ].copy()
    participant_oof = result[
        result["calibration_method"].astype(str).eq(
            "source_median_loo"
        )
    ].copy()

    if raw["external_trial_id"].duplicated().any():
        raise ValueError("Duplicate raw target trial rows.")
    if participant_oof[
        "external_trial_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate participant-calibrated trial rows."
        )

    raw_columns = [
        "external_trial_id",
        "participant_id",
        "label_nr",
        "raw_score",
        "frozen_threshold",
    ]
    if "task" in raw.columns:
        raw_columns.append("task")
    raw = raw[raw_columns].copy()

    participant_oof = participant_oof[
        [
            "external_trial_id",
            "calibrated_score",
            "prediction",
        ]
    ].rename(
        columns={
            "calibrated_score": (
                "participant_only_oof_score"
            ),
            "prediction": (
                "participant_only_oof_prediction"
            ),
        }
    )
    merged = raw.merge(
        participant_oof,
        on="external_trial_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(raw):
        raise ValueError(
            "Raw and participant-calibrated target sets differ."
        )
    return merged, result


def loo_median(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        raise ValueError(
            "At least three trials per participant are required."
        )
    output = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        output[index] = float(
            np.median(np.delete(values, index))
        )
    return output


def apply_target_methods(
    merged: pd.DataFrame,
    references: dict[str, float],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    participant_rows = []

    for participant_id, group in merged.groupby(
        "participant_id",
        observed=True,
        sort=False,
    ):
        group = group.copy().reset_index(drop=True)
        full_score = group[
            "reconstructed_full_score"
        ].to_numpy(dtype=float)
        quality = group[
            "quality_contribution"
        ].to_numpy(dtype=float)
        quality_removed = (
            full_score
            - quality
            + references[
                "quality_contribution_reference"
            ]
        )

        full_center = loo_median(full_score)
        quality_removed_center = loo_median(
            quality_removed
        )

        participant_final = (
            full_score
            - full_center
            + references["full_final_reference"]
        )
        quality_removed_plus_participant = (
            quality_removed
            - quality_removed_center
            + references[
                "quality_removed_final_reference"
            ]
        )

        method_scores = {
            "raw_frozen": full_score,
            "participant_only_oof": group[
                "participant_only_oof_score"
            ].to_numpy(dtype=float),
            "participant_only_final_reference": (
                participant_final
            ),
            "quality_removed": quality_removed,
            "quality_removed_plus_participant": (
                quality_removed_plus_participant
            ),
        }

        base_columns = [
            "external_trial_id",
            "participant_id",
            "label_nr",
        ]
        if "task" in group.columns:
            base_columns.append("task")

        for method in METHOD_ORDER:
            score = method_scores[method]
            part = group[base_columns].copy()
            part["method"] = method
            part["score"] = score
            part["frozen_threshold"] = threshold
            part["prediction"] = (
                score >= threshold
            ).astype(int)
            parts.append(part)

        participant_rows.append(
            {
                "participant_id": participant_id,
                "n_trials": len(group),
                "raw_full_score_median": float(
                    np.median(full_score)
                ),
                "quality_contribution_median": float(
                    np.median(quality)
                ),
                "quality_contribution_abs_median": float(
                    np.median(np.abs(quality))
                ),
                "quality_removed_score_median": float(
                    np.median(quality_removed)
                ),
                "participant_full_shift_median": float(
                    np.median(
                        references["full_final_reference"]
                        - full_center
                    )
                ),
                "participant_quality_removed_shift_median": float(
                    np.median(
                        references[
                            "quality_removed_final_reference"
                        ]
                        - quality_removed_center
                    )
                ),
            }
        )

    return (
        pd.concat(parts, ignore_index=True),
        pd.DataFrame(participant_rows),
    )


def safe_auc(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(
        roc_auc_score(
            y_true,
            score,
            sample_weight=sample_weight,
        )
    )


def safe_ap(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(
        average_precision_score(
            y_true,
            score,
            sample_weight=sample_weight,
        )
    )


def metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    if sample_weight is None:
        predicted_nr_rate = float(
            np.mean(prediction == 1)
        )
    else:
        predicted_nr_rate = float(
            np.sum(
                sample_weight * (prediction == 1)
            )
            / np.sum(sample_weight)
        )
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
                sample_weight=sample_weight,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
                sample_weight=sample_weight,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                sample_weight=sample_weight,
                zero_division=0,
            )
        ),
        "roc_auc": safe_auc(
            y_true,
            score,
            sample_weight,
        ),
        "pr_auc_nr": safe_ap(
            y_true,
            score,
            sample_weight,
        ),
        "predicted_nr_rate": predicted_nr_rate,
    }


def pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for method, group in predictions.groupby(
        "method",
        observed=True,
    ):
        result = metrics(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "method": method,
                "n_trials": len(group),
                "n_participants": group[
                    "participant_id"
                ].nunique(),
                **result,
            }
        )
    table = pd.DataFrame(rows)
    table["method"] = pd.Categorical(
        table["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    return table.sort_values("method").reset_index(
        drop=True
    )


def component_statistics(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for column in (
        "intercept_component",
        "content_contribution",
        "quality_contribution",
        "reconstructed_full_score",
    ):
        values = merged[column].to_numpy(dtype=float)
        rows.append(
            {
                "component": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "median_absolute_value": float(
                    np.median(np.abs(values))
                ),
                "q05": float(
                    np.quantile(values, 0.05)
                ),
                "q95": float(
                    np.quantile(values, 0.95)
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_deltas(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    participants = np.asarray(
        sorted(predictions["participant_id"].unique()),
        dtype=object,
    )
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc_nr",
        "predicted_nr_rate",
    )

    prepared: dict[str, dict[str, Any]] = {}
    for method, group in predictions.groupby(
        "method",
        observed=True,
    ):
        frame = group.reset_index(drop=True)
        prepared[str(method)] = {
            "frame": frame,
            "participants": frame[
                "participant_id"
            ].astype(str).to_numpy(),
        }

    comparisons = (
        ("raw_frozen", "participant_only_oof"),
        (
            "participant_only_final_reference",
            "quality_removed_plus_participant",
        ),
        ("raw_frozen", "quality_removed"),
        (
            "participant_only_oof",
            "quality_removed_plus_participant",
        ),
    )
    replicate_rows = []

    for replicate in range(n_bootstrap):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        ).astype(str)
        counts = Counter(sampled.tolist())

        values: dict[str, dict[str, float]] = {}
        for method, item in prepared.items():
            frame = item["frame"]
            weights = np.fromiter(
                (
                    counts.get(participant, 0)
                    for participant in item[
                        "participants"
                    ]
                ),
                dtype=float,
                count=len(frame),
            )
            values[method] = metrics(
                frame["label_nr"].to_numpy(dtype=int),
                frame["prediction"].to_numpy(dtype=int),
                frame["score"].to_numpy(dtype=float),
                sample_weight=weights,
            )

        for reference, comparison in comparisons:
            for metric_name in metric_names:
                replicate_rows.append(
                    {
                        "replicate": replicate,
                        "reference": reference,
                        "comparison": comparison,
                        "metric": metric_name,
                        "delta": (
                            values[comparison][metric_name]
                            - values[reference][metric_name]
                        ),
                    }
                )

    replicates = pd.DataFrame(replicate_rows)
    rows = []
    for (
        reference,
        comparison,
        metric_name,
    ), group in replicates.groupby(
        ["reference", "comparison", "metric"],
        observed=True,
    ):
        delta = group["delta"].to_numpy(dtype=float)
        rows.append(
            {
                "reference": reference,
                "comparison": comparison,
                "metric": metric_name,
                "delta_mean": float(np.nanmean(delta)),
                "ci_low": float(
                    np.nanquantile(delta, 0.025)
                ),
                "ci_high": float(
                    np.nanquantile(delta, 0.975)
                ),
                "probability_delta_gt_0": float(
                    np.nanmean(delta > 0)
                ),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    if args.score_tolerance <= 0:
        raise ValueError(
            "--score-tolerance must be positive."
        )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        frozen_pipeline,
        imputer,
        scaler,
        feature_columns,
        coefficient,
        intercept,
        threshold,
    ) = load_linear_artifact(args.frozen_model)

    source = pd.read_parquet(
        require_file(args.source_features)
    ).reset_index(drop=True)
    source_decomposition = exact_decomposition(
        source,
        imputer,
        scaler,
        feature_columns,
        coefficient,
        intercept,
    )
    source_pipeline_score = np.asarray(
        frozen_pipeline.decision_function(
            source[feature_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
        ),
        dtype=float,
    ).reshape(-1)
    source_error = float(
        np.max(
            np.abs(
                source_pipeline_score
                - source_decomposition[
                    "reconstructed_full_score"
                ].to_numpy(dtype=float)
            )
        )
    )
    if source_error > args.score_tolerance:
        raise RuntimeError(
            f"Source score reconstruction error={source_error} "
            f"> tolerance={args.score_tolerance}"
        )

    (
        source_participant_references,
        references,
    ) = source_references(
        source,
        source_decomposition,
    )

    target_features = normalize_target_features(
        pd.read_csv(
            require_file(args.target_features),
            low_memory=False,
        ),
        feature_columns=feature_columns,
    ).reset_index(drop=True)
    target_decomposition = exact_decomposition(
        target_features,
        imputer,
        scaler,
        feature_columns,
        coefficient,
        intercept,
    )
    target_features = pd.concat(
        [
            target_features.reset_index(drop=True),
            target_decomposition.reset_index(drop=True),
        ],
        axis=1,
    )

    metadata, _ = target_metadata(
        pd.read_csv(
            require_file(
                args.target_calibrated_predictions
            ),
            low_memory=False,
        )
    )
    merged = metadata.merge(
        target_features,
        on=[
            "external_trial_id",
            "participant_id",
        ],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(metadata):
        raise ValueError(
            "Target feature and prediction trial sets differ."
        )

    required_after_merge = {
        "external_trial_id",
        "participant_id",
        "label_nr",
        "raw_score",
        "frozen_threshold",
        "participant_only_oof_score",
        "content_contribution",
        "quality_contribution",
        "reconstructed_full_score",
    }
    missing_after_merge = sorted(
        required_after_merge - set(merged.columns)
    )
    if missing_after_merge:
        raise ValueError(
            "Merged target table lacks required columns: "
            f"{missing_after_merge}. "
            f"Available columns={list(merged.columns)}"
        )

    raw_score_error = float(
        np.max(
            np.abs(
                merged["raw_score"].to_numpy(dtype=float)
                - merged[
                    "reconstructed_full_score"
                ].to_numpy(dtype=float)
            )
        )
    )
    if raw_score_error > args.score_tolerance:
        raise RuntimeError(
            f"Target frozen score reconstruction error="
            f"{raw_score_error} > tolerance="
            f"{args.score_tolerance}"
        )

    predictions, participant_audit = (
        apply_target_methods(
            merged,
            references,
            threshold,
        )
    )
    pooled = pooled_metrics(predictions)
    statistics = component_statistics(merged)
    deltas = bootstrap_deltas(
        predictions,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    print(
        "=== EXACT LINEAR FEATURE-GROUP FACTORIZATION ==="
    )
    print(f"source trials: {len(source):,}")
    print(
        f"source participants: "
        f"{source['participant_id'].nunique()}"
    )
    print(f"target trials: {len(merged):,}")
    print(
        f"target participants: "
        f"{merged['participant_id'].nunique()}"
    )
    print(
        f"content features: {len(CONTENT_FEATURES)}"
    )
    print(
        f"quality features: {len(QUALITY_FEATURES)}"
    )
    print(
        f"source reconstruction max error: "
        f"{source_error:.3e}"
    )
    print(
        f"target reconstruction max error: "
        f"{raw_score_error:.3e}"
    )
    print(
        "target labels used for decomposition/correction: False"
    )
    print(
        "\n=== SOURCE REFERENCES ==="
    )
    print(
        json.dumps(
            references,
            ensure_ascii=False,
            indent=2,
        )
    )

    source_reference_path = (
        args.output_dir
        / "source_participant_score_references.csv"
    )
    target_component_path = (
        args.output_dir
        / "target_exact_score_components.csv"
    )
    participant_audit_path = (
        args.output_dir
        / "target_feature_group_participant_audit.csv"
    )
    predictions_path = (
        args.output_dir
        / "target_feature_group_predictions.csv"
    )
    pooled_path = (
        args.output_dir
        / "target_feature_group_pooled_metrics.csv"
    )
    statistics_path = (
        args.output_dir
        / "target_feature_group_component_statistics.csv"
    )
    deltas_path = (
        args.output_dir
        / "target_feature_group_bootstrap_deltas.csv"
    )
    manifest_path = (
        args.output_dir
        / "linear_feature_group_factorization_manifest.json"
    )

    source_participant_references.to_csv(
        source_reference_path,
        index=False,
        encoding="utf-8-sig",
    )
    merged[
        [
            "external_trial_id",
            "participant_id",
            "label_nr",
            *(
                ["task"]
                if "task" in merged.columns
                else []
            ),
            "intercept_component",
            "content_contribution",
            "quality_contribution",
            "reconstructed_full_score",
        ]
    ].to_csv(
        target_component_path,
        index=False,
        encoding="utf-8-sig",
    )
    participant_audit.to_csv(
        participant_audit_path,
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        pooled_path,
        index=False,
        encoding="utf-8-sig",
    )
    statistics.to_csv(
        statistics_path,
        index=False,
        encoding="utf-8-sig",
    )
    deltas.to_csv(
        deltas_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "analysis": (
            "Exact frozen linear score decomposition by "
            "content and membership/quality feature groups"
        ),
        "model_name": MODEL_NAME,
        "feature_groups": {
            "content": CONTENT_FEATURES,
            "membership_quality": QUALITY_FEATURES,
        },
        "exact_identity": (
            "decision_score = intercept + content_contribution "
            "+ quality_contribution"
        ),
        "source_references": references,
        "target_labels_used_for_decomposition": False,
        "target_labels_used_for_correction": False,
        "primary_reference": PRIMARY_REFERENCE,
        "primary_comparison": PRIMARY_COMPARISON,
        "interpretation_caveat": (
            "Membership/quality features are measurement-associated, "
            "not guaranteed to be a pure causal measurement factor. "
            "This is an exact model-score decomposition rather than "
            "causal disentanglement."
        ),
        "reconstruction_audit": {
            "source_max_absolute_error": source_error,
            "target_max_absolute_error": raw_score_error,
            "tolerance": args.score_tolerance,
        },
        "bootstrap": {
            "replicates": args.bootstrap,
            "unit": "participant",
            "paired_methods": True,
        },
        "outputs": {
            "source_references": str(
                source_reference_path
            ),
            "target_components": str(
                target_component_path
            ),
            "participant_audit": str(
                participant_audit_path
            ),
            "predictions": str(predictions_path),
            "pooled_metrics": str(pooled_path),
            "component_statistics": str(
                statistics_path
            ),
            "bootstrap_deltas": str(deltas_path),
        },
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== EXACT FEATURE-GROUP POOLED METRICS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
    ):
        print(pooled.to_string(index=False))

    print(
        "\n=== EXACT FEATURE-GROUP COMPONENT STATISTICS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(statistics.to_string(index=False))

    print(
        "\n=== EXACT FEATURE-GROUP BOOTSTRAP DELTAS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
    ):
        print(deltas.to_string(index=False))

    print(f"\nSaved: {pooled_path}")
    print(f"Saved: {statistics_path}")
    print(f"Saved: {deltas_path}")
    print(f"Saved: {manifest_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
