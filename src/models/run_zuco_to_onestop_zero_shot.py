#!/usr/bin/env python
r"""
Apply frozen ZuCo Soft-AOI models to OneStop without retraining or target tuning.

Primary zero-shot mapping:
- ZuCo NR label 1 -> OneStop NR label 1
- ZuCo TSR label 0 -> OneStop information-seeking (IS) label 0

The frozen sklearn pipeline, stored feature order, and stored decision threshold
are used exactly as saved. OneStop labels are used only after prediction for
evaluation. A target-oracle threshold is also reported in a separate diagnostic
table to distinguish ranking failure from threshold/marginal shift; it is NOT a
zero-shot result.

Example
-------
python -u src\models\run_zuco_to_onestop_zero_shot.py ^
  --features data\processed\onestop\soft_aoi_asc344\onestop_soft_aoi_all_features.parquet ^
  --frozen-dir outputs\frozen_zuco_external_validation_v1 ^
  --output-dir outputs\zuco_to_onestop_zero_shot_v1 ^
  --bootstrap 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


PRIMARY_MODELS = {
    "soft_compact_distribution_membership": (
        "models/soft_compact_distribution_membership.joblib"
    ),
    "soft_full_all": "models/soft_full_all.joblib",
}
SECONDARY_MODELS = {
    "soft_compact_all18_secondary": (
        "models/soft_compact_all18_secondary.joblib"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=1001,
    )
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Also evaluate soft_compact_all18_secondary.joblib.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def get_final_estimator(pipeline: Any) -> Any:
    if hasattr(pipeline, "steps") and pipeline.steps:
        return pipeline.steps[-1][1]
    return pipeline


def get_classes(pipeline: Any) -> np.ndarray:
    if hasattr(pipeline, "classes_"):
        return np.asarray(pipeline.classes_)
    estimator = get_final_estimator(pipeline)
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_)
    raise ValueError("Frozen pipeline does not expose classes_.")


def positive_decision_scores(
    pipeline: Any,
    x: pd.DataFrame,
) -> np.ndarray:
    classes = get_classes(pipeline)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(
            f"Frozen model classes are {classes.tolist()}, expected [0, 1]."
        )

    if hasattr(pipeline, "decision_function"):
        raw = np.asarray(pipeline.decision_function(x))
        if raw.ndim == 1:
            # sklearn binary decision_function is positive for classes_[1].
            if int(classes[1]) == 1:
                return raw.astype(float)
            return (-raw).astype(float)
        if raw.ndim == 2:
            positive_index = int(np.where(classes == 1)[0][0])
            return raw[:, positive_index].astype(float)

    if hasattr(pipeline, "predict_proba"):
        probability = np.asarray(pipeline.predict_proba(x))
        positive_index = int(np.where(classes == 1)[0][0])
        # A probability score is still valid for ranking and thresholding, but
        # the stored threshold must then be on the same probability scale.
        return probability[:, positive_index].astype(float)

    raise ValueError(
        "Frozen pipeline has neither decision_function nor predict_proba."
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


def compute_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    prediction: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    nr_mask = y_true == 1
    is_mask = y_true == 0

    if sample_weight is None:
        true_nr_rate = float(np.mean(nr_mask))
        predicted_nr_rate = float(np.mean(prediction == 1))
    else:
        total_weight = float(np.sum(sample_weight))
        true_nr_rate = float(
            np.sum(sample_weight * nr_mask) / total_weight
        )
        predicted_nr_rate = float(
            np.sum(sample_weight * (prediction == 1)) / total_weight
        )

    return {
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                sample_weight=sample_weight,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
                sample_weight=sample_weight,
            )
        ),
        "roc_auc": safe_auc(y_true, score, sample_weight),
        "pr_auc_nr": safe_ap(y_true, score, sample_weight),
        "true_nr_rate": true_nr_rate,
        "predicted_nr_rate": predicted_nr_rate,
        "score_median_nr": weighted_median(
            score[nr_mask],
            None if sample_weight is None else sample_weight[nr_mask],
        ),
        "score_median_is": weighted_median(
            score[is_mask],
            None if sample_weight is None else sample_weight[is_mask],
        ),
    }


def weighted_median(
    values: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(np.median(values))

    weights = np.asarray(weights, dtype=float)[finite]
    positive = weights > 0
    values = values[positive]
    weights = weights[positive]
    if values.size == 0:
        return float("nan")

    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(weights.sum())
    index = int(np.searchsorted(np.cumsum(weights), cutoff, side="left"))
    return float(values[min(index, len(values) - 1)])


def threshold_candidates(
    score: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        raise ValueError("No finite scores.")
    quantiles = np.linspace(0.0, 1.0, max(3, grid_size))
    candidates = np.quantile(finite, quantiles)
    candidates = np.unique(
        np.concatenate(
            [
                candidates,
                np.asarray(
                    [np.nextafter(finite.min(), -np.inf)]
                ),
                np.asarray(
                    [np.nextafter(finite.max(), np.inf)]
                ),
            ]
        )
    )
    return candidates


def select_oracle_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    grid_size: int,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.0
    best_metrics: dict[str, float] | None = None
    best_key = (-np.inf, -np.inf, -np.inf)

    for threshold in threshold_candidates(score, grid_size):
        prediction = (score >= threshold).astype(int)
        metrics = compute_metrics(y_true, score, prediction)
        key = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            -abs(float(threshold)),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def validate_artifact(
    artifact: Any,
    expected_name: str,
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise TypeError(
            f"{expected_name}: expected dict artifact, got {type(artifact)}"
        )
    required = {
        "pipeline",
        "feature_columns",
        "decision_threshold",
        "label_to_int",
        "int_to_label",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(
            f"{expected_name}: artifact lacks keys {missing}"
        )

    feature_columns = [str(item) for item in artifact["feature_columns"]]
    if not feature_columns:
        raise ValueError(f"{expected_name}: empty feature list.")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            f"{expected_name}: duplicate feature columns."
        )

    threshold = float(artifact["decision_threshold"])
    if not math.isfinite(threshold):
        raise ValueError(
            f"{expected_name}: non-finite decision threshold."
        )

    label_to_int = {
        str(key): int(value)
        for key, value in dict(artifact["label_to_int"]).items()
    }
    # ZuCo information-seeking may be named TSR, while OneStop uses IS.
    if label_to_int.get("NR") != 1:
        raise ValueError(
            f"{expected_name}: NR must map to 1, got {label_to_int}"
        )
    zero_labels = [
        key for key, value in label_to_int.items() if value == 0
    ]
    if not zero_labels:
        raise ValueError(
            f"{expected_name}: no label maps to class 0."
        )

    pipeline = artifact["pipeline"]
    classes = get_classes(pipeline)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(
            f"{expected_name}: classes={classes.tolist()}"
        )

    n_features = getattr(pipeline, "n_features_in_", None)
    if n_features is None:
        n_features = getattr(
            get_final_estimator(pipeline), "n_features_in_", None
        )
    if n_features is not None and int(n_features) != len(feature_columns):
        raise ValueError(
            f"{expected_name}: model expects {n_features} features but "
            f"artifact lists {len(feature_columns)}."
        )

    return {
        "feature_columns": feature_columns,
        "decision_threshold": threshold,
        "label_to_int": label_to_int,
        "zero_class_source_labels": zero_labels,
        "classes": classes.tolist(),
        "pipeline": pipeline,
    }


def participant_level(
    prediction_frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    participant = (
        prediction_frame.groupby(
            ["model_name", "participant_id"],
            observed=True,
        )
        .agg(
            label=("label", "first"),
            label_name=("label_name", "first"),
            score=("score", "median"),
            trials=("onestop_trial_id", "size"),
        )
        .reset_index()
    )
    participant["prediction"] = (
        participant["score"].to_numpy(dtype=float) >= threshold
    ).astype(int)
    return participant


def bootstrap_weights(
    participant_ids: np.ndarray,
    participant_labels: dict[str, int],
    rng: np.random.Generator,
) -> Counter[str]:
    labels_to_ids: dict[int, list[str]] = {0: [], 1: []}
    for participant in participant_ids:
        labels_to_ids[participant_labels[str(participant)]].append(
            str(participant)
        )

    sampled: list[str] = []
    for label in (0, 1):
        population = np.asarray(labels_to_ids[label], dtype=object)
        sampled.extend(
            rng.choice(
                population,
                size=len(population),
                replace=True,
            ).astype(str).tolist()
        )
    return Counter(sampled)


def bootstrap_cis_and_deltas(
    trial_predictions: pd.DataFrame,
    participant_predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    participant_label_frame = (
        trial_predictions[
            ["participant_id", "label"]
        ]
        .drop_duplicates()
    )
    if participant_label_frame["participant_id"].duplicated().any():
        raise ValueError("Participant has multiple labels.")
    participant_labels = {
        str(row.participant_id): int(row.label)
        for row in participant_label_frame.itertuples(index=False)
    }
    participant_ids = np.asarray(
        sorted(participant_labels),
        dtype=object,
    )

    model_names = sorted(trial_predictions["model_name"].unique())
    metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
        "predicted_nr_rate",
    ]

    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    for model_name in model_names:
        trial = trial_predictions[
            trial_predictions["model_name"].eq(model_name)
        ].reset_index(drop=True)
        participant = participant_predictions[
            participant_predictions["model_name"].eq(model_name)
        ].reset_index(drop=True)

        prepared[(model_name, "trial")] = {
            "frame": trial,
            "participant_array": trial[
                "participant_id"
            ].astype(str).to_numpy(),
        }
        prepared[(model_name, "participant")] = {
            "frame": participant,
            "participant_array": participant[
                "participant_id"
            ].astype(str).to_numpy(),
        }

    rows = []
    for replicate in range(n_bootstrap):
        counts = bootstrap_weights(
            participant_ids,
            participant_labels,
            rng,
        )
        for model_name in model_names:
            for level in ("trial", "participant"):
                prepared_item = prepared[(model_name, level)]
                frame = prepared_item["frame"]
                weights = np.fromiter(
                    (
                        counts.get(participant, 0)
                        for participant in prepared_item[
                            "participant_array"
                        ]
                    ),
                    dtype=float,
                    count=len(frame),
                )
                metrics = compute_metrics(
                    frame["label"].to_numpy(dtype=int),
                    frame["score"].to_numpy(dtype=float),
                    frame["prediction"].to_numpy(dtype=int),
                    sample_weight=weights,
                )
                rows.append(
                    {
                        "replicate": replicate,
                        "model_name": model_name,
                        "level": level,
                        **{
                            metric: metrics[metric]
                            for metric in metric_names
                        },
                    }
                )

    replicates = pd.DataFrame(rows)
    ci_rows = []
    for (model_name, level), group in replicates.groupby(
        ["model_name", "level"],
        observed=True,
    ):
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=float)
            ci_rows.append(
                {
                    "model_name": model_name,
                    "level": level,
                    "metric": metric,
                    "bootstrap_mean": float(np.nanmean(values)),
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    cis = pd.DataFrame(ci_rows)

    delta_rows = []
    if {
        "soft_compact_distribution_membership",
        "soft_full_all",
    }.issubset(model_names):
        compact = replicates[
            replicates["model_name"].eq(
                "soft_compact_distribution_membership"
            )
        ].set_index(["replicate", "level"])
        full = replicates[
            replicates["model_name"].eq("soft_full_all")
        ].set_index(["replicate", "level"])
        common = compact.index.intersection(full.index)
        for level in ("trial", "participant"):
            level_index = [
                index for index in common if index[1] == level
            ]
            for metric in metric_names:
                delta = (
                    compact.loc[level_index, metric].to_numpy(dtype=float)
                    - full.loc[level_index, metric].to_numpy(dtype=float)
                )
                delta_rows.append(
                    {
                        "reference": (
                            "soft_compact_distribution_membership"
                        ),
                        "comparison": "soft_full_all",
                        "level": level,
                        "metric": metric,
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
    deltas = pd.DataFrame(delta_rows)
    return cis, deltas


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(require_file(args.features))
    required_columns = {
        "onestop_trial_id",
        "participant_id",
        "label",
        "label_name",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"OneStop table lacks columns: {missing}")

    frame["participant_id"] = (
        frame["participant_id"].astype(str).str.strip().str.lower()
    )
    frame["label"] = pd.to_numeric(
        frame["label"], errors="raise"
    ).astype(int)
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("OneStop labels must be {0, 1}.")
    if frame["onestop_trial_id"].duplicated().any():
        raise ValueError("Duplicate OneStop trial IDs.")

    participant_label_counts = (
        frame[["participant_id", "label"]]
        .drop_duplicates()
        .groupby("label")
        .size()
        .to_dict()
    )

    model_paths = dict(PRIMARY_MODELS)
    if args.include_secondary:
        model_paths.update(SECONDARY_MODELS)

    print("=== ZUCO -> ONESTOP ZERO-SHOT INPUT ===")
    print(f"OneStop trials: {len(frame):,}")
    print(
        f"OneStop participants: "
        f"{frame['participant_id'].nunique()}"
    )
    print(
        "participant labels: "
        + json.dumps(
            {
                "IS_0": int(participant_label_counts.get(0, 0)),
                "NR_1": int(participant_label_counts.get(1, 0)),
            }
        )
    )
    print(
        "target labels are used only for evaluation; "
        "no model fitting or threshold tuning is performed."
    )

    trial_prediction_parts = []
    participant_prediction_parts = []
    frozen_metric_rows = []
    oracle_rows = []
    model_audit_rows = []

    for model_name, relative_path in model_paths.items():
        model_path = require_file(
            args.frozen_dir / relative_path
        )
        artifact = joblib.load(model_path)
        validated = validate_artifact(artifact, model_name)
        columns = validated["feature_columns"]
        missing_features = sorted(set(columns) - set(frame.columns))
        if missing_features:
            raise ValueError(
                f"{model_name}: OneStop lacks features "
                f"{missing_features}"
            )

        x = frame[columns].apply(pd.to_numeric, errors="coerce")
        pipeline = validated["pipeline"]
        threshold = validated["decision_threshold"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = positive_decision_scores(pipeline, x)

        if not np.isfinite(score).all():
            raise ValueError(
                f"{model_name}: non-finite prediction scores."
            )
        prediction = (score >= threshold).astype(int)

        trial = pd.DataFrame(
            {
                "onestop_trial_id": frame[
                    "onestop_trial_id"
                ].astype(str).to_numpy(),
                "participant_id": frame[
                    "participant_id"
                ].astype(str).to_numpy(),
                "label": frame["label"].to_numpy(dtype=int),
                "label_name": frame[
                    "label_name"
                ].astype(str).to_numpy(),
                "model_name": model_name,
                "score": score,
                "prediction": prediction,
                "frozen_threshold": threshold,
            }
        )
        participant = participant_level(trial, threshold)
        participant["frozen_threshold"] = threshold

        for level, evaluation in (
            ("trial", trial),
            ("participant", participant),
        ):
            metrics = compute_metrics(
                evaluation["label"].to_numpy(dtype=int),
                evaluation["score"].to_numpy(dtype=float),
                evaluation["prediction"].to_numpy(dtype=int),
            )
            frozen_metric_rows.append(
                {
                    "model_name": model_name,
                    "level": level,
                    "n_units": len(evaluation),
                    "n_participants": evaluation[
                        "participant_id"
                    ].nunique(),
                    "threshold_source": "frozen_zuco",
                    "decision_threshold": threshold,
                    **metrics,
                }
            )

            oracle_threshold, oracle_metrics = (
                select_oracle_threshold(
                    evaluation["label"].to_numpy(dtype=int),
                    evaluation["score"].to_numpy(dtype=float),
                    args.threshold_grid_size,
                )
            )
            oracle_rows.append(
                {
                    "model_name": model_name,
                    "level": level,
                    "n_units": len(evaluation),
                    "threshold_source": (
                        "target_label_oracle_diagnostic_not_zero_shot"
                    ),
                    "decision_threshold": oracle_threshold,
                    **oracle_metrics,
                }
            )

        model_audit_rows.append(
            {
                "model_name": model_name,
                "model_path": str(model_path),
                "artifact_version": to_builtin(
                    artifact.get("artifact_version")
                ),
                "training_scope": to_builtin(
                    artifact.get("training_scope")
                ),
                "feature_set": to_builtin(
                    artifact.get("feature_set")
                ),
                "n_features": len(columns),
                "decision_threshold": threshold,
                "prediction_rule": to_builtin(
                    artifact.get("prediction_rule")
                ),
                "c_value": to_builtin(artifact.get("c_value")),
                "seed": to_builtin(artifact.get("seed")),
                "classes": json.dumps(
                    validated["classes"],
                    ensure_ascii=False,
                ),
                "label_to_int": json.dumps(
                    validated["label_to_int"],
                    ensure_ascii=False,
                ),
                "zero_class_source_labels": json.dumps(
                    validated["zero_class_source_labels"],
                    ensure_ascii=False,
                ),
                "primary_external_comparison_eligible": (
                    to_builtin(
                        artifact.get(
                            "primary_external_comparison_eligible",
                            True,
                        )
                    )
                ),
                "feature_columns_json": json.dumps(
                    columns, ensure_ascii=False
                ),
            }
        )

        trial_prediction_parts.append(trial)
        participant_prediction_parts.append(participant)

        print(
            f"\n[{model_name}] features={len(columns)}, "
            f"frozen threshold={threshold:.6f}"
        )
        trial_metrics = frozen_metric_rows[-2]
        participant_metrics = frozen_metric_rows[-1]
        print(
            "  trial: "
            f"macroF1={trial_metrics['macro_f1']:.4f}, "
            f"BA={trial_metrics['balanced_accuracy']:.4f}, "
            f"AUC={trial_metrics['roc_auc']:.4f}, "
            f"PR-AUC(NR)={trial_metrics['pr_auc_nr']:.4f}, "
            f"pred_NR={trial_metrics['predicted_nr_rate']:.4f}"
        )
        print(
            "  participant: "
            f"macroF1={participant_metrics['macro_f1']:.4f}, "
            f"BA={participant_metrics['balanced_accuracy']:.4f}, "
            f"AUC={participant_metrics['roc_auc']:.4f}, "
            f"pred_NR={participant_metrics['predicted_nr_rate']:.4f}"
        )

    trial_predictions = pd.concat(
        trial_prediction_parts, ignore_index=True
    )
    participant_predictions = pd.concat(
        participant_prediction_parts, ignore_index=True
    )
    frozen_metrics = pd.DataFrame(frozen_metric_rows)
    oracle_diagnostics = pd.DataFrame(oracle_rows)
    model_audit = pd.DataFrame(model_audit_rows)

    cis, paired_deltas = bootstrap_cis_and_deltas(
        trial_predictions=trial_predictions,
        participant_predictions=participant_predictions,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    metrics_path = args.output_dir / "zero_shot_frozen_metrics.csv"
    oracle_path = (
        args.output_dir / "target_oracle_threshold_diagnostics.csv"
    )
    ci_path = args.output_dir / "zero_shot_bootstrap_cis.csv"
    delta_path = (
        args.output_dir / "zero_shot_compact_vs_full_deltas.csv"
    )
    trial_path = (
        args.output_dir / "zero_shot_trial_predictions.parquet"
    )
    participant_path = (
        args.output_dir / "zero_shot_participant_predictions.csv"
    )
    audit_csv_path = args.output_dir / "frozen_model_audit.csv"
    audit_json_path = (
        args.output_dir / "zuco_to_onestop_zero_shot_audit.json"
    )

    frozen_metrics.to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )
    oracle_diagnostics.to_csv(
        oracle_path, index=False, encoding="utf-8-sig"
    )
    cis.to_csv(ci_path, index=False, encoding="utf-8-sig")
    paired_deltas.to_csv(
        delta_path, index=False, encoding="utf-8-sig"
    )
    trial_predictions.to_parquet(
        trial_path, index=False, compression="zstd"
    )
    participant_predictions.to_csv(
        participant_path, index=False, encoding="utf-8-sig"
    )
    model_audit.to_csv(
        audit_csv_path, index=False, encoding="utf-8-sig"
    )

    audit = {
        "analysis": "Frozen ZuCo to OneStop zero-shot evaluation",
        "label_mapping": {
            "ZuCo_NR_1": "OneStop_NR_1",
            "ZuCo_information_seeking_class_0": "OneStop_IS_0",
        },
        "zero_shot_definition": (
            "Frozen pipeline, stored feature order, and stored ZuCo "
            "threshold. No fitting, calibration, feature selection, or "
            "threshold tuning on OneStop."
        ),
        "oracle_diagnostic_warning": (
            "Target-oracle thresholds use OneStop labels and are diagnostic "
            "only; they must not be reported as zero-shot performance."
        ),
        "n_trials": int(len(frame)),
        "n_participants": int(
            frame["participant_id"].nunique()
        ),
        "models": model_audit.to_dict(orient="records"),
        "bootstrap": {
            "replicates": args.bootstrap,
            "unit": "participant",
            "stratified_by_label": True,
        },
        "outputs": {
            "frozen_metrics": str(metrics_path),
            "target_oracle_diagnostics": str(oracle_path),
            "bootstrap_cis": str(ci_path),
            "paired_deltas": str(delta_path),
            "trial_predictions": str(trial_path),
            "participant_predictions": str(participant_path),
            "model_audit": str(audit_csv_path),
        },
    }
    audit_json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ZUCO -> ONESTOP FROZEN ZERO-SHOT METRICS ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 240
    ):
        print(
            frozen_metrics.sort_values(
                ["level", "model_name"]
            ).to_string(index=False)
        )

    print(
        "\n=== TARGET-ORACLE THRESHOLD DIAGNOSTIC "
        "(NOT ZERO-SHOT) ==="
    )
    with pd.option_context(
        "display.max_columns", None, "display.width", 240
    ):
        print(
            oracle_diagnostics.sort_values(
                ["level", "model_name"]
            ).to_string(index=False)
        )

    print("\n=== COMPACT VS FULL PAIRED BOOTSTRAP DELTAS ===")
    if paired_deltas.empty:
        print("No primary compact/full pair available.")
    else:
        with pd.option_context(
            "display.max_columns", None, "display.width", 240
        ):
            print(paired_deltas.to_string(index=False))

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {oracle_path}")
    print(f"Saved: {ci_path}")
    print(f"Saved: {delta_path}")
    print(f"Saved: {trial_path}")
    print(f"Saved: {audit_json_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
