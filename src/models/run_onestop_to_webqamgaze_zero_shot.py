#!/usr/bin/env python
r"""
Apply frozen OneStop Soft-AOI models to WebQAmGaze without target retraining.

Primary evaluation:
- OneStop IS=0 -> WebQAmGaze IS=0
- OneStop NR=1 -> WebQAmGaze NR=1
- Frozen OneStop pipeline, feature order, preprocessing, C, and threshold
- No WebQAmGaze fitting, calibration, feature selection, or threshold tuning

A target-oracle threshold table is written only as a diagnostic to distinguish
ranking transfer from score-location shift. It is not a zero-shot result.

Example
-------
python -u src\models\run_onestop_to_webqamgaze_zero_shot.py ^
  --compact-features outputs\webqamgaze_external_validation_stage3c\soft_compact_features_primary.csv ^
  --full-features outputs\webqamgaze_external_validation_stage3c\soft_full_features_primary.csv ^
  --frozen-dir outputs\frozen_onestop_external_validation_v1 ^
  --output-dir outputs\onestop_to_webqamgaze_zero_shot_v1 ^
  --bootstrap 2000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
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
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


MODEL_SPECS = {
    "soft_compact_distribution_membership": {
        "artifact": (
            "models/soft_compact_distribution_membership.joblib"
        ),
        "input": "compact",
    },
    "soft_full_all": {
        "artifact": "models/soft_full_all.joblib",
        "input": "full",
    },
}

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
TASK_ALIASES = (
    "task",
    "label_name",
    "condition",
)
LABEL_ALIASES = (
    "label_nr",
    "label",
    "y",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-features", type=Path, required=True)
    parser.add_argument("--full-features", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=1001,
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def first_existing(
    columns: pd.Index,
    aliases: tuple[str, ...],
    purpose: str,
) -> str:
    mapping = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in mapping:
            return mapping[alias.lower()]
    raise ValueError(
        f"Could not identify {purpose} column. "
        f"Tried {aliases}; available columns={list(columns)}"
    )


def normalize_task(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"NR", "NATURAL_READING", "ORDINARY"}:
        return "NR"
    if text in {
        "IS",
        "INFORMATION_SEEKING",
        "INFORMATION-SEEKING",
        "TSR",
    }:
        return "IS"
    return text


def normalize_external_table(
    frame: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    id_column = first_existing(
        frame.columns, ID_ALIASES, "trial ID"
    )
    participant_column = first_existing(
        frame.columns, PARTICIPANT_ALIASES, "participant ID"
    )
    task_column = first_existing(
        frame.columns, TASK_ALIASES, "task"
    )

    result = frame.copy()
    result["external_trial_id"] = (
        result[id_column].astype(str).str.strip()
    )
    result["participant_id"] = (
        result[participant_column].astype(str).str.strip()
    )
    result["task_canonical"] = result[task_column].map(normalize_task)

    label_column = ""
    lower_columns = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }
    for alias in LABEL_ALIASES:
        if alias.lower() in lower_columns:
            label_column = lower_columns[alias.lower()]
            break

    expected_from_task = result["task_canonical"].map(
        {"IS": 0, "NR": 1}
    )
    if expected_from_task.isna().any():
        unknown = sorted(
            result.loc[
                expected_from_task.isna(),
                "task_canonical",
            ].unique()
        )
        raise ValueError(
            f"{source_name}: unsupported task labels {unknown}"
        )

    if label_column:
        numeric_label = pd.to_numeric(
            result[label_column], errors="raise"
        ).astype(int)
        if not numeric_label.isin([0, 1]).all():
            raise ValueError(
                f"{source_name}: labels are not binary 0/1."
            )
        mismatch = numeric_label.ne(
            expected_from_task.astype(int)
        )
        if mismatch.any():
            preview = result.loc[
                mismatch,
                [
                    "external_trial_id",
                    "task_canonical",
                    label_column,
                ],
            ].head(20)
            raise ValueError(
                f"{source_name}: task/label direction mismatch:\n"
                + preview.to_string(index=False)
            )
        result["label_nr"] = numeric_label
    else:
        result["label_nr"] = expected_from_task.astype(int)

    if result["external_trial_id"].duplicated().any():
        duplicates = result.loc[
            result["external_trial_id"].duplicated(keep=False),
            "external_trial_id",
        ].head(20).tolist()
        raise ValueError(
            f"{source_name}: duplicate trial IDs {duplicates}"
        )
    if result["participant_id"].eq("").any():
        raise ValueError(f"{source_name}: empty participant ID.")
    if set(result["label_nr"].unique()) != {0, 1}:
        raise ValueError(
            f"{source_name}: both IS=0 and NR=1 are required."
        )
    return result


def get_classes(pipeline: Any) -> np.ndarray:
    if hasattr(pipeline, "classes_"):
        return np.asarray(pipeline.classes_)
    if hasattr(pipeline, "steps") and pipeline.steps:
        estimator = pipeline.steps[-1][1]
        if hasattr(estimator, "classes_"):
            return np.asarray(estimator.classes_)
    raise ValueError("Frozen pipeline does not expose classes_.")


def decision_score_for_nr(
    pipeline: Any,
    x: pd.DataFrame,
) -> np.ndarray:
    classes = get_classes(pipeline)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(
            f"Expected frozen classes [0,1], got {classes.tolist()}."
        )

    if hasattr(pipeline, "decision_function"):
        raw = np.asarray(pipeline.decision_function(x))
        if raw.ndim == 1:
            return (
                raw.astype(float)
                if int(classes[1]) == 1
                else (-raw).astype(float)
            )
        positive_index = int(np.where(classes == 1)[0][0])
        return raw[:, positive_index].astype(float)

    if hasattr(pipeline, "predict_proba"):
        probability = np.asarray(pipeline.predict_proba(x))
        positive_index = int(np.where(classes == 1)[0][0])
        return probability[:, positive_index].astype(float)

    raise ValueError(
        "Frozen pipeline has neither decision_function nor predict_proba."
    )


def validate_artifact(
    artifact: Any,
    model_name: str,
) -> tuple[Any, list[str], float, dict[str, Any]]:
    if not isinstance(artifact, dict):
        raise TypeError(
            f"{model_name}: expected dict artifact, got {type(artifact)}"
        )
    required = {
        "pipeline",
        "feature_columns",
        "decision_threshold",
        "label_to_int",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(
            f"{model_name}: artifact lacks keys {missing}"
        )

    columns = [str(value) for value in artifact["feature_columns"]]
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(
            f"{model_name}: invalid feature column definition."
        )
    threshold = float(artifact["decision_threshold"])
    if not math.isfinite(threshold):
        raise ValueError(
            f"{model_name}: non-finite threshold."
        )

    label_to_int = {
        str(key).upper(): int(value)
        for key, value in dict(artifact["label_to_int"]).items()
    }
    if label_to_int.get("IS") != 0 or label_to_int.get("NR") != 1:
        raise ValueError(
            f"{model_name}: expected IS=0, NR=1; got {label_to_int}"
        )

    pipeline = artifact["pipeline"]
    classes = get_classes(pipeline)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(
            f"{model_name}: classes={classes.tolist()}"
        )

    audit = {
        "artifact_version": artifact.get("artifact_version"),
        "feature_set": artifact.get("feature_set"),
        "training_scope": artifact.get("training_scope"),
        "n_features": len(columns),
        "decision_threshold": threshold,
        "c_value": artifact.get("c_value"),
        "classes": classes.tolist(),
        "label_to_int": label_to_int,
        "prediction_rule": artifact.get("prediction_rule"),
    }
    return pipeline, columns, threshold, audit


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
    prediction: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    if sample_weight is None:
        predicted_nr_rate = float(np.mean(prediction == 1))
        true_nr_rate = float(np.mean(y_true == 1))
    else:
        total = float(np.sum(sample_weight))
        predicted_nr_rate = float(
            np.sum(sample_weight * (prediction == 1)) / total
        )
        true_nr_rate = float(
            np.sum(sample_weight * (y_true == 1)) / total
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
            y_true, score, sample_weight
        ),
        "pr_auc_nr": safe_ap(
            y_true, score, sample_weight
        ),
        "true_nr_rate": true_nr_rate,
        "predicted_nr_rate": predicted_nr_rate,
        "score_mean": float(
            np.average(score, weights=sample_weight)
            if sample_weight is not None
            else np.mean(score)
        ),
    }


def subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, participant_id), group in predictions.groupby(
        ["model_name", "participant_id"],
        observed=True,
        sort=False,
    ):
        metrics = compute_metrics(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": model_name,
                "participant_id": participant_id,
                "n_trials": len(group),
                "n_is": int(group["label_nr"].eq(0).sum()),
                "n_nr": int(group["label_nr"].eq(1).sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_subject_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc_nr",
        "predicted_nr_rate",
    ]
    rows = []
    for model_name, group in frame.groupby(
        "model_name", observed=True
    ):
        row: dict[str, Any] = {
            "model_name": model_name,
            "participants": group["participant_id"].nunique(),
            "participants_with_both_labels": int(
                ((group["n_is"] > 0) & (group["n_nr"] > 0)).sum()
            ),
        }
        for metric in metric_columns:
            values = pd.to_numeric(
                group[metric], errors="coerce"
            )
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_candidates(
    score: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        raise ValueError("No finite scores.")
    candidates = np.quantile(
        finite,
        np.linspace(0.0, 1.0, max(3, grid_size)),
    )
    return np.unique(
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


def oracle_threshold_diagnostic(
    y_true: np.ndarray,
    score: np.ndarray,
    grid_size: int,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.0
    best_metrics: dict[str, float] | None = None
    best_key = (-np.inf, -np.inf, -np.inf)

    for threshold in threshold_candidates(score, grid_size):
        prediction = (score >= threshold).astype(int)
        metrics = compute_metrics(y_true, prediction, score)
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


def cluster_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    participants = np.asarray(
        sorted(predictions["participant_id"].unique()),
        dtype=object,
    )
    model_names = sorted(predictions["model_name"].unique())
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc_nr",
        "predicted_nr_rate",
    ]

    prepared: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        frame = predictions[
            predictions["model_name"].eq(model_name)
        ].reset_index(drop=True)
        prepared[model_name] = {
            "frame": frame,
            "participant_ids": frame[
                "participant_id"
            ].astype(str).to_numpy(),
        }

    rows = []
    for replicate in range(n_bootstrap):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        ).astype(str)
        counts = Counter(sampled.tolist())

        for model_name in model_names:
            item = prepared[model_name]
            frame = item["frame"]
            weights = np.fromiter(
                (
                    counts.get(participant_id, 0)
                    for participant_id in item["participant_ids"]
                ),
                dtype=float,
                count=len(frame),
            )
            metrics = compute_metrics(
                frame["label_nr"].to_numpy(dtype=int),
                frame["prediction"].to_numpy(dtype=int),
                frame["score"].to_numpy(dtype=float),
                sample_weight=weights,
            )
            rows.append(
                {
                    "replicate": replicate,
                    "model_name": model_name,
                    **{
                        metric: metrics[metric]
                        for metric in metric_names
                    },
                }
            )

    replicates = pd.DataFrame(rows)
    ci_rows = []
    for model_name, group in replicates.groupby(
        "model_name", observed=True
    ):
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=float)
            ci_rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "bootstrap_mean": float(np.nanmean(values)),
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    confidence_intervals = pd.DataFrame(ci_rows)

    delta_rows = []
    compact_name = "soft_compact_distribution_membership"
    full_name = "soft_full_all"
    if {compact_name, full_name}.issubset(model_names):
        compact = replicates[
            replicates["model_name"].eq(compact_name)
        ].set_index("replicate")
        full = replicates[
            replicates["model_name"].eq(full_name)
        ].set_index("replicate")
        common = compact.index.intersection(full.index)
        for metric in metric_names:
            delta = (
                compact.loc[common, metric]
                - full.loc[common, metric]
            ).to_numpy(dtype=float)
            delta_rows.append(
                {
                    "reference": compact_name,
                    "comparison": full_name,
                    "metric": metric,
                    "delta_mean": float(np.nanmean(delta)),
                    "ci_low": float(np.nanquantile(delta, 0.025)),
                    "ci_high": float(np.nanquantile(delta, 0.975)),
                    "probability_delta_gt_0": float(
                        np.nanmean(delta > 0)
                    ),
                    "n_bootstrap": n_bootstrap,
                }
            )
    deltas = pd.DataFrame(delta_rows)
    return replicates, confidence_intervals, deltas


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compact_raw = pd.read_csv(
        require_file(args.compact_features),
        low_memory=False,
    )
    full_raw = pd.read_csv(
        require_file(args.full_features),
        low_memory=False,
    )
    compact = normalize_external_table(
        compact_raw, "compact features"
    )
    full = normalize_external_table(full_raw, "full features")

    compact_ids = set(compact["external_trial_id"])
    full_ids = set(full["external_trial_id"])
    if compact_ids != full_ids:
        raise ValueError(
            "Compact/full WebQAmGaze trial sets differ: "
            f"compact_only={len(compact_ids-full_ids)}, "
            f"full_only={len(full_ids-compact_ids)}"
        )

    metadata_columns = [
        "external_trial_id",
        "participant_id",
        "task_canonical",
        "label_nr",
    ]
    metadata_check = compact[metadata_columns].merge(
        full[metadata_columns],
        on="external_trial_id",
        suffixes=("_compact", "_full"),
        validate="one_to_one",
    )
    for column in (
        "participant_id",
        "task_canonical",
        "label_nr",
    ):
        mismatch = metadata_check[
            f"{column}_compact"
        ].astype(str).ne(
            metadata_check[f"{column}_full"].astype(str)
        )
        if mismatch.any():
            raise ValueError(
                f"Compact/full metadata mismatch for {column}."
            )

    tables = {
        "compact": compact,
        "full": full,
    }

    print("=== ONESTOP -> WEBQAMGAZE ZERO-SHOT INPUT ===")
    print(f"trials: {len(compact):,}")
    print(
        f"participants: "
        f"{compact['participant_id'].nunique()}"
    )
    print(
        "task counts: "
        + json.dumps(
            compact["task_canonical"].value_counts().to_dict()
        )
    )
    print(
        "WebQAmGaze labels used before prediction: False"
    )
    print(
        "Frozen OneStop preprocessing and threshold used: True"
    )

    prediction_parts = []
    pooled_rows = []
    oracle_rows = []
    model_audits = []

    for model_name, specification in MODEL_SPECS.items():
        artifact_path = require_file(
            args.frozen_dir / specification["artifact"]
        )
        artifact = joblib.load(artifact_path)
        (
            pipeline,
            feature_columns,
            threshold,
            artifact_audit,
        ) = validate_artifact(artifact, model_name)

        table = tables[specification["input"]]
        missing_features = sorted(
            set(feature_columns) - set(table.columns)
        )
        if missing_features:
            raise ValueError(
                f"{model_name}: external table lacks features "
                f"{missing_features}"
            )

        x = table[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = decision_score_for_nr(pipeline, x)
        if not np.isfinite(score).all():
            raise ValueError(
                f"{model_name}: prediction scores contain non-finite values."
            )

        prediction = (score >= threshold).astype(int)
        result = pd.DataFrame(
            {
                "external_trial_id": table[
                    "external_trial_id"
                ].astype(str).to_numpy(),
                "participant_id": table[
                    "participant_id"
                ].astype(str).to_numpy(),
                "task": table[
                    "task_canonical"
                ].astype(str).to_numpy(),
                "label_nr": table[
                    "label_nr"
                ].to_numpy(dtype=int),
                "model_name": model_name,
                "score": score,
                "prediction": prediction,
                "frozen_threshold": threshold,
            }
        )
        prediction_parts.append(result)

        metrics = compute_metrics(
            result["label_nr"].to_numpy(dtype=int),
            result["prediction"].to_numpy(dtype=int),
            result["score"].to_numpy(dtype=float),
        )
        pooled_rows.append(
            {
                "model_name": model_name,
                "n_trials": len(result),
                "n_participants": result[
                    "participant_id"
                ].nunique(),
                "threshold_source": "frozen_onestop",
                "decision_threshold": threshold,
                **metrics,
            }
        )

        oracle_threshold, oracle_metrics = (
            oracle_threshold_diagnostic(
                result["label_nr"].to_numpy(dtype=int),
                result["score"].to_numpy(dtype=float),
                args.threshold_grid_size,
            )
        )
        oracle_rows.append(
            {
                "model_name": model_name,
                "n_trials": len(result),
                "threshold_source": (
                    "webqamgaze_label_oracle_diagnostic_not_zero_shot"
                ),
                "decision_threshold": oracle_threshold,
                **oracle_metrics,
            }
        )

        model_audits.append(
            {
                "model_name": model_name,
                "artifact_path": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "input_table": specification["input"],
                "input_path": str(
                    args.compact_features
                    if specification["input"] == "compact"
                    else args.full_features
                ),
                "feature_columns": feature_columns,
                **artifact_audit,
            }
        )

        print(
            f"\n[{model_name}] features={len(feature_columns)}, "
            f"threshold={threshold:.6f}"
        )
        print(
            f"  macroF1={metrics['macro_f1']:.4f}, "
            f"BA={metrics['balanced_accuracy']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC(NR)={metrics['pr_auc_nr']:.4f}, "
            f"pred_NR={metrics['predicted_nr_rate']:.4f}"
        )

    predictions = pd.concat(
        prediction_parts, ignore_index=True
    )
    pooled = pd.DataFrame(pooled_rows)
    oracle = pd.DataFrame(oracle_rows)
    by_subject = subject_metrics(predictions)
    subject_summary = summarize_subject_metrics(by_subject)
    (
        bootstrap_replicates,
        bootstrap_summary,
        bootstrap_deltas,
    ) = cluster_bootstrap(
        predictions=predictions,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    predictions_path = (
        args.output_dir / "external_predictions.csv"
    )
    pooled_path = (
        args.output_dir / "external_pooled_metrics.csv"
    )
    subject_metrics_path = (
        args.output_dir / "external_subject_metrics.csv"
    )
    subject_summary_path = (
        args.output_dir / "external_subject_summary.csv"
    )
    bootstrap_replicates_path = (
        args.output_dir / "external_bootstrap_replicates.csv"
    )
    bootstrap_summary_path = (
        args.output_dir / "external_bootstrap_summary.csv"
    )
    bootstrap_deltas_path = (
        args.output_dir
        / "compact_vs_full_bootstrap_differences.csv"
    )
    oracle_path = (
        args.output_dir / "target_oracle_threshold_diagnostics.csv"
    )
    audit_path = (
        args.output_dir
        / "onestop_to_webqamgaze_zero_shot_manifest.json"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        pooled_path, index=False, encoding="utf-8-sig"
    )
    by_subject.to_csv(
        subject_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    subject_summary.to_csv(
        subject_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_replicates.to_csv(
        bootstrap_replicates_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        bootstrap_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_deltas.to_csv(
        bootstrap_deltas_path,
        index=False,
        encoding="utf-8-sig",
    )
    oracle.to_csv(
        oracle_path, index=False, encoding="utf-8-sig"
    )

    manifest = {
        "analysis": "OneStop to WebQAmGaze frozen zero-shot validation",
        "label_mapping": {
            "OneStop_IS": 0,
            "OneStop_NR": 1,
            "WebQAmGaze_IS": 0,
            "WebQAmGaze_NR": 1,
        },
        "zero_shot_constraints": {
            "external_model_fitting": False,
            "external_preprocessing_fit": False,
            "external_feature_selection": False,
            "external_threshold_tuning": False,
            "external_calibration": False,
            "frozen_onestop_threshold": True,
        },
        "oracle_warning": (
            "The oracle threshold table uses WebQAmGaze labels and is "
            "diagnostic only. It is not zero-shot performance."
        ),
        "n_trials": int(len(compact)),
        "n_participants": int(
            compact["participant_id"].nunique()
        ),
        "task_counts": {
            str(key): int(value)
            for key, value in compact[
                "task_canonical"
            ].value_counts().items()
        },
        "model_audits": model_audits,
        "bootstrap": {
            "replicates": args.bootstrap,
            "cluster_unit": "participant_id",
            "paired_models": True,
        },
        "input_hashes": {
            "compact_features": sha256_file(
                args.compact_features
            ),
            "full_features": sha256_file(
                args.full_features
            ),
        },
        "outputs": {
            "predictions": str(predictions_path),
            "pooled_metrics": str(pooled_path),
            "subject_metrics": str(subject_metrics_path),
            "subject_summary": str(subject_summary_path),
            "bootstrap_replicates": str(
                bootstrap_replicates_path
            ),
            "bootstrap_summary": str(
                bootstrap_summary_path
            ),
            "bootstrap_differences": str(
                bootstrap_deltas_path
            ),
            "oracle_diagnostics": str(oracle_path),
        },
    }
    audit_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== ONESTOP -> WEBQAMGAZE FROZEN ZERO-SHOT "
        "POOLED METRICS ==="
    )
    with pd.option_context(
        "display.max_columns", None, "display.width", 240
    ):
        print(pooled.to_string(index=False))

    print(
        "\n=== ONESTOP -> WEBQAMGAZE SUBJECT-MEAN "
        "METRICS ==="
    )
    with pd.option_context(
        "display.max_columns", None, "display.width", 260
    ):
        print(subject_summary.to_string(index=False))

    print(
        "\n=== TARGET-ORACLE THRESHOLD DIAGNOSTIC "
        "(NOT ZERO-SHOT) ==="
    )
    with pd.option_context(
        "display.max_columns", None, "display.width", 240
    ):
        print(oracle.to_string(index=False))

    print(
        "\n=== COMPACT VS FULL PAIRED BOOTSTRAP "
        "DIFFERENCES ==="
    )
    with pd.option_context(
        "display.max_columns", None, "display.width", 240
    ):
        print(bootstrap_deltas.to_string(index=False))

    print(f"\nSaved: {predictions_path}")
    print(f"Saved: {pooled_path}")
    print(f"Saved: {subject_summary_path}")
    print(f"Saved: {bootstrap_summary_path}")
    print(f"Saved: {oracle_path}")
    print(f"Saved: {audit_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
