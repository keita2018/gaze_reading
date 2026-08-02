#!/usr/bin/env python
r"""
Evaluate label-free participant self-calibration on WebQAmGaze scores.

Inputs
------
1. OneStop participant-grouped OOF predictions used to freeze the source
   models. These provide a source-domain score-location reference.
2. Frozen OneStop -> WebQAmGaze zero-shot predictions.

Calibration methods
-------------------
raw_frozen
    Original frozen score and frozen threshold.

source_median_loo
    For every target trial, subtract the median score of the other trials
    from the same participant and add the median of source-participant median
    scores. The target trial's label and score are not used to estimate its
    own participant center.

source_mean_loo
    Mean-based sensitivity analysis using the same leave-one-trial-out rule.

threshold_median_loo
    Design-balanced sensitivity analysis. Subtract the target participant's
    leave-one-out median and place that median at the frozen decision
    threshold. This uses no target labels, but it assumes the participant's
    session is approximately balanced around the decision boundary. It must
    not be presented as the primary deployment setting.

The primary comparison is source_median_loo versus raw_frozen. All target
labels are used only after calibrated scores and predictions have been formed.

Example
-------
python -u src\models\run_webqamgaze_participant_self_calibration.py ^
  --source-oof outputs\frozen_onestop_external_validation_v1\onestop_selected_oof_predictions.parquet ^
  --target-predictions outputs\onestop_to_webqamgaze_zero_shot_v1\external_predictions.csv ^
  --output-dir outputs\webqamgaze_participant_self_calibration_v1 ^
  --bootstrap 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


METHOD_ORDER = (
    "raw_frozen",
    "source_median_loo",
    "source_mean_loo",
    "threshold_median_loo",
)
PRIMARY_METHOD = "source_median_loo"
REFERENCE_METHOD = "raw_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument(
        "--target-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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
        total = float(len(y_true))
        true_nr_rate = float(np.mean(y_true == 1))
        predicted_nr_rate = float(np.mean(prediction == 1))
    else:
        total = float(np.sum(sample_weight))
        true_nr_rate = float(
            np.sum(sample_weight * (y_true == 1)) / total
        )
        predicted_nr_rate = float(
            np.sum(sample_weight * (prediction == 1)) / total
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
        "true_nr_rate": true_nr_rate,
        "predicted_nr_rate": predicted_nr_rate,
    }


def validate_source(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "participant_id",
        "feature_set",
        "score",
        "decision_threshold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Source OOF table lacks columns: {missing}"
        )

    result = frame.copy()
    result["participant_id"] = (
        result["participant_id"].astype(str).str.strip()
    )
    result["feature_set"] = (
        result["feature_set"].astype(str).str.strip()
    )
    result["score"] = pd.to_numeric(
        result["score"], errors="raise"
    ).astype(float)
    result["decision_threshold"] = pd.to_numeric(
        result["decision_threshold"], errors="raise"
    ).astype(float)

    if not np.isfinite(result["score"]).all():
        raise ValueError("Source scores contain non-finite values.")
    if not np.isfinite(
        result["decision_threshold"]
    ).all():
        raise ValueError(
            "Source thresholds contain non-finite values."
        )
    return result


def validate_target(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "external_trial_id",
        "participant_id",
        "label_nr",
        "model_name",
        "score",
        "frozen_threshold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Target prediction table lacks columns: {missing}"
        )

    result = frame.copy()
    result["external_trial_id"] = (
        result["external_trial_id"].astype(str).str.strip()
    )
    result["participant_id"] = (
        result["participant_id"].astype(str).str.strip()
    )
    result["model_name"] = (
        result["model_name"].astype(str).str.strip()
    )
    result["label_nr"] = pd.to_numeric(
        result["label_nr"], errors="raise"
    ).astype(int)
    result["score"] = pd.to_numeric(
        result["score"], errors="raise"
    ).astype(float)
    result["frozen_threshold"] = pd.to_numeric(
        result["frozen_threshold"], errors="raise"
    ).astype(float)

    if set(result["label_nr"].unique()) != {0, 1}:
        raise ValueError("Target labels must be {0,1}.")
    if not np.isfinite(result["score"]).all():
        raise ValueError("Target scores contain non-finite values.")
    if result.duplicated(
        ["model_name", "external_trial_id"]
    ).any():
        raise ValueError("Duplicate target model/trial rows.")

    counts = result.groupby(
        ["model_name", "participant_id"],
        observed=True,
    ).size()
    if counts.lt(3).any():
        bad = counts[counts.lt(3)]
        raise ValueError(
            "At least three target trials per participant/model "
            "are required for LOO calibration:\n"
            + bad.head(20).to_string()
        )

    label_counts = (
        result.groupby(
            ["model_name", "participant_id"],
            observed=True,
        )["label_nr"]
        .nunique()
    )
    if label_counts.lt(2).any():
        bad = label_counts[label_counts.lt(2)]
        raise ValueError(
            "Every target participant/model must contain both labels "
            "for participant-level evaluation:\n"
            + bad.head(20).to_string()
        )
    return result


def source_references(
    source: pd.DataFrame,
) -> pd.DataFrame:
    participant = (
        source.groupby(
            ["feature_set", "participant_id"],
            observed=True,
        )
        .agg(
            participant_median_score=("score", "median"),
            participant_mean_score=("score", "mean"),
            trials=("score", "size"),
        )
        .reset_index()
    )

    rows = []
    for feature_set, group in participant.groupby(
        "feature_set",
        observed=True,
    ):
        thresholds = source.loc[
            source["feature_set"].eq(feature_set),
            "decision_threshold",
        ].unique()
        if len(thresholds) != 1:
            raise ValueError(
                f"{feature_set}: expected one source threshold, "
                f"got {thresholds.tolist()}"
            )
        rows.append(
            {
                "model_name": feature_set,
                "source_participants": group[
                    "participant_id"
                ].nunique(),
                "source_trials": int(group["trials"].sum()),
                "source_reference_median": float(
                    group["participant_median_score"].median()
                ),
                "source_reference_mean": float(
                    group["participant_mean_score"].mean()
                ),
                "source_threshold": float(thresholds[0]),
                "source_participant_median_iqr": float(
                    group["participant_median_score"].quantile(0.75)
                    - group["participant_median_score"].quantile(0.25)
                ),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_centers(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    if n < 3:
        raise ValueError("Need at least three values.")

    total = float(values.sum())
    loo_mean = (total - values) / float(n - 1)

    loo_median = np.empty(n, dtype=float)
    for index in range(n):
        loo_median[index] = float(
            np.median(np.delete(values, index))
        )
    return loo_median, loo_mean


def calibrate(
    target: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_lookup = references.set_index(
        "model_name",
        verify_integrity=True,
    )
    unknown_models = sorted(
        set(target["model_name"])
        - set(reference_lookup.index)
    )
    if unknown_models:
        raise ValueError(
            f"No source reference for target models: {unknown_models}"
        )

    calibrated_parts = []
    participant_audits = []

    for (model_name, participant_id), group in target.groupby(
        ["model_name", "participant_id"],
        observed=True,
        sort=False,
    ):
        group = group.copy().reset_index(drop=True)
        scores = group["score"].to_numpy(dtype=float)
        loo_median, loo_mean = leave_one_out_centers(scores)

        reference = reference_lookup.loc[model_name]
        frozen_thresholds = group[
            "frozen_threshold"
        ].unique()
        if len(frozen_thresholds) != 1:
            raise ValueError(
                f"{model_name}/{participant_id}: multiple target "
                f"thresholds={frozen_thresholds.tolist()}"
            )
        frozen_threshold = float(frozen_thresholds[0])
        if not math.isclose(
            frozen_threshold,
            float(reference["source_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"{model_name}: source/target frozen threshold mismatch "
                f"{reference['source_threshold']} vs {frozen_threshold}"
            )

        method_scores = {
            "raw_frozen": scores,
            "source_median_loo": (
                scores
                - loo_median
                + float(reference["source_reference_median"])
            ),
            "source_mean_loo": (
                scores
                - loo_mean
                + float(reference["source_reference_mean"])
            ),
            "threshold_median_loo": (
                scores
                - loo_median
                + frozen_threshold
            ),
        }

        for method in METHOD_ORDER:
            calibrated_score = np.asarray(
                method_scores[method],
                dtype=float,
            )
            prediction = (
                calibrated_score >= frozen_threshold
            ).astype(int)
            part = group[
                [
                    "external_trial_id",
                    "participant_id",
                    "label_nr",
                    "model_name",
                ]
            ].copy()
            if "task" in group.columns:
                part["task"] = group["task"].astype(str)
            part["calibration_method"] = method
            part["raw_score"] = scores
            part["calibrated_score"] = calibrated_score
            part["frozen_threshold"] = frozen_threshold
            part["prediction"] = prediction
            part["calibration_shift"] = (
                calibrated_score - scores
            )
            calibrated_parts.append(part)

        participant_audits.append(
            {
                "model_name": model_name,
                "participant_id": participant_id,
                "n_trials": len(group),
                "target_score_median": float(
                    np.median(scores)
                ),
                "target_score_mean": float(np.mean(scores)),
                "source_reference_median": float(
                    reference["source_reference_median"]
                ),
                "source_reference_mean": float(
                    reference["source_reference_mean"]
                ),
                "frozen_threshold": frozen_threshold,
                "median_shift_to_source_reference": float(
                    reference["source_reference_median"]
                    - np.median(scores)
                ),
                "mean_shift_to_source_reference": float(
                    reference["source_reference_mean"]
                    - np.mean(scores)
                ),
            }
        )

    calibrated = pd.concat(
        calibrated_parts,
        ignore_index=True,
    )
    audit = pd.DataFrame(participant_audits)
    return calibrated, audit


def pooled_metrics(
    calibrated: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (model_name, method), group in calibrated.groupby(
        ["model_name", "calibration_method"],
        observed=True,
    ):
        result = metrics(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["calibrated_score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": model_name,
                "calibration_method": method,
                "n_trials": len(group),
                "n_participants": group[
                    "participant_id"
                ].nunique(),
                **result,
                "prediction_change_rate_vs_raw": float(
                    np.mean(
                        group["prediction"].to_numpy(dtype=int)
                        != (
                            group["raw_score"].to_numpy(dtype=float)
                            >= group["frozen_threshold"].to_numpy(
                                dtype=float
                            )
                        ).astype(int)
                    )
                ),
                "median_absolute_shift": float(
                    np.median(
                        np.abs(
                            group["calibration_shift"].to_numpy(
                                dtype=float
                            )
                        )
                    )
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["calibration_method"] = pd.Categorical(
        result["calibration_method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    return result.sort_values(
        ["model_name", "calibration_method"]
    ).reset_index(drop=True)


def participant_metrics(
    calibrated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (
        model_name,
        method,
        participant_id,
    ), group in calibrated.groupby(
        [
            "model_name",
            "calibration_method",
            "participant_id",
        ],
        observed=True,
    ):
        result = metrics(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["calibrated_score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": model_name,
                "calibration_method": method,
                "participant_id": participant_id,
                "n_trials": len(group),
                **result,
            }
        )
    per_participant = pd.DataFrame(rows)

    summary_rows = []
    metric_columns = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc_nr",
        "predicted_nr_rate",
    )
    for (model_name, method), group in per_participant.groupby(
        ["model_name", "calibration_method"],
        observed=True,
    ):
        row: dict[str, Any] = {
            "model_name": model_name,
            "calibration_method": method,
            "participants": group["participant_id"].nunique(),
        }
        for metric_name in metric_columns:
            values = group[metric_name].to_numpy(dtype=float)
            row[f"{metric_name}_subject_mean"] = float(
                np.nanmean(values)
            )
            row[f"{metric_name}_subject_median"] = float(
                np.nanmedian(values)
            )
        summary_rows.append(row)
    return per_participant, pd.DataFrame(summary_rows)


def participant_cluster_bootstrap(
    calibrated: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    participants = np.asarray(
        sorted(calibrated["participant_id"].unique()),
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

    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    for (model_name, method), group in calibrated.groupby(
        ["model_name", "calibration_method"],
        observed=True,
    ):
        frame = group.reset_index(drop=True)
        prepared[(model_name, method)] = {
            "frame": frame,
            "participant_ids": frame[
                "participant_id"
            ].astype(str).to_numpy(),
        }

    replicate_rows = []
    for replicate in range(n_bootstrap):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        ).astype(str)
        counts = Counter(sampled.tolist())

        for (model_name, method), item in prepared.items():
            frame = item["frame"]
            weights = np.fromiter(
                (
                    counts.get(participant_id, 0)
                    for participant_id in item[
                        "participant_ids"
                    ]
                ),
                dtype=float,
                count=len(frame),
            )
            result = metrics(
                frame["label_nr"].to_numpy(dtype=int),
                frame["prediction"].to_numpy(dtype=int),
                frame["calibrated_score"].to_numpy(dtype=float),
                sample_weight=weights,
            )
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "model_name": model_name,
                    "calibration_method": method,
                    **{
                        metric_name: result[metric_name]
                        for metric_name in metric_names
                    },
                }
            )

    replicates = pd.DataFrame(replicate_rows)
    delta_rows = []
    comparisons = (
        PRIMARY_METHOD,
        "source_mean_loo",
        "threshold_median_loo",
    )

    for model_name, model_frame in replicates.groupby(
        "model_name",
        observed=True,
    ):
        reference = model_frame[
            model_frame["calibration_method"].eq(
                REFERENCE_METHOD
            )
        ].set_index("replicate")

        for method in comparisons:
            comparison = model_frame[
                model_frame["calibration_method"].eq(method)
            ].set_index("replicate")
            common = reference.index.intersection(
                comparison.index
            )
            for metric_name in metric_names:
                delta = (
                    comparison.loc[common, metric_name]
                    - reference.loc[common, metric_name]
                ).to_numpy(dtype=float)
                delta_rows.append(
                    {
                        "model_name": model_name,
                        "reference": REFERENCE_METHOD,
                        "comparison": method,
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

    return replicates, pd.DataFrame(delta_rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = validate_source(
        pd.read_parquet(require_file(args.source_oof))
    )
    target = validate_target(
        pd.read_csv(
            require_file(args.target_predictions),
            low_memory=False,
        )
    )
    references = source_references(source)

    print(
        "=== WEBQAMGAZE PARTICIPANT SELF-CALIBRATION INPUT ==="
    )
    print(f"source rows: {len(source):,}")
    print(
        f"source participants: "
        f"{source['participant_id'].nunique()}"
    )
    print(f"target rows: {len(target):,}")
    print(
        f"target participants: "
        f"{target['participant_id'].nunique()}"
    )
    print(
        "target labels used for calibration: False"
    )
    print(
        "primary calibration: source_median_loo"
    )

    print("\n=== SOURCE SCORE REFERENCES ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(references.to_string(index=False))

    calibrated, participant_audit = calibrate(
        target,
        references,
    )
    pooled = pooled_metrics(calibrated)
    (
        per_participant,
        participant_summary,
    ) = participant_metrics(calibrated)
    (
        bootstrap_replicates,
        bootstrap_deltas,
    ) = participant_cluster_bootstrap(
        calibrated,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    calibrated_path = (
        args.output_dir
        / "self_calibrated_trial_predictions.csv"
    )
    participant_audit_path = (
        args.output_dir
        / "self_calibration_participant_audit.csv"
    )
    references_path = (
        args.output_dir
        / "source_score_references.csv"
    )
    pooled_path = (
        args.output_dir
        / "self_calibration_pooled_metrics.csv"
    )
    per_participant_path = (
        args.output_dir
        / "self_calibration_participant_metrics.csv"
    )
    participant_summary_path = (
        args.output_dir
        / "self_calibration_participant_summary.csv"
    )
    bootstrap_replicates_path = (
        args.output_dir
        / "self_calibration_bootstrap_replicates.csv"
    )
    bootstrap_deltas_path = (
        args.output_dir
        / "self_calibration_bootstrap_deltas.csv"
    )
    manifest_path = (
        args.output_dir
        / "self_calibration_manifest.json"
    )

    calibrated.to_csv(
        calibrated_path,
        index=False,
        encoding="utf-8-sig",
    )
    participant_audit.to_csv(
        participant_audit_path,
        index=False,
        encoding="utf-8-sig",
    )
    references.to_csv(
        references_path,
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        pooled_path,
        index=False,
        encoding="utf-8-sig",
    )
    per_participant.to_csv(
        per_participant_path,
        index=False,
        encoding="utf-8-sig",
    )
    participant_summary.to_csv(
        participant_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_replicates.to_csv(
        bootstrap_replicates_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_deltas.to_csv(
        bootstrap_deltas_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "analysis": (
            "Label-free participant self-calibration on frozen "
            "OneStop-to-WebQAmGaze scores"
        ),
        "primary_method": PRIMARY_METHOD,
        "reference_method": REFERENCE_METHOD,
        "methods": {
            "raw_frozen": (
                "Original frozen score and threshold."
            ),
            "source_median_loo": (
                "Subtract target participant leave-one-out median and "
                "add median of source-participant median OOF scores."
            ),
            "source_mean_loo": (
                "Mean-based leave-one-out sensitivity analysis."
            ),
            "threshold_median_loo": (
                "Design-balanced sensitivity analysis placing the "
                "target participant median at the frozen threshold."
            ),
        },
        "target_labels_used_for_calibration": False,
        "target_trial_excluded_from_own_center": True,
        "source_reference_uses_oof_scores": True,
        "bootstrap": {
            "replicates": args.bootstrap,
            "unit": "participant_id",
            "paired_methods": True,
        },
        "caveat": (
            "threshold_median_loo assumes a session approximately "
            "balanced around the decision boundary and is secondary."
        ),
        "outputs": {
            "trial_predictions": str(calibrated_path),
            "participant_audit": str(
                participant_audit_path
            ),
            "source_references": str(references_path),
            "pooled_metrics": str(pooled_path),
            "participant_metrics": str(
                per_participant_path
            ),
            "participant_summary": str(
                participant_summary_path
            ),
            "bootstrap_replicates": str(
                bootstrap_replicates_path
            ),
            "bootstrap_deltas": str(
                bootstrap_deltas_path
            ),
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
        "\n=== PARTICIPANT SELF-CALIBRATION POOLED METRICS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
    ):
        print(pooled.to_string(index=False))

    print(
        "\n=== PARTICIPANT SELF-CALIBRATION SUBJECT-MEAN "
        "METRICS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        280,
    ):
        print(
            participant_summary.to_string(index=False)
        )

    print(
        "\n=== PARTICIPANT SELF-CALIBRATION "
        "BOOTSTRAP DELTAS VS RAW ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
    ):
        print(bootstrap_deltas.to_string(index=False))

    print(f"\nSaved: {pooled_path}")
    print(f"Saved: {participant_summary_path}")
    print(f"Saved: {bootstrap_deltas_path}")
    print(f"Saved: {manifest_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
