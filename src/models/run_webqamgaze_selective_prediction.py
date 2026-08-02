#!/usr/bin/env python
r"""
Evaluate label-free selective prediction after participant self-calibration.

Selection never uses WebQAmGaze labels. At each requested coverage, trials are
ranked only by confidence:

    confidence = abs(calibrated_score - frozen_threshold)

Two reporting modes are produced:
1. global:
   Retain the most confident trials across the full target set.
2. participant_balanced:
   Retain the most confident requested fraction within every participant, so
   selective coverage is not concentrated in a small subset of participants.

The primary comparison is:
    soft_full_all / source_median_loo
versus
    soft_full_all / raw_frozen

A participant-cluster bootstrap is used for the primary global comparison.

Example
-------
python -u src\models\run_webqamgaze_selective_prediction.py ^
  --predictions outputs\webqamgaze_participant_self_calibration_v1\self_calibrated_trial_predictions.csv ^
  --output-dir outputs\webqamgaze_selective_prediction_v1 ^
  --coverages 1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1 ^
  --bootstrap 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
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


PRIMARY_MODEL = "soft_full_all"
PRIMARY_METHOD = "source_median_loo"
REFERENCE_METHOD = "raw_frozen"
SELECTION_MODES = ("global", "participant_balanced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--coverages",
        default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_coverages(text: str) -> list[float]:
    values = sorted(
        {
            float(token.strip())
            for token in text.split(",")
            if token.strip()
        },
        reverse=True,
    )
    if not values or any(
        value <= 0.0 or value > 1.0 for value in values
    ):
        raise ValueError("Coverages must lie in (0, 1].")
    if 1.0 not in values:
        values.insert(0, 1.0)
    return values


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "external_trial_id",
        "participant_id",
        "label_nr",
        "model_name",
        "calibration_method",
        "calibrated_score",
        "frozen_threshold",
        "prediction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction table lacks columns: {missing}")

    result = frame.copy()
    for column in (
        "external_trial_id",
        "participant_id",
        "model_name",
        "calibration_method",
    ):
        result[column] = result[column].astype(str).str.strip()

    result["label_nr"] = pd.to_numeric(
        result["label_nr"], errors="raise"
    ).astype(int)
    result["prediction"] = pd.to_numeric(
        result["prediction"], errors="raise"
    ).astype(int)
    result["calibrated_score"] = pd.to_numeric(
        result["calibrated_score"], errors="raise"
    ).astype(float)
    result["frozen_threshold"] = pd.to_numeric(
        result["frozen_threshold"], errors="raise"
    ).astype(float)

    if set(result["label_nr"].unique()) != {0, 1}:
        raise ValueError("Expected target labels {0,1}.")
    if not result["prediction"].isin([0, 1]).all():
        raise ValueError("Predictions must be binary.")
    if not np.isfinite(
        result[
            ["calibrated_score", "frozen_threshold"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Scores or thresholds are non-finite.")
    if result.duplicated(
        [
            "model_name",
            "calibration_method",
            "external_trial_id",
        ]
    ).any():
        raise ValueError("Duplicate model/method/trial rows.")

    recomputed = (
        result["calibrated_score"].to_numpy(dtype=float)
        >= result["frozen_threshold"].to_numpy(dtype=float)
    ).astype(int)
    if not np.array_equal(
        recomputed,
        result["prediction"].to_numpy(dtype=int),
    ):
        raise ValueError(
            "Stored predictions disagree with calibrated score/threshold."
        )

    result["confidence"] = np.abs(
        result["calibrated_score"]
        - result["frozen_threshold"]
    )
    result["correct"] = (
        result["prediction"] == result["label_nr"]
    ).astype(int)

    required_pairs = {
        (PRIMARY_MODEL, PRIMARY_METHOD),
        (PRIMARY_MODEL, REFERENCE_METHOD),
    }
    observed_pairs = set(
        zip(
            result["model_name"],
            result["calibration_method"],
        )
    )
    missing_pairs = sorted(required_pairs - observed_pairs)
    if missing_pairs:
        raise ValueError(
            f"Primary comparison rows missing: {missing_pairs}"
        )
    return result


def safe_auc(
    y_true: np.ndarray,
    score: np.ndarray,
) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_ap(
    y_true: np.ndarray,
    score: np.ndarray,
) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["label_nr"].to_numpy(dtype=int)
    prediction = frame["prediction"].to_numpy(dtype=int)
    score = frame["calibrated_score"].to_numpy(dtype=float)
    accuracy = float(accuracy_score(y_true, prediction))
    return {
        "accuracy": accuracy,
        "selective_risk": 1.0 - accuracy,
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": safe_auc(y_true, score),
        "pr_auc_nr": safe_ap(y_true, score),
        "true_nr_rate": float(np.mean(y_true == 1)),
        "predicted_nr_rate": float(
            np.mean(prediction == 1)
        ),
        "mean_confidence": float(
            frame["confidence"].mean()
        ),
        "median_confidence": float(
            frame["confidence"].median()
        ),
    }


def select_global(
    frame: pd.DataFrame,
    coverage: float,
) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["confidence", "external_trial_id"],
        ascending=[False, True],
        kind="stable",
    )
    keep = max(1, int(math.ceil(coverage * len(ordered))))
    return ordered.iloc[:keep].copy()


def select_participant_balanced(
    frame: pd.DataFrame,
    coverage: float,
) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby(
        "participant_id",
        observed=True,
        sort=False,
    ):
        ordered = group.sort_values(
            ["confidence", "external_trial_id"],
            ascending=[False, True],
            kind="stable",
        )
        keep = max(
            1,
            int(math.ceil(coverage * len(ordered))),
        )
        parts.append(ordered.iloc[:keep])
    return pd.concat(parts, ignore_index=True)


def select_rows(
    frame: pd.DataFrame,
    coverage: float,
    mode: str,
) -> pd.DataFrame:
    if mode == "global":
        return select_global(frame, coverage)
    if mode == "participant_balanced":
        return select_participant_balanced(
            frame, coverage
        )
    raise ValueError(mode)


def selective_metrics(
    frame: pd.DataFrame,
    coverages: list[float],
) -> pd.DataFrame:
    rows = []
    for (
        model_name,
        calibration_method,
    ), group in frame.groupby(
        ["model_name", "calibration_method"],
        observed=True,
    ):
        for mode in SELECTION_MODES:
            for target_coverage in coverages:
                retained = select_rows(
                    group,
                    target_coverage,
                    mode,
                )
                result = compute_metrics(retained)
                rows.append(
                    {
                        "model_name": model_name,
                        "calibration_method": (
                            calibration_method
                        ),
                        "selection_mode": mode,
                        "target_coverage": target_coverage,
                        "actual_coverage": (
                            len(retained) / len(group)
                        ),
                        "n_retained": len(retained),
                        "n_total": len(group),
                        "participants_retained": retained[
                            "participant_id"
                        ].nunique(),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def aurc_statistics(frame: pd.DataFrame) -> dict[str, float]:
    ordered = frame.sort_values(
        ["confidence", "external_trial_id"],
        ascending=[False, True],
        kind="stable",
    )
    errors = (
        1
        - ordered["correct"].to_numpy(dtype=int)
    ).astype(float)
    cumulative_risk = np.cumsum(errors) / np.arange(
        1, len(errors) + 1
    )
    aurc = float(np.mean(cumulative_risk))

    total_errors = int(errors.sum())
    oracle_errors = np.concatenate(
        [
            np.zeros(len(errors) - total_errors),
            np.ones(total_errors),
        ]
    )
    oracle_cumulative_risk = (
        np.cumsum(oracle_errors)
        / np.arange(1, len(errors) + 1)
    )
    oracle_aurc = float(
        np.mean(oracle_cumulative_risk)
    )
    return {
        "aurc_error": aurc,
        "oracle_aurc_error": oracle_aurc,
        "excess_aurc_error": aurc - oracle_aurc,
        "confidence_correct_spearman": float(
            pd.Series(
                ordered["confidence"].to_numpy(dtype=float)
            ).corr(
                pd.Series(
                    ordered["correct"].to_numpy(dtype=float)
                ),
                method="spearman",
            )
        ),
    }


def aurc_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (
        model_name,
        calibration_method,
    ), group in frame.groupby(
        ["model_name", "calibration_method"],
        observed=True,
    ):
        rows.append(
            {
                "model_name": model_name,
                "calibration_method": calibration_method,
                "n_trials": len(group),
                **aurc_statistics(group),
            }
        )
    return pd.DataFrame(rows)


def numpy_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    accuracy = float(np.mean(y_true == prediction))

    tp = float(
        np.sum((y_true == 1) & (prediction == 1))
    )
    tn = float(
        np.sum((y_true == 0) & (prediction == 0))
    )
    fp = float(
        np.sum((y_true == 0) & (prediction == 1))
    )
    fn = float(
        np.sum((y_true == 1) & (prediction == 0))
    )

    tpr = 0.0 if tp + fn <= 0 else tp / (tp + fn)
    tnr = 0.0 if tn + fp <= 0 else tn / (tn + fp)
    balanced_accuracy = 0.5 * (tpr + tnr)

    positive_denominator = 2.0 * tp + fp + fn
    negative_denominator = 2.0 * tn + fp + fn
    positive_f1 = (
        0.0
        if positive_denominator <= 0
        else 2.0 * tp / positive_denominator
    )
    negative_f1 = (
        0.0
        if negative_denominator <= 0
        else 2.0 * tn / negative_denominator
    )
    return {
        "accuracy": accuracy,
        "selective_risk": 1.0 - accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": 0.5 * (
            positive_f1 + negative_f1
        ),
    }


def bootstrap_primary(
    frame: pd.DataFrame,
    coverages: list[float],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = frame[
        frame["model_name"].eq(PRIMARY_MODEL)
        & frame["calibration_method"].isin(
            [REFERENCE_METHOD, PRIMARY_METHOD]
        )
    ].copy()

    participants = np.asarray(
        sorted(primary["participant_id"].unique()),
        dtype=object,
    )
    prepared: dict[
        str, dict[str, dict[str, np.ndarray]]
    ] = {}

    for method in (
        REFERENCE_METHOD,
        PRIMARY_METHOD,
    ):
        method_frame = primary[
            primary["calibration_method"].eq(method)
        ]
        prepared[method] = {}
        for participant, group in method_frame.groupby(
            "participant_id",
            observed=True,
        ):
            prepared[method][str(participant)] = {
                "y": group["label_nr"].to_numpy(dtype=int),
                "prediction": group[
                    "prediction"
                ].to_numpy(dtype=int),
                "confidence": group[
                    "confidence"
                ].to_numpy(dtype=float),
                "tie": group[
                    "external_trial_id"
                ].astype(str).to_numpy(),
            }

    rng = np.random.default_rng(seed)
    replicate_rows = []

    for replicate in range(n_bootstrap):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        ).astype(str)

        method_results: dict[
            str, dict[float, dict[str, float]]
        ] = {}

        for method in (
            REFERENCE_METHOD,
            PRIMARY_METHOD,
        ):
            y_parts = []
            prediction_parts = []
            confidence_parts = []
            tie_parts = []

            for instance_index, participant in enumerate(
                sampled
            ):
                item = prepared[method][participant]
                y_parts.append(item["y"])
                prediction_parts.append(
                    item["prediction"]
                )
                confidence_parts.append(
                    item["confidence"]
                )
                tie_parts.append(
                    np.asarray(
                        [
                            f"{instance_index:04d}|{value}"
                            for value in item["tie"]
                        ],
                        dtype=object,
                    )
                )

            y = np.concatenate(y_parts)
            prediction = np.concatenate(
                prediction_parts
            )
            confidence = np.concatenate(
                confidence_parts
            )
            tie = np.concatenate(tie_parts)

            order = np.lexsort((tie, -confidence))
            y = y[order]
            prediction = prediction[order]

            method_results[method] = {}
            for coverage in coverages:
                keep = max(
                    1,
                    int(math.ceil(coverage * len(y))),
                )
                result = numpy_metrics(
                    y[:keep],
                    prediction[:keep],
                )
                method_results[method][
                    coverage
                ] = result

        for coverage in coverages:
            for metric_name in (
                "accuracy",
                "selective_risk",
                "balanced_accuracy",
                "macro_f1",
            ):
                reference_value = method_results[
                    REFERENCE_METHOD
                ][coverage][metric_name]
                primary_value = method_results[
                    PRIMARY_METHOD
                ][coverage][metric_name]
                delta = (
                    primary_value - reference_value
                )
                replicate_rows.append(
                    {
                        "replicate": replicate,
                        "model_name": PRIMARY_MODEL,
                        "reference": REFERENCE_METHOD,
                        "comparison": PRIMARY_METHOD,
                        "selection_mode": "global",
                        "target_coverage": coverage,
                        "metric": metric_name,
                        "reference_value": reference_value,
                        "comparison_value": primary_value,
                        "delta": delta,
                    }
                )

    replicates = pd.DataFrame(replicate_rows)
    summary_rows = []
    for (
        coverage,
        metric_name,
    ), group in replicates.groupby(
        ["target_coverage", "metric"],
        observed=True,
    ):
        delta = group["delta"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model_name": PRIMARY_MODEL,
                "reference": REFERENCE_METHOD,
                "comparison": PRIMARY_METHOD,
                "selection_mode": "global",
                "target_coverage": coverage,
                "metric": metric_name,
                "delta_mean": float(np.mean(delta)),
                "ci_low": float(
                    np.quantile(delta, 0.025)
                ),
                "ci_high": float(
                    np.quantile(delta, 0.975)
                ),
                "probability_delta_gt_0": float(
                    np.mean(delta > 0)
                ),
                "n_bootstrap": n_bootstrap,
            }
        )
    return replicates, pd.DataFrame(summary_rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    coverages = parse_coverages(args.coverages)
    predictions = validate_predictions(
        pd.read_csv(
            require_file(args.predictions),
            low_memory=False,
        )
    )

    print(
        "=== WEBQAMGAZE SELECTIVE PREDICTION INPUT ==="
    )
    print(f"rows: {len(predictions):,}")
    print(
        f"unique target trials: "
        f"{predictions['external_trial_id'].nunique():,}"
    )
    print(
        f"participants: "
        f"{predictions['participant_id'].nunique()}"
    )
    print(f"coverages: {coverages}")
    print(
        "target labels used for selection: False"
    )
    print(
        f"primary: {PRIMARY_MODEL} / "
        f"{PRIMARY_METHOD} vs {REFERENCE_METHOD}"
    )

    metrics_table = selective_metrics(
        predictions,
        coverages,
    )
    aurc = aurc_table(predictions)
    (
        bootstrap_replicates,
        bootstrap_summary,
    ) = bootstrap_primary(
        predictions,
        coverages=coverages,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    metrics_path = (
        args.output_dir
        / "selective_prediction_metrics.csv"
    )
    aurc_path = (
        args.output_dir
        / "selective_prediction_aurc.csv"
    )
    bootstrap_replicates_path = (
        args.output_dir
        / "selective_prediction_bootstrap_replicates.csv"
    )
    bootstrap_summary_path = (
        args.output_dir
        / "selective_prediction_bootstrap_deltas.csv"
    )
    manifest_path = (
        args.output_dir
        / "selective_prediction_manifest.json"
    )

    metrics_table.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    aurc.to_csv(
        aurc_path,
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

    manifest = {
        "analysis": (
            "Label-free confidence-based selective prediction "
            "on WebQAmGaze"
        ),
        "confidence": (
            "absolute calibrated decision margin"
        ),
        "target_labels_used_for_selection": False,
        "coverage_modes": {
            "global": (
                "retain globally highest-confidence trials"
            ),
            "participant_balanced": (
                "retain highest-confidence requested fraction "
                "within every participant"
            ),
        },
        "primary_model": PRIMARY_MODEL,
        "primary_calibration": PRIMARY_METHOD,
        "reference_calibration": REFERENCE_METHOD,
        "coverages": coverages,
        "bootstrap": {
            "replicates": args.bootstrap,
            "unit": "participant",
            "primary_mode": "global",
            "paired_methods": True,
        },
        "outputs": {
            "metrics": str(metrics_path),
            "aurc": str(aurc_path),
            "bootstrap_replicates": str(
                bootstrap_replicates_path
            ),
            "bootstrap_deltas": str(
                bootstrap_summary_path
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

    primary_metrics = metrics_table[
        metrics_table["model_name"].eq(
            PRIMARY_MODEL
        )
        & metrics_table[
            "calibration_method"
        ].isin(
            [REFERENCE_METHOD, PRIMARY_METHOD]
        )
    ].copy()

    print(
        "\n=== SELECTIVE PREDICTION PRIMARY METRICS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        280,
    ):
        print(
            primary_metrics.to_string(index=False)
        )

    print(
        "\n=== SELECTIVE PREDICTION AURC ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(
            aurc[
                aurc["model_name"].eq(PRIMARY_MODEL)
                & aurc[
                    "calibration_method"
                ].isin(
                    [
                        REFERENCE_METHOD,
                        PRIMARY_METHOD,
                    ]
                )
            ].to_string(index=False)
        )

    print(
        "\n=== SELECTIVE PREDICTION PRIMARY "
        "BOOTSTRAP DELTAS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
    ):
        print(
            bootstrap_summary.to_string(index=False)
        )

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {aurc_path}")
    print(f"Saved: {bootstrap_summary_path}")
    print(f"Saved: {manifest_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
