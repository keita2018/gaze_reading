#!/usr/bin/env python
"""Diagnose whether low-rate gains are genuine or feature-rate artifacts.

This script reuses the feature tables produced by
run_zuco_downsampling_robustness.py. It does not reread Parquet files.

Two protocols are evaluated:

1. rate_matched
   Train and test at the same sampling rate.

2. train500_test_degraded
   Train on 500 Hz trials from the 17 training subjects and test the held-out
   subject at the target lower rate. This measures robustness without
   retraining for the degraded device.

Rate-invariant feature sets deliberately exclude quantities defined per sample,
such as dx, dy, displacement per observation, and adjacent pupil differences.
Those quantities change mechanically as the sampling interval changes.

Feature sets:
- dummy_reference
- time_only
- speed_only
- gaze_rate_invariant
- pupil_rate_invariant
- strict_rate_invariant
- strict_rate_dependent_reference

The last condition is the previous strict_relative set and is retained only as
a diagnostic reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_zuco_raw_gaze_loso import (
    make_model,
    score_predictions,
)


BASE_RATE_HZ = 500
EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302


def parse_rates(text: str) -> list[int]:
    rates = sorted(
        {
            int(item.strip())
            for item in text.split(",")
            if item.strip()
        },
        reverse=True,
    )
    if BASE_RATE_HZ not in rates:
        rates.insert(0, BASE_RATE_HZ)
    return rates


def load_rate_tables(
    feature_dir: Path,
    rates: list[int],
) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    reference_ids: set[str] | None = None
    reference_subjects: set[str] | None = None

    for rate_hz in rates:
        path = (
            feature_dir
            / f"complete_raw_18_{rate_hz:04d}hz_features.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)

        data = pd.read_csv(path)
        data["subject_id"] = (
            data["subject_id"].astype(str).str.upper()
        )
        data["task"] = data["task"].astype(str).str.upper()
        data["label_nr"] = data["task"].map({"NR": 1, "TSR": 0})

        if len(data) != EXPECTED_TRIALS:
            raise ValueError(
                f"{rate_hz} Hz rows={len(data)}, "
                f"expected={EXPECTED_TRIALS}"
            )
        if data["subject_id"].nunique() != EXPECTED_SUBJECTS:
            raise ValueError(
                f"{rate_hz} Hz subjects="
                f"{data['subject_id'].nunique()}, "
                f"expected={EXPECTED_SUBJECTS}"
            )
        if data["trial_id"].duplicated().any():
            raise ValueError(
                f"{rate_hz} Hz has duplicate trial IDs."
            )

        trial_ids = set(data["trial_id"])
        subjects = set(data["subject_id"])
        if reference_ids is None:
            reference_ids = trial_ids
            reference_subjects = subjects
        else:
            if trial_ids != reference_ids:
                raise ValueError(
                    f"{rate_hz} Hz trial IDs differ from 500 Hz."
                )
            if subjects != reference_subjects:
                raise ValueError(
                    f"{rate_hz} Hz subjects differ from 500 Hz."
                )

        tables[rate_hz] = data

    return tables


def build_feature_sets(
    previous_feature_sets_path: Path,
) -> dict[str, list[str]]:
    previous = json.loads(
        previous_feature_sets_path.read_text(encoding="utf-8")
    )

    quality = [
        "valid_xy_rate",
        "valid_pupil_rate",
        "valid_basic_rate",
        "longest_invalid_run_ratio",
    ]

    position_dispersion = [
        "x_std",
        "x_iqr",
        "x_range90",
        "y_std",
        "y_iqr",
        "y_range90",
        "xy_correlation",
    ]

    speed_only = [
        "speed_mean_px_s",
        "speed_median_px_s",
        "speed_p90_px_s",
        "speed_p95_px_s",
    ]

    pupil_rate_invariant = [
        "pupil_std_rel",
        "pupil_iqr_rel",
        "pupil_range90_rel",
    ]

    gaze_rate_invariant = list(
        dict.fromkeys(
            quality + position_dispersion + speed_only
        )
    )
    strict_rate_invariant = list(
        dict.fromkeys(
            gaze_rate_invariant + pupil_rate_invariant
        )
    )

    return {
        "dummy_reference": quality[:1],
        "time_only": ["duration_sec", "n_samples"],
        "speed_only": speed_only,
        "gaze_rate_invariant": gaze_rate_invariant,
        "pupil_rate_invariant": pupil_rate_invariant,
        "strict_rate_invariant": strict_rate_invariant,
        "strict_rate_dependent_reference": previous[
            "strict_relative"
        ],
    }


def validate_features(
    tables: dict[int, pd.DataFrame],
    feature_sets: dict[str, list[str]],
) -> None:
    missing: dict[int, dict[str, list[str]]] = {}
    for rate_hz, data in tables.items():
        missing_at_rate = {
            name: sorted(set(columns) - set(data.columns))
            for name, columns in feature_sets.items()
            if set(columns) - set(data.columns)
        }
        if missing_at_rate:
            missing[rate_hz] = missing_at_rate

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def evaluate_protocol(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    protocol: str,
    sampling_rate_hz: int,
    feature_set_name: str,
    columns: list[str],
    model_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subjects = sorted(train_table["subject_id"].unique())
    if set(subjects) != set(test_table["subject_id"].unique()):
        raise ValueError("Train and test tables have different subjects.")

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold_index, held_out_subject in enumerate(
        subjects,
        start=1,
    ):
        train = train_table[
            train_table["subject_id"] != held_out_subject
        ].copy()
        test = test_table[
            test_table["subject_id"] == held_out_subject
        ].copy()

        if len(test) != 739:
            raise ValueError(
                f"{held_out_subject} test rows={len(test)}, expected 739."
            )

        y_train = train["label_nr"].astype(int).to_numpy()
        y_test = test["label_nr"].astype(int).to_numpy()

        model = make_model(model_name, seed)
        model.fit(train[columns], y_train)
        y_pred = model.predict(test[columns]).astype(int)

        if hasattr(model, "decision_function"):
            scores = np.asarray(
                model.decision_function(test[columns]),
                dtype=float,
            )
        elif hasattr(model, "predict_proba"):
            scores = np.asarray(
                model.predict_proba(test[columns])[:, 1],
                dtype=float,
            )
        else:
            scores = y_pred.astype(float)

        metrics = score_predictions(y_test, y_pred, scores)
        fold_rows.append(
            {
                "protocol": protocol,
                "sampling_rate_hz": sampling_rate_hz,
                "feature_set": feature_set_name,
                "model": model_name,
                "fold_index": fold_index,
                "held_out_subject": held_out_subject,
                "n_train": len(train),
                "n_test": len(test),
                **metrics,
            }
        )

        for index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "protocol": protocol,
                    "sampling_rate_hz": sampling_rate_hz,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "fold_index": fold_index,
                    "held_out_subject": held_out_subject,
                    "trial_id": row["trial_id"],
                    "true_label_nr": int(y_test[index]),
                    "predicted_label_nr": int(y_pred[index]),
                    "score_nr": float(scores[index]),
                }
            )

    return fold_rows, prediction_rows


def summarize(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "roc_auc",
        "pr_auc_nr",
    ]

    rows: list[dict[str, Any]] = []
    group_columns = [
        "protocol",
        "sampling_rate_hz",
        "feature_set",
        "model",
    ]

    for keys, fold_group in folds.groupby(
        group_columns,
        sort=False,
    ):
        protocol, rate_hz, feature_set, model = keys
        pred_group = predictions[
            predictions["protocol"].eq(protocol)
            & predictions["sampling_rate_hz"].eq(rate_hz)
            & predictions["feature_set"].eq(feature_set)
            & predictions["model"].eq(model)
        ]

        y_true = pred_group["true_label_nr"].astype(int).to_numpy()
        y_pred = (
            pred_group["predicted_label_nr"].astype(int).to_numpy()
        )
        scores = pred_group["score_nr"].astype(float).to_numpy()
        pooled = score_predictions(y_true, y_pred, scores)

        row: dict[str, Any] = {
            "protocol": protocol,
            "sampling_rate_hz": int(rate_hz),
            "feature_set": feature_set,
            "model": model,
            "n_folds": len(fold_group),
            "n_oof_predictions": len(pred_group),
        }

        for metric in metric_names:
            values = pd.to_numeric(
                fold_group[metric],
                errors="coerce",
            )
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_sd"] = float(values.std(ddof=1))
            row[f"{metric}_pooled_oof"] = float(pooled[metric])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "protocol",
            "feature_set",
            "sampling_rate_hz",
        ],
        ascending=[True, True, False],
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0

    for rank, original_index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[original_index])
        running = max(running, value)
        adjusted[original_index] = running

    return adjusted.tolist()


def paired_tests_vs_500(
    folds: pd.DataFrame,
) -> pd.DataFrame:
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    rows: list[dict[str, Any]] = []

    for protocol in sorted(folds["protocol"].unique()):
        for feature_set in sorted(folds["feature_set"].unique()):
            for metric in metrics:
                baseline = (
                    folds[
                        folds["protocol"].eq(protocol)
                        & folds["feature_set"].eq(feature_set)
                        & folds["sampling_rate_hz"].eq(
                            BASE_RATE_HZ
                        )
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )

                lower_rates = sorted(
                    [
                        int(rate)
                        for rate in folds.loc[
                            folds["protocol"].eq(protocol)
                            & folds["feature_set"].eq(feature_set),
                            "sampling_rate_hz",
                        ].unique()
                        if int(rate) != BASE_RATE_HZ
                    ],
                    reverse=True,
                )

                indices: list[int] = []
                p_values: list[float] = []

                for rate_hz in lower_rates:
                    current = (
                        folds[
                            folds["protocol"].eq(protocol)
                            & folds["feature_set"].eq(feature_set)
                            & folds["sampling_rate_hz"].eq(rate_hz)
                        ]
                        .set_index("held_out_subject")[metric]
                        .sort_index()
                    )

                    paired = pd.concat(
                        [
                            current.rename("current"),
                            baseline.rename("baseline"),
                        ],
                        axis=1,
                        join="inner",
                    ).dropna()
                    differences = (
                        paired["current"] - paired["baseline"]
                    )

                    if np.allclose(
                        differences.to_numpy(),
                        0.0,
                    ):
                        statistic = 0.0
                        p_value = 1.0
                    else:
                        test = wilcoxon(
                            paired["current"],
                            paired["baseline"],
                            alternative="two-sided",
                            zero_method="wilcox",
                        )
                        statistic = float(test.statistic)
                        p_value = float(test.pvalue)

                    rows.append(
                        {
                            "protocol": protocol,
                            "feature_set": feature_set,
                            "metric": metric,
                            "sampling_rate_hz": rate_hz,
                            "reference_rate_hz": BASE_RATE_HZ,
                            "n_subjects": len(paired),
                            "current_mean": float(
                                paired["current"].mean()
                            ),
                            "baseline_mean": float(
                                paired["baseline"].mean()
                            ),
                            "mean_difference": float(
                                differences.mean()
                            ),
                            "wins": int(np.sum(differences > 0)),
                            "ties": int(
                                np.sum(np.isclose(differences, 0))
                            ),
                            "losses": int(np.sum(differences < 0)),
                            "wilcoxon_statistic": statistic,
                            "p_value": p_value,
                            "p_holm_within_protocol_feature_metric": math.nan,
                        }
                    )
                    indices.append(len(rows) - 1)
                    p_values.append(p_value)

                if p_values:
                    adjusted = holm_adjust(p_values)
                    for row_index, adjusted_value in zip(
                        indices,
                        adjusted,
                    ):
                        rows[row_index][
                            "p_holm_within_protocol_feature_metric"
                        ] = adjusted_value

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir",
        default="outputs/downsampling_robustness",
    )
    parser.add_argument(
        "--previous-feature-sets",
        default="outputs/raw_loso/feature_sets.json",
    )
    parser.add_argument(
        "--rates",
        default="500,250,125,60,30,15",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/downsampling_rate_invariant",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rates = parse_rates(args.rates)
    tables = load_rate_tables(Path(args.feature_dir), rates)
    feature_sets = build_feature_sets(
        Path(args.previous_feature_sets)
    )
    validate_features(tables, feature_sets)

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for rate_hz in rates:
        for feature_set_name, columns in feature_sets.items():
            model_name = (
                "dummy_most_frequent"
                if feature_set_name == "dummy_reference"
                else "linear_svm"
            )

            matched_rows, matched_predictions = evaluate_protocol(
                train_table=tables[rate_hz],
                test_table=tables[rate_hz],
                protocol="rate_matched",
                sampling_rate_hz=rate_hz,
                feature_set_name=feature_set_name,
                columns=columns,
                model_name=model_name,
                seed=args.seed,
            )
            fold_rows.extend(matched_rows)
            prediction_rows.extend(matched_predictions)

            cross_rows, cross_predictions = evaluate_protocol(
                train_table=tables[BASE_RATE_HZ],
                test_table=tables[rate_hz],
                protocol="train500_test_degraded",
                sampling_rate_hz=rate_hz,
                feature_set_name=feature_set_name,
                columns=columns,
                model_name=model_name,
                seed=args.seed,
            )
            fold_rows.extend(cross_rows)
            prediction_rows.extend(cross_predictions)

        print(f"Completed {rate_hz} Hz.")

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize(folds, predictions)
    paired = paired_tests_vs_500(folds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets_path = output_dir / "rate_invariant_feature_sets.json"
    folds_path = output_dir / "rate_invariant_fold_metrics.csv"
    predictions_path = output_dir / "rate_invariant_predictions.csv"
    summary_path = output_dir / "rate_invariant_summary.csv"
    paired_path = output_dir / "rate_invariant_paired_tests_vs_500.csv"

    feature_sets_path.write_text(
        json.dumps(
            feature_sets,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        paired_path,
        index=False,
        encoding="utf-8-sig",
    )

    display_sets = [
        "dummy_reference",
        "time_only",
        "gaze_rate_invariant",
        "pupil_rate_invariant",
        "strict_rate_invariant",
        "strict_rate_dependent_reference",
    ]
    display = summary[
        summary["feature_set"].isin(display_sets)
    ][
        [
            "protocol",
            "sampling_rate_hz",
            "feature_set",
            "macro_f1_subject_mean",
            "macro_f1_subject_sd",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
            "macro_f1_pooled_oof",
        ]
    ]

    print("=== RATE-INVARIANT DIAGNOSTIC SUMMARY ===")
    print(display.to_string(index=False))
    print("\nWritten:")
    print(f"  {feature_sets_path}")
    print(f"  {folds_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    print(f"  {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
