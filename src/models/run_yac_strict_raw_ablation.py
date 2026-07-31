#!/usr/bin/env python
"""Run strict diagnostic ablations on existing YAC raw-gaze statistics.

Purpose
-------
The first baseline showed that absolute pupil and raw statistical features can
classify NR vs TSR for one subject. However, NR and TSR are recorded in
different task blocks, so absolute gaze position and absolute pupil level may
encode block/session conditions rather than reading purpose.

This script reuses:
- YAC_raw_stat_features.csv
- YAC_sent_gaze_sacc_common_trials.csv
- YAC_raw_baseline_fold_design.csv

It evaluates diagnostic feature sets without rereading the 500 Hz Parquet data.

Key feature sets
----------------
time_only:
    duration and sample count.
quality_only:
    gaze/pupil validity and longest invalid-run ratio.
position_location:
    absolute x/y location statistics.
position_dispersion:
    within-trial x/y spread; excludes absolute location.
dynamics_local:
    adjacent-sample motion and speed; excludes cumulative path length.
pupil_level:
    absolute pupil level.
pupil_variation_absolute:
    absolute within-trial pupil variation.
pupil_relative:
    scale-normalized pupil variability ratios.
strict_relative:
    quality + position dispersion + local dynamics + relative pupil;
    excludes duration, sample count, absolute position, absolute pupil level,
    and cumulative path length.
strict_absolute:
    quality + absolute/dispersion position + local dynamics + absolute pupil.

Both unweighted fold means and pooled out-of-fold metrics are saved.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_MAP = {"NR": 1, "TSR": 0}


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    epsilon: float = 1e-8,
) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    valid = (
        numerator.notna()
        & denominator.notna()
        & denominator.abs().gt(epsilon)
    )
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    result.loc[valid] = (
        numerator.loc[valid] / denominator.loc[valid].abs()
    )
    return result


def add_relative_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    result["pupil_std_rel"] = safe_ratio(
        result["pupil_std"],
        result["pupil_median"],
    )
    result["pupil_iqr_rel"] = safe_ratio(
        result["pupil_iqr"],
        result["pupil_median"],
    )
    result["pupil_range90_rel"] = safe_ratio(
        result["pupil_q95"] - result["pupil_q05"],
        result["pupil_median"],
    )
    result["pupil_diff_abs_mean_rel"] = safe_ratio(
        result["pupil_diff_abs_mean"],
        result["pupil_median"],
    )
    result["pupil_diff_abs_median_rel"] = safe_ratio(
        result["pupil_diff_abs_median"],
        result["pupil_median"],
    )
    result["pupil_diff_abs_p90_rel"] = safe_ratio(
        result["pupil_diff_abs_p90"],
        result["pupil_median"],
    )

    return result


def build_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    time_only = [
        "duration_sec",
        "n_samples",
    ]

    quality_only = [
        "valid_xy_rate",
        "valid_pupil_rate",
        "valid_basic_rate",
        "longest_invalid_run_ratio",
    ]

    position_location = [
        "x_mean",
        "x_q05",
        "x_q25",
        "x_median",
        "x_q75",
        "x_q95",
        "y_mean",
        "y_q05",
        "y_q25",
        "y_median",
        "y_q75",
        "y_q95",
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

    dynamics_local = [
        "dx_mean",
        "dx_std",
        "dx_abs_mean",
        "dx_abs_median",
        "dx_abs_p90",
        "dx_p10",
        "dx_p90",
        "dx_positive_rate",
        "dx_negative_rate",
        "dy_mean",
        "dy_std",
        "dy_abs_mean",
        "dy_abs_median",
        "dy_abs_p90",
        "dy_p10",
        "dy_p90",
        "displacement_mean",
        "displacement_median",
        "displacement_p90",
        "speed_mean_px_s",
        "speed_median_px_s",
        "speed_p90_px_s",
        "speed_p95_px_s",
        "path_length_px_per_sec",
    ]

    pupil_level = [
        "pupil_mean",
        "pupil_q05",
        "pupil_q25",
        "pupil_median",
        "pupil_q75",
        "pupil_q95",
    ]

    pupil_variation_absolute = [
        "pupil_std",
        "pupil_iqr",
        "pupil_diff_abs_mean",
        "pupil_diff_abs_median",
        "pupil_diff_abs_p90",
    ]

    pupil_relative = [
        "pupil_std_rel",
        "pupil_iqr_rel",
        "pupil_range90_rel",
        "pupil_diff_abs_mean_rel",
        "pupil_diff_abs_median_rel",
        "pupil_diff_abs_p90_rel",
    ]

    strict_relative = list(
        dict.fromkeys(
            quality_only
            + position_dispersion
            + dynamics_local
            + pupil_relative
        )
    )

    strict_absolute = list(
        dict.fromkeys(
            quality_only
            + position_location
            + position_dispersion
            + dynamics_local
            + pupil_level
            + pupil_variation_absolute
        )
    )

    feature_sets = {
        "dummy_reference": quality_only,
        "time_only": time_only,
        "quality_only": quality_only,
        "position_location": position_location,
        "position_dispersion": position_dispersion,
        "dynamics_local": dynamics_local,
        "pupil_level": pupil_level,
        "pupil_variation_absolute": pupil_variation_absolute,
        "pupil_relative": pupil_relative,
        "strict_relative": strict_relative,
        "strict_absolute": strict_absolute,
    }

    missing = {
        set_name: sorted(set(columns) - set(data.columns))
        for set_name, columns in feature_sets.items()
        if set(columns) - set(data.columns)
    }
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return feature_sets


def make_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "linear_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_name == "dummy_most_frequent":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DummyClassifier(strategy="most_frequent"),
                ),
            ]
        )

    raise ValueError(model_name)


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = math.nan

    try:
        pr_auc_nr = float(average_precision_score(y_true, scores))
    except ValueError:
        pr_auc_nr = math.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc,
        "pr_auc_nr": pr_auc_nr,
        "tn_tsr_correct": int(tn),
        "fp_tsr_as_nr": int(fp),
        "fn_nr_as_tsr": int(fn),
        "tp_nr_correct": int(tp),
    }


def make_trial_assignments(
    data: pd.DataFrame,
    fold_design: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    design = fold_design[
        fold_design["dataset"] == dataset_name
    ].copy()

    if design.empty:
        raise ValueError(f"No fold design for {dataset_name}")

    mapping = {
        (str(row.task).upper(), int(row.block_id)): int(row.fold_id)
        for row in design.itertuples(index=False)
    }

    assignments = data[
        ["trial_id", "task", "block_id", "label_nr"]
    ].copy()
    assignments["task"] = assignments["task"].astype(str).str.upper()
    assignments["block_id"] = assignments["block_id"].astype(int)
    assignments["fold_id"] = [
        mapping.get((task, block), np.nan)
        for task, block in zip(
            assignments["task"],
            assignments["block_id"],
        )
    ]

    if assignments["fold_id"].isna().any():
        missing_groups = (
            assignments.loc[
                assignments["fold_id"].isna(),
                ["task", "block_id"],
            ]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            f"Trials outside fold design for {dataset_name}: "
            f"{missing_groups}"
        )

    assignments["fold_id"] = assignments["fold_id"].astype(int)
    return assignments


def evaluate(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    dataset_name: str,
    feature_set_name: str,
    columns: list[str],
    model_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged = data.merge(
        assignments[["trial_id", "fold_id"]],
        on="trial_id",
        how="inner",
        validate="one_to_one",
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold_id in sorted(merged["fold_id"].unique()):
        train = merged[merged["fold_id"] != fold_id]
        test = merged[merged["fold_id"] == fold_id]

        train_labels = set(train["label_nr"].astype(int).unique())
        test_labels = set(test["label_nr"].astype(int).unique())
        if train_labels != {0, 1} or test_labels != {0, 1}:
            raise ValueError(
                f"Invalid classes: dataset={dataset_name}, "
                f"fold={fold_id}, train={train_labels}, "
                f"test={test_labels}"
            )

        x_train = train[columns]
        y_train = train["label_nr"].astype(int).to_numpy()
        x_test = test[columns]
        y_test = test["label_nr"].astype(int).to_numpy()

        model = make_model(model_name, seed)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test).astype(int)

        if hasattr(model, "decision_function"):
            scores = np.asarray(
                model.decision_function(x_test),
                dtype=float,
            )
        elif hasattr(model, "predict_proba"):
            scores = np.asarray(
                model.predict_proba(x_test)[:, 1],
                dtype=float,
            )
        else:
            scores = y_pred.astype(float)

        metrics = score_predictions(y_test, y_pred, scores)
        fold_rows.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "fold_id": int(fold_id),
                "n_train": len(train),
                "n_test": len(test),
                "n_test_nr": int(np.sum(y_test == 1)),
                "n_test_tsr": int(np.sum(y_test == 0)),
                **metrics,
            }
        )

        for index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "fold_id": int(fold_id),
                    "trial_id": row["trial_id"],
                    "task": row["task"],
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
    group_columns = ["dataset", "feature_set", "model"]

    for keys, fold_group in folds.groupby(
        group_columns,
        sort=False,
    ):
        dataset, feature_set, model = keys
        pred_group = predictions[
            (predictions["dataset"] == dataset)
            & (predictions["feature_set"] == feature_set)
            & (predictions["model"] == model)
        ]

        y_true = pred_group["true_label_nr"].astype(int).to_numpy()
        y_pred = (
            pred_group["predicted_label_nr"].astype(int).to_numpy()
        )
        scores = pred_group["score_nr"].astype(float).to_numpy()
        pooled = score_predictions(y_true, y_pred, scores)

        row: dict[str, Any] = {
            "dataset": dataset,
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
            row[f"{metric}_fold_mean"] = float(values.mean())
            row[f"{metric}_fold_sd"] = float(values.std(ddof=1))
            row[f"{metric}_pooled_oof"] = float(pooled[metric])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["dataset", "macro_f1_pooled_oof"],
        ascending=[True, False],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default=(
            "outputs/raw_baseline/YAC/"
            "YAC_raw_stat_features.csv"
        ),
    )
    parser.add_argument(
        "--common-map",
        default=(
            "data/processed/zuco2/YAC/"
            "YAC_sent_gaze_sacc_common_trials.csv"
        ),
    )
    parser.add_argument(
        "--fold-design",
        default=(
            "outputs/raw_baseline/YAC/"
            "YAC_raw_baseline_fold_design.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/raw_baseline/YAC/strict_ablation",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    features["task"] = features["task"].astype(str).str.upper()
    features["label_nr"] = features["task"].map(LABEL_MAP)
    features = add_relative_features(features)

    common_map = pd.read_csv(args.common_map)
    common_ids = set(common_map["trial_id"])
    common = features[
        features["trial_id"].isin(common_ids)
    ].copy()

    fold_design = pd.read_csv(args.fold_design)
    feature_sets = build_feature_sets(features)

    full_assignments = make_trial_assignments(
        features,
        fold_design,
        "full_739",
    )
    common_assignments = make_trial_assignments(
        common,
        fold_design,
        "common_521",
    )

    datasets = {
        "full_739": (features, full_assignments),
        "common_521": (common, common_assignments),
    }

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for dataset_name, (data, assignments) in datasets.items():
        for feature_set_name, columns in feature_sets.items():
            model_name = (
                "dummy_most_frequent"
                if feature_set_name == "dummy_reference"
                else "linear_svm"
            )
            rows, predictions = evaluate(
                data=data,
                assignments=assignments,
                dataset_name=dataset_name,
                feature_set_name=feature_set_name,
                columns=columns,
                model_name=model_name,
                seed=args.seed,
            )
            fold_rows.extend(rows)
            prediction_rows.extend(predictions)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize(folds, predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets_path = output_dir / "feature_sets.txt"
    fold_path = output_dir / "strict_ablation_fold_metrics.csv"
    prediction_path = output_dir / "strict_ablation_predictions.csv"
    summary_path = output_dir / "strict_ablation_summary.csv"

    feature_sets_path.write_text(
        "\n\n".join(
            f"[{name}]\n" + "\n".join(columns)
            for name, columns in feature_sets.items()
        )
        + "\n",
        encoding="utf-8",
    )
    folds.to_csv(fold_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "dataset",
        "feature_set",
        "model",
        "n_folds",
        "accuracy_pooled_oof",
        "macro_f1_pooled_oof",
        "roc_auc_pooled_oof",
        "pr_auc_nr_pooled_oof",
        "macro_f1_fold_mean",
        "macro_f1_fold_sd",
    ]

    print("=== STRICT ABLATION SUMMARY ===")
    print(summary[display_columns].to_string(index=False))
    print("\nWritten:")
    print(f"  {feature_sets_path}")
    print(f"  {fold_path}")
    print(f"  {prediction_path}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
