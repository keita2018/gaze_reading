#!/usr/bin/env python
"""Compare ZuCo 2.0 RawStat and Soft-AOI features with nested subject-wise CV.

Outer evaluation
----------------
Leave-one-subject-out (LOSO) across all available subjects.

Inner model selection
---------------------
For each outer fold, choose LinearSVC C using GroupKFold on the training
subjects only. Imputation and scaling are fitted inside each inner/outer
training split, preventing leakage from held-out subjects.

Feature sets
------------
- dummy_reference: most-frequent classifier
- rawstat_content: 30 Hz strict content-only RawStat features
- soft_aoi_content: Soft-AOI content features
- combined_content: RawStat content + Soft-AOI content
- rawstat_all: RawStat content + RawStat quality features
- soft_aoi_all: Soft-AOI content + membership/quality features
- combined_all: all RawStat + all Soft-AOI features

NR is encoded as 1 and TSR as 0.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS_PER_SUBJECT = 739
EXPECTED_NR_PER_SUBJECT = 349
EXPECTED_TSR_PER_SUBJECT = 390

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "roc_auc",
    "pr_auc_nr",
]


def parse_float_list(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("All C values must be positive.")
    return values


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_trial_structure(frame: pd.DataFrame, name: str) -> None:
    require_columns(frame, ["trial_id", "subject_id", "task"], name)
    if frame["trial_id"].duplicated().any():
        examples = frame.loc[
            frame["trial_id"].duplicated(keep=False), "trial_id"
        ].head(20).tolist()
        raise ValueError(f"{name} contains duplicate trial IDs: {examples}")

    frame["subject_id"] = frame["subject_id"].astype(str).str.upper()
    frame["task"] = frame["task"].astype(str).str.upper()
    unexpected = sorted(set(frame["task"]) - {"NR", "TSR"})
    if unexpected:
        raise ValueError(f"{name} contains unexpected task labels: {unexpected}")

    subjects = sorted(frame["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"{name}: subjects={len(subjects)}, expected={EXPECTED_SUBJECTS}."
        )

    for subject, group in frame.groupby("subject_id"):
        counts = group.groupby("task").size().to_dict()
        total = len(group)
        nr = int(counts.get("NR", 0))
        tsr = int(counts.get("TSR", 0))
        if (
            total != EXPECTED_TRIALS_PER_SUBJECT
            or nr != EXPECTED_NR_PER_SUBJECT
            or tsr != EXPECTED_TSR_PER_SUBJECT
        ):
            raise ValueError(
                f"{name}/{subject}: total={total}, NR={nr}, TSR={tsr}."
            )


def ensure_numeric(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[columns].to_numpy(dtype=float)
    inf_count = int(np.isinf(values).sum())
    if inf_count:
        raise ValueError(f"{name} contains {inf_count} infinite feature values.")


def load_data(
    raw_path: Path,
    raw_sets_path: Path,
    soft_path: Path,
    soft_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    for path in (raw_path, raw_sets_path, soft_path, soft_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = pd.read_csv(raw_path)
    soft = pd.read_csv(soft_path)
    validate_trial_structure(raw, "RawStat")
    validate_trial_structure(soft, "Soft-AOI")

    raw_sets = json.loads(raw_sets_path.read_text(encoding="utf-8"))
    soft_manifest = json.loads(
        soft_manifest_path.read_text(encoding="utf-8")
    )

    quality_columns = [
        "valid_xy_rate",
        "valid_pupil_rate",
        "valid_basic_rate",
        "longest_invalid_run_ratio",
    ]

    raw_all = list(raw_sets.get("strict_rate_invariant", []))
    raw_content_source = ""
    if raw_sets.get("strict_content_only"):
        # Compatibility with feature-set manifests that explicitly store the
        # content-only representation.
        raw_content = list(raw_sets["strict_content_only"])
        raw_content_source = "strict_content_only"
    elif raw_sets.get("gaze_rate_invariant") and raw_sets.get(
        "pupil_rate_invariant"
    ):
        # The existing rate-invariant manifest stores gaze and pupil groups
        # separately. Remove tracking-quality fields from gaze, then combine
        # the remaining gaze content with pupil content.
        gaze_content = [
            column
            for column in raw_sets["gaze_rate_invariant"]
            if column not in quality_columns
        ]
        raw_content = list(
            dict.fromkeys(gaze_content + list(raw_sets["pupil_rate_invariant"]))
        )
        raw_content_source = (
            "gaze_rate_invariant_without_quality_plus_pupil_rate_invariant"
        )
    elif raw_all:
        # Conservative fallback for manifests containing only the complete
        # rate-invariant set.
        raw_content = [
            column for column in raw_all if column not in quality_columns
        ]
        raw_content_source = "strict_rate_invariant_without_quality"
    else:
        raw_content = []

    soft_content = list(soft_manifest.get("content_feature_columns", []))
    soft_quality = list(
        soft_manifest.get("membership_quality_feature_columns", [])
    )
    soft_all = list(dict.fromkeys(soft_content + soft_quality))

    if not raw_all:
        raise ValueError(
            "Raw feature-set JSON must contain strict_rate_invariant. "
            f"Available keys: {sorted(raw_sets)}"
        )
    if not raw_content:
        raise ValueError(
            "Could not derive a RawStat content-only feature set. "
            f"Available keys: {sorted(raw_sets)}"
        )
    if not soft_content or not soft_all:
        raise ValueError(
            "Soft-AOI manifest must contain content_feature_columns and "
            "membership_quality_feature_columns."
        )

    require_columns(raw, raw_all, "RawStat")
    require_columns(soft, soft_all, "Soft-AOI")
    ensure_numeric(raw, raw_all, "RawStat")
    ensure_numeric(soft, soft_all, "Soft-AOI")

    raw_rename = {column: f"raw__{column}" for column in raw_all}
    soft_rename = {column: f"soft__{column}" for column in soft_all}

    raw_small = raw[["trial_id", "subject_id", "task"] + raw_all].rename(
        columns=raw_rename
    )
    soft_small = soft[["trial_id", "subject_id", "task"] + soft_all].rename(
        columns=soft_rename
    )

    merged = raw_small.merge(
        soft_small,
        on=["trial_id", "subject_id", "task"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    merge_counts = merged["_merge"].value_counts().to_dict()
    if merge_counts.get("both", 0) != EXPECTED_SUBJECTS * EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(f"Raw/Soft trial merge failed: {merge_counts}")
    merged = merged.drop(columns="_merge")
    merged["label_nr"] = merged["task"].map({"NR": 1, "TSR": 0}).astype(int)

    feature_sets = {
        "rawstat_content": [raw_rename[column] for column in raw_content],
        "soft_aoi_content": [soft_rename[column] for column in soft_content],
        "combined_content": (
            [raw_rename[column] for column in raw_content]
            + [soft_rename[column] for column in soft_content]
        ),
        "rawstat_all": [raw_rename[column] for column in raw_all],
        "soft_aoi_all": [soft_rename[column] for column in soft_all],
        "combined_all": (
            [raw_rename[column] for column in raw_all]
            + [soft_rename[column] for column in soft_all]
        ),
    }
    for feature_name, columns in feature_sets.items():
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate columns in {feature_name}.")
        require_columns(merged, columns, feature_name)

    metadata = {
        "raw_content_columns": raw_content,
        "raw_content_source": raw_content_source,
        "raw_quality_columns": [
            column for column in quality_columns if column in raw_all
        ],
        "raw_all_columns": raw_all,
        "soft_content_columns": soft_content,
        "soft_quality_columns": soft_quality,
        "feature_sets_prefixed": feature_sets,
        "raw_nan_cells": int(raw[raw_all].isna().sum().sum()),
        "soft_nan_cells": int(soft[soft_all].isna().sum().sum()),
    }
    return merged, feature_sets, metadata


def make_svm(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=float(c_value),
                    dual=False,
                    max_iter=30000,
                    random_state=seed,
                ),
            ),
        ]
    )


def decision_scores(model: Any, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float).reshape(-1)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)]
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = math.nan
    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except ValueError:
        pr_auc = math.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "roc_auc": roc_auc,
        "pr_auc_nr": pr_auc,
        "tn_tsr_correct": int(tn),
        "fp_tsr_as_nr": int(fp),
        "fn_nr_as_tsr": int(fn),
        "tp_nr_correct": int(tp),
    }


def tune_c(
    train: pd.DataFrame,
    feature_columns: list[str],
    c_values: list[float],
    inner_splits: int,
    seed: int,
    outer_subject: str,
    feature_set: str,
) -> tuple[float, list[dict[str, Any]]]:
    groups = train["subject_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(int(inner_splits), len(unique_groups))
    if n_splits < 2:
        raise ValueError("Inner CV requires at least two training subjects.")

    splitter = GroupKFold(n_splits=n_splits)
    x = train[feature_columns]
    y = train["label_nr"].astype(int).to_numpy()
    records: list[dict[str, Any]] = []

    for c_value in c_values:
        fold_scores: list[float] = []
        for inner_fold, (fit_indices, valid_indices) in enumerate(
            splitter.split(x, y, groups), start=1
        ):
            model = make_svm(c_value, seed)
            model.fit(x.iloc[fit_indices], y[fit_indices])
            predictions = model.predict(x.iloc[valid_indices]).astype(int)
            macro_f1 = float(
                f1_score(
                    y[valid_indices],
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )
            fold_scores.append(macro_f1)
            records.append(
                {
                    "held_out_subject": outer_subject,
                    "feature_set": feature_set,
                    "c_value": float(c_value),
                    "inner_fold": inner_fold,
                    "n_inner_train": len(fit_indices),
                    "n_inner_valid": len(valid_indices),
                    "inner_macro_f1": macro_f1,
                }
            )

    c_summary = (
        pd.DataFrame(records)
        .groupby("c_value", as_index=False)["inner_macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    best_mean = float(c_summary["mean"].max())
    candidates = c_summary[
        np.isclose(c_summary["mean"], best_mean, rtol=1e-12, atol=1e-12)
    ]
    best_c = float(candidates["c_value"].min())
    for record in records:
        record["selected_c"] = best_c
        record["selected_for_outer_fold"] = bool(
            math.isclose(record["c_value"], best_c)
        )
    return best_c, records


def evaluate_nested_loso(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    c_values: list[float],
    inner_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = sorted(data["subject_id"].unique())
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    evaluated_sets = ["dummy_reference"] + list(feature_sets.keys())
    for feature_index, feature_set in enumerate(evaluated_sets, start=1):
        columns = feature_sets.get(feature_set, [])
        print("\n" + "=" * 78)
        print(
            f"Feature set {feature_index}/{len(evaluated_sets)}: "
            f"{feature_set} ({len(columns)} input columns)"
        )

        for outer_fold, held_out in enumerate(subjects, start=1):
            print(
                f"  Outer fold {outer_fold:02d}/{len(subjects)}: "
                f"held out {held_out}"
            )
            train = data[data["subject_id"] != held_out].copy()
            test = data[data["subject_id"] == held_out].copy()
            y_train = train["label_nr"].astype(int).to_numpy()
            y_test = test["label_nr"].astype(int).to_numpy()

            if feature_set == "dummy_reference":
                model = DummyClassifier(strategy="most_frequent")
                x_train = np.zeros((len(train), 1), dtype=float)
                x_test = np.zeros((len(test), 1), dtype=float)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test).astype(int)
                scores = decision_scores(model, x_test)
                selected_c = math.nan
            else:
                selected_c, records = tune_c(
                    train=train,
                    feature_columns=columns,
                    c_values=c_values,
                    inner_splits=inner_splits,
                    seed=seed,
                    outer_subject=held_out,
                    feature_set=feature_set,
                )
                tuning_rows.extend(records)
                model = make_svm(selected_c, seed)
                model.fit(train[columns], y_train)
                y_pred = model.predict(test[columns]).astype(int)
                scores = decision_scores(model, test[columns])

            metrics = score_predictions(y_test, y_pred, scores)
            fold_rows.append(
                {
                    "feature_set": feature_set,
                    "outer_fold": outer_fold,
                    "held_out_subject": held_out,
                    "n_input_features": len(columns),
                    "selected_c": selected_c,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_test_nr": int(np.sum(y_test == 1)),
                    "n_test_tsr": int(np.sum(y_test == 0)),
                    **metrics,
                }
            )

            ordered_test = test.reset_index(drop=True)
            for row_index, row in ordered_test.iterrows():
                prediction_rows.append(
                    {
                        "feature_set": feature_set,
                        "held_out_subject": held_out,
                        "trial_id": row["trial_id"],
                        "subject_id": row["subject_id"],
                        "task": row["task"],
                        "label_nr": int(row["label_nr"]),
                        "prediction_nr": int(y_pred[row_index]),
                        "decision_score_nr": float(scores[row_index]),
                        "selected_c": selected_c,
                    }
                )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
    )


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in folds.groupby("feature_set", sort=False):
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "n_outer_folds": len(group),
            "n_input_features": int(group["n_input_features"].iloc[0]),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_std"] = float(values.std(ddof=1))
            row[f"{metric}_subject_median"] = float(values.median())
        selected = pd.to_numeric(group["selected_c"], errors="coerce").dropna()
        row["selected_c_median"] = (
            float(selected.median()) if len(selected) else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "macro_f1_subject_mean", ascending=False
    )


def summarize_pooled(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in predictions.groupby("feature_set", sort=False):
        metrics = score_predictions(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction_nr"].to_numpy(dtype=int),
            group["decision_score_nr"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "feature_set": feature_set,
                "n_predictions": len(group),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result.tolist()
    order = finite[np.argsort(values[finite])]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (m - rank) * values[index])
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def build_pairwise_tests(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_sets = list(folds["feature_set"].drop_duplicates())
    for metric in ["macro_f1", "balanced_accuracy", "roc_auc", "pr_auc_nr"]:
        metric_row_indices: list[int] = []
        metric_p_values: list[float] = []
        for first, second in combinations(feature_sets, 2):
            a = folds[folds["feature_set"] == first].set_index(
                "held_out_subject"
            )[metric]
            b = folds[folds["feature_set"] == second].set_index(
                "held_out_subject"
            )[metric]
            paired = pd.concat(
                [a.rename("first"), b.rename("second")], axis=1
            ).dropna()
            differences = paired["second"] - paired["first"]
            if len(paired) == 0:
                statistic = math.nan
                p_value = math.nan
            elif np.allclose(differences.to_numpy(), 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    paired["second"],
                    paired["first"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            rows.append(
                {
                    "metric": metric,
                    "reference_feature_set": first,
                    "candidate_feature_set": second,
                    "n_subjects": len(paired),
                    "reference_mean": float(paired["first"].mean()),
                    "candidate_mean": float(paired["second"].mean()),
                    "mean_difference_candidate_minus_reference": float(
                        differences.mean()
                    ),
                    "wins": int(np.sum(differences > 0)),
                    "ties": int(np.sum(np.isclose(differences, 0.0))),
                    "losses": int(np.sum(differences < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_within_metric": math.nan,
                }
            )
            metric_row_indices.append(len(rows) - 1)
            metric_p_values.append(p_value)

        adjusted = holm_adjust(metric_p_values)
        for row_index, value in zip(metric_row_indices, adjusted):
            rows[row_index]["p_holm_within_metric"] = value
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-features",
        default=(
            "outputs/downsampling_robustness/"
            "complete_raw_18_0030hz_features.csv"
        ),
    )
    parser.add_argument(
        "--raw-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--soft-features",
        default=(
            "data/processed/zuco2/soft_aoi_all_subjects/"
            "zuco2_soft_aoi_features_30hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--soft-feature-manifest",
        default=(
            "data/processed/zuco2/YAC/soft_aoi/"
            "YAC_soft_aoi_feature_manifest_30hz.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/soft_aoi_rawstat_nested_loso_30hz",
    )
    parser.add_argument("--c-values", default="0.01,0.1,1,10")
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    c_values = parse_float_list(args.c_values)
    if args.inner_splits < 2:
        raise ValueError("--inner-splits must be at least 2.")

    data, feature_sets, input_metadata = load_data(
        raw_path=Path(args.raw_features),
        raw_sets_path=Path(args.raw_feature_sets),
        soft_path=Path(args.soft_features),
        soft_manifest_path=Path(args.soft_feature_manifest),
    )
    print("=== INPUT VALIDATION ===")
    print(f"subjects: {data['subject_id'].nunique()}")
    print(f"trials: {len(data)}")
    print(f"RawStat NaN cells: {input_metadata['raw_nan_cells']}")
    print(f"Soft-AOI NaN cells: {input_metadata['soft_nan_cells']}")
    print("Feature sets:")
    for name, columns in feature_sets.items():
        print(f"  {name}: {len(columns)}")

    folds, predictions, tuning = evaluate_nested_loso(
        data=data,
        feature_sets=feature_sets,
        c_values=c_values,
        inner_splits=args.inner_splits,
        seed=args.seed,
    )
    summary = summarize_folds(folds)
    pooled = summarize_pooled(predictions)
    pairwise = build_pairwise_tests(folds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = output_dir / "nested_loso_outer_folds.csv"
    predictions_path = output_dir / "nested_loso_predictions.csv"
    tuning_path = output_dir / "nested_loso_inner_tuning.csv"
    summary_path = output_dir / "nested_loso_subject_summary.csv"
    pooled_path = output_dir / "nested_loso_pooled_summary.csv"
    pairwise_path = output_dir / "nested_loso_pairwise_tests.csv"
    manifest_path = output_dir / "nested_loso_manifest.json"

    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    tuning.to_csv(tuning_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")
    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")

    manifest = {
        "raw_features": str(args.raw_features),
        "raw_feature_sets": str(args.raw_feature_sets),
        "soft_features": str(args.soft_features),
        "soft_feature_manifest": str(args.soft_feature_manifest),
        "n_subjects": int(data["subject_id"].nunique()),
        "n_trials": len(data),
        "outer_cv": "leave-one-subject-out",
        "inner_cv": f"GroupKFold(n_splits={args.inner_splits}) on training subjects",
        "selection_metric": "macro_f1",
        "c_values": c_values,
        "imputer": "median with missing indicators, fitted within each split",
        "scaler": "StandardScaler fitted within each split",
        "classifier": "LinearSVC",
        "seed": args.seed,
        "feature_sets": input_metadata["feature_sets_prefixed"],
        "raw_nan_cells": input_metadata["raw_nan_cells"],
        "soft_nan_cells": input_metadata["soft_nan_cells"],
        "outputs": {
            "outer_folds": str(folds_path),
            "predictions": str(predictions_path),
            "inner_tuning": str(tuning_path),
            "subject_summary": str(summary_path),
            "pooled_summary": str(pooled_path),
            "pairwise_tests": str(pairwise_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    display_columns = [
        "feature_set",
        "n_input_features",
        "macro_f1_subject_mean",
        "balanced_accuracy_subject_mean",
        "roc_auc_subject_mean",
        "pr_auc_nr_subject_mean",
        "selected_c_median",
    ]
    print("\n=== NESTED LOSO SUBJECT-MEAN SUMMARY ===")
    print(summary[display_columns].to_string(index=False))
    print("\nWritten:")
    for path in (
        folds_path,
        predictions_path,
        tuning_path,
        summary_path,
        pooled_path,
        pairwise_path,
        manifest_path,
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
