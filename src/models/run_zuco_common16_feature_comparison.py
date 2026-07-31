#!/usr/bin/env python
"""Compare raw-gaze and official sent_gaze_sacc features on identical trials.

This script creates a strict subject/trial-matched comparison for the 16
ZuCo 2.0 subjects that have both:

1. complete continuous raw eye-tracking trials; and
2. an official <SUBJECT>_sent_gaze_sacc.npy feature dictionary.

For every subject, official feature keys are mapped to the corresponding raw
trial using the original task-level sentence index. Only trials available in
both representations are retained.

Evaluation
----------
Leave-one-subject-out (LOSO) over the same 16 subjects and exactly the same test
trials for every feature condition.

Compared conditions
-------------------
- dummy_reference
- raw_gaze_relative
- raw_pupil_relative
- raw_strict_relative
- raw_strict_absolute
- official_sent_gaze_sacc

The same LinearSVC pipeline is used for every non-dummy condition so that the
comparison primarily reflects the feature representation rather than a
different classifier.

Outputs
-------
- common16_trial_map.csv
- common16_subject_counts.csv
- common16_loso_fold_metrics.csv
- common16_loso_predictions.csv
- common16_loso_summary.csv
- common16_paired_comparisons_vs_official.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


BLOCK_RANGES = {
    "NR": {
        1: (0, 49),
        2: (50, 99),
        3: (100, 150),
        4: (151, 200),
        5: (201, 250),
        6: (251, 299),
        7: (300, 348),
    },
    "TSR": {
        1: (0, 44),
        2: (45, 116),
        3: (117, 170),
        4: (171, 235),
        5: (236, 289),
        6: (290, 349),
        7: (350, 389),
    },
}

FEATURE_KEY_RE = re.compile(
    r"^(?P<subject>[^_]+)_(?P<label>.+)_(?P<sentence_index>\d+)_"
    r"(?P<full_index>\d+)$"
)

LABEL_MAP = {"NR": 1, "TSR": 0}


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def infer_task(label: str) -> str:
    upper = label.upper()
    if upper == "NR" or upper.startswith("NR_"):
        return "NR"
    if upper == "TSR" or upper.startswith("TSR_"):
        return "TSR"
    raise ValueError(f"Cannot infer task from official label: {label!r}")


def add_sentence_indices(trials: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "block_id",
        "trial_index_in_block",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Trial metadata columns missing: {missing}")

    result = trials.copy()
    result["subject_id"] = result["subject_id"].astype(str).str.upper()
    result["task"] = result["task"].astype(str).str.upper()
    result["block_id"] = result["block_id"].astype(int)
    result["trial_index_in_block"] = (
        result["trial_index_in_block"].astype(int)
    )

    indices: list[int] = []
    for row in result.itertuples(index=False):
        task = row.task
        block = int(row.block_id)
        within_block = int(row.trial_index_in_block)

        if task not in BLOCK_RANGES or block not in BLOCK_RANGES[task]:
            raise ValueError(f"Unexpected task/block: {task}{block}")

        start, end = BLOCK_RANGES[task][block]
        expected_count = end - start + 1
        if not 0 <= within_block < expected_count:
            raise ValueError(
                f"Trial index outside block range: "
                f"{row.trial_id}, index={within_block}, "
                f"expected 0..{expected_count - 1}"
            )
        indices.append(start + within_block)

    result["task_sentence_index"] = indices

    key_columns = ["subject_id", "task", "task_sentence_index"]
    if result.duplicated(key_columns).any():
        raise ValueError("Duplicate raw sentence keys detected.")

    return result


def load_official_features(
    feature_path: Path,
    expected_subject: str,
) -> pd.DataFrame:
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)

    loaded = np.load(feature_path, allow_pickle=True)
    dictionary = loaded.item()
    if not isinstance(dictionary, dict):
        raise TypeError(f"{feature_path} does not contain a dictionary.")

    rows: list[dict[str, Any]] = []
    dimensions: set[int] = set()

    for key, value in dictionary.items():
        match = FEATURE_KEY_RE.match(str(key))
        if match is None:
            raise ValueError(f"Unexpected official key: {key!r}")

        subject = match.group("subject").upper()
        if subject != expected_subject:
            raise ValueError(
                f"{feature_path} contains {subject}, expected "
                f"{expected_subject}."
            )

        task = infer_task(match.group("label"))
        sentence_index = int(match.group("sentence_index"))
        full_index = int(match.group("full_index"))

        values = list(value)
        if len(values) < 2:
            raise ValueError(f"Official feature row too short: {key}")

        numeric_values = values[:-1]
        stored_label = str(values[-1])
        dimensions.add(len(numeric_values))

        row: dict[str, Any] = {
            "subject_id": subject,
            "task": task,
            "task_sentence_index": sentence_index,
            "official_feature_key": str(key),
            "official_feature_full_index": full_index,
            "official_feature_stored_label": stored_label,
        }
        for index, item in enumerate(numeric_values):
            row[f"official_feature_{index}"] = safe_float(item)
        rows.append(row)

    if len(dimensions) != 1:
        raise ValueError(
            f"Inconsistent official feature dimensions in {feature_path}: "
            f"{sorted(dimensions)}"
        )

    result = pd.DataFrame(rows)
    key_columns = ["subject_id", "task", "task_sentence_index"]
    if result.duplicated(key_columns).any():
        raise ValueError(
            f"Duplicate official sentence keys in {feature_path}."
        )
    return result


def build_common_table(
    raw_features: pd.DataFrame,
    trials: pd.DataFrame,
    feature_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    indexed_trials = add_sentence_indices(trials)

    if "has_local_official_sent_gaze_sacc" not in indexed_trials.columns:
        raise ValueError(
            "Trial metadata lacks has_local_official_sent_gaze_sacc."
        )

    official_subjects = sorted(
        indexed_trials.loc[
            pd.to_numeric(
                indexed_trials[
                    "has_local_official_sent_gaze_sacc"
                ],
                errors="coerce",
            ).fillna(0).astype(int).eq(1),
            "subject_id",
        ].unique()
    )
    if len(official_subjects) != 16:
        raise ValueError(
            f"Expected 16 official-overlap subjects, found "
            f"{len(official_subjects)}: {official_subjects}"
        )

    raw_features = raw_features.copy()
    raw_features["subject_id"] = (
        raw_features["subject_id"].astype(str).str.upper()
    )
    raw_features["task"] = raw_features["task"].astype(str).str.upper()

    raw_with_index = raw_features.merge(
        indexed_trials[
            [
                "trial_id",
                "task_sentence_index",
                "has_local_official_sent_gaze_sacc",
            ]
        ],
        on="trial_id",
        how="left",
        validate="one_to_one",
    )

    subject_tables: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    global_dimensions: set[int] = set()

    for subject in official_subjects:
        official_path = feature_dir / f"{subject}_sent_gaze_sacc.npy"
        official = load_official_features(official_path, subject)

        raw_subject = raw_with_index[
            raw_with_index["subject_id"] == subject
        ].copy()
        if len(raw_subject) != 739:
            raise ValueError(
                f"{subject} has {len(raw_subject)} raw feature rows, "
                "expected 739."
            )

        common = raw_subject.merge(
            official,
            on=["subject_id", "task", "task_sentence_index"],
            how="inner",
            validate="one_to_one",
        )

        official_columns = [
            column
            for column in common.columns
            if re.fullmatch(r"official_feature_\d+", column)
        ]
        global_dimensions.add(len(official_columns))

        counts = common.groupby("task").size().to_dict()
        nr = int(counts.get("NR", 0))
        tsr = int(counts.get("TSR", 0))
        if nr == 0 or tsr == 0:
            raise ValueError(
                f"{subject} common subset lacks a class: "
                f"NR={nr}, TSR={tsr}"
            )
        if len(common) != len(official):
            raise ValueError(
                f"{subject}: not all official rows mapped to raw trials: "
                f"official={len(official)}, common={len(common)}"
            )

        count_rows.append(
            {
                "subject_id": subject,
                "common_trials": len(common),
                "common_nr": nr,
                "common_tsr": tsr,
                "raw_only_trials": 739 - len(common),
                "official_feature_dimension": len(official_columns),
            }
        )
        subject_tables.append(common)

    if len(global_dimensions) != 1:
        raise ValueError(
            "Official feature dimensions differ across subjects: "
            f"{sorted(global_dimensions)}"
        )

    combined = pd.concat(subject_tables, ignore_index=True)
    counts = pd.DataFrame(count_rows)

    if combined["trial_id"].duplicated().any():
        raise ValueError("Duplicate trial IDs in common table.")

    official_columns = sorted(
        [
            column
            for column in combined.columns
            if re.fullmatch(r"official_feature_\d+", column)
        ],
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    return combined, counts, official_columns


def make_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "linear_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        C=1.0,
                        dual=False,
                        max_iter=20000,
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
        pr_auc_nr = float(
            average_precision_score(y_true, scores)
        )
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


def evaluate_loso(
    data: pd.DataFrame,
    feature_set_name: str,
    columns: list[str],
    model_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    subjects = sorted(data["subject_id"].unique())

    for fold_index, held_out_subject in enumerate(subjects, start=1):
        train = data[data["subject_id"] != held_out_subject]
        test = data[data["subject_id"] == held_out_subject]

        y_train = train["label_nr"].astype(int).to_numpy()
        y_test = test["label_nr"].astype(int).to_numpy()
        if set(np.unique(y_train)) != {0, 1}:
            raise ValueError(
                f"Training set lacks a class for {held_out_subject}."
            )
        if set(np.unique(y_test)) != {0, 1}:
            raise ValueError(
                f"Test set lacks a class for {held_out_subject}."
            )

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
                "feature_set": feature_set_name,
                "model": model_name,
                "fold_index": fold_index,
                "held_out_subject": held_out_subject,
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
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "fold_index": fold_index,
                    "held_out_subject": held_out_subject,
                    "trial_id": row["trial_id"],
                    "task": row["task"],
                    "true_label_nr": int(y_test[index]),
                    "predicted_label_nr": int(y_pred[index]),
                    "score_nr": safe_float(scores[index]),
                }
            )

        print(
            f"{feature_set_name} | test={held_out_subject} | "
            f"n={len(test)} | macro_f1={metrics['macro_f1']:.4f} | "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    return fold_rows, prediction_rows


def summarize_results(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "roc_auc",
        "pr_auc_nr",
    ]
    rows: list[dict[str, Any]] = []

    for (feature_set, model), fold_group in folds.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        pred_group = predictions[
            (predictions["feature_set"] == feature_set)
            & (predictions["model"] == model)
        ]
        y_true = pred_group["true_label_nr"].astype(int).to_numpy()
        y_pred = (
            pred_group["predicted_label_nr"].astype(int).to_numpy()
        )
        scores = pred_group["score_nr"].astype(float).to_numpy()
        pooled = score_predictions(y_true, y_pred, scores)

        row: dict[str, Any] = {
            "feature_set": feature_set,
            "model": model,
            "n_folds": len(fold_group),
            "n_oof_predictions": len(pred_group),
        }
        for metric in metrics:
            values = pd.to_numeric(
                fold_group[metric],
                errors="coerce",
            )
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_sd"] = float(values.std(ddof=1))
            row[f"{metric}_pooled_oof"] = float(pooled[metric])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "macro_f1_subject_mean",
        ascending=False,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0

    for rank, original_index in enumerate(order):
        raw = p_values[original_index]
        value = min(1.0, (n - rank) * raw)
        running = max(running, value)
        adjusted[original_index] = running
    return adjusted.tolist()


def paired_comparisons(
    folds: pd.DataFrame,
    official_name: str = "official_sent_gaze_sacc",
) -> pd.DataFrame:
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    candidate_names = [
        name
        for name in folds["feature_set"].unique()
        if name not in {official_name, "dummy_reference"}
    ]

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        official = (
            folds[folds["feature_set"] == official_name]
            .set_index("held_out_subject")[metric]
            .sort_index()
        )

        metric_row_indices: list[int] = []
        p_values: list[float] = []

        for candidate in candidate_names:
            candidate_values = (
                folds[folds["feature_set"] == candidate]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            paired = pd.concat(
                [
                    candidate_values.rename("candidate"),
                    official.rename("official"),
                ],
                axis=1,
                join="inner",
            ).dropna()

            differences = (
                paired["candidate"] - paired["official"]
            )

            if len(paired) == 0:
                statistic = math.nan
                p_value = math.nan
            elif np.allclose(differences.to_numpy(), 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    paired["candidate"],
                    paired["official"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            row = {
                "metric": metric,
                "candidate": candidate,
                "reference": official_name,
                "n_subjects": len(paired),
                "candidate_mean": float(
                    paired["candidate"].mean()
                ),
                "official_mean": float(
                    paired["official"].mean()
                ),
                "mean_difference": float(differences.mean()),
                "median_difference": float(differences.median()),
                "wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(np.isclose(differences, 0))),
                "losses": int(np.sum(differences < 0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "p_holm_within_metric": math.nan,
            }
            rows.append(row)
            metric_row_indices.append(len(rows) - 1)
            p_values.append(p_value)

        finite_positions = [
            index
            for index, value in enumerate(p_values)
            if math.isfinite(value)
        ]
        if finite_positions:
            adjusted = holm_adjust(
                [p_values[index] for index in finite_positions]
            )
            for local_index, adjusted_value in zip(
                finite_positions,
                adjusted,
            ):
                rows[
                    metric_row_indices[local_index]
                ]["p_holm_within_metric"] = adjusted_value

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-features",
        default=(
            "outputs/raw_loso/"
            "complete_raw_18_features.csv"
        ),
    )
    parser.add_argument(
        "--all-trials",
        default=(
            "data/interim/zuco2/all_subjects/"
            "complete_raw_all_trials.csv"
        ),
    )
    parser.add_argument(
        "--raw-feature-sets",
        default="outputs/raw_loso/feature_sets.json",
    )
    parser.add_argument(
        "--official-feature-dir",
        default="external/zuco-benchmark/src/features",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/common16_comparison",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    raw_features = pd.read_csv(args.raw_features)
    trials = pd.read_csv(args.all_trials)
    feature_dir = Path(args.official_feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_features["subject_id"] = (
        raw_features["subject_id"].astype(str).str.upper()
    )
    raw_features["task"] = raw_features["task"].astype(str).str.upper()
    raw_features["label_nr"] = (
        raw_features["task"].map(LABEL_MAP).astype(int)
    )

    common, counts, official_columns = build_common_table(
        raw_features=raw_features,
        trials=trials,
        feature_dir=feature_dir,
    )

    feature_sets = json.loads(
        Path(args.raw_feature_sets).read_text(encoding="utf-8")
    )
    selected_raw_sets = {
        "raw_gaze_relative": feature_sets["gaze_relative"],
        "raw_pupil_relative": feature_sets["pupil_relative"],
        "raw_strict_relative": feature_sets["strict_relative"],
        "raw_strict_absolute": feature_sets["strict_absolute"],
    }

    comparison_sets: dict[str, list[str]] = {
        "dummy_reference": selected_raw_sets[
            "raw_strict_relative"
        ][:1],
        **selected_raw_sets,
        "official_sent_gaze_sacc": official_columns,
    }

    missing = {
        name: sorted(set(columns) - set(common.columns))
        for name, columns in comparison_sets.items()
        if set(columns) - set(common.columns)
    }
    if missing:
        raise ValueError(f"Comparison feature columns missing: {missing}")

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    print("=== COMMON 16 SUBJECT COUNTS ===")
    print(counts.to_string(index=False))
    print(f"\nTotal common trials: {len(common)}")
    print(f"Official feature dimension: {len(official_columns)}")

    for feature_set_name, columns in comparison_sets.items():
        model_name = (
            "dummy_most_frequent"
            if feature_set_name == "dummy_reference"
            else "linear_svm"
        )
        rows, predictions = evaluate_loso(
            data=common,
            feature_set_name=feature_set_name,
            columns=columns,
            model_name=model_name,
            seed=args.seed,
        )
        fold_rows.extend(rows)
        prediction_rows.extend(predictions)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_results(folds, predictions)
    paired = paired_comparisons(folds)

    common_path = output_dir / "common16_trial_map.csv"
    counts_path = output_dir / "common16_subject_counts.csv"
    folds_path = output_dir / "common16_loso_fold_metrics.csv"
    predictions_path = output_dir / "common16_loso_predictions.csv"
    summary_path = output_dir / "common16_loso_summary.csv"
    paired_path = (
        output_dir
        / "common16_paired_comparisons_vs_official.csv"
    )

    common.to_csv(common_path, index=False, encoding="utf-8-sig")
    counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
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

    display_columns = [
        "feature_set",
        "model",
        "n_folds",
        "accuracy_subject_mean",
        "macro_f1_subject_mean",
        "macro_f1_subject_sd",
        "roc_auc_subject_mean",
        "pr_auc_nr_subject_mean",
        "macro_f1_pooled_oof",
    ]
    print("\n=== MATCHED COMMON-TRIAL LOSO SUMMARY ===")
    print(summary[display_columns].to_string(index=False))

    print("\n=== PAIRED COMPARISONS VS OFFICIAL ===")
    print(
        paired[
            [
                "metric",
                "candidate",
                "mean_difference",
                "wins",
                "ties",
                "losses",
                "p_value",
                "p_holm_within_metric",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    print(f"  {common_path}")
    print(f"  {counts_path}")
    print(f"  {folds_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    print(f"  {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
