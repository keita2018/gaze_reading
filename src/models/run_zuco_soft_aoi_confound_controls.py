#!/usr/bin/env python
"""Run sentence/layout confound controls for ZuCo 2.0 Soft-AOI features.

This script performs two complementary analyses.

1. Subject-LOSO static-layout control
   The model receives only sentence text-length and AOI-layout descriptors.
   Because the same sentences are available from the training subjects, this
   estimates how much of the ordinary subject-LOSO result can be explained by
   static sentence/layout differences rather than gaze behaviour.

2. Dual holdout: unseen subject AND unseen sentence
   Unique NR and TSR sentences are stratified into sentence folds. For each
   held-out subject and sentence fold, training excludes:
     - every trial from the held-out subject; and
     - every trial from the held-out sentence fold for all other subjects.
   The test set is the intersection of the held-out subject and held-out
   sentence fold. Across all sentence folds, every trial is predicted exactly
   once for each feature set.

The primary dual-holdout classifier uses a pre-specified LinearSVC C value.
No hyperparameter is selected from dual-holdout test results. Median imputation,
missing indicators, and scaling are fitted on each training split only.

NR is encoded as 1 and TSR as 0. No fixation or saccade event is used.
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

EXPECTED_SUBJECTS = 18
EXPECTED_SENTENCES = 739
EXPECTED_TRIALS = 13302
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

QUALITY_COLUMNS = [
    "valid_xy_rate",
    "valid_pupil_rate",
    "valid_basic_rate",
    "longest_invalid_run_ratio",
]


def parse_float_list(text: str) -> list[float]:
    values = sorted(
        {float(item.strip()) for item in text.split(",") if item.strip()}
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("All C values must be positive.")
    return values


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def finite_or_nan(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def safe_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else math.nan


def safe_std(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=0)) if array.size else math.nan


def safe_median(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else math.nan


def safe_quantile(values: list[float] | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else math.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0:
        return math.nan
    return float(numerator / denominator)


def validate_trial_structure(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    require_columns(frame, ["trial_id", "subject_id", "task"], name)
    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = result["subject_id"].astype(str).str.upper()
    result["task"] = result["task"].astype(str).str.upper()

    if result["trial_id"].duplicated().any():
        examples = result.loc[
            result["trial_id"].duplicated(keep=False), "trial_id"
        ].head(20).tolist()
        raise ValueError(f"{name} contains duplicate trial IDs: {examples}")

    unexpected = sorted(set(result["task"]) - {"NR", "TSR"})
    if unexpected:
        raise ValueError(f"{name} contains unexpected task labels: {unexpected}")
    if len(result) != EXPECTED_TRIALS:
        raise ValueError(
            f"{name}: rows={len(result)}, expected={EXPECTED_TRIALS}."
        )
    if result["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"{name}: subjects={result['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    for subject, group in result.groupby("subject_id"):
        counts = group.groupby("task").size().to_dict()
        nr = int(counts.get("NR", 0))
        tsr = int(counts.get("TSR", 0))
        if (
            len(group) != EXPECTED_TRIALS_PER_SUBJECT
            or nr != EXPECTED_NR_PER_SUBJECT
            or tsr != EXPECTED_TSR_PER_SUBJECT
        ):
            raise ValueError(
                f"{name}/{subject}: total={len(group)}, NR={nr}, TSR={tsr}."
            )
    return result


def derive_raw_feature_sets(
    raw_sets: dict[str, Any],
) -> tuple[list[str], list[str], str]:
    raw_all = list(raw_sets.get("strict_rate_invariant", []))
    if not raw_all:
        raise ValueError(
            "Raw feature-set JSON lacks strict_rate_invariant. "
            f"Available keys: {sorted(raw_sets)}"
        )

    if raw_sets.get("strict_content_only"):
        raw_content = list(raw_sets["strict_content_only"])
        source = "strict_content_only"
    elif raw_sets.get("gaze_rate_invariant") and raw_sets.get(
        "pupil_rate_invariant"
    ):
        gaze_content = [
            column
            for column in raw_sets["gaze_rate_invariant"]
            if column not in QUALITY_COLUMNS
        ]
        raw_content = list(
            dict.fromkeys(
                gaze_content + list(raw_sets["pupil_rate_invariant"])
            )
        )
        source = (
            "gaze_rate_invariant_without_quality_plus_"
            "pupil_rate_invariant"
        )
    else:
        raw_content = [
            column for column in raw_all if column not in QUALITY_COLUMNS
        ]
        source = "strict_rate_invariant_without_quality"

    if not raw_content:
        raise ValueError("Could not derive RawStat content columns.")
    return raw_content, raw_all, source


def build_layout_table(
    word_template_path: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not word_template_path.is_file():
        raise FileNotFoundError(word_template_path)
    words = pd.read_csv(word_template_path)
    required = [
        "task",
        "task_sentence_index",
        "word_index",
        "word_text",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "line_index",
    ]
    require_columns(words, required, "Word template")
    words["task"] = words["task"].astype(str).str.upper()
    words["task_sentence_index"] = pd.to_numeric(
        words["task_sentence_index"], errors="raise"
    ).astype(int)
    words["word_index"] = pd.to_numeric(
        words["word_index"], errors="raise"
    ).astype(int)
    words["line_index"] = pd.to_numeric(
        words["line_index"], errors="raise"
    ).astype(int)

    if "valid_box" in words.columns:
        valid_box = words["valid_box"]
        if valid_box.dtype == bool:
            words = words[valid_box].copy()
        else:
            words = words[
                valid_box.astype(str).str.lower().isin(["true", "1"])
            ].copy()

    for column in ["x_min", "y_min", "x_max", "y_max"]:
        words[column] = pd.to_numeric(words[column], errors="coerce")
    finite = np.isfinite(
        words[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
    ).all(axis=1)
    positive_extent = (
        (words["x_max"] > words["x_min"])
        & (words["y_max"] > words["y_min"])
    )
    words = words[finite & positive_extent].copy()

    rows: list[dict[str, Any]] = []
    group_columns = ["task", "task_sentence_index"]
    for (task, sentence_index), group in words.groupby(
        group_columns, sort=True
    ):
        group = group.sort_values("word_index").copy()
        if group["word_index"].duplicated().any():
            raise ValueError(
                f"Duplicate word indices for {task}/{sentence_index}."
            )

        x_min = group["x_min"].to_numpy(dtype=float)
        x_max = group["x_max"].to_numpy(dtype=float)
        y_min = group["y_min"].to_numpy(dtype=float)
        y_max = group["y_max"].to_numpy(dtype=float)
        widths = x_max - x_min
        heights = y_max - y_min
        areas = widths * heights
        envelope_width = float(np.max(x_max) - np.min(x_min))
        envelope_height = float(np.max(y_max) - np.min(y_min))
        envelope_area = envelope_width * envelope_height

        word_text = group["word_text"].fillna("").astype(str)
        word_chars = word_text.str.len().to_numpy(dtype=float)
        sentence_text = (
            str(group["sentence_text"].iloc[0])
            if "sentence_text" in group.columns
            else " ".join(word_text.tolist())
        )

        line_word_counts: list[float] = []
        line_widths: list[float] = []
        line_heights: list[float] = []
        line_top_bottom: list[tuple[float, float]] = []
        horizontal_gaps: list[float] = []

        for _, line in group.groupby("line_index", sort=True):
            line = line.sort_values(["x_min", "word_index"])
            line_word_counts.append(float(len(line)))
            line_widths.append(
                float(line["x_max"].max() - line["x_min"].min())
            )
            line_heights.append(
                float(line["y_max"].max() - line["y_min"].min())
            )
            line_top_bottom.append(
                (float(line["y_min"].min()), float(line["y_max"].max()))
            )
            if len(line) >= 2:
                current_x_max = line["x_max"].to_numpy(dtype=float)[:-1]
                next_x_min = line["x_min"].to_numpy(dtype=float)[1:]
                horizontal_gaps.extend((next_x_min - current_x_max).tolist())

        vertical_gaps: list[float] = []
        ordered_lines = sorted(line_top_bottom, key=lambda pair: pair[0])
        for previous, current in zip(ordered_lines[:-1], ordered_lines[1:]):
            vertical_gaps.append(float(current[0] - previous[1]))

        n_words = int(len(group))
        n_lines = int(group["line_index"].nunique())
        row = {
            "task": str(task),
            "task_sentence_index": int(sentence_index),
            "sentence_key": f"{task}:{int(sentence_index):03d}",
            "label_nr": 1 if str(task) == "NR" else 0,
            "layout_n_words": float(n_words),
            "layout_n_lines": float(n_lines),
            "layout_sentence_char_count": float(len(sentence_text)),
            "layout_sentence_nonspace_char_count": float(
                len("".join(sentence_text.split()))
            ),
            "layout_word_char_mean": safe_mean(word_chars),
            "layout_word_char_std": safe_std(word_chars),
            "layout_word_char_median": safe_median(word_chars),
            "layout_word_char_q90": safe_quantile(word_chars, 0.90),
            "layout_envelope_width_px": envelope_width,
            "layout_envelope_height_px": envelope_height,
            "layout_envelope_area_px2": envelope_area,
            "layout_envelope_aspect_ratio": safe_ratio(
                envelope_width, envelope_height
            ),
            "layout_total_word_box_area_px2": float(np.sum(areas)),
            "layout_word_box_density": safe_ratio(
                float(np.sum(areas)), envelope_area
            ),
            "layout_word_width_mean_px": safe_mean(widths),
            "layout_word_width_std_px": safe_std(widths),
            "layout_word_width_median_px": safe_median(widths),
            "layout_word_width_q90_px": safe_quantile(widths, 0.90),
            "layout_word_height_mean_px": safe_mean(heights),
            "layout_word_height_std_px": safe_std(heights),
            "layout_word_height_median_px": safe_median(heights),
            "layout_line_word_count_mean": safe_mean(line_word_counts),
            "layout_line_word_count_std": safe_std(line_word_counts),
            "layout_line_word_count_max": (
                float(np.max(line_word_counts))
                if line_word_counts
                else math.nan
            ),
            "layout_line_width_mean_px": safe_mean(line_widths),
            "layout_line_width_std_px": safe_std(line_widths),
            "layout_line_width_max_px": (
                float(np.max(line_widths)) if line_widths else math.nan
            ),
            "layout_line_height_mean_px": safe_mean(line_heights),
            "layout_horizontal_gap_mean_px": safe_mean(horizontal_gaps),
            "layout_horizontal_gap_std_px": safe_std(horizontal_gaps),
            "layout_vertical_gap_mean_px": safe_mean(vertical_gaps),
            "layout_vertical_gap_std_px": safe_std(vertical_gaps),
        }
        rows.append(row)

    layout = pd.DataFrame(rows).sort_values(
        ["task", "task_sentence_index"]
    ).reset_index(drop=True)
    if len(layout) != EXPECTED_SENTENCES:
        raise ValueError(
            f"Layout rows={len(layout)}, expected={EXPECTED_SENTENCES}."
        )
    if layout[["task", "task_sentence_index"]].duplicated().any():
        raise ValueError("Layout table contains duplicate sentence keys.")
    counts = layout.groupby("task").size().to_dict()
    if int(counts.get("NR", 0)) != 349 or int(counts.get("TSR", 0)) != 390:
        raise ValueError(f"Unexpected layout task counts: {counts}")

    length_columns = [
        "layout_n_words",
        "layout_n_lines",
        "layout_sentence_char_count",
        "layout_sentence_nonspace_char_count",
        "layout_word_char_mean",
        "layout_word_char_std",
        "layout_word_char_median",
        "layout_word_char_q90",
    ]
    layout_columns = [
        column for column in layout.columns if column.startswith("layout_")
    ]
    forbidden = [
        column
        for column in layout_columns
        if any(token in column for token in ["task", "label", "sentence_index"])
    ]
    if forbidden:
        raise ValueError(f"Forbidden layout feature names: {forbidden}")

    values = layout[layout_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Layout table contains infinite values.")
    return layout, length_columns, layout_columns


def load_merged_data(
    raw_path: Path,
    raw_sets_path: Path,
    soft_path: Path,
    soft_manifest_path: Path,
    word_template_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, Any]]:
    for path in (
        raw_path,
        raw_sets_path,
        soft_path,
        soft_manifest_path,
        word_template_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = validate_trial_structure(pd.read_csv(raw_path), "RawStat")
    soft = validate_trial_structure(pd.read_csv(soft_path), "Soft-AOI")
    require_columns(soft, ["task_sentence_index"], "Soft-AOI")
    soft["task_sentence_index"] = pd.to_numeric(
        soft["task_sentence_index"], errors="raise"
    ).astype(int)

    raw_sets = json.loads(raw_sets_path.read_text(encoding="utf-8"))
    soft_manifest = json.loads(
        soft_manifest_path.read_text(encoding="utf-8")
    )
    raw_content, raw_all, raw_content_source = derive_raw_feature_sets(raw_sets)
    soft_content = list(soft_manifest.get("content_feature_columns", []))
    soft_quality = list(
        soft_manifest.get("membership_quality_feature_columns", [])
    )
    soft_all = list(dict.fromkeys(soft_content + soft_quality))
    if not soft_content or not soft_all:
        raise ValueError(
            "Soft manifest lacks content or membership/quality columns."
        )

    require_columns(raw, raw_all, "RawStat")
    require_columns(soft, soft_all, "Soft-AOI")
    for column in raw_all:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    for column in soft_all:
        soft[column] = pd.to_numeric(soft[column], errors="coerce")
    if np.isinf(raw[raw_all].to_numpy(dtype=float)).any():
        raise ValueError("RawStat features contain infinity.")
    if np.isinf(soft[soft_all].to_numpy(dtype=float)).any():
        raise ValueError("Soft-AOI features contain infinity.")

    raw_rename = {column: f"raw__{column}" for column in raw_all}
    soft_rename = {column: f"soft__{column}" for column in soft_all}
    raw_small = raw[["trial_id", "subject_id", "task"] + raw_all].rename(
        columns=raw_rename
    )
    soft_small = soft[
        [
            "trial_id",
            "subject_id",
            "task",
            "task_sentence_index",
        ]
        + soft_all
    ].rename(columns=soft_rename)

    merged = raw_small.merge(
        soft_small,
        on=["trial_id", "subject_id", "task"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    merge_counts = merged["_merge"].value_counts().to_dict()
    if int(merge_counts.get("both", 0)) != EXPECTED_TRIALS:
        raise ValueError(f"Raw/Soft merge failed: {merge_counts}")
    merged = merged.drop(columns="_merge")

    layout, length_columns, layout_columns = build_layout_table(
        word_template_path
    )
    layout_rename = {
        column: f"layout__{column}" for column in layout_columns
    }
    layout_small = layout[
        ["task", "task_sentence_index", "sentence_key"] + layout_columns
    ].rename(columns=layout_rename)
    merged = merged.merge(
        layout_small,
        on=["task", "task_sentence_index"],
        how="left",
        validate="many_to_one",
    )
    if merged["sentence_key"].isna().any():
        examples = merged.loc[merged["sentence_key"].isna(), "trial_id"].head(20)
        raise ValueError(
            f"Trials without layout mapping: {examples.tolist()}"
        )
    if len(merged) != EXPECTED_TRIALS:
        raise ValueError(f"Merged rows={len(merged)}, expected={EXPECTED_TRIALS}.")
    merged["label_nr"] = merged["task"].map({"NR": 1, "TSR": 0}).astype(int)

    prefixed_length = [layout_rename[column] for column in length_columns]
    prefixed_layout = [layout_rename[column] for column in layout_columns]
    feature_sets = {
        "sentence_length_only": prefixed_length,
        "layout_only": prefixed_layout,
        "rawstat_all": [raw_rename[column] for column in raw_all],
        "soft_aoi_all": [soft_rename[column] for column in soft_all],
        "combined_all": (
            [raw_rename[column] for column in raw_all]
            + [soft_rename[column] for column in soft_all]
        ),
    }
    for name, columns in feature_sets.items():
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate columns in feature set {name}.")
        require_columns(merged, columns, name)

    metadata = {
        "raw_content_source": raw_content_source,
        "raw_content_columns": raw_content,
        "raw_all_columns": raw_all,
        "soft_content_columns": soft_content,
        "soft_quality_columns": soft_quality,
        "soft_all_columns": soft_all,
        "layout_length_columns": length_columns,
        "layout_all_columns": layout_columns,
        "feature_sets_prefixed": feature_sets,
        "raw_nan_cells": int(raw[raw_all].isna().sum().sum()),
        "soft_nan_cells": int(soft[soft_all].isna().sum().sum()),
        "layout_nan_cells": int(layout[layout_columns].isna().sum().sum()),
    }
    return merged, feature_sets, layout, metadata


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
                    max_iter=20000,
                    random_state=seed,
                ),
            ),
        ]
    )


def decision_scores(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float).reshape(-1)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        classes = np.asarray(model.classes_)
        position = int(np.flatnonzero(classes == 1)[0])
        return probabilities[:, position]
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
        pr_auc_nr = float(average_precision_score(y_true, scores))
    except ValueError:
        pr_auc_nr = math.nan
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
        "pr_auc_nr": pr_auc_nr,
        "tn_tsr_correct": int(tn),
        "fp_tsr_as_nr": int(fp),
        "fn_nr_as_tsr": int(fn),
        "tp_nr_correct": int(tp),
    }


def tune_c_subject_groups(
    train: pd.DataFrame,
    feature_columns: list[str],
    c_values: list[float],
    inner_splits: int,
    seed: int,
    outer_subject: str,
    feature_set: str,
) -> tuple[float, list[dict[str, Any]]]:
    groups = train["subject_id"].astype(str).to_numpy()
    n_splits = min(int(inner_splits), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Inner subject CV requires at least two groups.")
    splitter = GroupKFold(n_splits=n_splits)
    x = train[feature_columns]
    y = train["label_nr"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []

    for c_value in c_values:
        for inner_fold, (fit_idx, valid_idx) in enumerate(
            splitter.split(x, y, groups), start=1
        ):
            model = make_svm(c_value, seed)
            model.fit(x.iloc[fit_idx], y[fit_idx])
            pred = model.predict(x.iloc[valid_idx]).astype(int)
            score = float(
                f1_score(
                    y[valid_idx], pred, average="macro", zero_division=0
                )
            )
            rows.append(
                {
                    "held_out_subject": outer_subject,
                    "feature_set": feature_set,
                    "c_value": float(c_value),
                    "inner_fold": int(inner_fold),
                    "n_inner_train": int(len(fit_idx)),
                    "n_inner_valid": int(len(valid_idx)),
                    "inner_macro_f1": score,
                }
            )

    tuning = pd.DataFrame(rows)
    means = tuning.groupby("c_value")["inner_macro_f1"].mean()
    best_mean = float(means.max())
    candidates = means[np.isclose(means, best_mean, rtol=1e-12, atol=1e-12)]
    best_c = float(min(candidates.index.astype(float)))
    for row in rows:
        row["selected_c"] = best_c
        row["selected_for_outer_fold"] = bool(
            math.isclose(float(row["c_value"]), best_c)
        )
    return best_c, rows


def evaluate_layout_subject_loso(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    c_values: list[float],
    inner_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = sorted(data["subject_id"].unique())
    evaluated = ["dummy_reference", "sentence_length_only", "layout_only"]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for set_index, feature_set in enumerate(evaluated, start=1):
        columns = feature_sets.get(feature_set, [])
        print("\n" + "=" * 78)
        print(
            f"Subject-LOSO layout control {set_index}/{len(evaluated)}: "
            f"{feature_set} ({len(columns)} columns)"
        )
        for outer_fold, held_out in enumerate(subjects, start=1):
            print(
                f"  Subject fold {outer_fold:02d}/{len(subjects)}: {held_out}"
            )
            train = data[data["subject_id"] != held_out].copy()
            test = data[data["subject_id"] == held_out].copy()
            y_train = train["label_nr"].to_numpy(dtype=int)
            y_test = test["label_nr"].to_numpy(dtype=int)

            if feature_set == "dummy_reference":
                model = DummyClassifier(strategy="most_frequent")
                x_train = np.zeros((len(train), 1), dtype=float)
                x_test = np.zeros((len(test), 1), dtype=float)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test).astype(int)
                scores = decision_scores(model, x_test)
                selected_c = math.nan
            else:
                selected_c, tuning = tune_c_subject_groups(
                    train=train,
                    feature_columns=columns,
                    c_values=c_values,
                    inner_splits=inner_splits,
                    seed=seed,
                    outer_subject=held_out,
                    feature_set=feature_set,
                )
                tuning_rows.extend(tuning)
                model = make_svm(selected_c, seed)
                model.fit(train[columns], y_train)
                y_pred = model.predict(test[columns]).astype(int)
                scores = decision_scores(model, test[columns])

            metrics = score_predictions(y_test, y_pred, scores)
            fold_rows.append(
                {
                    "protocol": "subject_loso_static_layout",
                    "feature_set": feature_set,
                    "outer_fold": int(outer_fold),
                    "held_out_subject": held_out,
                    "n_input_features": int(len(columns)),
                    "selected_c": selected_c,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_test_nr": int(np.sum(y_test == 1)),
                    "n_test_tsr": int(np.sum(y_test == 0)),
                    **metrics,
                }
            )
            ordered_test = test.reset_index(drop=True)
            for index, row in ordered_test.iterrows():
                prediction_rows.append(
                    {
                        "protocol": "subject_loso_static_layout",
                        "feature_set": feature_set,
                        "held_out_subject": held_out,
                        "trial_id": row["trial_id"],
                        "subject_id": row["subject_id"],
                        "task": row["task"],
                        "task_sentence_index": int(row["task_sentence_index"]),
                        "sentence_key": row["sentence_key"],
                        "label_nr": int(row["label_nr"]),
                        "prediction_nr": int(y_pred[index]),
                        "decision_score_nr": float(scores[index]),
                        "selected_c": selected_c,
                    }
                )
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
    )


def assign_sentence_folds(
    layout: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    if n_splits < 2:
        raise ValueError("Sentence folds must be at least 2.")
    if n_splits > min(349, 390):
        raise ValueError("Too many sentence folds for stratification.")
    sentence = layout[
        ["task", "task_sentence_index", "sentence_key", "label_nr"]
    ].copy()
    sentence = sentence.sort_values(
        ["task", "task_sentence_index"]
    ).reset_index(drop=True)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    fold_values = np.full(len(sentence), -1, dtype=int)
    x_dummy = np.zeros((len(sentence), 1), dtype=float)
    y = sentence["label_nr"].to_numpy(dtype=int)
    for fold_index, (_, test_index) in enumerate(
        splitter.split(x_dummy, y), start=0
    ):
        fold_values[test_index] = fold_index
    if np.any(fold_values < 0):
        raise ValueError("Sentence-fold assignment is incomplete.")
    sentence["sentence_fold"] = fold_values

    for fold, group in sentence.groupby("sentence_fold"):
        counts = group.groupby("task").size().to_dict()
        if not counts.get("NR", 0) or not counts.get("TSR", 0):
            raise ValueError(
                f"Sentence fold {fold} lacks a task class: {counts}"
            )
    return sentence


def evaluate_dual_holdout(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    sentence_folds: pd.DataFrame,
    c_value: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_map = sentence_folds[["sentence_key", "sentence_fold"]]
    work = data.merge(
        fold_map,
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if work["sentence_fold"].isna().any():
        raise ValueError("Some trials lack sentence-fold assignments.")
    work["sentence_fold"] = work["sentence_fold"].astype(int)

    subjects = sorted(work["subject_id"].unique())
    folds = sorted(work["sentence_fold"].unique())
    evaluated = [
        "dummy_reference",
        "sentence_length_only",
        "layout_only",
        "rawstat_all",
        "soft_aoi_all",
        "combined_all",
    ]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for subject_index, held_out in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"Dual holdout subject {subject_index:02d}/{len(subjects)}: "
            f"{held_out}"
        )
        for sentence_fold in folds:
            train = work[
                (work["subject_id"] != held_out)
                & (work["sentence_fold"] != sentence_fold)
            ].copy()
            test = work[
                (work["subject_id"] == held_out)
                & (work["sentence_fold"] == sentence_fold)
            ].copy()
            if test.empty:
                raise ValueError(
                    f"Empty dual test: subject={held_out}, fold={sentence_fold}."
                )
            train_sentence_keys = set(train["sentence_key"])
            test_sentence_keys = set(test["sentence_key"])
            if train_sentence_keys & test_sentence_keys:
                raise ValueError("Sentence leakage in dual holdout.")
            if held_out in set(train["subject_id"]):
                raise ValueError("Subject leakage in dual holdout.")
            y_train = train["label_nr"].to_numpy(dtype=int)
            y_test = test["label_nr"].to_numpy(dtype=int)
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError("Dual training split lacks both classes.")
            if set(np.unique(y_test)) != {0, 1}:
                raise ValueError("Dual test split lacks both classes.")

            print(
                f"  Sentence fold {sentence_fold + 1}/{len(folds)}: "
                f"train={len(train)}, test={len(test)}"
            )
            for feature_set in evaluated:
                columns = feature_sets.get(feature_set, [])
                if feature_set == "dummy_reference":
                    model = DummyClassifier(strategy="most_frequent")
                    x_train = np.zeros((len(train), 1), dtype=float)
                    x_test = np.zeros((len(test), 1), dtype=float)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test).astype(int)
                    scores = decision_scores(model, x_test)
                    used_c = math.nan
                else:
                    model = make_svm(c_value, seed)
                    model.fit(train[columns], y_train)
                    y_pred = model.predict(test[columns]).astype(int)
                    scores = decision_scores(model, test[columns])
                    used_c = float(c_value)

                metrics = score_predictions(y_test, y_pred, scores)
                fold_rows.append(
                    {
                        "protocol": "dual_unseen_subject_unseen_sentence",
                        "feature_set": feature_set,
                        "held_out_subject": held_out,
                        "sentence_fold": int(sentence_fold),
                        "n_input_features": int(len(columns)),
                        "c_value": used_c,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "n_train_subjects": int(train["subject_id"].nunique()),
                        "n_train_sentences": int(train["sentence_key"].nunique()),
                        "n_test_sentences": int(test["sentence_key"].nunique()),
                        "n_test_nr": int(np.sum(y_test == 1)),
                        "n_test_tsr": int(np.sum(y_test == 0)),
                        **metrics,
                    }
                )

                ordered_test = test.reset_index(drop=True)
                for index, row in ordered_test.iterrows():
                    prediction_rows.append(
                        {
                            "protocol": (
                                "dual_unseen_subject_unseen_sentence"
                            ),
                            "feature_set": feature_set,
                            "held_out_subject": held_out,
                            "sentence_fold": int(sentence_fold),
                            "trial_id": row["trial_id"],
                            "subject_id": row["subject_id"],
                            "task": row["task"],
                            "task_sentence_index": int(
                                row["task_sentence_index"]
                            ),
                            "sentence_key": row["sentence_key"],
                            "label_nr": int(row["label_nr"]),
                            "prediction_nr": int(y_pred[index]),
                            "decision_score_nr": float(scores[index]),
                            "c_value": used_c,
                        }
                    )

    folds_frame = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    expected = EXPECTED_TRIALS * len(evaluated)
    if len(predictions) != expected:
        raise ValueError(
            f"Dual predictions={len(predictions)}, expected={expected}."
        )
    if predictions[["feature_set", "trial_id"]].duplicated().any():
        raise ValueError("Duplicate dual predictions for a feature/trial pair.")
    counts = predictions.groupby("feature_set").size()
    if not counts.eq(EXPECTED_TRIALS).all():
        raise ValueError(f"Incomplete dual prediction counts: {counts.to_dict()}")
    return folds_frame, predictions


def subject_metrics_from_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature_set, subject), group in predictions.groupby(
        ["feature_set", "held_out_subject"], sort=False
    ):
        metrics = score_predictions(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction_nr"].to_numpy(dtype=int),
            group["decision_score_nr"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "feature_set": feature_set,
                "held_out_subject": subject,
                "n_predictions": int(len(group)),
                "n_input_features": int(
                    0
                    if feature_set == "dummy_reference"
                    else group.get("n_input_features", pd.Series([math.nan])).iloc[0]
                    if "n_input_features" in group.columns
                    else -1
                ),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    for feature_set, group in result.groupby("feature_set"):
        if len(group) != EXPECTED_SUBJECTS:
            raise ValueError(
                f"{feature_set}: subject metrics={len(group)}, "
                f"expected={EXPECTED_SUBJECTS}."
            )
        if not group["n_predictions"].eq(EXPECTED_TRIALS_PER_SUBJECT).all():
            raise ValueError(
                f"{feature_set}: not every subject has 739 predictions."
            )
    return result


def summarize_subject_metrics(
    subject_metrics: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    feature_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in subject_metrics.groupby(
        "feature_set", sort=False
    ):
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "n_subjects": int(len(group)),
            "n_input_features": int(
                (feature_counts or {}).get(feature_set, 0)
            ),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_std"] = float(values.std(ddof=1))
            row[f"{metric}_subject_median"] = float(values.median())
        if predictions is not None:
            pred = predictions[predictions["feature_set"] == feature_set]
            pooled = score_predictions(
                pred["label_nr"].to_numpy(dtype=int),
                pred["prediction_nr"].to_numpy(dtype=int),
                pred["decision_score_nr"].to_numpy(dtype=float),
            )
            for metric in METRICS:
                row[f"{metric}_pooled"] = float(pooled[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "macro_f1_subject_mean", ascending=False
    )


def summarize_outer_folds(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in folds.groupby("feature_set", sort=False):
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "n_subjects": int(len(group)),
            "n_input_features": int(group["n_input_features"].iloc[0]),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_std"] = float(values.std(ddof=1))
            row[f"{metric}_subject_median"] = float(values.median())
        selected = pd.to_numeric(group.get("selected_c"), errors="coerce").dropna()
        row["selected_c_median"] = (
            float(selected.median()) if len(selected) else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "macro_f1_subject_mean", ascending=False
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return output.tolist()
    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * values[index])
        running = max(running, value)
        output[index] = running
    return output.tolist()


def build_pairwise_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_sets = list(subject_metrics["feature_set"].drop_duplicates())
    for metric in ["macro_f1", "balanced_accuracy", "roc_auc", "pr_auc_nr"]:
        indices: list[int] = []
        p_values: list[float] = []
        for first, second in combinations(feature_sets, 2):
            a = subject_metrics[
                subject_metrics["feature_set"] == first
            ].set_index("held_out_subject")[metric]
            b = subject_metrics[
                subject_metrics["feature_set"] == second
            ].set_index("held_out_subject")[metric]
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
                    "n_subjects": int(len(paired)),
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
            indices.append(len(rows) - 1)
            p_values.append(p_value)
        adjusted = holm_adjust(p_values)
        for row_index, adjusted_value in zip(indices, adjusted):
            rows[row_index]["p_holm_within_metric"] = adjusted_value
    return pd.DataFrame(rows)


def combine_with_previous_subject_loso(
    layout_folds: pd.DataFrame,
    previous_outer_folds_path: Path,
) -> pd.DataFrame:
    keep_layout = layout_folds[
        layout_folds["feature_set"].isin(
            ["sentence_length_only", "layout_only"]
        )
    ].copy()
    if not previous_outer_folds_path.is_file():
        dummy = layout_folds[
            layout_folds["feature_set"] == "dummy_reference"
        ].copy()
        return pd.concat([dummy, keep_layout], ignore_index=True)

    previous = pd.read_csv(previous_outer_folds_path)
    required = [
        "feature_set",
        "held_out_subject",
        "n_input_features",
        "macro_f1",
    ]
    require_columns(previous, required, "Previous nested LOSO folds")
    selected = previous[
        previous["feature_set"].isin(
            [
                "dummy_reference",
                "rawstat_all",
                "soft_aoi_all",
                "combined_all",
            ]
        )
    ].copy()
    selected["protocol"] = "subject_loso_previous_nested"
    combined = pd.concat([selected, keep_layout], ignore_index=True, sort=False)
    for feature_set, group in combined.groupby("feature_set"):
        if len(group) != EXPECTED_SUBJECTS:
            raise ValueError(
                f"Combined subject LOSO/{feature_set}: rows={len(group)}."
            )
    return combined


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
        "--word-template",
        default=(
            "data/processed/zuco2/soft_aoi_template/"
            "zuco_soft_aoi_word_template.csv"
        ),
    )
    parser.add_argument(
        "--previous-outer-folds",
        default=(
            "outputs/soft_aoi_rawstat_nested_loso_30hz/"
            "nested_loso_outer_folds.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/soft_aoi_confound_controls_30hz",
    )
    parser.add_argument("--c-values", default="0.01,0.1,1,10")
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--sentence-folds", type=int, default=5)
    parser.add_argument("--dual-c-value", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    c_values = parse_float_list(args.c_values)
    if args.inner_splits < 2:
        raise ValueError("--inner-splits must be at least 2.")
    if args.sentence_folds < 2:
        raise ValueError("--sentence-folds must be at least 2.")
    if args.dual_c_value <= 0:
        raise ValueError("--dual-c-value must be positive.")

    data, feature_sets, layout, metadata = load_merged_data(
        raw_path=Path(args.raw_features),
        raw_sets_path=Path(args.raw_feature_sets),
        soft_path=Path(args.soft_features),
        soft_manifest_path=Path(args.soft_feature_manifest),
        word_template_path=Path(args.word_template),
    )

    print("=== INPUT VALIDATION ===")
    print(f"subjects: {data['subject_id'].nunique()}")
    print(f"trials: {len(data)}")
    print(f"sentences: {data['sentence_key'].nunique()}")
    print(f"RawStat NaN cells: {metadata['raw_nan_cells']}")
    print(f"Soft-AOI NaN cells: {metadata['soft_nan_cells']}")
    print(f"Layout NaN cells: {metadata['layout_nan_cells']}")
    print("Feature sets:")
    for name, columns in feature_sets.items():
        print(f"  {name}: {len(columns)}")

    layout_folds, layout_predictions, layout_tuning = (
        evaluate_layout_subject_loso(
            data=data,
            feature_sets=feature_sets,
            c_values=c_values,
            inner_splits=args.inner_splits,
            seed=args.seed,
        )
    )
    layout_summary = summarize_outer_folds(layout_folds)

    subject_loso_all_folds = combine_with_previous_subject_loso(
        layout_folds=layout_folds,
        previous_outer_folds_path=Path(args.previous_outer_folds),
    )
    subject_loso_all_summary = summarize_outer_folds(
        subject_loso_all_folds
    )
    subject_loso_all_pairwise = build_pairwise_tests(
        subject_loso_all_folds.rename(
            columns={"held_out_subject": "held_out_subject"}
        )
    )

    sentence_folds = assign_sentence_folds(
        layout=layout,
        n_splits=args.sentence_folds,
        seed=args.seed,
    )
    dual_folds, dual_predictions = evaluate_dual_holdout(
        data=data,
        feature_sets=feature_sets,
        sentence_folds=sentence_folds,
        c_value=args.dual_c_value,
        seed=args.seed,
    )

    feature_count_map = {
        "dummy_reference": 0,
        **{name: len(columns) for name, columns in feature_sets.items()},
    }
    dual_subject_metrics = subject_metrics_from_predictions(dual_predictions)
    dual_summary = summarize_subject_metrics(
        subject_metrics=dual_subject_metrics,
        predictions=dual_predictions,
        feature_counts=feature_count_map,
    )
    dual_pairwise = build_pairwise_tests(dual_subject_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "layout_table": output_dir / "sentence_layout_feature_table.csv",
        "sentence_folds": output_dir / "dual_holdout_sentence_folds.csv",
        "layout_outer_folds": output_dir / "layout_subject_loso_outer_folds.csv",
        "layout_predictions": output_dir / "layout_subject_loso_predictions.csv",
        "layout_tuning": output_dir / "layout_subject_loso_inner_tuning.csv",
        "layout_summary": output_dir / "layout_subject_loso_summary.csv",
        "subject_loso_all_folds": output_dir / "subject_loso_confound_outer_folds.csv",
        "subject_loso_all_summary": output_dir / "subject_loso_confound_summary.csv",
        "subject_loso_all_pairwise": output_dir / "subject_loso_confound_pairwise.csv",
        "dual_folds": output_dir / "dual_holdout_fold_metrics.csv",
        "dual_predictions": output_dir / "dual_holdout_predictions.csv",
        "dual_subject_metrics": output_dir / "dual_holdout_subject_metrics.csv",
        "dual_summary": output_dir / "dual_holdout_summary.csv",
        "dual_pairwise": output_dir / "dual_holdout_pairwise_tests.csv",
        "manifest": output_dir / "confound_control_manifest.json",
    }

    layout.to_csv(paths["layout_table"], index=False, encoding="utf-8-sig")
    sentence_folds.to_csv(
        paths["sentence_folds"], index=False, encoding="utf-8-sig"
    )
    layout_folds.to_csv(
        paths["layout_outer_folds"], index=False, encoding="utf-8-sig"
    )
    layout_predictions.to_csv(
        paths["layout_predictions"], index=False, encoding="utf-8-sig"
    )
    layout_tuning.to_csv(
        paths["layout_tuning"], index=False, encoding="utf-8-sig"
    )
    layout_summary.to_csv(
        paths["layout_summary"], index=False, encoding="utf-8-sig"
    )
    subject_loso_all_folds.to_csv(
        paths["subject_loso_all_folds"], index=False, encoding="utf-8-sig"
    )
    subject_loso_all_summary.to_csv(
        paths["subject_loso_all_summary"], index=False, encoding="utf-8-sig"
    )
    subject_loso_all_pairwise.to_csv(
        paths["subject_loso_all_pairwise"], index=False, encoding="utf-8-sig"
    )
    dual_folds.to_csv(
        paths["dual_folds"], index=False, encoding="utf-8-sig"
    )
    dual_predictions.to_csv(
        paths["dual_predictions"], index=False, encoding="utf-8-sig"
    )
    dual_subject_metrics.to_csv(
        paths["dual_subject_metrics"], index=False, encoding="utf-8-sig"
    )
    dual_summary.to_csv(
        paths["dual_summary"], index=False, encoding="utf-8-sig"
    )
    dual_pairwise.to_csv(
        paths["dual_pairwise"], index=False, encoding="utf-8-sig"
    )

    manifest = {
        "inputs": {
            "raw_features": args.raw_features,
            "raw_feature_sets": args.raw_feature_sets,
            "soft_features": args.soft_features,
            "soft_feature_manifest": args.soft_feature_manifest,
            "word_template": args.word_template,
            "previous_outer_folds": args.previous_outer_folds,
        },
        "n_subjects": int(data["subject_id"].nunique()),
        "n_trials": int(len(data)),
        "n_sentences": int(data["sentence_key"].nunique()),
        "subject_loso_layout_control": {
            "outer_cv": "leave-one-subject-out",
            "inner_cv": (
                f"GroupKFold(n_splits={args.inner_splits}) on training subjects"
            ),
            "c_values": c_values,
            "selection_metric": "macro_f1",
        },
        "dual_holdout": {
            "description": (
                "For each held-out subject and stratified sentence fold, "
                "exclude the held-out subject and all held-out sentences "
                "from training; test their intersection."
            ),
            "sentence_folds": int(args.sentence_folds),
            "fixed_c": float(args.dual_c_value),
            "c_selection": (
                "Pre-specified; not selected using dual-holdout test results."
            ),
        },
        "preprocessing": {
            "imputer": (
                "Median with missing indicators, fitted on each training split"
            ),
            "scaler": "StandardScaler fitted on each training split",
            "classifier": "LinearSVC(dual=False)",
        },
        "feature_sets": metadata["feature_sets_prefixed"],
        "layout_feature_definitions": {
            "sentence_length_only": metadata["layout_length_columns"],
            "layout_only": metadata["layout_all_columns"],
        },
        "nan_cells": {
            "raw": metadata["raw_nan_cells"],
            "soft": metadata["soft_nan_cells"],
            "layout": metadata["layout_nan_cells"],
        },
        "seed": int(args.seed),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(
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
    ]
    print("\n=== SUBJECT-LOSO CONFOUND SUMMARY ===")
    print(subject_loso_all_summary[display_columns].to_string(index=False))
    print("\n=== DUAL HOLDOUT: UNSEEN SUBJECT + UNSEEN SENTENCE ===")
    print(dual_summary[display_columns].to_string(index=False))

    key_pairs = {
        ("layout_only", "soft_aoi_all"),
        ("sentence_length_only", "soft_aoi_all"),
        ("rawstat_all", "soft_aoi_all"),
        ("rawstat_all", "combined_all"),
        ("soft_aoi_all", "combined_all"),
    }
    key_tests = dual_pairwise[
        dual_pairwise["metric"].eq("macro_f1")
        & dual_pairwise.apply(
            lambda row: (
                row["reference_feature_set"],
                row["candidate_feature_set"],
            )
            in key_pairs,
            axis=1,
        )
    ]
    if not key_tests.empty:
        print("\n=== DUAL HOLDOUT KEY MACRO-F1 TESTS ===")
        print(
            key_tests[
                [
                    "reference_feature_set",
                    "candidate_feature_set",
                    "reference_mean",
                    "candidate_mean",
                    "mean_difference_candidate_minus_reference",
                    "wins",
                    "ties",
                    "losses",
                    "p_value",
                    "p_holm_within_metric",
                ]
            ].to_string(index=False)
        )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
