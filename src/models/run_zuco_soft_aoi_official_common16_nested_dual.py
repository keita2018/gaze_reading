#!/usr/bin/env python
"""Final matched comparison of Soft-AOI, RawStat, and official ZuCo features.

This is the confirmatory clean-condition comparison for the current study.

All representations are evaluated on exactly the same:
- 16 subjects;
- 11,324 trials for which official sent_gaze_sacc features are available;
- unseen-subject + unseen-sentence outer test splits;
- nested, split-local hyperparameter selection procedure.

Compared representations
------------------------
dummy_reference
    Most-frequent training-label prediction.

rawstat_all
    Eighteen rate-invariant RawStat features.

soft_full_all
    All 41 Soft-AOI content and membership/quality features.

soft_compact_distribution_membership
    The 27-dimensional rate-robust Soft-AOI candidate:
    15 distribution features + 12 membership/quality features.

official_sent_gaze_sacc
    Official ZuCo fixation/saccade-derived sentence feature vector.

Leakage control
---------------
Each outer test split contains one held-out subject and one held-out sentence
fold. Outer training excludes:
1. every trial from that subject;
2. every occurrence of every sentence in the test sentence fold.

C is selected independently for each representation and outer split. Each
inner validation split also holds out both a group of subjects and a sentence
fold. Median imputation and standardization are fit only on the corresponding
inner or outer training data.

The compact feature set was selected in earlier exploratory ablation and rate
robustness analyses. This matched official comparison should therefore be
reported transparently as a final confirmatory comparison of the frozen
27-feature candidate.
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
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_zuco_soft_aoi_ablation import DISTRIBUTION_BASE
from run_zuco_soft_aoi_confound_controls import (
    METRICS,
    derive_raw_feature_sets,
    score_predictions,
)


EXPECTED_SUBJECTS = 16
EXPECTED_COMMON_TRIALS = 11324
COMPACT_NAME = "soft_compact_distribution_membership"
FEATURE_SET_ORDER = [
    "dummy_reference",
    "rawstat_all",
    "soft_full_all",
    COMPACT_NAME,
    "official_sent_gaze_sacc",
]


def parse_c_grid(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("Every C value must be positive.")
    return values


def prepare_labels(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"trial_id", "subject_id", "task"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks columns: {missing}")

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()
    result["label_nr"] = result["task"].map(
        {"NR": 1, "TSR": 0}
    )
    if result["label_nr"].isna().any():
        bad = sorted(
            result.loc[
                result["label_nr"].isna(),
                "task",
            ].unique()
        )
        raise ValueError(f"{name} has unexpected tasks: {bad}")
    result["label_nr"] = result["label_nr"].astype(int)
    return result


def load_common_map(
    path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    common = prepare_labels(
        pd.read_csv(path),
        "Common16 official map",
    )
    required = {"task_sentence_index"}
    missing = sorted(required - set(common.columns))
    if missing:
        raise ValueError(
            f"Common16 official map lacks columns: {missing}"
        )

    official_columns = sorted(
        [
            column
            for column in common.columns
            if column.startswith("official_feature_")
            and column.rsplit("_", 1)[-1].isdigit()
        ],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )

    if len(common) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Common rows={len(common)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if common["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Common subjects={common['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )
    if common["trial_id"].duplicated().any():
        raise ValueError(
            "Common16 official map contains duplicate trial IDs."
        )
    if not official_columns:
        raise ValueError(
            "Official sent_gaze_sacc feature columns were not found."
        )

    common["task_sentence_index"] = pd.to_numeric(
        common["task_sentence_index"],
        errors="raise",
    ).astype(int)
    common["sentence_key"] = common.apply(
        lambda row: (
            f"{str(row['task']).upper()}:"
            f"{int(row['task_sentence_index']):03d}"
        ),
        axis=1,
    )
    return common, official_columns


def load_soft_manifest(
    path: Path,
) -> tuple[list[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    content = list(
        manifest.get("content_feature_columns", [])
    )
    quality = list(
        manifest.get(
            "membership_quality_feature_columns",
            [],
        )
    )
    if len(content) != 29:
        raise ValueError(
            f"Soft content feature count={len(content)}, expected=29."
        )
    if len(quality) != 12:
        raise ValueError(
            f"Soft quality feature count={len(quality)}, expected=12."
        )

    compact = list(DISTRIBUTION_BASE) + quality
    if len(compact) != 27:
        raise ValueError(
            f"Compact Soft-AOI count={len(compact)}, expected=27."
        )
    missing_distribution = sorted(
        set(DISTRIBUTION_BASE) - set(content)
    )
    if missing_distribution:
        raise ValueError(
            "Distribution features are absent from Soft content: "
            f"{missing_distribution}"
        )
    return content, quality


def load_sentence_folds(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    folds = pd.read_csv(path)
    required = {"sentence_key", "sentence_fold"}
    missing = sorted(required - set(folds.columns))
    if missing:
        raise ValueError(
            f"Sentence-fold file lacks columns: {missing}"
        )

    folds = folds[
        ["sentence_key", "sentence_fold"]
    ].copy()
    folds["sentence_key"] = folds["sentence_key"].astype(str)
    folds["sentence_fold"] = pd.to_numeric(
        folds["sentence_fold"],
        errors="raise",
    ).astype(int)

    if folds["sentence_key"].duplicated().any():
        raise ValueError(
            "Sentence-fold file contains duplicate sentence keys."
        )
    if folds["sentence_fold"].nunique() < 2:
        raise ValueError(
            "Sentence-fold file must contain at least two folds."
        )
    return folds


def load_matched_table(
    common: pd.DataFrame,
    official_columns: list[str],
    raw_path: Path,
    raw_sets_path: Path,
    soft_path: Path,
    soft_manifest_path: Path,
    sentence_folds_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, list[str]],
    dict[str, Any],
]:
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not raw_sets_path.is_file():
        raise FileNotFoundError(raw_sets_path)
    if not soft_path.is_file():
        raise FileNotFoundError(soft_path)

    common_ids = set(common["trial_id"])

    raw = prepare_labels(
        pd.read_csv(raw_path),
        "RawStat table",
    )
    raw = raw[raw["trial_id"].isin(common_ids)].copy()
    if len(raw) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Matched RawStat rows={len(raw)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if raw["trial_id"].duplicated().any():
        raise ValueError(
            "Matched RawStat table contains duplicate trial IDs."
        )
    if set(raw["trial_id"]) != common_ids:
        raise ValueError(
            "Matched RawStat trial IDs differ from common map."
        )

    raw_sets = json.loads(
        raw_sets_path.read_text(encoding="utf-8")
    )
    _, raw_all, raw_source = derive_raw_feature_sets(raw_sets)
    if len(raw_all) != 18:
        raise ValueError(
            f"RawStat all count={len(raw_all)}, expected=18."
        )
    missing_raw = sorted(set(raw_all) - set(raw.columns))
    if missing_raw:
        raise ValueError(
            f"RawStat table lacks selected features: {missing_raw}"
        )

    soft = prepare_labels(
        pd.read_csv(soft_path),
        "Soft-AOI table",
    )
    soft = soft[soft["trial_id"].isin(common_ids)].copy()
    if len(soft) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Matched Soft-AOI rows={len(soft)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if soft["trial_id"].duplicated().any():
        raise ValueError(
            "Matched Soft-AOI table contains duplicate trial IDs."
        )
    if set(soft["trial_id"]) != common_ids:
        raise ValueError(
            "Matched Soft-AOI trial IDs differ from common map."
        )

    soft_content, soft_quality = load_soft_manifest(
        soft_manifest_path
    )
    soft_full = soft_content + soft_quality
    soft_compact = list(DISTRIBUTION_BASE) + soft_quality
    missing_soft = sorted(set(soft_full) - set(soft.columns))
    if missing_soft:
        raise ValueError(
            f"Soft-AOI table lacks selected features: {missing_soft}"
        )

    for column in raw_all:
        raw[column] = pd.to_numeric(
            raw[column],
            errors="coerce",
        )
    for column in soft_full:
        soft[column] = pd.to_numeric(
            soft[column],
            errors="coerce",
        )
    for column in official_columns:
        common[column] = pd.to_numeric(
            common[column],
            errors="coerce",
        )

    if np.isinf(raw[raw_all].to_numpy(dtype=float)).any():
        raise ValueError("RawStat selected features contain infinity.")
    if np.isinf(
        soft[soft_full].to_numpy(dtype=float)
    ).any():
        raise ValueError("Soft-AOI selected features contain infinity.")
    if np.isinf(
        common[official_columns].to_numpy(dtype=float)
    ).any():
        raise ValueError("Official features contain infinity.")

    raw_rename = {
        column: f"raw__{column}" for column in raw_all
    }
    soft_rename = {
        column: f"soft__{column}" for column in soft_full
    }
    official_rename = {
        column: f"official__{column}"
        for column in official_columns
    }

    base_columns = [
        "trial_id",
        "subject_id",
        "task",
        "label_nr",
        "task_sentence_index",
        "sentence_key",
    ]
    base = common[
        base_columns + official_columns
    ].rename(columns=official_rename)

    raw_small = raw[
        ["trial_id", "subject_id", "task", "label_nr"] + raw_all
    ].rename(columns=raw_rename)
    soft_small = soft[
        ["trial_id", "subject_id", "task", "label_nr"] + soft_full
    ].rename(columns=soft_rename)

    merged = base.merge(
        raw_small.drop(columns=["label_nr"]),
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        soft_small.drop(columns=["label_nr"]),
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Merged common table rows={len(merged)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )

    sentence_folds = load_sentence_folds(
        sentence_folds_path
    )
    merged = merged.merge(
        sentence_folds,
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if merged["sentence_fold"].isna().any():
        missing_keys = sorted(
            merged.loc[
                merged["sentence_fold"].isna(),
                "sentence_key",
            ].unique()
        )
        raise ValueError(
            "Common trials lack sentence-fold assignments: "
            f"{missing_keys[:30]}"
        )
    merged["sentence_fold"] = merged[
        "sentence_fold"
    ].astype(int)

    feature_sets = {
        "rawstat_all": [
            raw_rename[column] for column in raw_all
        ],
        "soft_full_all": [
            soft_rename[column] for column in soft_full
        ],
        COMPACT_NAME: [
            soft_rename[column] for column in soft_compact
        ],
        "official_sent_gaze_sacc": [
            official_rename[column]
            for column in official_columns
        ],
    }
    metadata = {
        "raw_feature_source": raw_source,
        "raw_feature_count": len(raw_all),
        "soft_full_feature_count": len(soft_full),
        "soft_compact_feature_count": len(soft_compact),
        "official_feature_count": len(official_columns),
        "n_sentence_folds": int(
            merged["sentence_fold"].nunique()
        ),
        "n_unique_sentences": int(
            merged["sentence_key"].nunique()
        ),
        "subjects": sorted(
            merged["subject_id"].unique()
        ),
    }
    return (
        merged.sort_values("trial_id").reset_index(drop=True),
        feature_sets,
        metadata,
    )


def fit_preprocessor(
    x_train: pd.DataFrame,
    x_other: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )
    train_imputed = imputer.fit_transform(x_train)
    other_imputed = imputer.transform(x_other)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    other_scaled = scaler.transform(other_imputed)
    return train_scaled, other_scaled, imputer, scaler


def fit_linear_svc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    seed: int,
) -> LinearSVC:
    model = LinearSVC(
        C=float(c_value),
        dual=False,
        max_iter=30000,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def subject_mean_macro_f1(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> float:
    scored = frame[
        ["subject_id", "label_nr"]
    ].copy().reset_index(drop=True)
    scored["prediction_nr"] = np.asarray(
        predictions,
        dtype=int,
    )

    values: list[float] = []
    for _, group in scored.groupby(
        "subject_id",
        sort=False,
    ):
        values.append(
            float(
                f1_score(
                    group["label_nr"].to_numpy(dtype=int),
                    group["prediction_nr"].to_numpy(dtype=int),
                    average="macro",
                    labels=[0, 1],
                    zero_division=0,
                )
            )
        )
    return float(np.mean(values)) if values else math.nan


def select_c_nested_dual_holdout(
    outer_train: pd.DataFrame,
    feature_columns: list[str],
    c_grid: list[float],
    inner_subject_splits: int,
    seed: int,
    outer_subject: str,
    outer_sentence_fold: int,
    feature_set: str,
) -> tuple[float, list[dict[str, Any]]]:
    subjects = np.asarray(
        sorted(outer_train["subject_id"].unique()),
        dtype=object,
    )
    n_splits = min(
        int(inner_subject_splits),
        int(len(subjects)),
    )
    if n_splits < 2:
        raise ValueError(
            "At least two inner subject folds are required."
        )

    group_splitter = GroupKFold(n_splits=n_splits)
    subject_positions = np.arange(len(subjects))
    sentence_folds = sorted(
        outer_train["sentence_fold"].unique()
    )
    if len(sentence_folds) < 2:
        raise ValueError(
            "Outer training data must contain at least two sentence folds."
        )

    score_lists = {
        float(c_value): [] for c_value in c_grid
    }
    rows: list[dict[str, Any]] = []

    for subject_fold, (
        inner_train_positions,
        inner_validation_positions,
    ) in enumerate(
        group_splitter.split(
            subject_positions,
            groups=subjects,
        ),
        start=1,
    ):
        inner_train_subjects = set(
            subjects[inner_train_positions]
        )
        inner_validation_subjects = set(
            subjects[inner_validation_positions]
        )

        for validation_sentence_fold in sentence_folds:
            inner_train = outer_train[
                outer_train["subject_id"].isin(
                    inner_train_subjects
                )
                & outer_train["sentence_fold"].ne(
                    validation_sentence_fold
                )
            ].copy()
            inner_validation = outer_train[
                outer_train["subject_id"].isin(
                    inner_validation_subjects
                )
                & outer_train["sentence_fold"].eq(
                    validation_sentence_fold
                )
            ].copy()

            if inner_train.empty or inner_validation.empty:
                continue
            if set(inner_train["subject_id"]) & set(
                inner_validation["subject_id"]
            ):
                raise ValueError(
                    "Inner subject leakage detected."
                )
            if set(inner_train["sentence_key"]) & set(
                inner_validation["sentence_key"]
            ):
                raise ValueError(
                    "Inner sentence leakage detected."
                )

            y_train = inner_train[
                "label_nr"
            ].to_numpy(dtype=int)
            y_validation = inner_validation[
                "label_nr"
            ].to_numpy(dtype=int)
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError(
                    "Inner training split lacks both classes."
                )

            x_train, x_validation, _, _ = fit_preprocessor(
                inner_train[feature_columns],
                inner_validation[feature_columns],
            )

            for c_value in c_grid:
                model = fit_linear_svc(
                    x_train=x_train,
                    y_train=y_train,
                    c_value=c_value,
                    seed=seed,
                )
                predictions = model.predict(
                    x_validation
                ).astype(int)
                score = subject_mean_macro_f1(
                    inner_validation,
                    predictions,
                )
                score_lists[float(c_value)].append(score)
                rows.append(
                    {
                        "outer_subject": outer_subject,
                        "outer_sentence_fold": int(
                            outer_sentence_fold
                        ),
                        "feature_set": feature_set,
                        "inner_subject_fold": int(
                            subject_fold
                        ),
                        "inner_validation_sentence_fold": int(
                            validation_sentence_fold
                        ),
                        "c_value": float(c_value),
                        "n_inner_train": int(
                            len(inner_train)
                        ),
                        "n_inner_validation": int(
                            len(inner_validation)
                        ),
                        "n_inner_validation_subjects": int(
                            inner_validation[
                                "subject_id"
                            ].nunique()
                        ),
                        "inner_subject_mean_macro_f1": (
                            float(score)
                        ),
                    }
                )

    aggregate_rows: list[dict[str, Any]] = []
    for c_value in c_grid:
        scores = np.asarray(
            score_lists[float(c_value)],
            dtype=float,
        )
        if scores.size == 0:
            raise ValueError(
                f"No valid inner scores for C={c_value}."
            )
        aggregate_rows.append(
            {
                "c_value": float(c_value),
                "mean_score": float(
                    np.nanmean(scores)
                ),
                "n_inner_splits": int(
                    np.sum(np.isfinite(scores))
                ),
            }
        )

    aggregate_rows.sort(
        key=lambda row: (
            -row["mean_score"],
            row["c_value"],
        )
    )
    best_c = float(
        aggregate_rows[0]["c_value"]
    )

    for aggregate in aggregate_rows:
        rows.append(
            {
                "outer_subject": outer_subject,
                "outer_sentence_fold": int(
                    outer_sentence_fold
                ),
                "feature_set": feature_set,
                "inner_subject_fold": 0,
                "inner_validation_sentence_fold": -1,
                "c_value": float(
                    aggregate["c_value"]
                ),
                "n_inner_train": math.nan,
                "n_inner_validation": math.nan,
                "n_inner_validation_subjects": math.nan,
                "inner_subject_mean_macro_f1": float(
                    aggregate["mean_score"]
                ),
                "n_aggregated_inner_splits": int(
                    aggregate["n_inner_splits"]
                ),
            }
        )

    return best_c, rows


def dummy_scores(
    model: DummyClassifier,
    x_test: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(x_test),
        dtype=float,
    )
    classes = np.asarray(model.classes_)
    if 1 in classes:
        position = int(
            np.flatnonzero(classes == 1)[0]
        )
        return probabilities[:, position]
    return np.zeros(len(x_test), dtype=float)


def evaluate_outer_dual_holdout(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    c_grid: list[float],
    inner_subject_splits: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    subjects = sorted(data["subject_id"].unique())
    sentence_folds = sorted(
        data["sentence_fold"].unique()
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []

    total_outer = len(subjects) * len(sentence_folds)
    outer_index = 0

    for held_out_subject in subjects:
        for held_out_sentence_fold in sentence_folds:
            outer_index += 1
            print(
                f"[Outer {outer_index:03d}/{total_outer}] "
                f"subject={held_out_subject}, "
                f"sentence_fold={held_out_sentence_fold}"
            )

            outer_train = data[
                data["subject_id"].ne(
                    held_out_subject
                )
                & data["sentence_fold"].ne(
                    held_out_sentence_fold
                )
            ].copy()
            outer_test = data[
                data["subject_id"].eq(
                    held_out_subject
                )
                & data["sentence_fold"].eq(
                    held_out_sentence_fold
                )
            ].copy()

            if outer_test.empty:
                raise ValueError(
                    f"Empty outer test split: "
                    f"{held_out_subject}, "
                    f"fold={held_out_sentence_fold}."
                )
            if held_out_subject in set(
                outer_train["subject_id"]
            ):
                raise ValueError(
                    "Outer subject leakage detected."
                )
            if set(outer_train["sentence_key"]) & set(
                outer_test["sentence_key"]
            ):
                raise ValueError(
                    "Outer sentence leakage detected."
                )

            y_train = outer_train[
                "label_nr"
            ].to_numpy(dtype=int)
            y_test = outer_test[
                "label_nr"
            ].to_numpy(dtype=int)
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError(
                    "Outer training split lacks both classes."
                )

            # Dummy reference.
            dummy = DummyClassifier(
                strategy="most_frequent"
            )
            x_dummy_train = np.zeros(
                (len(outer_train), 1),
                dtype=float,
            )
            x_dummy_test = np.zeros(
                (len(outer_test), 1),
                dtype=float,
            )
            dummy.fit(
                x_dummy_train,
                y_train,
            )
            dummy_predictions = dummy.predict(
                x_dummy_test
            ).astype(int)
            dummy_decision = dummy_scores(
                dummy,
                x_dummy_test,
            )
            dummy_metrics = score_predictions(
                y_test,
                dummy_predictions,
                dummy_decision,
            )
            fold_rows.append(
                {
                    "feature_set": "dummy_reference",
                    "held_out_subject": held_out_subject,
                    "held_out_sentence_fold": int(
                        held_out_sentence_fold
                    ),
                    "n_input_features": 0,
                    "selected_c": math.nan,
                    "n_train": int(len(outer_train)),
                    "n_test": int(len(outer_test)),
                    **dummy_metrics,
                }
            )
            ordered_test = outer_test.reset_index(
                drop=True
            )
            for index, row in ordered_test.iterrows():
                prediction_rows.append(
                    {
                        "feature_set": "dummy_reference",
                        "held_out_subject": (
                            held_out_subject
                        ),
                        "held_out_sentence_fold": int(
                            held_out_sentence_fold
                        ),
                        "trial_id": row["trial_id"],
                        "subject_id": row["subject_id"],
                        "task": row["task"],
                        "sentence_key": row[
                            "sentence_key"
                        ],
                        "true_label_nr": int(
                            row["label_nr"]
                        ),
                        "predicted_label_nr": int(
                            dummy_predictions[index]
                        ),
                        "score_nr": float(
                            dummy_decision[index]
                        ),
                    }
                )

            # Learned representations.
            for feature_set in FEATURE_SET_ORDER[1:]:
                columns = feature_sets[feature_set]
                best_c, tuning_rows = (
                    select_c_nested_dual_holdout(
                        outer_train=outer_train,
                        feature_columns=columns,
                        c_grid=c_grid,
                        inner_subject_splits=(
                            inner_subject_splits
                        ),
                        seed=seed,
                        outer_subject=held_out_subject,
                        outer_sentence_fold=(
                            held_out_sentence_fold
                        ),
                        feature_set=feature_set,
                    )
                )
                inner_rows.extend(tuning_rows)
                selected_rows.append(
                    {
                        "feature_set": feature_set,
                        "held_out_subject": (
                            held_out_subject
                        ),
                        "held_out_sentence_fold": int(
                            held_out_sentence_fold
                        ),
                        "selected_c": float(best_c),
                        "n_input_features": int(
                            len(columns)
                        ),
                    }
                )

                x_train, x_test, _, _ = fit_preprocessor(
                    outer_train[columns],
                    outer_test[columns],
                )
                model = fit_linear_svc(
                    x_train=x_train,
                    y_train=y_train,
                    c_value=best_c,
                    seed=seed,
                )
                predictions = model.predict(
                    x_test
                ).astype(int)
                scores = np.asarray(
                    model.decision_function(x_test),
                    dtype=float,
                ).reshape(-1)
                metrics = score_predictions(
                    y_test,
                    predictions,
                    scores,
                )
                fold_rows.append(
                    {
                        "feature_set": feature_set,
                        "held_out_subject": (
                            held_out_subject
                        ),
                        "held_out_sentence_fold": int(
                            held_out_sentence_fold
                        ),
                        "n_input_features": int(
                            len(columns)
                        ),
                        "selected_c": float(
                            best_c
                        ),
                        "n_train": int(
                            len(outer_train)
                        ),
                        "n_test": int(
                            len(outer_test)
                        ),
                        **metrics,
                    }
                )

                for index, row in (
                    ordered_test.iterrows()
                ):
                    prediction_rows.append(
                        {
                            "feature_set": feature_set,
                            "held_out_subject": (
                                held_out_subject
                            ),
                            "held_out_sentence_fold": int(
                                held_out_sentence_fold
                            ),
                            "trial_id": row["trial_id"],
                            "subject_id": row["subject_id"],
                            "task": row["task"],
                            "sentence_key": row[
                                "sentence_key"
                            ],
                            "true_label_nr": int(
                                row["label_nr"]
                            ),
                            "predicted_label_nr": int(
                                predictions[index]
                            ),
                            "score_nr": float(
                                scores[index]
                            ),
                        }
                    )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(selected_rows)
    inner = pd.DataFrame(inner_rows)

    expected_prediction_rows = (
        len(FEATURE_SET_ORDER)
        * EXPECTED_COMMON_TRIALS
    )
    if len(predictions) != expected_prediction_rows:
        raise ValueError(
            f"Prediction rows={len(predictions)}, "
            f"expected={expected_prediction_rows}."
        )
    if predictions[
        ["feature_set", "trial_id"]
    ].duplicated().any():
        raise ValueError(
            "Predictions contain duplicate feature-set/trial rows."
        )
    return folds, predictions, selected, inner


def build_subject_metrics(
    predictions: pd.DataFrame,
    feature_counts: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (
        feature_set,
        subject,
    ), group in predictions.groupby(
        ["feature_set", "held_out_subject"],
        sort=False,
    ):
        metrics = score_predictions(
            group["true_label_nr"].to_numpy(dtype=int),
            group[
                "predicted_label_nr"
            ].to_numpy(dtype=int),
            group["score_nr"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "feature_set": feature_set,
                "held_out_subject": subject,
                "n_input_features": int(
                    feature_counts[feature_set]
                ),
                "n_predictions": int(len(group)),
                **metrics,
            }
        )

    result = pd.DataFrame(rows)
    counts = result.groupby(
        "feature_set"
    )["held_out_subject"].nunique()
    if not counts.eq(EXPECTED_SUBJECTS).all():
        raise ValueError(
            f"Subject metric counts are invalid: "
            f"{counts.to_dict()}"
        )
    return result


def summarize_subject_metrics(
    subject_metrics: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in subject_metrics.groupby(
        "feature_set",
        sort=False,
    ):
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "n_subjects": int(len(group)),
            "n_input_features": int(
                group["n_input_features"].iloc[0]
            ),
        }
        for metric in METRICS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )
            row[f"{metric}_subject_mean"] = float(
                values.mean()
            )
            row[f"{metric}_subject_std"] = float(
                values.std(ddof=1)
            )

        selected_values = selected.loc[
            selected["feature_set"].eq(
                feature_set
            ),
            "selected_c",
        ]
        row["selected_c_median"] = (
            float(selected_values.median())
            if len(selected_values)
            else math.nan
        )
        row["selected_c_mode"] = (
            float(
                selected_values.value_counts()
                .sort_index()
                .idxmax()
            )
            if len(selected_values)
            else math.nan
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "macro_f1_subject_mean",
        ascending=False,
    )


def summarize_pooled(
    predictions: pd.DataFrame,
    feature_counts: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in predictions.groupby(
        "feature_set",
        sort=False,
    ):
        metrics = score_predictions(
            group["true_label_nr"].to_numpy(dtype=int),
            group[
                "predicted_label_nr"
            ].to_numpy(dtype=int),
            group["score_nr"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "feature_set": feature_set,
                "n_input_features": int(
                    feature_counts[feature_set]
                ),
                "n_predictions": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "macro_f1",
        ascending=False,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return result.tolist()

    order = finite[np.argsort(values[finite])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        adjusted = min(
            1.0,
            (total - rank) * values[index],
        )
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def build_pairwise_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    feature_sets = [
        feature_set
        for feature_set in FEATURE_SET_ORDER
        if feature_set
        in set(subject_metrics["feature_set"])
    ]
    rows: list[dict[str, Any]] = []

    for metric in [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ]:
        metric_positions: list[int] = []
        p_values: list[float] = []

        for reference_index, reference in enumerate(
            feature_sets
        ):
            for candidate in feature_sets[
                reference_index + 1 :
            ]:
                reference_values = (
                    subject_metrics[
                        subject_metrics[
                            "feature_set"
                        ].eq(reference)
                    ]
                    .set_index(
                        "held_out_subject"
                    )[metric]
                    .sort_index()
                )
                candidate_values = (
                    subject_metrics[
                        subject_metrics[
                            "feature_set"
                        ].eq(candidate)
                    ]
                    .set_index(
                        "held_out_subject"
                    )[metric]
                    .sort_index()
                )
                paired = pd.concat(
                    [
                        reference_values.rename(
                            "reference"
                        ),
                        candidate_values.rename(
                            "candidate"
                        ),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                differences = (
                    paired["candidate"]
                    - paired["reference"]
                )

                if len(paired) == 0:
                    statistic = math.nan
                    p_value = math.nan
                elif np.allclose(
                    differences.to_numpy(
                        dtype=float
                    ),
                    0.0,
                ):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    test = wilcoxon(
                        paired["candidate"],
                        paired["reference"],
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                    statistic = float(
                        test.statistic
                    )
                    p_value = float(
                        test.pvalue
                    )

                rows.append(
                    {
                        "metric": metric,
                        "reference_feature_set": (
                            reference
                        ),
                        "candidate_feature_set": (
                            candidate
                        ),
                        "n_subjects": int(
                            len(paired)
                        ),
                        "reference_mean": float(
                            paired["reference"].mean()
                        ),
                        "candidate_mean": float(
                            paired["candidate"].mean()
                        ),
                        "mean_difference_candidate_minus_reference": float(
                            differences.mean()
                        ),
                        "wins": int(
                            np.sum(differences > 0)
                        ),
                        "ties": int(
                            np.sum(
                                np.isclose(
                                    differences,
                                    0.0,
                                )
                            )
                        ),
                        "losses": int(
                            np.sum(differences < 0)
                        ),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "p_holm_within_metric": math.nan,
                    }
                )
                metric_positions.append(
                    len(rows) - 1
                )
                p_values.append(p_value)

        adjusted = holm_adjust(p_values)
        for row_position, adjusted_value in zip(
            metric_positions,
            adjusted,
        ):
            rows[row_position][
                "p_holm_within_metric"
            ] = adjusted_value

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common-map",
        default=(
            "outputs/common16_comparison/"
            "common16_trial_map.csv"
        ),
    )
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
        "--sentence-folds-file",
        default=(
            "outputs/soft_aoi_confound_controls_30hz/"
            "dual_holdout_sentence_folds.csv"
        ),
    )
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1,10",
    )
    parser.add_argument(
        "--inner-subject-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "soft_aoi_official_common16_nested_dual"
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    c_grid = parse_c_grid(args.c_grid)
    if args.inner_subject_splits < 2:
        raise ValueError(
            "--inner-subject-splits must be at least 2."
        )

    common, official_columns = load_common_map(
        Path(args.common_map)
    )
    data, feature_sets, metadata = (
        load_matched_table(
            common=common,
            official_columns=official_columns,
            raw_path=Path(args.raw_features),
            raw_sets_path=Path(
                args.raw_feature_sets
            ),
            soft_path=Path(args.soft_features),
            soft_manifest_path=Path(
                args.soft_feature_manifest
            ),
            sentence_folds_path=Path(
                args.sentence_folds_file
            ),
        )
    )

    print(
        "=== SOFT-AOI VS OFFICIAL COMMON16 INPUT ==="
    )
    print(
        f"subjects: {data['subject_id'].nunique()}"
    )
    print(f"trials: {len(data)}")
    print(
        f"unique sentences: "
        f"{data['sentence_key'].nunique()}"
    )
    print(
        f"sentence folds: "
        f"{data['sentence_fold'].nunique()}"
    )
    print(f"C grid: {c_grid}")
    print(
        f"inner subject folds: "
        f"{args.inner_subject_splits}"
    )
    for feature_set in FEATURE_SET_ORDER:
        count = (
            0
            if feature_set == "dummy_reference"
            else len(feature_sets[feature_set])
        )
        print(
            f"  {feature_set}: {count} features"
        )

    (
        fold_metrics,
        predictions,
        selected,
        inner_scores,
    ) = evaluate_outer_dual_holdout(
        data=data,
        feature_sets=feature_sets,
        c_grid=c_grid,
        inner_subject_splits=(
            args.inner_subject_splits
        ),
        seed=args.seed,
    )

    feature_counts = {
        "dummy_reference": 0,
        **{
            name: len(columns)
            for name, columns in feature_sets.items()
        },
    }
    subject_metrics = build_subject_metrics(
        predictions,
        feature_counts,
    )
    summary = summarize_subject_metrics(
        subject_metrics,
        selected,
    )
    pooled = summarize_pooled(
        predictions,
        feature_counts,
    )
    tests = build_pairwise_tests(
        subject_metrics
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths = {
        "fold_metrics": (
            output_dir
            / "common16_nested_dual_outer_folds.csv"
        ),
        "predictions": (
            output_dir
            / "common16_nested_dual_predictions.csv"
        ),
        "selected": (
            output_dir
            / "common16_nested_dual_selected_c.csv"
        ),
        "inner_scores": (
            output_dir
            / "common16_nested_dual_inner_scores.csv"
        ),
        "subject_metrics": (
            output_dir
            / "common16_nested_dual_subject_metrics.csv"
        ),
        "summary": (
            output_dir
            / "common16_nested_dual_summary.csv"
        ),
        "pooled": (
            output_dir
            / "common16_nested_dual_pooled.csv"
        ),
        "tests": (
            output_dir
            / "common16_nested_dual_pairwise_tests.csv"
        ),
        "feature_sets": (
            output_dir
            / "common16_nested_dual_feature_sets.json"
        ),
        "manifest": (
            output_dir
            / "common16_nested_dual_manifest.json"
        ),
    }

    fold_metrics.to_csv(
        paths["fold_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(
        paths["selected"],
        index=False,
        encoding="utf-8-sig",
    )
    inner_scores.to_csv(
        paths["inner_scores"],
        index=False,
        encoding="utf-8-sig",
    )
    subject_metrics.to_csv(
        paths["subject_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        paths["summary"],
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        paths["pooled"],
        index=False,
        encoding="utf-8-sig",
    )
    tests.to_csv(
        paths["tests"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["feature_sets"].write_text(
        json.dumps(
            {
                "feature_counts": feature_counts,
                "feature_columns": feature_sets,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "analysis_status": (
                    "confirmatory_comparison_of_frozen_"
                    "compact_candidate"
                ),
                "protocol": (
                    "nested_unseen_subject_and_"
                    "unseen_sentence"
                ),
                "n_subjects": EXPECTED_SUBJECTS,
                "n_trials": EXPECTED_COMMON_TRIALS,
                "n_unique_sentences": metadata[
                    "n_unique_sentences"
                ],
                "n_sentence_folds": metadata[
                    "n_sentence_folds"
                ],
                "c_grid": c_grid,
                "inner_subject_splits": int(
                    args.inner_subject_splits
                ),
                "seed": int(args.seed),
                "feature_counts": feature_counts,
                "raw_feature_source": metadata[
                    "raw_feature_source"
                ],
                "inputs": {
                    "common_map": args.common_map,
                    "raw_features": args.raw_features,
                    "raw_feature_sets": (
                        args.raw_feature_sets
                    ),
                    "soft_features": args.soft_features,
                    "soft_feature_manifest": (
                        args.soft_feature_manifest
                    ),
                    "sentence_folds_file": (
                        args.sentence_folds_file
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== COMMON16 NESTED DUAL-HOLDOUT SUMMARY ==="
    )
    print(
        summary[
            [
                "feature_set",
                "n_input_features",
                "macro_f1_subject_mean",
                "balanced_accuracy_subject_mean",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
                "selected_c_median",
                "selected_c_mode",
            ]
        ].to_string(index=False)
    )

    key_pairs = {
        ("rawstat_all", COMPACT_NAME),
        ("soft_full_all", COMPACT_NAME),
        (COMPACT_NAME, "official_sent_gaze_sacc"),
        ("soft_full_all", "official_sent_gaze_sacc"),
        ("rawstat_all", "official_sent_gaze_sacc"),
    }
    key_tests = tests[
        tests["metric"].eq("macro_f1")
        & tests.apply(
            lambda row: (
                row["reference_feature_set"],
                row["candidate_feature_set"],
            )
            in key_pairs,
            axis=1,
        )
    ][
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
    ]
    print(
        "\n=== KEY COMMON16 MACRO-F1 TESTS ==="
    )
    print(key_tests.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
