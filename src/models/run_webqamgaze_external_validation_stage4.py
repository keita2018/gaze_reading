#!/usr/bin/env python
"""Apply the five frozen ZuCo Common16 models to WebQAmGaze.

This stage is the first stage that produces external-validation predictions.

Before prediction, it corrects the transferred official ``sent_gaze_sacc``
vector to the exact order used by the official ZuCo benchmark implementation:

    0 omission_rate
    1 fixation_number
    2 reading_speed
    3 mean_sacc_dur
    4 max_sacc_velocity
    5 mean_sacc_velocity
    6 max_sacc_dur
    7 mean_sacc_amp
    8 max_sacc_amp

The benchmark implementation forms ``sent_gaze_sacc`` in precisely this order.
Its missing saccade aggregates are replaced with 0.0. The WebQAmGaze saccade
terms remain transferred approximations from consecutive filtered fixations;
they are not EyeLink-detected saccades and must be reported as nonidentical.

No WebQAmGaze label is used for:
- model fitting;
- imputation or scaling;
- feature selection;
- threshold selection;
- calibration;
- AOI selection;
- official-feature unit adjustment.

The already frozen C value, ZuCo-fitted preprocessing, and decision threshold
are loaded directly from each trusted local joblib artifact.

Outputs
-------
corrected_official_formula_features.csv
corrected_official_features_primary.csv
corrected_official_feature_audit.csv
feature_transport_audit.csv
external_predictions.csv
external_pooled_metrics.csv
external_subject_metrics.csv
external_summary.csv
external_confusion_matrices.csv
external_bootstrap_replicates.csv
external_bootstrap_summary.csv
compact_pairwise_subject_tests.csv
compact_pairwise_bootstrap_differences.csv
stage4_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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


MODEL_ORDER = [
    "dummy_reference",
    "rawstat_all",
    "soft_full_all",
    "soft_compact_distribution_membership",
    "official_sent_gaze_sacc",
]
PRIMARY_COMPACT = "soft_compact_distribution_membership"

OFFICIAL_BENCHMARK_ORDER = [
    "omission_rate",
    "fixation_number",
    "reading_speed",
    "mean_sacc_dur",
    "max_sacc_velocity",
    "mean_sacc_velocity",
    "max_sacc_dur",
    "mean_sacc_amp",
    "max_sacc_amp",
]

SOURCE_BROWSER_WIDTH = 1280.0
SOURCE_BROWSER_HEIGHT = 720.0
ZUCO_WIDTH_PX = 800.0
ZUCO_HEIGHT_PX = 600.0
ZUCO_ACTIVE_WIDTH_MM = 400.0
ZUCO_ACTIVE_HEIGHT_MM = 298.9
ZUCO_VIEWING_DISTANCE_MM = 680.0

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc_nr",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def find_model_artifact(frozen_dir: Path, feature_set: str) -> Path:
    expected = frozen_dir / "models" / f"{feature_set}.joblib"
    if expected.is_file():
        return expected
    matches = list(frozen_dir.rglob(f"{feature_set}.joblib"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {feature_set}.joblib under {frozen_dir}"
        )
    raise ValueError(
        f"Multiple artifacts found for {feature_set}: {matches}"
    )


def rectangle_membership(
    x: np.ndarray,
    y: np.ndarray,
    aoi: pd.DataFrame,
) -> np.ndarray:
    return (
        (x[:, None] >= aoi["x_min"].to_numpy(dtype=float)[None, :])
        & (x[:, None] <= aoi["x_max"].to_numpy(dtype=float)[None, :])
        & (y[:, None] >= aoi["y_min"].to_numpy(dtype=float)[None, :])
        & (y[:, None] <= aoi["y_max"].to_numpy(dtype=float)[None, :])
    )


def zuco_degrees_per_pixel() -> tuple[float, float]:
    horizontal_fov_deg = math.degrees(
        2.0
        * math.atan(
            (ZUCO_ACTIVE_WIDTH_MM / 2.0)
            / ZUCO_VIEWING_DISTANCE_MM
        )
    )
    vertical_fov_deg = math.degrees(
        2.0
        * math.atan(
            (ZUCO_ACTIVE_HEIGHT_MM / 2.0)
            / ZUCO_VIEWING_DISTANCE_MM
        )
    )
    return (
        horizontal_fov_deg / ZUCO_WIDTH_PX,
        vertical_fov_deg / ZUCO_HEIGHT_PX,
    )


def prepare_fixations(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("x_px", "y_px", "t_ms", "duration_ms"):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )
    valid = (
        result["valid_fixation"].fillna(False).astype(bool)
        & np.isfinite(result["x_px"])
        & np.isfinite(result["y_px"])
        & np.isfinite(result["t_ms"])
        & np.isfinite(result["duration_ms"])
        & result["duration_ms"].ge(0)
        & result["x_px"].between(0, SOURCE_BROWSER_WIDTH)
        & result["y_px"].between(0, SOURCE_BROWSER_HEIGHT)
    )
    sort_columns = ["t_ms"]
    if "fixation_index" in result.columns:
        sort_columns.append("fixation_index")
    return result[valid].sort_values(sort_columns).reset_index(
        drop=True
    )


def transferred_official_vector(
    fixations: pd.DataFrame,
    aoi: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Create the transferred vector in the benchmark's actual array order."""
    n_words = int(len(aoi))
    if n_words <= 0:
        raise ValueError("Official feature extraction requires word AOIs.")

    frame = prepare_fixations(fixations)
    n_fixations = len(frame)

    if n_fixations:
        x = frame["x_px"].to_numpy(dtype=float)
        y = frame["y_px"].to_numpy(dtype=float)
        membership = rectangle_membership(x, y, aoi)
        inside_any = membership.any(axis=1)

        word_assignments = np.full(n_fixations, -1, dtype=int)
        areas = (
            (aoi["x_max"] - aoi["x_min"])
            * (aoi["y_max"] - aoi["y_min"])
        ).to_numpy(dtype=float)
        for index in np.flatnonzero(inside_any):
            candidates = np.flatnonzero(membership[index])
            word_assignments[index] = int(
                candidates[np.argmin(areas[candidates])]
            )
    else:
        inside_any = np.zeros(0, dtype=bool)
        word_assignments = np.zeros(0, dtype=int)

    n_inside = int(np.sum(inside_any))
    unique_fixated_words = (
        np.unique(word_assignments[word_assignments >= 0])
        if n_inside
        else np.asarray([], dtype=int)
    )

    omission_rate = 1.0 - (
        len(unique_fixated_words) / float(n_words)
    )
    fixation_number = n_inside / float(n_words)

    # Exact benchmark-code-compatible transfer of:
    #   (sum(duration_samples) * 2 / 100) / word_count
    # WebQAmGaze durations are milliseconds. At the ZuCo 500 Hz reference,
    # duration_samples = duration_ms / 2, therefore:
    #   duration_samples * 2 / 100 = duration_ms / 100.
    duration_inside_ms = (
        float(frame.loc[inside_any, "duration_ms"].sum())
        if n_inside
        else 0.0
    )
    reading_speed = duration_inside_ms / 100.0 / float(n_words)

    if n_fixations >= 2:
        starts = frame["t_ms"].to_numpy(dtype=float)
        durations = frame["duration_ms"].to_numpy(dtype=float)
        ends = starts + durations
        saccade_duration_ms_all = starts[1:] - ends[:-1]

        x_zuco = (
            frame["x_px"].to_numpy(dtype=float)
            / SOURCE_BROWSER_WIDTH
            * ZUCO_WIDTH_PX
        )
        y_zuco = (
            frame["y_px"].to_numpy(dtype=float)
            / SOURCE_BROWSER_HEIGHT
            * ZUCO_HEIGHT_PX
        )
        deg_x, deg_y = zuco_degrees_per_pixel()
        amplitude_deg_all = np.hypot(
            np.diff(x_zuco) * deg_x,
            np.diff(y_zuco) * deg_y,
        )
        valid_saccade = (
            np.isfinite(saccade_duration_ms_all)
            & (saccade_duration_ms_all > 0)
            & np.isfinite(amplitude_deg_all)
        )
        saccade_duration_ms = saccade_duration_ms_all[
            valid_saccade
        ]
        amplitude_deg = amplitude_deg_all[valid_saccade]
        velocity_deg_s = (
            amplitude_deg / (saccade_duration_ms / 1000.0)
            if len(saccade_duration_ms)
            else np.asarray([], dtype=float)
        )
    else:
        saccade_duration_ms = np.asarray([], dtype=float)
        amplitude_deg = np.asarray([], dtype=float)
        velocity_deg_s = np.asarray([], dtype=float)

    # The official benchmark extractor replaces absent/NaN saccade aggregates
    # with 0.0. We follow that behavior rather than median-imputing these six
    # transferred event features.
    if len(saccade_duration_ms):
        mean_sacc_dur = float(np.mean(saccade_duration_ms))
        max_sacc_dur = float(np.max(saccade_duration_ms))
        mean_sacc_velocity = float(np.mean(velocity_deg_s))
        max_sacc_velocity = float(np.max(velocity_deg_s))
        mean_sacc_amp = float(np.mean(amplitude_deg))
        max_sacc_amp = float(np.max(amplitude_deg))
    else:
        mean_sacc_dur = 0.0
        max_sacc_dur = 0.0
        mean_sacc_velocity = 0.0
        max_sacc_velocity = 0.0
        mean_sacc_amp = 0.0
        max_sacc_amp = 0.0

    values = {
        "omission_rate": float(omission_rate),
        "fixation_number": float(fixation_number),
        "reading_speed": float(reading_speed),
        "mean_sacc_dur": mean_sacc_dur,
        "max_sacc_velocity": max_sacc_velocity,
        "mean_sacc_velocity": mean_sacc_velocity,
        "max_sacc_dur": max_sacc_dur,
        "mean_sacc_amp": mean_sacc_amp,
        "max_sacc_amp": max_sacc_amp,
    }

    audit = {
        "n_word_boxes": n_words,
        "n_valid_browser_fixations": n_fixations,
        "n_fixations_inside_words": n_inside,
        "n_unique_fixated_words": len(unique_fixated_words),
        "duration_inside_words_ms": duration_inside_ms,
        "n_transferred_saccades": len(saccade_duration_ms),
        "saccade_zero_fill_applied": int(
            len(saccade_duration_ms) == 0
        ),
        "reading_speed_formula": (
            "sum(word_fixation_duration_ms)/100/word_count; "
            "benchmark-code-compatible transferred scale"
        ),
        "saccade_definition": (
            "positive gap from filtered-fixation end to next "
            "filtered-fixation start"
        ),
    }
    return values, audit


def build_corrected_official_table(
    *,
    trials: pd.DataFrame,
    fixations: pd.DataFrame,
    aoi_all: pd.DataFrame,
    official_artifact: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    official_columns = list(
        official_artifact.get("feature_columns", [])
    )
    if len(official_columns) != 9:
        raise ValueError(
            "Frozen official artifact does not have nine feature columns."
        )

    aoi_required = {
        "set_name",
        "trial_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    }
    missing_aoi = sorted(aoi_required - set(aoi_all.columns))
    if missing_aoi:
        raise ValueError(
            f"Set-specific AOI table lacks columns: {missing_aoi}"
        )

    aoi_lookup = {
        (str(set_name), str(trial_name)): group.copy()
        for (set_name, trial_name), group in aoi_all.groupby(
            ["set_name", "trial_name"],
            sort=False,
        )
    }
    fixation_groups = fixations.groupby(
        "external_trial_id",
        sort=False,
    )

    descriptive_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for trial in trials.itertuples(index=False):
        trial_id = str(trial.external_trial_id)
        key = (str(trial.set_name), str(trial.trial_name))
        if key not in aoi_lookup:
            raise ValueError(f"Missing set-specific AOI: {key}")
        trial_fixations = (
            fixation_groups.get_group(trial_id)
            if trial_id in fixation_groups.groups
            else pd.DataFrame(columns=fixations.columns)
        )
        values, audit = transferred_official_vector(
            fixations=trial_fixations,
            aoi=aoi_lookup[key],
        )

        metadata = {
            "external_trial_id": trial_id,
            "trial_id": trial_id,
            "subject_id": str(trial.participant_id),
            "participant_id": str(trial.participant_id),
            "set_name": str(trial.set_name),
            "trial_index": int(trial.trial_index),
            "trial_name": str(trial.trial_name),
            "task": str(trial.task).upper(),
            "label_nr": int(trial.label_nr),
        }
        descriptive_rows.append({**metadata, **values})

        model_row = dict(metadata)
        for index, feature_column in enumerate(official_columns):
            model_row[feature_column] = values[
                OFFICIAL_BENCHMARK_ORDER[index]
            ]
        model_rows.append(model_row)
        audit_rows.append({**metadata, **audit})

    descriptive = pd.DataFrame(descriptive_rows)
    model_ready = pd.DataFrame(model_rows)
    audit = pd.DataFrame(audit_rows)

    values = model_ready[official_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "Corrected official feature table contains non-finite values."
        )

    # Internal order/aggregation consistency checks.
    if (
        descriptive["max_sacc_velocity"]
        < descriptive["mean_sacc_velocity"] - 1e-12
    ).any():
        raise RuntimeError(
            "Corrected official table has max velocity below mean velocity."
        )
    if (
        descriptive["max_sacc_dur"]
        < descriptive["mean_sacc_dur"] - 1e-12
    ).any():
        raise RuntimeError(
            "Corrected official table has max duration below mean duration."
        )
    if (
        descriptive["max_sacc_amp"]
        < descriptive["mean_sacc_amp"] - 1e-12
    ).any():
        raise RuntimeError(
            "Corrected official table has max amplitude below mean amplitude."
        )

    return descriptive, model_ready, audit


def safe_binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    return (
        float(roc_auc_score(y_true, score))
        if len(np.unique(y_true)) == 2
        else math.nan
    )


def safe_binary_pr_auc(
    y_true: np.ndarray,
    score: np.ndarray,
) -> float:
    return (
        float(average_precision_score(y_true, score))
        if len(np.unique(y_true)) == 2
        else math.nan
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
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
        "precision_nr": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_nr": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_nr": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "precision_is": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_is": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_is": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "roc_auc": safe_binary_auc(y_true, score),
        "pr_auc_nr": safe_binary_pr_auc(y_true, score),
        "positive_rate_nr": float(np.mean(y_true == 1)),
        "predicted_nr_rate": float(np.mean(y_pred == 1)),
    }


def load_feature_table(
    path: Path,
    master_ids: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "external_trial_id" not in frame.columns:
        raise ValueError(
            f"{path} lacks external_trial_id."
        )
    if frame["external_trial_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate trial IDs.")
    indexed = frame.set_index(
        frame["external_trial_id"].astype(str)
    )
    missing = sorted(set(master_ids) - set(indexed.index))
    extra = sorted(set(indexed.index) - set(master_ids))
    if missing or extra:
        raise ValueError(
            f"{path} trial IDs differ from master: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return indexed.loc[master_ids].reset_index(drop=True)


def dummy_scores(
    artifact: dict[str, Any],
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = artifact["model"]
    x = np.zeros((n_rows, 1), dtype=float)
    prediction = np.asarray(
        model.predict(x),
        dtype=int,
    ).reshape(-1)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(x),
            dtype=float,
        )
        classes = list(model.classes_)
        if 1 in classes:
            score = probabilities[:, classes.index(1)]
        else:
            score = np.zeros(n_rows, dtype=float)
    else:
        score = prediction.astype(float)
    return prediction, score


def apply_artifact(
    artifact: dict[str, Any],
    feature_table: pd.DataFrame | None,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    feature_set = str(artifact["feature_set"])
    if feature_set == "dummy_reference":
        prediction, score = dummy_scores(artifact, n_rows)
        return prediction, score, 0

    feature_columns = list(artifact["feature_columns"])
    if feature_table is None:
        raise ValueError(f"{feature_set} requires a feature table.")
    missing = sorted(set(feature_columns) - set(feature_table.columns))
    if missing:
        raise ValueError(
            f"{feature_set} input lacks columns: {missing}"
        )

    x = feature_table[feature_columns].copy()
    for column in feature_columns:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    values = x.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{feature_set} input contains infinity.")

    pipeline = artifact["pipeline"]
    score = np.asarray(
        pipeline.decision_function(x),
        dtype=float,
    ).reshape(-1)
    threshold = float(artifact["decision_threshold"])
    prediction = (score >= threshold).astype(int)

    if len(prediction) != n_rows:
        raise RuntimeError(
            f"{feature_set} prediction length mismatch."
        )
    if not np.isfinite(score).all():
        raise RuntimeError(
            f"{feature_set} produced non-finite scores."
        )
    return prediction, score, len(feature_columns)


def build_transport_audit(
    *,
    artifact: dict[str, Any],
    feature_table: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    feature_set = str(artifact["feature_set"])
    if feature_set == "dummy_reference":
        return []

    feature_columns = list(artifact["feature_columns"])
    pipeline = artifact["pipeline"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    statistics = np.asarray(imputer.statistics_, dtype=float)

    if len(statistics) != len(feature_columns):
        raise ValueError(
            f"{feature_set}: imputer statistics length mismatch."
        )

    rows = []
    for index, feature in enumerate(feature_columns):
        external = pd.to_numeric(
            feature_table[feature],
            errors="coerce",
        )
        missing_count = int(external.isna().sum())
        external_median_raw = (
            float(external.median())
            if external.notna().any()
            else math.nan
        )
        imputed = external.fillna(statistics[index])
        external_median_after_imputation = float(
            imputed.median()
        )
        scaler_mean = float(scaler.mean_[index])
        scaler_scale = float(scaler.scale_[index])
        standardized_external_median = (
            (external_median_after_imputation - scaler_mean)
            / scaler_scale
            if scaler_scale > 0
            else math.nan
        )
        rows.append(
            {
                "feature_set": feature_set,
                "feature_index": index,
                "feature_name": feature,
                "n_external_rows": len(external),
                "n_external_missing": missing_count,
                "external_missing_rate": (
                    missing_count / float(len(external))
                ),
                "external_median_raw": external_median_raw,
                "zuco_training_imputer_median": float(
                    statistics[index]
                ),
                "external_median_after_frozen_imputation": (
                    external_median_after_imputation
                ),
                "zuco_training_scaler_mean": scaler_mean,
                "zuco_training_scaler_scale": scaler_scale,
                "external_median_standardized_by_zuco": (
                    standardized_external_median
                ),
            }
        )
    return rows


def build_prediction_outputs(
    *,
    trials: pd.DataFrame,
    artifacts: dict[str, dict[str, Any]],
    feature_tables: dict[str, pd.DataFrame | None],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    y_true = trials["label_nr"].to_numpy(dtype=int)
    prediction_frames: list[pd.DataFrame] = []
    pooled_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []

    metadata = trials[
        [
            "external_trial_id",
            "participant_id",
            "set_name",
            "trial_index",
            "trial_name",
            "task",
            "label_nr",
        ]
    ].copy()

    for feature_set in MODEL_ORDER:
        artifact = artifacts[feature_set]
        prediction, score, n_features = apply_artifact(
            artifact=artifact,
            feature_table=feature_tables[feature_set],
            n_rows=len(trials),
        )
        predictions = metadata.copy()
        predictions.insert(0, "feature_set", feature_set)
        predictions["decision_score_nr"] = score
        predictions["decision_threshold"] = artifact.get(
            "decision_threshold",
            math.nan,
        )
        predictions["predicted_label_nr"] = prediction
        predictions["predicted_task"] = np.where(
            prediction == 1,
            "NR",
            "IS",
        )
        predictions["correct"] = (
            prediction == y_true
        ).astype(int)
        prediction_frames.append(predictions)

        pooled = compute_metrics(y_true, prediction, score)
        pooled_rows.append(
            {
                "feature_set": feature_set,
                "n_input_features": n_features,
                "n_trials": len(trials),
                "n_participants": int(
                    trials["participant_id"].nunique()
                ),
                **pooled,
            }
        )

        cm = confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        )
        for true_label, true_name in ((0, "IS"), (1, "NR")):
            for predicted_label, predicted_name in (
                (0, "IS"),
                (1, "NR"),
            ):
                confusion_rows.append(
                    {
                        "feature_set": feature_set,
                        "true_label": true_label,
                        "true_task": true_name,
                        "predicted_label": predicted_label,
                        "predicted_task": predicted_name,
                        "count": int(
                            cm[true_label, predicted_label]
                        ),
                    }
                )

        for participant_id, group in predictions.groupby(
            "participant_id",
            sort=True,
        ):
            participant_y = group["label_nr"].to_numpy(dtype=int)
            participant_pred = group[
                "predicted_label_nr"
            ].to_numpy(dtype=int)
            participant_score = group[
                "decision_score_nr"
            ].to_numpy(dtype=float)
            metrics = compute_metrics(
                participant_y,
                participant_pred,
                participant_score,
            )
            subject_rows.append(
                {
                    "feature_set": feature_set,
                    "participant_id": participant_id,
                    "n_trials": len(group),
                    "n_nr": int(np.sum(participant_y == 1)),
                    "n_is": int(np.sum(participant_y == 0)),
                    **metrics,
                }
            )

        transport_rows.extend(
            build_transport_audit(
                artifact=artifact,
                feature_table=feature_tables[feature_set],
            )
        )

    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(pooled_rows),
        pd.DataFrame(subject_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(transport_rows),
    )


def build_external_summary(
    pooled: pd.DataFrame,
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    pooled_lookup = pooled.set_index("feature_set")
    for feature_set in MODEL_ORDER:
        group = subject_metrics[
            subject_metrics["feature_set"].eq(feature_set)
        ]
        pooled_row = pooled_lookup.loc[feature_set]
        row = {
            "feature_set": feature_set,
            "n_input_features": int(
                pooled_row["n_input_features"]
            ),
            "n_trials": int(pooled_row["n_trials"]),
            "n_participants": int(
                pooled_row["n_participants"]
            ),
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "pr_auc_nr",
            "f1_nr",
            "f1_is",
        ):
            row[f"{metric}_pooled"] = float(
                pooled_row[metric]
            )
            row[f"{metric}_subject_mean"] = float(
                group[metric].mean()
            )
            row[f"{metric}_subject_std"] = float(
                group[metric].std(ddof=1)
            )
        row["predicted_nr_rate_pooled"] = float(
            pooled_row["predicted_nr_rate"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite_indexes = np.flatnonzero(np.isfinite(values))
    if not len(finite_indexes):
        return adjusted.tolist()

    order = finite_indexes[
        np.argsort(values[finite_indexes])
    ]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def build_compact_subject_tests(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in ("macro_f1", "balanced_accuracy"):
        compact = (
            subject_metrics[
                subject_metrics["feature_set"].eq(
                    PRIMARY_COMPACT
                )
            ][["participant_id", metric]]
            .rename(columns={metric: "compact_value"})
        )
        metric_rows: list[dict[str, Any]] = []
        for comparator in MODEL_ORDER:
            if comparator == PRIMARY_COMPACT:
                continue
            other = (
                subject_metrics[
                    subject_metrics["feature_set"].eq(
                        comparator
                    )
                ][["participant_id", metric]]
                .rename(columns={metric: "comparator_value"})
            )
            paired = compact.merge(
                other,
                on="participant_id",
                how="inner",
                validate="one_to_one",
            ).dropna()
            differences = (
                paired["compact_value"]
                - paired["comparator_value"]
            ).to_numpy(dtype=float)

            if not len(differences):
                statistic = math.nan
                p_value = math.nan
            elif np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            metric_rows.append(
                {
                    "metric": metric,
                    "reference_feature_set": comparator,
                    "candidate_feature_set": PRIMARY_COMPACT,
                    "n_paired_participants": len(paired),
                    "reference_mean": float(
                        paired["comparator_value"].mean()
                    ),
                    "candidate_mean": float(
                        paired["compact_value"].mean()
                    ),
                    "mean_difference_candidate_minus_reference": (
                        float(np.mean(differences))
                    ),
                    "median_difference_candidate_minus_reference": (
                        float(np.median(differences))
                    ),
                    "wins": int(np.sum(differences > 0)),
                    "ties": int(np.sum(differences == 0)),
                    "losses": int(np.sum(differences < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
        adjusted = holm_adjust(
            row["p_value"] for row in metric_rows
        )
        for row, p_holm in zip(metric_rows, adjusted):
            row["p_holm_within_metric"] = p_holm
        rows.extend(metric_rows)
    return pd.DataFrame(rows)


def cluster_bootstrap(
    *,
    predictions: pd.DataFrame,
    trials: pd.DataFrame,
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    participants = sorted(
        trials["participant_id"].astype(str).unique()
    )

    model_arrays: dict[str, dict[str, Any]] = {}
    for feature_set in MODEL_ORDER:
        group = predictions[
            predictions["feature_set"].eq(feature_set)
        ].copy()
        per_participant: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for participant_id, participant_group in group.groupby(
            "participant_id",
            sort=False,
        ):
            per_participant[str(participant_id)] = (
                participant_group["label_nr"].to_numpy(dtype=int),
                participant_group[
                    "predicted_label_nr"
                ].to_numpy(dtype=int),
                participant_group[
                    "decision_score_nr"
                ].to_numpy(dtype=float),
            )
        model_arrays[feature_set] = {
            "per_participant": per_participant
        }

    replicate_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []

    for replicate in range(n_bootstraps):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        replicate_metrics: dict[str, dict[str, float]] = {}
        for feature_set in MODEL_ORDER:
            per_participant = model_arrays[feature_set][
                "per_participant"
            ]
            y_parts = []
            pred_parts = []
            score_parts = []
            for participant_id in sampled:
                y, pred, score = per_participant[
                    str(participant_id)
                ]
                y_parts.append(y)
                pred_parts.append(pred)
                score_parts.append(score)

            y_boot = np.concatenate(y_parts)
            pred_boot = np.concatenate(pred_parts)
            score_boot = np.concatenate(score_parts)
            metrics = compute_metrics(
                y_boot,
                pred_boot,
                score_boot,
            )
            replicate_metrics[feature_set] = metrics
            replicate_rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "feature_set": feature_set,
                    "n_sampled_participant_clusters": len(sampled),
                    "n_trials_with_cluster_multiplicity": len(
                        y_boot
                    ),
                    **{
                        metric: metrics[metric]
                        for metric in METRIC_NAMES
                    },
                }
            )

        for comparator in MODEL_ORDER:
            if comparator == PRIMARY_COMPACT:
                continue
            for metric in (
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc_nr",
            ):
                difference_rows.append(
                    {
                        "bootstrap_replicate": replicate,
                        "metric": metric,
                        "reference_feature_set": comparator,
                        "candidate_feature_set": PRIMARY_COMPACT,
                        "difference_candidate_minus_reference": (
                            replicate_metrics[PRIMARY_COMPACT][
                                metric
                            ]
                            - replicate_metrics[comparator][metric]
                        ),
                    }
                )

    replicates = pd.DataFrame(replicate_rows)
    summary_rows = []
    for (feature_set, metric), group in replicates.melt(
        id_vars=[
            "bootstrap_replicate",
            "feature_set",
            "n_sampled_participant_clusters",
            "n_trials_with_cluster_multiplicity",
        ],
        value_vars=METRIC_NAMES,
        var_name="metric",
        value_name="value",
    ).groupby(["feature_set", "metric"], sort=False):
        values = group["value"].dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "feature_set": feature_set,
                "metric": metric,
                "n_bootstrap_values": len(values),
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_std": float(
                    np.std(values, ddof=1)
                ),
                "ci95_lower": float(
                    np.quantile(values, 0.025)
                ),
                "ci95_upper": float(
                    np.quantile(values, 0.975)
                ),
            }
        )

    difference_replicates = pd.DataFrame(difference_rows)
    difference_summary_rows = []
    for keys, group in difference_replicates.groupby(
        [
            "metric",
            "reference_feature_set",
            "candidate_feature_set",
        ],
        sort=False,
    ):
        metric, reference, candidate = keys
        values = group[
            "difference_candidate_minus_reference"
        ].dropna().to_numpy(dtype=float)
        difference_summary_rows.append(
            {
                "metric": metric,
                "reference_feature_set": reference,
                "candidate_feature_set": candidate,
                "n_bootstrap_values": len(values),
                "mean_difference_candidate_minus_reference": float(
                    np.mean(values)
                ),
                "ci95_lower": float(
                    np.quantile(values, 0.025)
                ),
                "ci95_upper": float(
                    np.quantile(values, 0.975)
                ),
                "bootstrap_probability_candidate_gt_reference": (
                    float(np.mean(values > 0))
                ),
            }
        )

    difference_summary = pd.DataFrame(
        difference_summary_rows
    )
    return (
        replicates,
        pd.DataFrame(summary_rows),
        difference_summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage2"
        ),
    )
    parser.add_argument(
        "--stage3a-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage3a"
        ),
    )
    parser.add_argument(
        "--stage3c-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage3c"
        ),
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path(
            "outputs/frozen_zuco_external_validation_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage4"
        ),
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=2000,
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.n_bootstraps < 100:
        raise ValueError("--n-bootstraps must be at least 100.")

    stage2_dir = args.stage2_dir.resolve()
    stage3a_dir = args.stage3a_dir.resolve()
    stage3c_dir = args.stage3c_dir.resolve()
    frozen_dir = args.frozen_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "trials": require_file(
            stage2_dir
            / "trial_manifest_primary_candidates.csv"
        ),
        "fixations": require_file(
            stage2_dir / "filtered_fixation_events.parquet"
        ),
        "rawstat": require_file(
            stage3a_dir / "rawstat_features_primary.csv"
        ),
        "soft_full": require_file(
            stage3c_dir / "soft_full_features_primary.csv"
        ),
        "soft_compact": require_file(
            stage3c_dir / "soft_compact_features_primary.csv"
        ),
        "set_specific_aoi": require_file(
            stage3c_dir / "set_specific_word_aoi.csv"
        ),
        "stage3c_manifest": require_file(
            stage3c_dir / "stage3c_manifest.json"
        ),
    }

    trials = pd.read_csv(input_paths["trials"])
    trials["external_trial_id"] = trials[
        "external_trial_id"
    ].astype(str)
    trials["participant_id"] = trials[
        "participant_id"
    ].astype(str)
    trials["task"] = trials["task"].astype(str).str.upper()
    trials["label_nr"] = pd.to_numeric(
        trials["label_nr"],
        errors="raise",
    ).astype(int)
    trials = trials.sort_values(
        ["participant_id", "trial_index"]
    ).reset_index(drop=True)

    if set(trials["label_nr"].unique()) != {0, 1}:
        raise ValueError("External primary trials lack both labels.")
    if trials["external_trial_id"].duplicated().any():
        raise ValueError("External trial manifest has duplicates.")

    fixations = pd.read_parquet(input_paths["fixations"])
    aoi_all = pd.read_csv(input_paths["set_specific_aoi"])

    artifact_paths = {
        feature_set: find_model_artifact(
            frozen_dir,
            feature_set,
        )
        for feature_set in MODEL_ORDER
    }
    artifacts = {
        feature_set: joblib.load(path)
        for feature_set, path in artifact_paths.items()
    }

    for feature_set in MODEL_ORDER:
        artifact = artifacts[feature_set]
        if artifact.get("feature_set") != feature_set:
            raise ValueError(
                f"Artifact identity mismatch: requested={feature_set}, "
                f"stored={artifact.get('feature_set')}"
            )
        if (
            feature_set != "dummy_reference"
            and float(artifact["decision_threshold"]) != 0.0
        ):
            raise ValueError(
                f"{feature_set} threshold is not frozen at 0.0."
            )

    (
        corrected_official_descriptive,
        corrected_official_ready,
        corrected_official_audit,
    ) = build_corrected_official_table(
        trials=trials,
        fixations=fixations,
        aoi_all=aoi_all,
        official_artifact=artifacts[
            "official_sent_gaze_sacc"
        ],
    )

    master_ids = trials["external_trial_id"].tolist()
    feature_tables: dict[str, pd.DataFrame | None] = {
        "dummy_reference": None,
        "rawstat_all": load_feature_table(
            input_paths["rawstat"],
            master_ids,
        ),
        "soft_full_all": load_feature_table(
            input_paths["soft_full"],
            master_ids,
        ),
        "soft_compact_distribution_membership": (
            load_feature_table(
                input_paths["soft_compact"],
                master_ids,
            )
        ),
        "official_sent_gaze_sacc": load_feature_table(
            output_dir
            / "_temporary_corrected_official.csv",
            master_ids,
        )
        if False
        else corrected_official_ready.set_index(
            corrected_official_ready[
                "external_trial_id"
            ].astype(str)
        ).loc[master_ids].reset_index(drop=True),
    }

    (
        predictions,
        pooled,
        subject_metrics,
        confusion_matrices,
        transport_audit,
    ) = build_prediction_outputs(
        trials=trials,
        artifacts=artifacts,
        feature_tables=feature_tables,
    )
    summary = build_external_summary(
        pooled,
        subject_metrics,
    )
    subject_tests = build_compact_subject_tests(
        subject_metrics
    )

    (
        bootstrap_replicates,
        bootstrap_summary,
        bootstrap_differences,
    ) = cluster_bootstrap(
        predictions=predictions,
        trials=trials,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )

    paths = {
        "corrected_official_descriptive": (
            output_dir
            / "corrected_official_formula_features.csv"
        ),
        "corrected_official_ready": (
            output_dir
            / "corrected_official_features_primary.csv"
        ),
        "corrected_official_audit": (
            output_dir
            / "corrected_official_feature_audit.csv"
        ),
        "transport_audit": (
            output_dir / "feature_transport_audit.csv"
        ),
        "predictions": (
            output_dir / "external_predictions.csv"
        ),
        "pooled_metrics": (
            output_dir / "external_pooled_metrics.csv"
        ),
        "subject_metrics": (
            output_dir / "external_subject_metrics.csv"
        ),
        "summary": (
            output_dir / "external_summary.csv"
        ),
        "confusion": (
            output_dir / "external_confusion_matrices.csv"
        ),
        "bootstrap_replicates": (
            output_dir / "external_bootstrap_replicates.csv"
        ),
        "bootstrap_summary": (
            output_dir / "external_bootstrap_summary.csv"
        ),
        "subject_tests": (
            output_dir / "compact_pairwise_subject_tests.csv"
        ),
        "bootstrap_differences": (
            output_dir
            / "compact_pairwise_bootstrap_differences.csv"
        ),
        "manifest": output_dir / "stage4_manifest.json",
    }

    corrected_official_descriptive.to_csv(
        paths["corrected_official_descriptive"],
        index=False,
        encoding="utf-8-sig",
    )
    corrected_official_ready.to_csv(
        paths["corrected_official_ready"],
        index=False,
        encoding="utf-8-sig",
    )
    corrected_official_audit.to_csv(
        paths["corrected_official_audit"],
        index=False,
        encoding="utf-8-sig",
    )
    transport_audit.to_csv(
        paths["transport_audit"],
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        paths["pooled_metrics"],
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
    confusion_matrices.to_csv(
        paths["confusion"],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_replicates.to_csv(
        paths["bootstrap_replicates"],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        paths["bootstrap_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    subject_tests.to_csv(
        paths["subject_tests"],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_differences.to_csv(
        paths["bootstrap_differences"],
        index=False,
        encoding="utf-8-sig",
    )

    official_medians = {
        feature: float(
            corrected_official_descriptive[feature].median()
        )
        for feature in OFFICIAL_BENCHMARK_ORDER
    }
    artifact_hashes = {
        feature_set: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for feature_set, path in artifact_paths.items()
    }
    input_hashes = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in input_paths.items()
    }

    manifest = {
        "stage": "webqamgaze_external_validation_stage4_frozen_application",
        "version": "1.1",
        "created_at_utc": utc_now_iso(),
        "n_primary_trials": len(trials),
        "n_primary_participants": int(
            trials["participant_id"].nunique()
        ),
        "task_counts": {
            str(task): int(count)
            for task, count in trials["task"].value_counts().items()
        },
        "model_order": MODEL_ORDER,
        "primary_candidate": PRIMARY_COMPACT,
        "fixed_protocol": {
            "label_mapping": {"IS": 0, "NR": 1},
            "decision_threshold": 0.0,
            "no_external_retraining": True,
            "no_external_threshold_tuning": True,
            "no_external_imputation_fit": True,
            "cluster_bootstrap_unit": "participant",
            "n_bootstraps": args.n_bootstraps,
            "seed": args.seed,
        },
        "official_transferred_condition": {
            "benchmark_array_order": OFFICIAL_BENCHMARK_ORDER,
            "reading_speed_formula": (
                "sum(word_fixation_duration_ms)/100/word_count"
            ),
            "missing_saccade_aggregates": "0.0 as in benchmark extractor",
            "saccade_event_source": (
                "positive gaps between consecutive WebQAmGaze filtered "
                "fixations; not EyeLink events"
            ),
            "external_medians": official_medians,
            "reporting_label": (
                "transferred event-based official-feature-formula condition"
            ),
        },
        "artifact_files": artifact_hashes,
        "input_files": input_hashes,
        "outputs": {
            key: str(path)
            for key, path in paths.items()
            if key != "manifest"
        },
        "constraints": [
            "No WebQAmGaze sample was used to fit or modify a model.",
            "All preprocessing pipelines are the frozen ZuCo pipelines.",
            "All non-dummy predictions use each artifact's frozen threshold.",
            "Official transferred features are nonidentical to EyeLink official features.",
            "The official condition is included because it was prespecified, but should be interpreted as a transfer baseline.",
        ],
    }
    paths["manifest"].write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE EXTERNAL VALIDATION STAGE 4 ===")
    print(
        f"External primary data: "
        f"{trials['participant_id'].nunique()} participants, "
        f"{len(trials)} trials"
    )
    print(
        "Corrected official benchmark array order: "
        + " | ".join(OFFICIAL_BENCHMARK_ORDER)
    )
    official_feature_columns = list(
        artifacts["official_sent_gaze_sacc"]["feature_columns"]
    )
    corrected_official_missing_cells = int(
        corrected_official_ready[official_feature_columns]
        .isna()
        .sum()
        .sum()
    )
    zero_filled_saccade_trials = int(
        corrected_official_audit[
            "saccade_zero_fill_applied"
        ].sum()
    )
    print(
        "Corrected official missing cells: "
        f"{corrected_official_missing_cells}"
    )
    print(
        "Trials with zero-filled absent transferred saccades: "
        f"{zero_filled_saccade_trials}"
    )
    print("Corrected external official medians:")
    for feature in OFFICIAL_BENCHMARK_ORDER:
        print(f"  {feature}: {official_medians[feature]:.6g}")

    print("=== POOLED EXTERNAL METRICS ===")
    display_columns = [
        "feature_set",
        "n_input_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc_nr",
        "predicted_nr_rate",
    ]
    print(
        pooled[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("=== SUBJECT-MEAN EXTERNAL METRICS ===")
    display_summary = summary[
        [
            "feature_set",
            "macro_f1_subject_mean",
            "balanced_accuracy_subject_mean",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
        ]
    ]
    print(
        display_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("=== COMPACT PAIRED SUBJECT TESTS ===")
    print(
        subject_tests.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    print("=== COMPACT CLUSTER-BOOTSTRAP DIFFERENCES ===")
    print(
        bootstrap_differences.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    print("Written:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
