#!/usr/bin/env python
r"""
Factor frozen OneStop-to-WebQAmGaze scores into measurement, participant, and
residual goal-related components without using target labels for correction.

Source-side measurement model
-----------------------------
For each OneStop trial and each synthetic quality view:

    score_shift = frozen_score(view) - frozen_score(clean)
    quality_delta = quality_features(view) - quality_features(clean)

A ridge model with no intercept predicts score_shift from quality_delta. Alpha
is selected by participant-grouped source OOF RMSE. Because both quantities are
paired within the same source trial, source goal and participant baselines
cancel before fitting the measurement model.

Target-side factorization
-------------------------
For each WebQAmGaze trial:

1. Estimate trial-varying quality deviation relative to the leave-one-trial-out
   participant median quality vector.
2. Predict and subtract the source-learned measurement score bias.
3. Estimate and remove the leave-one-trial-out participant score offset.
4. Add the frozen OneStop source score reference back.

Thus, for each trial:

    raw_score
      = measurement_component
      + participant_component
      + factorized_residual_score

The factorized residual is evaluated with the original frozen threshold.
WebQAmGaze labels are used only after all corrections and predictions have been
formed.

Primary comparison
------------------
measurement_plus_participant versus participant_only for soft_full_all.

Example smoke test
------------------
python -u src\models\run_goal_measurement_participant_factorization.py ^
  --source-views data\processed\onestop\quality_views_asc344_v1\onestop_quality_views_all_features.parquet ^
  --source-oof outputs\frozen_onestop_external_validation_v1\onestop_selected_oof_predictions.parquet ^
  --frozen-model outputs\frozen_onestop_external_validation_v1\models\soft_full_all.joblib ^
  --target-features outputs\webqamgaze_external_validation_stage3c\soft_full_features_primary.csv ^
  --target-calibrated-predictions outputs\webqamgaze_participant_self_calibration_v1\self_calibrated_trial_predictions.csv ^
  --output-dir outputs\goal_measurement_participant_factorization_smoke ^
  --alpha-grid 0.001,0.01,0.1,1,10,100,1000 ^
  --source-cv-splits 5 --source-participant-limit 40 ^
  --bootstrap 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

try:
    import extract_zuco_soft_aoi_features as soft
except ImportError as exc:
    raise ImportError(
        "Could not import src/data/extract_zuco_soft_aoi_features.py."
    ) from exc


MODEL_NAME = "soft_full_all"
QUALITY_FEATURES = list(
    dict.fromkeys(soft.MEMBERSHIP_QUALITY_FEATURES)
)
METHOD_ORDER = (
    "raw_frozen",
    "measurement_only",
    "participant_only",
    "measurement_plus_participant",
)
PRIMARY_REFERENCE = "participant_only"
PRIMARY_COMPARISON = "measurement_plus_participant"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-views", type=Path, required=True)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument(
        "--target-calibrated-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alpha-grid",
        default="0.001,0.01,0.1,1,10,100,1000",
    )
    parser.add_argument("--source-cv-splits", type=int, default=5)
    parser.add_argument(
        "--source-participant-limit",
        type=int,
        default=0,
        help=(
            "Deterministically retain at most this many source "
            "participants for a smoke test; 0 uses all."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_alpha_grid(text: str) -> list[float]:
    values = sorted(
        {
            float(token.strip())
            for token in text.split(",")
            if token.strip()
        }
    )
    if not values or any(value < 0 for value in values):
        raise ValueError("Alpha grid must contain nonnegative values.")
    return values


def first_existing(
    columns: pd.Index,
    aliases: tuple[str, ...],
    purpose: str,
) -> str:
    mapping = {
        str(column).strip().lower(): str(column)
        for column in columns
    }
    for alias in aliases:
        if alias.lower() in mapping:
            return mapping[alias.lower()]
    raise ValueError(
        f"Could not identify {purpose}; aliases={aliases}, "
        f"columns={list(columns)}"
    )


def get_classes(pipeline: Any) -> np.ndarray:
    if hasattr(pipeline, "classes_"):
        return np.asarray(pipeline.classes_)
    if hasattr(pipeline, "steps") and pipeline.steps:
        estimator = pipeline.steps[-1][1]
        if hasattr(estimator, "classes_"):
            return np.asarray(estimator.classes_)
    raise ValueError("Frozen pipeline does not expose classes_.")


def score_for_nr(
    pipeline: Any,
    frame: pd.DataFrame,
) -> np.ndarray:
    classes = get_classes(pipeline)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(
            f"Expected classes {{0,1}}, got {classes.tolist()}."
        )
    raw = np.asarray(pipeline.decision_function(frame))
    if raw.ndim == 1:
        return (
            raw.astype(float)
            if int(classes[1]) == 1
            else (-raw).astype(float)
        )
    positive_index = int(np.where(classes == 1)[0][0])
    return raw[:, positive_index].astype(float)


def load_artifact(
    path: Path,
) -> tuple[Any, list[str], float, dict[str, Any]]:
    artifact = joblib.load(require_file(path))
    if not isinstance(artifact, dict):
        raise TypeError("Frozen model artifact must be a dict.")
    required = {
        "pipeline",
        "feature_columns",
        "decision_threshold",
        "label_to_int",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"Frozen artifact lacks keys: {missing}")

    feature_columns = [
        str(value) for value in artifact["feature_columns"]
    ]
    if len(feature_columns) != 41:
        raise ValueError(
            f"Expected 41 model features, got {len(feature_columns)}."
        )
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Frozen feature columns are not unique.")
    if not set(QUALITY_FEATURES).issubset(feature_columns):
        raise ValueError(
            "Quality features are not a subset of model features."
        )

    labels = {
        str(key).upper(): int(value)
        for key, value in dict(
            artifact["label_to_int"]
        ).items()
    }
    if labels.get("IS") != 0 or labels.get("NR") != 1:
        raise ValueError(
            f"Expected source labels IS=0, NR=1; got {labels}."
        )
    threshold = float(artifact["decision_threshold"])
    if not math.isfinite(threshold):
        raise ValueError("Frozen threshold is not finite.")

    audit = {
        "artifact_version": artifact.get("artifact_version"),
        "feature_set": artifact.get("feature_set"),
        "training_scope": artifact.get("training_scope"),
        "n_features": len(feature_columns),
        "quality_features": QUALITY_FEATURES,
        "decision_threshold": threshold,
        "c_value": artifact.get("c_value"),
    }
    return (
        artifact["pipeline"],
        feature_columns,
        threshold,
        audit,
    )


def select_source_participants(
    frame: pd.DataFrame,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    if limit <= 0:
        return frame
    participants = np.asarray(
        sorted(frame["participant_id"].unique()),
        dtype=object,
    )
    if limit >= len(participants):
        return frame
    rng = np.random.default_rng(seed)
    selected = set(
        rng.choice(
            participants,
            size=limit,
            replace=False,
        ).astype(str)
    )
    return frame[
        frame["participant_id"].isin(selected)
    ].reset_index(drop=True)


def prepare_source_pairs(
    source_views: pd.DataFrame,
    pipeline: Any,
    feature_columns: list[str],
    participant_limit: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "onestop_trial_id",
        "participant_id",
        "view_condition",
        "corruption_repeat",
        *feature_columns,
    }
    missing = sorted(required - set(source_views.columns))
    if missing:
        raise ValueError(f"Source views lack columns: {missing}")

    frame = source_views.copy()
    frame["onestop_trial_id"] = (
        frame["onestop_trial_id"].astype(str)
    )
    frame["participant_id"] = (
        frame["participant_id"].astype(str).str.strip().str.lower()
    )
    frame["view_condition"] = (
        frame["view_condition"].astype(str).str.strip()
    )
    frame = select_source_participants(
        frame,
        participant_limit,
        seed,
    )

    if frame.duplicated(
        [
            "onestop_trial_id",
            "view_condition",
            "corruption_repeat",
        ]
    ).any():
        raise ValueError("Source view keys are duplicated.")

    numeric = frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame["frozen_score"] = score_for_nr(
            pipeline,
            numeric,
        )

    clean = frame[
        frame["view_condition"].eq("clean")
    ][
        [
            "onestop_trial_id",
            "frozen_score",
            *QUALITY_FEATURES,
        ]
    ].copy()
    if clean["onestop_trial_id"].duplicated().any():
        raise ValueError("Multiple clean source views per trial.")
    clean = clean.rename(
        columns={
            "frozen_score": "clean_score",
            **{
                feature: f"clean__{feature}"
                for feature in QUALITY_FEATURES
            },
        }
    )

    paired = frame.merge(
        clean,
        on="onestop_trial_id",
        how="inner",
        validate="many_to_one",
    )
    if len(paired) != len(frame):
        raise ValueError("Some source views lack a clean pair.")

    paired["score_shift"] = (
        paired["frozen_score"]
        - paired["clean_score"]
    )
    delta_columns = []
    for feature in QUALITY_FEATURES:
        column = f"delta__{feature}"
        paired[column] = (
            pd.to_numeric(
                paired[feature],
                errors="coerce",
            )
            - pd.to_numeric(
                paired[f"clean__{feature}"],
                errors="coerce",
            )
        )
        delta_columns.append(column)

    if np.isinf(
        paired[
            [*delta_columns, "score_shift"]
        ].to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "Source paired deltas contain infinity."
        )
    return paired, pd.DataFrame(
        {
            "quality_feature": QUALITY_FEATURES,
            "delta_column": delta_columns,
        }
    )


def participant_folds(
    participants: list[str],
    n_splits: int,
    seed: int,
) -> list[tuple[set[str], set[str]]]:
    if n_splits < 2:
        raise ValueError("source-cv-splits must be at least 2.")
    if len(participants) < n_splits:
        raise ValueError(
            f"Only {len(participants)} source participants for "
            f"{n_splits} folds."
        )
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(
        participants,
        dtype=object,
    )[rng.permutation(len(participants))]
    fold_arrays = np.array_split(shuffled, n_splits)
    splits = []
    all_set = set(shuffled.astype(str))
    for fold_array in fold_arrays:
        valid = set(fold_array.astype(str))
        train = all_set - valid
        if train.intersection(valid):
            raise RuntimeError("Source participant leakage.")
        splits.append((train, valid))
    return splits


def make_measurement_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(with_mean=False),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                    fit_intercept=False,
                ),
            ),
        ]
    )


def source_cv(
    paired: pd.DataFrame,
    delta_columns: list[str],
    alphas: list[float],
    n_splits: int,
    seed: int,
) -> tuple[
    float,
    pd.DataFrame,
    pd.DataFrame,
    Pipeline,
]:
    participants = sorted(
        paired["participant_id"].unique()
    )
    folds = participant_folds(
        participants,
        n_splits,
        seed,
    )
    selection_rows = []
    selected_predictions: pd.DataFrame | None = None

    for alpha in alphas:
        parts = []
        for fold, (train_ids, valid_ids) in enumerate(
            folds,
            start=1,
        ):
            train = paired[
                paired["participant_id"].isin(
                    train_ids
                )
            ]
            valid = paired[
                paired["participant_id"].isin(
                    valid_ids
                )
            ]
            model = make_measurement_model(alpha)
            model.fit(
                train[delta_columns],
                train["score_shift"].to_numpy(
                    dtype=float
                ),
            )
            prediction = model.predict(
                valid[delta_columns]
            ).astype(float)
            part = valid[
                [
                    "onestop_trial_id",
                    "participant_id",
                    "view_condition",
                    "corruption_repeat",
                    "score_shift",
                ]
            ].copy()
            part["predicted_score_shift"] = prediction
            part["alpha"] = alpha
            part["fold"] = fold
            parts.append(part)

        oof = pd.concat(parts, ignore_index=True)
        y = oof["score_shift"].to_numpy(dtype=float)
        prediction = oof[
            "predicted_score_shift"
        ].to_numpy(dtype=float)
        rmse = float(
            mean_squared_error(
                y,
                prediction,
            )
            ** 0.5
        )
        mae = float(
            mean_absolute_error(y, prediction)
        )
        r2 = float(r2_score(y, prediction))
        pearson = float(
            np.corrcoef(y, prediction)[0, 1]
            if np.std(y) > 0
            and np.std(prediction) > 0
            else np.nan
        )
        selection_rows.append(
            {
                "alpha": alpha,
                "oof_rmse": rmse,
                "oof_mae": mae,
                "oof_r2": r2,
                "oof_pearson": pearson,
                "n_rows": len(oof),
                "n_participants": oof[
                    "participant_id"
                ].nunique(),
            }
        )

    selection = pd.DataFrame(selection_rows).sort_values(
        [
            "oof_rmse",
            "oof_mae",
            "alpha",
        ],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected_alpha = float(selection.iloc[0]["alpha"])

    # Recreate selected OOF predictions for output.
    parts = []
    for fold, (train_ids, valid_ids) in enumerate(
        folds,
        start=1,
    ):
        train = paired[
            paired["participant_id"].isin(train_ids)
        ]
        valid = paired[
            paired["participant_id"].isin(valid_ids)
        ]
        model = make_measurement_model(
            selected_alpha
        )
        model.fit(
            train[delta_columns],
            train["score_shift"].to_numpy(dtype=float),
        )
        prediction = model.predict(
            valid[delta_columns]
        ).astype(float)
        part = valid[
            [
                "onestop_trial_id",
                "participant_id",
                "view_condition",
                "corruption_repeat",
                "score_shift",
            ]
        ].copy()
        part["predicted_score_shift"] = prediction
        part["residual"] = (
            part["score_shift"]
            - part["predicted_score_shift"]
        )
        part["alpha"] = selected_alpha
        part["fold"] = fold
        parts.append(part)
    selected_predictions = pd.concat(
        parts,
        ignore_index=True,
    )

    final_model = make_measurement_model(
        selected_alpha
    )
    final_model.fit(
        paired[delta_columns],
        paired["score_shift"].to_numpy(dtype=float),
    )
    return (
        selected_alpha,
        selection,
        selected_predictions,
        final_model,
    )


def source_condition_metrics(
    oof: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for condition, group in oof.groupby(
        "view_condition",
        observed=True,
    ):
        y = group["score_shift"].to_numpy(dtype=float)
        prediction = group[
            "predicted_score_shift"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "view_condition": condition,
                "n_rows": len(group),
                "rmse": float(
                    mean_squared_error(
                        y,
                        prediction,
                    )
                    ** 0.5
                ),
                "mae": float(
                    mean_absolute_error(
                        y,
                        prediction,
                    )
                ),
                "r2": float(
                    r2_score(y, prediction)
                    if np.std(y) > 0
                    else np.nan
                ),
                "pearson": float(
                    np.corrcoef(
                        y,
                        prediction,
                    )[0, 1]
                    if np.std(y) > 0
                    and np.std(prediction) > 0
                    else np.nan
                ),
                "true_shift_mean": float(
                    np.mean(y)
                ),
                "predicted_shift_mean": float(
                    np.mean(prediction)
                ),
            }
        )
    return pd.DataFrame(rows)


def normalize_target_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    trial_column = first_existing(
        frame.columns,
        ID_ALIASES,
        "target trial ID",
    )
    participant_column = first_existing(
        frame.columns,
        PARTICIPANT_ALIASES,
        "target participant ID",
    )
    required = set(QUALITY_FEATURES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Target full features lack quality columns: {missing}"
        )

    result = frame.copy()
    result["external_trial_id"] = (
        result[trial_column].astype(str).str.strip()
    )
    result["participant_id"] = (
        result[participant_column].astype(str).str.strip()
    )
    if result["external_trial_id"].duplicated().any():
        raise ValueError(
            "Target full features contain duplicate trial IDs."
        )
    result[QUALITY_FEATURES] = result[
        QUALITY_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if np.isinf(
        result[QUALITY_FEATURES].to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "Target quality features contain infinity."
        )
    return result[
        [
            "external_trial_id",
            "participant_id",
            *QUALITY_FEATURES,
        ]
    ].copy()


def prepare_target_predictions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "external_trial_id",
        "participant_id",
        "label_nr",
        "model_name",
        "calibration_method",
        "raw_score",
        "calibrated_score",
        "frozen_threshold",
        "prediction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Target calibrated predictions lack columns: {missing}"
        )

    result = frame.copy()
    result = result[
        result["model_name"].astype(str).eq(
            MODEL_NAME
        )
    ].copy()
    raw = result[
        result["calibration_method"].astype(str).eq(
            "raw_frozen"
        )
    ].copy()
    participant = result[
        result["calibration_method"].astype(str).eq(
            "source_median_loo"
        )
    ].copy()

    for name, table in (
        ("raw", raw),
        ("participant", participant),
    ):
        if table["external_trial_id"].duplicated().any():
            raise ValueError(
                f"Duplicate {name} target trial predictions."
            )

    columns = [
        "external_trial_id",
        "participant_id",
        "label_nr",
        "raw_score",
        "frozen_threshold",
    ]
    if "task" in raw.columns:
        columns.append("task")
    raw = raw[columns].copy()
    raw["label_nr"] = pd.to_numeric(
        raw["label_nr"],
        errors="raise",
    ).astype(int)
    raw["raw_score"] = pd.to_numeric(
        raw["raw_score"],
        errors="raise",
    ).astype(float)
    raw["frozen_threshold"] = pd.to_numeric(
        raw["frozen_threshold"],
        errors="raise",
    ).astype(float)

    participant = participant[
        [
            "external_trial_id",
            "calibrated_score",
            "prediction",
        ]
    ].rename(
        columns={
            "calibrated_score": (
                "participant_only_score"
            ),
            "prediction": (
                "participant_only_prediction"
            ),
        }
    )
    merged = raw.merge(
        participant,
        on="external_trial_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(raw):
        raise ValueError(
            "Raw and participant-only trial sets differ."
        )
    return merged, result


def loo_median_matrix(
    values: np.ndarray,
) -> np.ndarray:
    n_rows, n_columns = values.shape
    if n_rows < 3:
        raise ValueError(
            "At least three target trials per participant "
            "are required."
        )
    output = np.empty_like(values, dtype=float)
    for row in range(n_rows):
        output[row, :] = np.nanmedian(
            np.delete(values, row, axis=0),
            axis=0,
        )
    return output


def loo_median_vector(
    values: np.ndarray,
) -> np.ndarray:
    if len(values) < 3:
        raise ValueError(
            "At least three scores per participant are required."
        )
    output = np.empty(len(values), dtype=float)
    for row in range(len(values)):
        output[row] = float(
            np.median(np.delete(values, row))
        )
    return output


def source_reference_median(
    source_oof: pd.DataFrame,
) -> float:
    required = {
        "participant_id",
        "feature_set",
        "score",
    }
    missing = sorted(required - set(source_oof.columns))
    if missing:
        raise ValueError(
            f"Source OOF table lacks columns: {missing}"
        )
    subset = source_oof[
        source_oof["feature_set"].astype(str).eq(
            MODEL_NAME
        )
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No source OOF rows for {MODEL_NAME}."
        )
    subset["score"] = pd.to_numeric(
        subset["score"],
        errors="raise",
    ).astype(float)
    participant_medians = subset.groupby(
        "participant_id",
        observed=True,
    )["score"].median()
    return float(participant_medians.median())


def apply_factorization(
    target: pd.DataFrame,
    measurement_model: Pipeline,
    source_reference: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    participant_rows = []

    for participant_id, group in target.groupby(
        "participant_id",
        observed=True,
        sort=False,
    ):
        group = group.copy().reset_index(drop=True)
        quality = group[
            QUALITY_FEATURES
        ].to_numpy(dtype=float)
        loo_quality_median = loo_median_matrix(
            quality
        )
        quality_delta = (
            quality - loo_quality_median
        )
        delta_frame = pd.DataFrame(
            quality_delta,
            columns=[
                f"delta__{feature}"
                for feature in QUALITY_FEATURES
            ],
        )
        predicted_measurement = (
            measurement_model.predict(
                delta_frame
            ).astype(float)
        )
        raw_score = group[
            "raw_score"
        ].to_numpy(dtype=float)
        measurement_score = (
            raw_score - predicted_measurement
        )
        loo_measurement_median = loo_median_vector(
            measurement_score
        )
        participant_component = (
            loo_measurement_median - source_reference
        )
        factorized_score = (
            measurement_score
            - participant_component
        )
        threshold = group[
            "frozen_threshold"
        ].to_numpy(dtype=float)

        base = group[
            [
                "external_trial_id",
                "participant_id",
                "label_nr",
                "frozen_threshold",
            ]
            + (["task"] if "task" in group.columns else [])
        ].copy()
        base["raw_score"] = raw_score
        base["measurement_component"] = (
            predicted_measurement
        )
        base["measurement_corrected_score"] = (
            measurement_score
        )
        base["participant_component"] = (
            participant_component
        )
        base["factorized_residual_score"] = (
            factorized_score
        )
        base["quality_delta_l2"] = np.sqrt(
            np.nansum(
                np.square(quality_delta),
                axis=1,
            )
        )
        parts.append(base)

        participant_rows.append(
            {
                "participant_id": participant_id,
                "n_trials": len(group),
                "raw_score_median": float(
                    np.median(raw_score)
                ),
                "measurement_corrected_score_median": float(
                    np.median(measurement_score)
                ),
                "participant_component_median": float(
                    np.median(participant_component)
                ),
                "measurement_component_abs_median": float(
                    np.median(
                        np.abs(predicted_measurement)
                    )
                ),
                "factorized_score_median": float(
                    np.median(factorized_score)
                ),
            }
        )

    components = pd.concat(parts, ignore_index=True)
    participant_audit = pd.DataFrame(
        participant_rows
    )
    return components, participant_audit


def create_method_predictions(
    components: pd.DataFrame,
    participant_only: pd.DataFrame,
) -> pd.DataFrame:
    participant_lookup = participant_only[
        [
            "external_trial_id",
            "participant_only_score",
            "participant_only_prediction",
        ]
    ].copy()
    frame = components.merge(
        participant_lookup,
        on="external_trial_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(components):
        raise ValueError(
            "Participant-only lookup is incomplete."
        )

    methods = {
        "raw_frozen": "raw_score",
        "measurement_only": (
            "measurement_corrected_score"
        ),
        "participant_only": (
            "participant_only_score"
        ),
        "measurement_plus_participant": (
            "factorized_residual_score"
        ),
    }
    parts = []
    for method, score_column in methods.items():
        part = frame[
            [
                "external_trial_id",
                "participant_id",
                "label_nr",
                "frozen_threshold",
            ]
            + (["task"] if "task" in frame.columns else [])
        ].copy()
        part["method"] = method
        part["score"] = pd.to_numeric(
            frame[score_column],
            errors="raise",
        ).astype(float)
        part["prediction"] = (
            part["score"].to_numpy(dtype=float)
            >= part["frozen_threshold"].to_numpy(
                dtype=float
            )
        ).astype(int)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


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


def classification_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    if sample_weight is None:
        predicted_nr_rate = float(
            np.mean(prediction == 1)
        )
    else:
        predicted_nr_rate = float(
            np.sum(
                sample_weight
                * (prediction == 1)
            )
            / np.sum(sample_weight)
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
        "predicted_nr_rate": (
            predicted_nr_rate
        ),
    }


def pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for method, group in predictions.groupby(
        "method",
        observed=True,
    ):
        result = classification_metrics(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "method": method,
                "n_trials": len(group),
                "n_participants": group[
                    "participant_id"
                ].nunique(),
                **result,
            }
        )
    table = pd.DataFrame(rows)
    table["method"] = pd.Categorical(
        table["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    return table.sort_values("method").reset_index(
        drop=True
    )


def component_summary(
    components: pd.DataFrame,
) -> pd.DataFrame:
    columns = {
        "raw_score": "raw_score",
        "measurement_component": (
            "measurement_component"
        ),
        "participant_component": (
            "participant_component"
        ),
        "factorized_residual_score": (
            "factorized_residual_score"
        ),
        "quality_delta_l2": "quality_delta_l2",
    }
    rows = []
    for label, column in columns.items():
        values = pd.to_numeric(
            components[column],
            errors="coerce",
        ).to_numpy(dtype=float)
        rows.append(
            {
                "component": label,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "median": float(np.nanmedian(values)),
                "median_absolute_value": float(
                    np.nanmedian(np.abs(values))
                ),
                "q05": float(
                    np.nanquantile(values, 0.05)
                ),
                "q95": float(
                    np.nanquantile(values, 0.95)
                ),
            }
        )

    component_values = components[
        [
            "measurement_component",
            "participant_component",
            "factorized_residual_score",
        ]
    ].astype(float)
    correlation = component_values.corr()
    correlation_rows = []
    for row_name in correlation.index:
        for column_name in correlation.columns:
            correlation_rows.append(
                {
                    "component_a": row_name,
                    "component_b": column_name,
                    "pearson_correlation": float(
                        correlation.loc[
                            row_name,
                            column_name,
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(
        correlation_rows
    )


def bootstrap_deltas(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    participants = np.asarray(
        sorted(predictions["participant_id"].unique()),
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

    prepared: dict[str, dict[str, Any]] = {}
    for method, group in predictions.groupby(
        "method",
        observed=True,
    ):
        frame = group.reset_index(drop=True)
        prepared[str(method)] = {
            "frame": frame,
            "participants": frame[
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

        values_by_method: dict[
            str, dict[str, float]
        ] = {}
        for method, item in prepared.items():
            frame = item["frame"]
            weights = np.fromiter(
                (
                    counts.get(participant, 0)
                    for participant in item[
                        "participants"
                    ]
                ),
                dtype=float,
                count=len(frame),
            )
            values_by_method[method] = (
                classification_metrics(
                    frame[
                        "label_nr"
                    ].to_numpy(dtype=int),
                    frame[
                        "prediction"
                    ].to_numpy(dtype=int),
                    frame[
                        "score"
                    ].to_numpy(dtype=float),
                    sample_weight=weights,
                )
            )

        comparisons = (
            ("raw_frozen", "measurement_only"),
            ("raw_frozen", "participant_only"),
            (
                "participant_only",
                "measurement_plus_participant",
            ),
            (
                "raw_frozen",
                "measurement_plus_participant",
            ),
        )
        for reference, comparison in comparisons:
            for metric_name in metric_names:
                replicate_rows.append(
                    {
                        "replicate": replicate,
                        "reference": reference,
                        "comparison": comparison,
                        "metric": metric_name,
                        "delta": (
                            values_by_method[
                                comparison
                            ][metric_name]
                            - values_by_method[
                                reference
                            ][metric_name]
                        ),
                    }
                )

    replicates = pd.DataFrame(replicate_rows)
    rows = []
    for (
        reference,
        comparison,
        metric_name,
    ), group in replicates.groupby(
        ["reference", "comparison", "metric"],
        observed=True,
    ):
        delta = group["delta"].to_numpy(dtype=float)
        rows.append(
            {
                "reference": reference,
                "comparison": comparison,
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
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.source_cv_splits < 2:
        raise ValueError(
            "--source-cv-splits must be at least 2."
        )
    if args.bootstrap < 1:
        raise ValueError(
            "--bootstrap must be positive."
        )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        frozen_pipeline,
        feature_columns,
        frozen_threshold,
        frozen_audit,
    ) = load_artifact(args.frozen_model)
    alphas = parse_alpha_grid(args.alpha_grid)

    source_views = pd.read_parquet(
        require_file(args.source_views)
    )
    paired, delta_schema = prepare_source_pairs(
        source_views=source_views,
        pipeline=frozen_pipeline,
        feature_columns=feature_columns,
        participant_limit=(
            args.source_participant_limit
        ),
        seed=args.seed,
    )
    delta_columns = delta_schema[
        "delta_column"
    ].astype(str).tolist()

    print(
        "=== GOAL / MEASUREMENT / PARTICIPANT "
        "FACTORIZATION INPUT ==="
    )
    print(
        f"source paired view rows: {len(paired):,}"
    )
    print(
        f"source clean trials: "
        f"{paired['onestop_trial_id'].nunique():,}"
    )
    print(
        f"source participants: "
        f"{paired['participant_id'].nunique()}"
    )
    print(
        f"measurement features: "
        f"{len(QUALITY_FEATURES)}"
    )
    print(f"alpha grid: {alphas}")
    print(
        "target labels used for measurement model: False"
    )

    (
        selected_alpha,
        source_selection,
        source_oof,
        measurement_model,
    ) = source_cv(
        paired=paired,
        delta_columns=delta_columns,
        alphas=alphas,
        n_splits=args.source_cv_splits,
        seed=args.seed,
    )
    source_condition = source_condition_metrics(
        source_oof
    )

    print(
        f"\nSelected source measurement alpha: "
        f"{selected_alpha:g}"
    )
    print("\n=== SOURCE MEASUREMENT MODEL CV ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(source_selection.to_string(index=False))

    print(
        "\n=== SOURCE MEASUREMENT MODEL BY CONDITION ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(source_condition.to_string(index=False))

    target_features = normalize_target_features(
        pd.read_csv(
            require_file(args.target_features),
            low_memory=False,
        )
    )
    target_raw, calibrated_all = (
        prepare_target_predictions(
            pd.read_csv(
                require_file(
                    args.target_calibrated_predictions
                ),
                low_memory=False,
            )
        )
    )
    target = target_raw.merge(
        target_features,
        on=[
            "external_trial_id",
            "participant_id",
        ],
        how="inner",
        validate="one_to_one",
    )
    if len(target) != len(target_raw):
        raise ValueError(
            "Target features/predictions trial sets differ."
        )

    source_reference = source_reference_median(
        pd.read_parquet(
            require_file(args.source_oof)
        )
    )
    components, participant_audit = (
        apply_factorization(
            target=target,
            measurement_model=measurement_model,
            source_reference=source_reference,
        )
    )
    predictions = create_method_predictions(
        components=components,
        participant_only=target_raw,
    )
    pooled = pooled_metrics(predictions)
    (
        component_statistics,
        component_correlations,
    ) = component_summary(components)
    deltas = bootstrap_deltas(
        predictions=predictions,
        n_bootstrap=args.bootstrap,
        seed=args.seed + 100000,
    )

    measurement_model_path = (
        args.output_dir
        / "source_measurement_bias_model.joblib"
    )
    source_selection_path = (
        args.output_dir
        / "source_measurement_model_selection.csv"
    )
    source_oof_path = (
        args.output_dir
        / "source_measurement_oof_predictions.parquet"
    )
    source_condition_path = (
        args.output_dir
        / "source_measurement_condition_metrics.csv"
    )
    delta_schema_path = (
        args.output_dir
        / "measurement_delta_schema.csv"
    )
    components_path = (
        args.output_dir
        / "target_factorized_components.csv"
    )
    participant_audit_path = (
        args.output_dir
        / "target_participant_component_audit.csv"
    )
    predictions_path = (
        args.output_dir
        / "target_factorized_predictions.csv"
    )
    pooled_path = (
        args.output_dir
        / "target_factorized_pooled_metrics.csv"
    )
    component_statistics_path = (
        args.output_dir
        / "target_component_statistics.csv"
    )
    component_correlations_path = (
        args.output_dir
        / "target_component_correlations.csv"
    )
    deltas_path = (
        args.output_dir
        / "target_factorized_bootstrap_deltas.csv"
    )
    manifest_path = (
        args.output_dir
        / "goal_measurement_participant_manifest.json"
    )

    joblib.dump(
        {
            "artifact_version": (
                "goal_measurement_participant_factorization_v1"
            ),
            "model": measurement_model,
            "alpha": selected_alpha,
            "quality_features": QUALITY_FEATURES,
            "delta_columns": delta_columns,
            "target": (
                "frozen score(view) - frozen score(clean)"
            ),
            "fit_intercept": False,
            "source_participants": paired[
                "participant_id"
            ].nunique(),
            "source_rows": len(paired),
            "frozen_model_audit": frozen_audit,
        },
        measurement_model_path,
        compress=3,
    )
    source_selection.to_csv(
        source_selection_path,
        index=False,
        encoding="utf-8-sig",
    )
    source_oof.to_parquet(
        source_oof_path,
        index=False,
        compression="zstd",
    )
    source_condition.to_csv(
        source_condition_path,
        index=False,
        encoding="utf-8-sig",
    )
    delta_schema.to_csv(
        delta_schema_path,
        index=False,
        encoding="utf-8-sig",
    )
    components.to_csv(
        components_path,
        index=False,
        encoding="utf-8-sig",
    )
    participant_audit.to_csv(
        participant_audit_path,
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        pooled_path,
        index=False,
        encoding="utf-8-sig",
    )
    component_statistics.to_csv(
        component_statistics_path,
        index=False,
        encoding="utf-8-sig",
    )
    component_correlations.to_csv(
        component_correlations_path,
        index=False,
        encoding="utf-8-sig",
    )
    deltas.to_csv(
        deltas_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "analysis": (
            "Goal / measurement / participant score factorization"
        ),
        "source_measurement_target": (
            "within-trial frozen score shift between quality view "
            "and clean view"
        ),
        "source_quality_delta": (
            "within-trial quality feature difference"
        ),
        "target_quality_delta": (
            "trial quality vector minus leave-one-trial-out "
            "participant median quality vector"
        ),
        "participant_component": (
            "leave-one-trial-out median measurement-corrected "
            "score minus source OOF reference median"
        ),
        "factorized_residual": (
            "raw score minus predicted measurement component "
            "minus participant component"
        ),
        "target_labels_used_for_factorization": False,
        "primary_reference": PRIMARY_REFERENCE,
        "primary_comparison": PRIMARY_COMPARISON,
        "selected_alpha": selected_alpha,
        "source_reference_median": source_reference,
        "frozen_threshold": frozen_threshold,
        "quality_features": QUALITY_FEATURES,
        "bootstrap": {
            "replicates": args.bootstrap,
            "unit": "participant",
            "paired_methods": True,
        },
        "interpretation_caveat": (
            "The residual score is goal-related after the modeled "
            "measurement and participant components are removed; it "
            "is not claimed to be a perfectly pure causal goal factor."
        ),
        "outputs": {
            "measurement_model": str(
                measurement_model_path
            ),
            "source_selection": str(
                source_selection_path
            ),
            "source_oof": str(source_oof_path),
            "source_condition_metrics": str(
                source_condition_path
            ),
            "target_components": str(
                components_path
            ),
            "target_predictions": str(
                predictions_path
            ),
            "target_metrics": str(pooled_path),
            "bootstrap_deltas": str(
                deltas_path
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
        "\n=== TARGET FACTORIZED POOLED METRICS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(pooled.to_string(index=False))

    print(
        "\n=== TARGET COMPONENT STATISTICS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(
            component_statistics.to_string(
                index=False
            )
        )

    print(
        "\n=== TARGET FACTORIZED BOOTSTRAP DELTAS ==="
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(deltas.to_string(index=False))

    print(f"\nSaved: {source_selection_path}")
    print(f"Saved: {components_path}")
    print(f"Saved: {pooled_path}")
    print(f"Saved: {deltas_path}")
    print(f"Saved: {manifest_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
