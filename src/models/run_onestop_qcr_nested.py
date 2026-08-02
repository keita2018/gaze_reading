#!/usr/bin/env python
r"""
Nested participant-independent evaluation of Quality-Consistency Regularization
(QCR) on paired OneStop clean/degraded Soft-AOI views.

Methods
-------
clean_only
    Linear SVM trained only on clean views. Hyperparameters and threshold are
    selected using clean inner-OOF predictions only.

augmentation
    Linear SVM trained on all clean/degraded views without quality
    regularization.
    Hyperparameters and threshold are selected using training-participant
    inner-OOF predictions.

qcr
    1. Fit median imputation and standardization on clean training views only.
    2. Form paired differences z_degraded - z_clean for the same trial.
    3. Estimate a condition-balanced quality scatter matrix using training
       participants only.
    4. Continuously shrink sensitivity along quality-change directions with
       penalty strength lambda, without deleting any direction.
    5. Train a Linear SVM on all regularized clean/degraded views.

All outer and inner splits are participant-independent. No held-out participant
is used to fit imputation, scaling, quality directions, SVM parameters, or the
decision threshold.

The primary selection target for augmentation and qcr is mean macro-F1 over
the degraded conditions. clean_only is selected using clean macro-F1, because
it must not use synthetic degradation during model selection.

Smoke test example
------------------
python -u src\models\run_onestop_qcr_nested.py ^
  --views data\processed\onestop\quality_views_asc344_v1\onestop_quality_views_all_features.parquet ^
  --schema outputs\frozen_onestop_external_validation_v1\feature_schema.csv ^
  --output-dir outputs\onestop_qcr_smoke ^
  --outer-splits 3 --inner-splits 3 ^
  --c-grid 0.01,0.1 --lambda-grid 0.1,1,10 ^
  --max-participants-per-label 20 ^
  --bootstrap 200 --seed 42

Final example
-------------
python -u src\models\run_onestop_qcr_nested.py ^
  --views data\processed\onestop\quality_views_asc344_v1\onestop_quality_views_all_features.parquet ^
  --schema outputs\frozen_onestop_external_validation_v1\feature_schema.csv ^
  --output-dir outputs\onestop_qcr_nested_v1 ^
  --outer-splits 5 --inner-splits 5 ^
  --c-grid 0.001,0.01,0.1,1 ^
  --lambda-grid 0.01,0.1,1,10,100 ^
  --bootstrap 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


CLEAN = "clean"
DEFAULT_DEGRADED = (
    "rate10",
    "noise50",
    "random40",
    "burst40",
    "high_random",
    "high_burst",
)
METHODS = ("clean_only", "augmentation", "qcr")


@dataclass
class Preprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        imputed = self.imputer.transform(frame)
        return self.scaler.transform(imputed).astype(np.float64)


@dataclass
class Candidate:
    method: str
    c_value: float
    lambda_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument(
        "--c-grid",
        default="0.001,0.01,0.1,1",
    )
    parser.add_argument(
        "--lambda-grid",
        default="0.01,0.1,1,10,100",
        help="Positive quality-consistency penalty strengths.",
    )
    parser.add_argument(
        "--conditions",
        default="clean,rate10,noise50,random40,burst40,"
        "high_random,high_burst",
    )
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=201,
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-participants-per-label",
        type=int,
        default=0,
        help=(
            "Deterministically retain at most this many participants per "
            "label. Use 0 for all participants."
        ),
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_float_grid(text: str) -> list[float]:
    values = sorted(
        {
            float(token.strip())
            for token in text.split(",")
            if token.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("C grid must contain positive values.")
    return values


def parse_lambda_grid(text: str) -> list[float]:
    values = sorted(
        {
            float(token.strip())
            for token in text.split(",")
            if token.strip()
        }
    )
    if not values or any(
        (not math.isfinite(value)) or value <= 0
        for value in values
    ):
        raise ValueError(
            "lambda grid must contain positive finite values."
        )
    return values


def parse_conditions(text: str) -> list[str]:
    conditions = [
        token.strip()
        for token in text.split(",")
        if token.strip()
    ]
    conditions = list(dict.fromkeys(conditions))
    if CLEAN not in conditions:
        raise ValueError("conditions must include clean.")
    if len(conditions) < 2:
        raise ValueError("At least one degraded condition is required.")
    return conditions


def load_full41(path: Path) -> list[str]:
    schema = pd.read_csv(require_file(path), low_memory=False)
    required = {"feature_set", "feature_name"}
    missing = sorted(required - set(schema.columns))
    if missing:
        raise ValueError(f"Schema lacks columns: {missing}")

    rows = schema.loc[
        schema["feature_set"].astype(str).eq("soft_full_all"),
        "feature_name",
    ].astype(str).tolist()
    features = list(dict.fromkeys(rows))
    if len(features) != 41:
        raise ValueError(
            f"soft_full_all resolves to {len(features)} features, expected 41."
        )
    return features


def deterministic_participant_subset(
    participant_table: pd.DataFrame,
    max_per_label: int,
    seed: int,
) -> pd.DataFrame:
    if max_per_label <= 0:
        return participant_table.copy()

    rng = np.random.default_rng(seed)
    parts = []
    for label, group in participant_table.groupby(
        "label", observed=True, sort=True
    ):
        identifiers = group["participant_id"].astype(str).to_numpy()
        order = rng.permutation(len(identifiers))
        keep = identifiers[order[: min(max_per_label, len(order))]]
        parts.append(
            participant_table[
                participant_table["participant_id"].isin(keep)
            ]
        )
    return pd.concat(parts, ignore_index=True)


def validate_and_prepare(
    frame: pd.DataFrame,
    features: list[str],
    conditions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "onestop_trial_id",
        "participant_id",
        "label",
        "label_name",
        "view_condition",
        "corruption_repeat",
        *features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"View table lacks columns: {missing}")

    result = frame.copy()
    result["onestop_trial_id"] = (
        result["onestop_trial_id"].astype(str)
    )
    result["participant_id"] = (
        result["participant_id"].astype(str).str.strip().str.lower()
    )
    result["label"] = pd.to_numeric(
        result["label"], errors="raise"
    ).astype(int)
    result["view_condition"] = (
        result["view_condition"].astype(str).str.strip()
    )
    result["corruption_repeat"] = pd.to_numeric(
        result["corruption_repeat"], errors="raise"
    ).astype(int)

    if set(result["label"].unique()) != {0, 1}:
        raise ValueError("Expected labels {0,1}.")
    if result.duplicated(
        [
            "onestop_trial_id",
            "view_condition",
            "corruption_repeat",
        ]
    ).any():
        raise ValueError("Duplicate trial/view keys.")

    result = result[
        result["view_condition"].isin(conditions)
    ].copy()

    observed_conditions = set(result["view_condition"].unique())
    if observed_conditions != set(conditions):
        raise ValueError(
            f"Condition mismatch: observed={sorted(observed_conditions)}, "
            f"expected={sorted(conditions)}"
        )

    trial_audit = (
        result.groupby("onestop_trial_id", observed=True)
        .agg(
            participant_nunique=("participant_id", "nunique"),
            label_nunique=("label", "nunique"),
            n_rows=("view_condition", "size"),
            n_conditions=("view_condition", "nunique"),
            clean_rows=(
                "view_condition",
                lambda values: int(
                    pd.Series(values).eq(CLEAN).sum()
                ),
            ),
        )
        .reset_index()
    )
    expected_rows = len(conditions)
    bad = trial_audit[
        trial_audit["participant_nunique"].ne(1)
        | trial_audit["label_nunique"].ne(1)
        | trial_audit["n_rows"].ne(expected_rows)
        | trial_audit["n_conditions"].ne(len(conditions))
        | trial_audit["clean_rows"].ne(1)
    ]
    if not bad.empty:
        raise ValueError(
            "Incomplete or inconsistent trial views:\n"
            + bad.head(20).to_string(index=False)
        )

    participant_table = (
        result.loc[
            result["view_condition"].eq(CLEAN),
            ["participant_id", "label", "label_name"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if participant_table["participant_id"].duplicated().any():
        bad_participants = (
            participant_table[
                participant_table["participant_id"].duplicated(
                    keep=False
                )
            ]
            .sort_values("participant_id")
        )
        raise ValueError(
            "Participant has multiple labels:\n"
            + bad_participants.head(20).to_string(index=False)
        )

    result[features] = result[features].apply(
        pd.to_numeric, errors="coerce"
    )
    if np.isinf(result[features].to_numpy(dtype=float)).any():
        raise ValueError("Feature matrix contains infinity.")
    return result, participant_table


def fit_preprocessor(
    clean_train: pd.DataFrame,
    features: list[str],
) -> Preprocessor:
    imputer = SimpleImputer(strategy="median")
    clean_imputed = imputer.fit_transform(clean_train[features])
    if np.isnan(clean_imputed).any():
        raise ValueError(
            "A feature is entirely missing in clean training data."
        )
    scaler = StandardScaler()
    scaler.fit(clean_imputed)
    return Preprocessor(imputer=imputer, scaler=scaler)


def transform_frame(
    frame: pd.DataFrame,
    features: list[str],
    preprocessor: Preprocessor,
) -> np.ndarray:
    return preprocessor.transform(frame[features])


def fit_quality_basis(
    train_frame: pd.DataFrame,
    transformed: np.ndarray,
    conditions: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Estimate condition-balanced nuisance directions.

    For each degraded condition, differences are divided by that condition's
    RMS magnitude before contributing to the scatter matrix. Thus every
    condition contributes equally even when one degradation has larger raw
    magnitude.
    """
    work = train_frame[
        ["onestop_trial_id", "view_condition"]
    ].copy()
    work["_row_index"] = np.arange(len(work), dtype=np.int64)

    clean_rows = work[
        work["view_condition"].eq(CLEAN)
    ][["onestop_trial_id", "_row_index"]].rename(
        columns={"_row_index": "clean_index"}
    )
    if clean_rows["onestop_trial_id"].duplicated().any():
        raise ValueError("Multiple clean rows for a training trial.")

    n_features = transformed.shape[1]
    scatter = np.zeros((n_features, n_features), dtype=np.float64)
    audit_rows = []
    degraded_conditions = [
        condition for condition in conditions if condition != CLEAN
    ]

    for condition in degraded_conditions:
        degraded_rows = work[
            work["view_condition"].eq(condition)
        ][["onestop_trial_id", "_row_index"]].rename(
            columns={"_row_index": "degraded_index"}
        )
        paired = clean_rows.merge(
            degraded_rows,
            on="onestop_trial_id",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(clean_rows):
            raise ValueError(
                f"Incomplete clean/{condition} pairs in training data."
            )

        difference = (
            transformed[
                paired["degraded_index"].to_numpy(dtype=np.int64)
            ]
            - transformed[
                paired["clean_index"].to_numpy(dtype=np.int64)
            ]
        )
        rms = float(np.sqrt(np.mean(np.square(difference))))
        if not math.isfinite(rms) or rms <= 1e-12:
            raise ValueError(
                f"Condition {condition} has degenerate difference RMS={rms}."
            )
        normalized = difference / rms
        scatter += (
            normalized.T @ normalized / float(len(normalized))
        )
        audit_rows.append(
            {
                "view_condition": condition,
                "pairs": len(paired),
                "difference_rms": rms,
                "difference_l2_median": float(
                    np.median(
                        np.linalg.norm(difference, axis=1)
                    )
                ),
            }
        )

    scatter /= float(len(degraded_conditions))
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    explained = (
        eigenvalues / total
        if total > 0
        else np.zeros_like(eigenvalues)
    )
    return basis, explained, pd.DataFrame(audit_rows)


def apply_quality_regularization(
    transformed: np.ndarray,
    basis: np.ndarray,
    explained: np.ndarray,
    lambda_value: float,
) -> np.ndarray:
    """
    Apply A^{-1/2} to features, where
    A = I + lambda * Q and Q is the quality scatter normalized so its largest
    eigenvalue is one. This is equivalent to a linear classifier objective
    with an additional score-consistency penalty along paired quality-change
    directions, without hard deletion of any feature dimension.
    """
    if lambda_value <= 0:
        return transformed
    if basis.shape[0] != transformed.shape[1]:
        raise ValueError("Quality basis dimension mismatch.")
    if explained.size != transformed.shape[1]:
        raise ValueError("Quality spectrum dimension mismatch.")
    maximum = float(np.max(explained))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Degenerate quality spectrum.")
    normalized = np.clip(explained / maximum, 0.0, None)
    shrink = 1.0 / np.sqrt(1.0 + lambda_value * normalized)
    return (transformed @ basis) * shrink


def make_classifier(c_value: float, seed: int) -> LinearSVC:
    return LinearSVC(
        C=c_value,
        class_weight="balanced",
        dual=False,
        max_iter=100000,
        random_state=seed,
    )


def fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
    seed: int,
) -> LinearSVC:
    model = make_classifier(c_value, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    return model


def safe_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_ap(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def compute_metrics_from_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc": safe_auc(y_true, scores),
        "pr_auc_nr": safe_ap(y_true, scores),
        "predicted_nr_rate": float(np.mean(predictions == 1)),
    }


def confusion_counts(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    return tp, tn, fp, fn


def macro_f1_from_counts(
    tp: float,
    tn: float,
    fp: float,
    fn: float,
) -> float:
    denom_pos = 2.0 * tp + fp + fn
    denom_neg = 2.0 * tn + fp + fn
    f1_pos = 0.0 if denom_pos <= 0 else 2.0 * tp / denom_pos
    f1_neg = 0.0 if denom_neg <= 0 else 2.0 * tn / denom_neg
    return 0.5 * (f1_pos + f1_neg)


def ba_from_counts(
    tp: float,
    tn: float,
    fp: float,
    fn: float,
) -> float:
    tpr = 0.0 if tp + fn <= 0 else tp / (tp + fn)
    tnr = 0.0 if tn + fp <= 0 else tn / (tn + fp)
    return 0.5 * (tpr + tnr)


def threshold_candidates(
    scores: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("No finite scores.")
    candidates = np.quantile(
        finite,
        np.linspace(0.0, 1.0, max(3, grid_size)),
    )
    return np.unique(
        np.concatenate(
            [
                candidates,
                np.asarray([0.0]),
                np.asarray(
                    [np.nextafter(finite.min(), -np.inf)]
                ),
                np.asarray(
                    [np.nextafter(finite.max(), np.inf)]
                ),
            ]
        )
    )


def select_threshold_pooled(
    predictions: pd.DataFrame,
    conditions_for_threshold: list[str],
    grid_size: int,
) -> tuple[float, dict[str, float]]:
    subset = predictions[
        predictions["view_condition"].isin(
            conditions_for_threshold
        )
    ]
    y = subset["label"].to_numpy(dtype=int)
    score = subset["score"].to_numpy(dtype=float)

    best_threshold = 0.0
    best_metrics: dict[str, float] | None = None
    best_key = (-np.inf, -np.inf, -np.inf)

    for threshold in threshold_candidates(score, grid_size):
        pred = (score >= threshold).astype(int)
        metrics = compute_metrics_from_predictions(
            y, pred, score
        )
        key = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            -abs(float(threshold)),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def metrics_by_condition(
    predictions: pd.DataFrame,
    threshold: float,
    conditions: list[str],
) -> pd.DataFrame:
    work = predictions.copy()
    work["prediction"] = (
        work["score"].to_numpy(dtype=float) >= threshold
    ).astype(int)
    rows = []
    for condition in conditions:
        group = work[
            work["view_condition"].eq(condition)
        ]
        metrics = compute_metrics_from_predictions(
            group["label"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "view_condition": condition,
                "n_trials": len(group),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def robust_summary(
    condition_metrics: pd.DataFrame,
    degraded_conditions: list[str],
) -> dict[str, float]:
    clean_row = condition_metrics[
        condition_metrics["view_condition"].eq(CLEAN)
    ].iloc[0]
    degraded = condition_metrics[
        condition_metrics["view_condition"].isin(
            degraded_conditions
        )
    ]
    return {
        "clean_macro_f1": float(clean_row["macro_f1"]),
        "clean_balanced_accuracy": float(
            clean_row["balanced_accuracy"]
        ),
        "clean_roc_auc": float(clean_row["roc_auc"]),
        "mean_degraded_macro_f1": float(
            degraded["macro_f1"].mean()
        ),
        "worst_degraded_macro_f1": float(
            degraded["macro_f1"].min()
        ),
        "mean_degraded_balanced_accuracy": float(
            degraded["balanced_accuracy"].mean()
        ),
        "mean_degraded_roc_auc": float(
            degraded["roc_auc"].mean()
        ),
    }


def candidate_key(
    candidate: Candidate,
) -> tuple[str, float, float]:
    return (
        candidate.method,
        candidate.c_value,
        candidate.lambda_value,
    )


def build_candidates(
    c_grid: list[float],
    lambda_grid: list[float],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for c_value in c_grid:
        candidates.append(
            Candidate("clean_only", c_value, 0.0)
        )
        candidates.append(
            Candidate("augmentation", c_value, 0.0)
        )
        for lambda_value in lambda_grid:
            candidates.append(
                Candidate("qcr", c_value, lambda_value)
            )
    return candidates


def participant_splits(
    participant_table: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[tuple[set[str], set[str]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    minimum_class = int(
        participant_table["label"].value_counts().min()
    )
    if minimum_class < n_splits:
        raise ValueError(
            f"Need at least {n_splits} participants per class; "
            f"minimum={minimum_class}."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    participant_ids = participant_table[
        "participant_id"
    ].to_numpy(dtype=object)
    labels = participant_table["label"].to_numpy(dtype=int)
    splits = []
    for train_index, test_index in splitter.split(
        participant_ids, labels
    ):
        train_ids = set(
            participant_ids[train_index].astype(str)
        )
        test_ids = set(
            participant_ids[test_index].astype(str)
        )
        if train_ids.intersection(test_ids):
            raise RuntimeError("Participant leakage.")
        splits.append((train_ids, test_ids))
    return splits


def predict_candidate(
    candidate: Candidate,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    z_train: np.ndarray,
    z_valid: np.ndarray,
    basis: np.ndarray,
    explained: np.ndarray,
    seed: int,
) -> np.ndarray:
    if candidate.method == "clean_only":
        train_mask = train_frame["view_condition"].eq(CLEAN).to_numpy()
        x_train = z_train[train_mask]
    elif candidate.method == "augmentation":
        train_mask = np.ones(len(train_frame), dtype=bool)
        x_train = z_train
    elif candidate.method == "qcr":
        train_mask = np.ones(len(train_frame), dtype=bool)
        x_train = apply_quality_regularization(
            z_train,
            basis,
            explained,
            candidate.lambda_value,
        )
    else:
        raise ValueError(candidate.method)

    y_train = train_frame.loc[
        train_mask, "label"
    ].to_numpy(dtype=int)

    if candidate.method == "qcr":
        x_valid = apply_quality_regularization(
            z_valid,
            basis,
            explained,
            candidate.lambda_value,
        )
    else:
        x_valid = z_valid

    model = fit_classifier(
        x=x_train,
        y=y_train,
        c_value=candidate.c_value,
        seed=seed,
    )
    return model.decision_function(x_valid).astype(float)


def inner_select_methods(
    outer_train: pd.DataFrame,
    outer_train_participants: pd.DataFrame,
    features: list[str],
    conditions: list[str],
    degraded_conditions: list[str],
    candidates: list[Candidate],
    inner_splits: int,
    threshold_grid_size: int,
    seed: int,
) -> tuple[
    dict[str, dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
]:
    splits = participant_splits(
        outer_train_participants,
        n_splits=inner_splits,
        seed=seed,
    )

    prediction_parts: dict[
        tuple[str, float, float], list[pd.DataFrame]
    ] = {
        candidate_key(candidate): []
        for candidate in candidates
    }
    basis_audits = []

    for inner_fold, (fit_ids, valid_ids) in enumerate(
        splits, start=1
    ):
        fit_frame = outer_train[
            outer_train["participant_id"].isin(fit_ids)
        ].reset_index(drop=True)
        valid_frame = outer_train[
            outer_train["participant_id"].isin(valid_ids)
        ].reset_index(drop=True)

        clean_fit = fit_frame[
            fit_frame["view_condition"].eq(CLEAN)
        ]
        preprocessor = fit_preprocessor(
            clean_fit, features
        )
        z_fit = transform_frame(
            fit_frame, features, preprocessor
        )
        z_valid = transform_frame(
            valid_frame, features, preprocessor
        )
        basis, explained, condition_audit = fit_quality_basis(
            train_frame=fit_frame,
            transformed=z_fit,
            conditions=conditions,
        )
        condition_audit.insert(0, "inner_fold", inner_fold)
        basis_audits.append(condition_audit)

        for candidate_index, candidate in enumerate(
            candidates, start=1
        ):
            scores = predict_candidate(
                candidate=candidate,
                train_frame=fit_frame,
                valid_frame=valid_frame,
                z_train=z_fit,
                z_valid=z_valid,
                basis=basis,
                explained=explained,
                seed=(
                    seed
                    + inner_fold * 10000
                    + candidate_index
                ),
            )
            prediction_parts[candidate_key(candidate)].append(
                pd.DataFrame(
                    {
                        "onestop_trial_id": valid_frame[
                            "onestop_trial_id"
                        ].astype(str).to_numpy(),
                        "participant_id": valid_frame[
                            "participant_id"
                        ].astype(str).to_numpy(),
                        "label": valid_frame[
                            "label"
                        ].to_numpy(dtype=int),
                        "view_condition": valid_frame[
                            "view_condition"
                        ].astype(str).to_numpy(),
                        "score": scores,
                        "inner_fold": inner_fold,
                    }
                )
            )

        print(
            f"    inner fold {inner_fold}/{inner_splits}: "
            f"fit participants={len(fit_ids)}, "
            f"valid participants={len(valid_ids)}, "
            f"quality top1={explained[0]:.4f}, "
            f"top4={explained[:4].sum():.4f}"
        )

    selection_rows = []
    selected: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        key = candidate_key(candidate)
        predictions = pd.concat(
            prediction_parts[key],
            ignore_index=True,
        )
        threshold_conditions = (
            [CLEAN]
            if candidate.method == "clean_only"
            else conditions
        )
        threshold, threshold_metrics = (
            select_threshold_pooled(
                predictions=predictions,
                conditions_for_threshold=threshold_conditions,
                grid_size=threshold_grid_size,
            )
        )
        condition_metrics = metrics_by_condition(
            predictions,
            threshold,
            conditions,
        )
        summary = robust_summary(
            condition_metrics,
            degraded_conditions,
        )
        row = {
            "method": candidate.method,
            "C": candidate.c_value,
            "lambda_quality": candidate.lambda_value,
            "decision_threshold": threshold,
            "threshold_fit_conditions": ",".join(
                threshold_conditions
            ),
            "threshold_macro_f1": (
                threshold_metrics["macro_f1"]
            ),
            **summary,
        }
        selection_rows.append(row)

    selection_table = pd.DataFrame(selection_rows)

    for method in METHODS:
        group = selection_table[
            selection_table["method"].eq(method)
        ].copy()
        if method == "clean_only":
            sort_columns = [
                "clean_macro_f1",
                "clean_balanced_accuracy",
                "clean_roc_auc",
                "C",
            ]
            ascending = [False, False, False, True]
        else:
            sort_columns = [
                "mean_degraded_macro_f1",
                "worst_degraded_macro_f1",
                "clean_macro_f1",
                "mean_degraded_roc_auc",
                "lambda_quality",
                "C",
            ]
            ascending = [
                False,
                False,
                False,
                False,
                True,
                True,
            ]

        ranked = group.sort_values(
            sort_columns,
            ascending=ascending,
            kind="stable",
        ).reset_index(drop=True)
        ranked["rank_within_method"] = (
            np.arange(len(ranked)) + 1
        )
        best = ranked.iloc[0].to_dict()
        selected[method] = best
        method_mask = selection_table["method"].eq(method)
        rank_values = (
            selection_table.loc[
                method_mask,
                ["method", "C", "lambda_quality"],
            ]
            .merge(
                ranked[
                    [
                        "method",
                        "C",
                        "lambda_quality",
                        "rank_within_method",
                    ]
                ],
                on=["method", "C", "lambda_quality"],
                how="left",
                validate="one_to_one",
                sort=False,
            )["rank_within_method"]
            .to_numpy()
        )
        if pd.isna(rank_values).any():
            raise RuntimeError(
                f"Failed to assign candidate ranks for method={method}."
            )
        selection_table.loc[
            method_mask,
            "rank_within_method",
        ] = rank_values

    return (
        selected,
        selection_table,
        pd.concat(basis_audits, ignore_index=True),
    )


def fit_outer_and_predict(
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    features: list[str],
    conditions: list[str],
    selected: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_train = outer_train[
        outer_train["view_condition"].eq(CLEAN)
    ]
    preprocessor = fit_preprocessor(
        clean_train, features
    )
    z_train = transform_frame(
        outer_train, features, preprocessor
    )
    z_test = transform_frame(
        outer_test, features, preprocessor
    )
    basis, explained, condition_audit = fit_quality_basis(
        train_frame=outer_train,
        transformed=z_train,
        conditions=conditions,
    )

    prediction_parts = []
    model_rows = []

    for method_index, method in enumerate(
        METHODS, start=1
    ):
        choice = selected[method]
        c_value = float(choice["C"])
        lambda_value = float(choice["lambda_quality"])
        threshold = float(choice["decision_threshold"])

        if method == "clean_only":
            train_mask = outer_train[
                "view_condition"
            ].eq(CLEAN).to_numpy()
            x_train = z_train[train_mask]
            x_test = z_test
        elif method == "augmentation":
            train_mask = np.ones(
                len(outer_train), dtype=bool
            )
            x_train = z_train
            x_test = z_test
        elif method == "qcr":
            train_mask = np.ones(
                len(outer_train), dtype=bool
            )
            x_train = apply_quality_regularization(
                z_train,
                basis,
                explained,
                lambda_value,
            )
            x_test = apply_quality_regularization(
                z_test,
                basis,
                explained,
                lambda_value,
            )
        else:
            raise ValueError(method)

        y_train = outer_train.loc[
            train_mask, "label"
        ].to_numpy(dtype=int)
        model = fit_classifier(
            x=x_train,
            y=y_train,
            c_value=c_value,
            seed=seed + method_index,
        )
        score = model.decision_function(
            x_test
        ).astype(float)
        prediction = (score >= threshold).astype(int)
        margin = score - threshold

        prediction_parts.append(
            pd.DataFrame(
                {
                    "onestop_trial_id": outer_test[
                        "onestop_trial_id"
                    ].astype(str).to_numpy(),
                    "participant_id": outer_test[
                        "participant_id"
                    ].astype(str).to_numpy(),
                    "label": outer_test[
                        "label"
                    ].to_numpy(dtype=int),
                    "label_name": outer_test[
                        "label_name"
                    ].astype(str).to_numpy(),
                    "view_condition": outer_test[
                        "view_condition"
                    ].astype(str).to_numpy(),
                    "method": method,
                    "score": score,
                    "decision_threshold": threshold,
                    "margin": margin,
                    "prediction": prediction,
                    "selected_C": c_value,
                    "selected_lambda": lambda_value,
                }
            )
        )
        model_rows.append(
            {
                "method": method,
                "selected_C": c_value,
                "selected_lambda": lambda_value,
                "selected_threshold": threshold,
                "train_rows": int(train_mask.sum()),
                "train_participants": outer_train[
                    "participant_id"
                ].nunique(),
                "test_rows": len(outer_test),
                "test_participants": outer_test[
                    "participant_id"
                ].nunique(),
            }
        )

    quality_spectrum = pd.DataFrame(
        {
            "component": np.arange(
                1, len(explained) + 1
            ),
            "explained_quality_scatter_ratio": explained,
            "cumulative_ratio": np.cumsum(explained),
        }
    )
    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.DataFrame(model_rows),
        pd.concat(
            [
                condition_audit,
            ],
            ignore_index=True,
        ),
        quality_spectrum,
    )


def pooled_metrics(
    predictions: pd.DataFrame,
    conditions: list[str],
) -> pd.DataFrame:
    rows = []
    for (method, condition), group in predictions.groupby(
        ["method", "view_condition"],
        observed=True,
    ):
        metrics = compute_metrics_from_predictions(
            group["label"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "method": method,
                "view_condition": condition,
                "n_trials": len(group),
                "n_participants": group[
                    "participant_id"
                ].nunique(),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    result["view_condition"] = pd.Categorical(
        result["view_condition"],
        categories=conditions,
        ordered=True,
    )
    return result.sort_values(
        ["method", "view_condition"]
    ).reset_index(drop=True)


def participant_level_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_predictions = (
        predictions.groupby(
            [
                "method",
                "view_condition",
                "participant_id",
            ],
            observed=True,
        )
        .agg(
            label=("label", "first"),
            median_margin=("margin", "median"),
            median_score=("score", "median"),
            trials=("onestop_trial_id", "size"),
        )
        .reset_index()
    )
    participant_predictions["prediction"] = (
        participant_predictions[
            "median_margin"
        ].to_numpy(dtype=float)
        >= 0.0
    ).astype(int)

    rows = []
    for (method, condition), group in (
        participant_predictions.groupby(
            ["method", "view_condition"],
            observed=True,
        )
    ):
        metrics = compute_metrics_from_predictions(
            group["label"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["median_score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "method": method,
                "view_condition": condition,
                "n_participants": len(group),
                **metrics,
            }
        )
    return participant_predictions, pd.DataFrame(rows)


def robust_metric_table(
    pooled: pd.DataFrame,
    degraded_conditions: list[str],
) -> pd.DataFrame:
    rows = []
    for method, group in pooled.groupby(
        "method", observed=True
    ):
        rows.append(
            {
                "method": method,
                **robust_summary(
                    group, degraded_conditions
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_confusion_summaries(
    predictions: pd.DataFrame,
    degraded_conditions: list[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()

    participant_ids = sorted(
        predictions["participant_id"].unique()
    )
    participant_index = {
        participant: index
        for index, participant in enumerate(
            participant_ids
        )
    }
    methods = list(METHODS)
    conditions = [CLEAN, *degraded_conditions]

    # counts[method, condition, participant, TP/TN/FP/FN]
    counts = np.zeros(
        (
            len(methods),
            len(conditions),
            len(participant_ids),
            4,
        ),
        dtype=np.float64,
    )
    method_index = {
        method: index
        for index, method in enumerate(methods)
    }
    condition_index = {
        condition: index
        for index, condition in enumerate(conditions)
    }

    for (
        method,
        condition,
        participant,
    ), group in predictions.groupby(
        ["method", "view_condition", "participant_id"],
        observed=True,
    ):
        counts[
            method_index[method],
            condition_index[condition],
            participant_index[participant],
            :,
        ] = confusion_counts(
            group["label"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
        )

    rng = np.random.default_rng(seed)
    replicate_rows = []

    for replicate in range(n_bootstrap):
        sampled = rng.integers(
            0,
            len(participant_ids),
            size=len(participant_ids),
        )
        multiplicity = np.bincount(
            sampled,
            minlength=len(participant_ids),
        ).astype(np.float64)
        summed = np.einsum(
            "mcpf,p->mcf",
            counts,
            multiplicity,
        )

        for method_i, method in enumerate(methods):
            condition_macro = []
            condition_ba = []
            for condition_i, condition in enumerate(
                conditions
            ):
                tp, tn, fp, fn = summed[
                    method_i, condition_i, :
                ]
                macro = macro_f1_from_counts(
                    tp, tn, fp, fn
                )
                ba = ba_from_counts(
                    tp, tn, fp, fn
                )
                condition_macro.append(macro)
                condition_ba.append(ba)
                replicate_rows.append(
                    {
                        "replicate": replicate,
                        "method": method,
                        "scope": condition,
                        "metric": "macro_f1",
                        "value": macro,
                    }
                )
                replicate_rows.append(
                    {
                        "replicate": replicate,
                        "method": method,
                        "scope": condition,
                        "metric": "balanced_accuracy",
                        "value": ba,
                    }
                )

            degraded_macro = np.asarray(
                condition_macro[1:], dtype=float
            )
            degraded_ba = np.asarray(
                condition_ba[1:], dtype=float
            )
            for scope, metric, value in (
                (
                    "mean_degraded",
                    "macro_f1",
                    float(degraded_macro.mean()),
                ),
                (
                    "worst_degraded",
                    "macro_f1",
                    float(degraded_macro.min()),
                ),
                (
                    "mean_degraded",
                    "balanced_accuracy",
                    float(degraded_ba.mean()),
                ),
                (
                    "worst_degraded",
                    "balanced_accuracy",
                    float(degraded_ba.min()),
                ),
            ):
                replicate_rows.append(
                    {
                        "replicate": replicate,
                        "method": method,
                        "scope": scope,
                        "metric": metric,
                        "value": value,
                    }
                )

    replicates = pd.DataFrame(replicate_rows)
    summary_rows = []
    for (
        method,
        scope,
        metric,
    ), group in replicates.groupby(
        ["method", "scope", "metric"],
        observed=True,
    ):
        values = group["value"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "method": method,
                "scope": scope,
                "metric": metric,
                "bootstrap_mean": float(
                    np.mean(values)
                ),
                "ci_low": float(
                    np.quantile(values, 0.025)
                ),
                "ci_high": float(
                    np.quantile(values, 0.975)
                ),
                "n_bootstrap": n_bootstrap,
            }
        )
    summary = pd.DataFrame(summary_rows)

    delta_rows = []
    qcr = replicates[
        replicates["method"].eq("qcr")
    ].set_index(["replicate", "scope", "metric"])
    for comparison_method in (
        "clean_only",
        "augmentation",
    ):
        comparison = replicates[
            replicates["method"].eq(
                comparison_method
            )
        ].set_index(
            ["replicate", "scope", "metric"]
        )
        common = qcr.index.intersection(
            comparison.index
        )
        delta = (
            qcr.loc[common, "value"]
            - comparison.loc[common, "value"]
        ).reset_index(name="delta")
        for (
            scope,
            metric,
        ), group in delta.groupby(
            ["scope", "metric"],
            observed=True,
        ):
            values = group["delta"].to_numpy(dtype=float)
            delta_rows.append(
                {
                    "reference": "qcr",
                    "comparison": comparison_method,
                    "scope": scope,
                    "metric": metric,
                    "delta_mean": float(
                        np.mean(values)
                    ),
                    "ci_low": float(
                        np.quantile(values, 0.025)
                    ),
                    "ci_high": float(
                        np.quantile(values, 0.975)
                    ),
                    "probability_delta_gt_0": float(
                        np.mean(values > 0)
                    ),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return summary, pd.DataFrame(delta_rows)


def main() -> int:
    args = parse_args()
    if args.outer_splits < 2 or args.inner_splits < 2:
        raise ValueError(
            "outer and inner splits must be at least 2."
        )
    if args.threshold_grid_size < 3:
        raise ValueError(
            "threshold grid size must be at least 3."
        )
    if args.bootstrap < 0:
        raise ValueError("bootstrap cannot be negative.")

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    features = load_full41(args.schema)
    c_grid = parse_float_grid(args.c_grid)
    lambda_grid = parse_lambda_grid(
        args.lambda_grid
    )
    conditions = parse_conditions(args.conditions)
    degraded_conditions = [
        condition
        for condition in conditions
        if condition != CLEAN
    ]

    frame = pd.read_parquet(require_file(args.views))
    frame, participant_table = validate_and_prepare(
        frame, features, conditions
    )
    participant_table = deterministic_participant_subset(
        participant_table,
        max_per_label=args.max_participants_per_label,
        seed=args.seed,
    )
    selected_participants = set(
        participant_table["participant_id"].astype(str)
    )
    frame = frame[
        frame["participant_id"].isin(
            selected_participants
        )
    ].reset_index(drop=True)

    candidates = build_candidates(
        c_grid, lambda_grid
    )
    outer_splits = participant_splits(
        participant_table,
        n_splits=args.outer_splits,
        seed=args.seed,
    )

    print("=== ONESTOP QCR NESTED INPUT ===")
    print(f"rows: {len(frame):,}")
    print(
        f"clean trials: "
        f"{frame.loc[frame['view_condition'].eq(CLEAN), 'onestop_trial_id'].nunique():,}"
    )
    print(
        f"participants: "
        f"{participant_table['participant_id'].nunique()}"
    )
    print(
        "participant labels: "
        + json.dumps(
            participant_table[
                "label_name"
            ].value_counts().to_dict()
        )
    )
    print(f"conditions: {conditions}")
    print(f"features: {len(features)}")
    print(f"C grid: {c_grid}")
    print(f"lambda grid: {lambda_grid}")
    print(
        f"outer={args.outer_splits}, "
        f"inner={args.inner_splits}, "
        f"bootstrap={args.bootstrap}"
    )

    outer_prediction_parts = []
    outer_model_parts = []
    selection_parts = []
    quality_condition_parts = []
    quality_spectrum_parts = []

    for outer_fold, (
        train_ids,
        test_ids,
    ) in enumerate(outer_splits, start=1):
        print(
            f"\n=== OUTER FOLD "
            f"{outer_fold}/{args.outer_splits} ==="
        )
        outer_train = frame[
            frame["participant_id"].isin(train_ids)
        ].reset_index(drop=True)
        outer_test = frame[
            frame["participant_id"].isin(test_ids)
        ].reset_index(drop=True)
        outer_train_participants = participant_table[
            participant_table["participant_id"].isin(
                train_ids
            )
        ].reset_index(drop=True)

        (
            selected,
            selection_table,
            inner_quality_audit,
        ) = inner_select_methods(
            outer_train=outer_train,
            outer_train_participants=(
                outer_train_participants
            ),
            features=features,
            conditions=conditions,
            degraded_conditions=degraded_conditions,
            candidates=candidates,
            inner_splits=args.inner_splits,
            threshold_grid_size=(
                args.threshold_grid_size
            ),
            seed=args.seed + outer_fold * 1000,
        )
        selection_table.insert(
            0, "outer_fold", outer_fold
        )
        selection_parts.append(selection_table)

        inner_quality_audit.insert(
            0, "outer_fold", outer_fold
        )
        inner_quality_audit.insert(
            1, "stage", "inner"
        )
        quality_condition_parts.append(
            inner_quality_audit
        )

        print("  selected:")
        for method in METHODS:
            choice = selected[method]
            print(
                f"    {method}: C={choice['C']:g}, "
                f"lambda={float(choice['lambda_quality']):g}, "
                f"threshold="
                f"{choice['decision_threshold']:.6f}, "
                f"cleanF1="
                f"{choice['clean_macro_f1']:.4f}, "
                f"meanDegradedF1="
                f"{choice['mean_degraded_macro_f1']:.4f}"
            )

        (
            fold_predictions,
            fold_models,
            outer_quality_audit,
            quality_spectrum,
        ) = fit_outer_and_predict(
            outer_train=outer_train,
            outer_test=outer_test,
            features=features,
            conditions=conditions,
            selected=selected,
            seed=args.seed + outer_fold * 100000,
        )
        fold_predictions.insert(
            0, "outer_fold", outer_fold
        )
        fold_models.insert(
            0, "outer_fold", outer_fold
        )
        outer_quality_audit.insert(
            0, "outer_fold", outer_fold
        )
        outer_quality_audit.insert(
            1, "stage", "outer"
        )
        quality_spectrum.insert(
            0, "outer_fold", outer_fold
        )

        outer_prediction_parts.append(
            fold_predictions
        )
        outer_model_parts.append(fold_models)
        quality_condition_parts.append(
            outer_quality_audit
        )
        quality_spectrum_parts.append(
            quality_spectrum
        )

        fold_metric_table = pooled_metrics(
            fold_predictions,
            conditions,
        )
        fold_robust = robust_metric_table(
            fold_metric_table,
            degraded_conditions,
        )
        print("  outer robust summary:")
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            200,
        ):
            print(
                fold_robust.to_string(
                    index=False
                )
            )

    predictions = pd.concat(
        outer_prediction_parts,
        ignore_index=True,
    )
    selection_table = pd.concat(
        selection_parts,
        ignore_index=True,
    )
    outer_models = pd.concat(
        outer_model_parts,
        ignore_index=True,
    )
    quality_condition_audit = pd.concat(
        quality_condition_parts,
        ignore_index=True,
    )
    quality_spectrum = pd.concat(
        quality_spectrum_parts,
        ignore_index=True,
    )

    expected_rows = (
        len(frame)
        * len(METHODS)
    )
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"OOF prediction rows={len(predictions)}, "
            f"expected={expected_rows}"
        )
    if predictions.duplicated(
        [
            "method",
            "onestop_trial_id",
            "view_condition",
            "corruption_repeat"
            if "corruption_repeat" in predictions.columns
            else "outer_fold",
        ]
    ).any():
        # One view per condition is expected in v1. The fallback keeps this
        # audit compatible with the compact prediction table.
        duplicate = predictions.duplicated(
            [
                "method",
                "onestop_trial_id",
                "view_condition",
            ]
        )
        if duplicate.any():
            raise RuntimeError(
                "Duplicate outer OOF predictions."
            )

    pooled = pooled_metrics(
        predictions, conditions
    )
    robust = robust_metric_table(
        pooled, degraded_conditions
    )
    (
        participant_predictions,
        participant_metrics,
    ) = participant_level_metrics(predictions)
    (
        bootstrap_summary,
        bootstrap_deltas,
    ) = bootstrap_confusion_summaries(
        predictions=predictions,
        degraded_conditions=degraded_conditions,
        n_bootstrap=args.bootstrap,
        seed=args.seed + 900000,
    )

    predictions_path = (
        args.output_dir
        / "qcr_outer_oof_predictions.parquet"
    )
    pooled_path = (
        args.output_dir
        / "qcr_pooled_condition_metrics.csv"
    )
    robust_path = (
        args.output_dir
        / "qcr_robust_summary.csv"
    )
    participant_prediction_path = (
        args.output_dir
        / "qcr_participant_predictions.csv"
    )
    participant_metric_path = (
        args.output_dir
        / "qcr_participant_condition_metrics.csv"
    )
    selection_path = (
        args.output_dir
        / "qcr_inner_selection_table.csv"
    )
    outer_model_path = (
        args.output_dir
        / "qcr_outer_selected_models.csv"
    )
    quality_condition_path = (
        args.output_dir
        / "qcr_quality_difference_audit.csv"
    )
    quality_spectrum_path = (
        args.output_dir
        / "qcr_quality_spectrum.csv"
    )
    bootstrap_summary_path = (
        args.output_dir
        / "qcr_bootstrap_summary.csv"
    )
    bootstrap_delta_path = (
        args.output_dir
        / "qcr_bootstrap_deltas.csv"
    )
    manifest_path = (
        args.output_dir
        / "qcr_nested_manifest.json"
    )

    predictions.to_parquet(
        predictions_path,
        index=False,
        compression="zstd",
    )
    pooled.to_csv(
        pooled_path,
        index=False,
        encoding="utf-8-sig",
    )
    robust.to_csv(
        robust_path,
        index=False,
        encoding="utf-8-sig",
    )
    participant_predictions.to_csv(
        participant_prediction_path,
        index=False,
        encoding="utf-8-sig",
    )
    participant_metrics.to_csv(
        participant_metric_path,
        index=False,
        encoding="utf-8-sig",
    )
    selection_table.to_csv(
        selection_path,
        index=False,
        encoding="utf-8-sig",
    )
    outer_models.to_csv(
        outer_model_path,
        index=False,
        encoding="utf-8-sig",
    )
    quality_condition_audit.to_csv(
        quality_condition_path,
        index=False,
        encoding="utf-8-sig",
    )
    quality_spectrum.to_csv(
        quality_spectrum_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        bootstrap_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_deltas.to_csv(
        bootstrap_delta_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "analysis": (
            "Nested participant-independent Quality-Contrastive "
            "Subspace Removal"
        ),
        "n_rows": int(len(frame)),
        "n_clean_trials": int(
            frame.loc[
                frame["view_condition"].eq(CLEAN),
                "onestop_trial_id",
            ].nunique()
        ),
        "n_participants": int(
            participant_table[
                "participant_id"
            ].nunique()
        ),
        "conditions": conditions,
        "degraded_conditions": degraded_conditions,
        "feature_set": "soft_full_all",
        "n_features": len(features),
        "feature_columns": features,
        "methods": {
            "clean_only": (
                "train clean only; select C/threshold on clean inner OOF"
            ),
            "augmentation": (
                "train all views without quality-consistency regularization"
            ),
            "qcr": (
                "condition-balanced paired quality scatter; continuously "
                "penalize classifier sensitivity along quality-change "
                "directions; train on all views"
            ),
        },
        "preprocessing": (
            "median imputation and standardization fit on clean training "
            "participants only"
        ),
        "quality_basis_fit": (
            "training-participant clean/degraded paired differences only; "
            "each degradation normalized by its RMS before averaging scatter"
        ),
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "c_grid": c_grid,
        "lambda_grid": lambda_grid,
        "threshold_grid_size": args.threshold_grid_size,
        "bootstrap": {
            "replicates": args.bootstrap,
            "unit": "participant",
            "metrics": [
                "macro_f1",
                "balanced_accuracy",
            ],
        },
        "seed": args.seed,
        "interpretation_caveat": (
            "OneStop condition and participant cohort are confounded. This "
            "experiment evaluates participant-independent robustness and "
            "quality invariance, not within-person task causality."
        ),
        "outputs": {
            "predictions": str(predictions_path),
            "pooled_metrics": str(pooled_path),
            "robust_summary": str(robust_path),
            "participant_predictions": str(
                participant_prediction_path
            ),
            "participant_metrics": str(
                participant_metric_path
            ),
            "inner_selection": str(selection_path),
            "outer_selected_models": str(
                outer_model_path
            ),
            "quality_difference_audit": str(
                quality_condition_path
            ),
            "quality_spectrum": str(
                quality_spectrum_path
            ),
            "bootstrap_summary": str(
                bootstrap_summary_path
            ),
            "bootstrap_deltas": str(
                bootstrap_delta_path
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

    print("\n=== QCR POOLED CONDITION METRICS ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(pooled.to_string(index=False))

    print("\n=== QCR ROBUST SUMMARY ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(robust.to_string(index=False))

    print("\n=== QCR OUTER SELECTED MODELS ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(
            outer_models.to_string(index=False)
        )

    print("\n=== QCR BOOTSTRAP DELTAS ===")
    if bootstrap_deltas.empty:
        print("Bootstrap disabled.")
    else:
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            240,
        ):
            print(
                bootstrap_deltas.to_string(
                    index=False
                )
            )

    print(f"\nSaved: {predictions_path}")
    print(f"Saved: {pooled_path}")
    print(f"Saved: {robust_path}")
    print(f"Saved: {selection_path}")
    print(f"Saved: {bootstrap_delta_path}")
    print(f"Saved: {manifest_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
