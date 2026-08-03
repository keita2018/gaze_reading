from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


MODEL_REL = Path(
    "outputs/frozen_onestop_external_validation_v1/models/soft_full_all.joblib"
)
SOURCE_REL = Path(
    "data/processed/onestop/quality_views_asc344_v1/"
    "onestop_quality_views_all_features.parquet"
)
SOURCE_OOF_REL = Path(
    "outputs/frozen_onestop_external_validation_v1/"
    "onestop_selected_oof_predictions.parquet"
)
TARGET_REL = Path(
    "outputs/webqamgaze_external_validation_stage3c/"
    "soft_full_features_primary.csv"
)
FACTOR_MANIFEST_REL = Path(
    "outputs/linear_feature_group_factorization_v1/"
    "linear_feature_group_factorization_manifest.json"
)
EXACT_COMPONENTS_REL = Path(
    "outputs/linear_feature_group_factorization_v1/"
    "target_exact_score_components.csv"
)

QUALITY_FEATURES = [
    "soft_aoi_valid_rate",
    "soft_aoi_active_rate",
    "soft_aoi_word_box_hit_rate",
    "soft_aoi_sentence_envelope_hit_rate",
    "soft_aoi_qmax_mean",
    "soft_aoi_qmax_median",
    "soft_aoi_qmax_q10",
    "soft_aoi_qmax_ge_0_5_rate",
    "soft_aoi_nearest_distance_mean_word_heights",
    "soft_aoi_nearest_distance_median_word_heights",
    "soft_aoi_nearest_distance_q90_word_heights",
    "soft_aoi_nearest_distance_q95_word_heights",
]

PARTICIPANT_ALIASES = [
    "participant_id", "participant", "subject_id", "subject",
    "worker_id", "worker", "reader_id", "reader",
]
TRIAL_ALIASES = [
    "trial_id", "trial", "trial_index", "sample_id", "item_id",
    "record_id", "trial_uid", "trial_key",
]
LABEL_ALIASES = [
    "label_nr", "label", "y", "target", "class", "y_true",
    "true_label", "reading_goal", "goal_label", "condition_label",
]
SCORE_ALIASES = [
    "score", "decision_score", "decision_function", "raw_score",
    "frozen_score",
]

BOOTSTRAP_METHODS = [
    "frozen_full41",
    "participant_median_loo",
    "participant_mean_loo",
    "participant_median_mad",
    "participant_rank_normalization",
    "content_only_29",
    "qcn_only",
    "qcn_pc",
]


@dataclass
class MethodMetric:
    method: str
    n_trials: int
    n_participants: int
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc_nr: float
    nr_prediction_rate: float
    mean_within_participant_auc: float


def norm(value: Any) -> str:
    return str(value).strip().lower()


def first_existing(columns: Iterable[str], aliases: Sequence[str]) -> str:
    lower = {norm(c): str(c) for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_binary_label(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int).to_numpy()

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique = set(numeric.astype(int).unique().tolist())
        if unique.issubset({0, 1}):
            return numeric.astype(int).to_numpy()

    positive = {
        "1", "nr", "natural_reading", "natural reading",
        "ordinary", "ordinary_reading", "ordinary reading",
    }
    negative = {
        "0", "is", "information_seeking", "information seeking",
        "tsr", "task_specific_reading", "task-specific reading",
    }
    mapped: list[int] = []
    bad: set[str] = set()
    for value in series:
        token = norm(value)
        if token in positive:
            mapped.append(1)
        elif token in negative:
            mapped.append(0)
        else:
            bad.add(token)
    if bad:
        raise ValueError(
            f"Cannot normalize binary labels in {name}; unknown values: "
            f"{sorted(bad)[:20]}"
        )
    return np.asarray(mapped, dtype=int)


def load_frozen_model(path: Path) -> tuple[dict[str, Any], Any, list[str], float]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected dict artifact, found {type(artifact)}")
    pipeline = artifact["pipeline"]
    features = list(artifact["feature_columns"])
    threshold = float(artifact["decision_threshold"])

    if len(features) != 41:
        raise RuntimeError(f"Expected 41 Full features, found {len(features)}")
    if features[-12:] != QUALITY_FEATURES:
        raise RuntimeError(
            "Frozen feature order does not end with the locked 12 quality features."
        )
    if list(pipeline.classes_) != [0, 1]:
        raise RuntimeError(f"Unexpected classes_: {pipeline.classes_}")
    return artifact, pipeline, features, threshold


def transform_before_model(pipeline: Any, frame: pd.DataFrame) -> np.ndarray:
    values: Any = frame
    steps = list(pipeline.steps)
    if not steps or steps[-1][0] != "model":
        raise RuntimeError(f"Unexpected pipeline steps: {[name for name, _ in steps]}")
    for _, transformer in steps[:-1]:
        values = transformer.transform(values)
    return np.asarray(values, dtype=float)


def exact_score_components(
    pipeline: Any,
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = transform_before_model(pipeline, frame[features])
    model = pipeline.named_steps["model"]
    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])

    if z.shape[1] != len(features) or coef.size != len(features):
        raise RuntimeError(
            f"Shape mismatch: z={z.shape}, coef={coef.shape}, features={len(features)}"
        )

    contributions = z * coef[None, :]
    quality_idx = np.asarray([features.index(f) for f in QUALITY_FEATURES])
    content_idx = np.asarray([i for i in range(len(features)) if i not in quality_idx])

    quality = contributions[:, quality_idx].sum(axis=1)
    content = contributions[:, content_idx].sum(axis=1)
    reconstructed = intercept + content + quality
    raw = np.asarray(pipeline.decision_function(frame[features]), dtype=float)

    max_error = float(np.max(np.abs(raw - reconstructed)))
    if max_error > 1e-8:
        raise RuntimeError(
            f"Exact linear decomposition failed; max score error={max_error:.3e}"
        )
    return raw, content, quality, contributions


def parse_factor_references(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    refs = data["source_references"]
    required = {
        "full_final_reference",
        "quality_contribution_reference",
        "quality_removed_final_reference",
    }
    missing = required - set(refs)
    if missing:
        raise KeyError(f"Missing factorization references: {sorted(missing)}")
    return {key: float(refs[key]) for key in required}


def leave_one_out_center(
    scores: np.ndarray,
    participant: np.ndarray,
    statistic: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.full(scores.shape, np.nan, dtype=float)
    fallback = 0

    for pid in pd.unique(participant):
        idx = np.flatnonzero(participant == pid)
        vals = scores[idx]
        if len(idx) < 2:
            raise ValueError(f"Participant {pid!r} has fewer than two trials.")

        if statistic == "median":
            for local, global_idx in enumerate(idx):
                out[global_idx] = float(np.median(np.delete(vals, local)))
        elif statistic == "mean":
            total = float(np.sum(vals))
            out[idx] = (total - vals) / (len(vals) - 1)
        else:
            raise ValueError(statistic)

    return out, {"fallback_count": fallback}


def robust_mad(values: np.ndarray) -> float:
    med = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - med)))


def participant_median_mad_scores(
    scores: np.ndarray,
    participant: np.ndarray,
    source_location: float,
    source_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full(scores.shape, np.nan, dtype=float)
    fallback_count = 0

    for pid in pd.unique(participant):
        idx = np.flatnonzero(participant == pid)
        vals = scores[idx]
        if len(idx) < 3:
            raise ValueError(
                f"Median/MAD control requires >=3 trials; participant={pid!r}"
            )
        full_scale = robust_mad(vals)
        for local, global_idx in enumerate(idx):
            other = np.delete(vals, local)
            center = float(np.median(other))
            scale = robust_mad(other)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = full_scale
                fallback_count += 1
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = source_scale
                fallback_count += 1
            result[global_idx] = (
                (scores[global_idx] - center) / scale * source_scale
                + source_location
            )

    return result, {"fallback_count": fallback_count}


def participant_rank_scores(
    scores: np.ndarray,
    participant: np.ndarray,
    source_scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full(scores.shape, np.nan, dtype=float)
    min_n = 10**9

    for pid in pd.unique(participant):
        idx = np.flatnonzero(participant == pid)
        vals = scores[idx]
        min_n = min(min_n, len(vals))
        if len(vals) < 2:
            raise ValueError(f"Participant {pid!r} has fewer than two trials.")
        for local, global_idx in enumerate(idx):
            other = np.delete(vals, local)
            value = vals[local]
            less = np.sum(other < value)
            equal = np.sum(other == value)
            percentile = (less + 0.5 * equal) / len(other)
            percentile = float(np.clip(percentile, 0.005, 0.995))
            result[global_idx] = float(np.quantile(source_scores, percentile))

    return result, {"minimum_participant_trials": int(min_n)}


def average_precision_nr(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y, score)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def mean_within_participant_auc(
    y: np.ndarray,
    score: np.ndarray,
    participant: np.ndarray,
) -> float:
    aucs: list[float] = []
    for pid in pd.unique(participant):
        idx = participant == pid
        if np.unique(y[idx]).size == 2:
            aucs.append(float(roc_auc_score(y[idx], score[idx])))
    return float(np.mean(aucs)) if aucs else float("nan")


def calculate_metrics(
    method: str,
    y: np.ndarray,
    score: np.ndarray,
    participant: np.ndarray,
    threshold: float,
) -> MethodMetric:
    pred = (score >= threshold).astype(int)
    return MethodMetric(
        method=method,
        n_trials=int(len(y)),
        n_participants=int(pd.unique(participant).size),
        accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        roc_auc=float(roc_auc_score(y, score)),
        pr_auc_nr=average_precision_nr(y, score),
        nr_prediction_rate=float(np.mean(pred)),
        mean_within_participant_auc=mean_within_participant_auc(
            y, score, participant
        ),
    )


def exact_best_macro_f1_threshold(
    y: np.ndarray,
    score: np.ndarray,
) -> tuple[float, float]:
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]

    pos = (sorted_y == 1).astype(np.int64)
    neg = 1 - pos
    cum_pos = np.concatenate([[0], np.cumsum(pos)])
    cum_neg = np.concatenate([[0], np.cumsum(neg)])

    total_pos = int(cum_pos[-1])
    total_neg = int(cum_neg[-1])
    tp = total_pos - cum_pos
    fn = cum_pos
    tn = cum_neg
    fp = total_neg - cum_neg

    denom_pos = 2 * tp + fp + fn
    denom_neg = 2 * tn + fn + fp
    f1_pos = np.divide(
        2 * tp,
        denom_pos,
        out=np.zeros_like(tp, dtype=float),
        where=denom_pos != 0,
    )
    f1_neg = np.divide(
        2 * tn,
        denom_neg,
        out=np.zeros_like(tn, dtype=float),
        where=denom_neg != 0,
    )
    macro = 0.5 * (f1_pos + f1_neg)

    valid_k = np.ones(len(score) + 1, dtype=bool)
    if len(score) > 1:
        valid_k[1:-1] = sorted_score[:-1] < sorted_score[1:]
    candidate_idx = np.flatnonzero(valid_k)
    best_value = float(np.max(macro[candidate_idx]))
    tied = candidate_idx[np.isclose(macro[candidate_idx], best_value)]

    thresholds: list[float] = []
    for k in tied:
        if k == 0:
            threshold = float(np.nextafter(sorted_score[0], -np.inf))
        elif k == len(score):
            threshold = float(np.nextafter(sorted_score[-1], np.inf))
        else:
            threshold = float((sorted_score[k - 1] + sorted_score[k]) / 2)
        thresholds.append(threshold)

    best_threshold = min(thresholds, key=lambda value: (abs(value), value))
    return best_threshold, best_value


def find_clean_column(
    frame: pd.DataFrame,
    feature_columns: set[str],
) -> tuple[str, Any, dict[str, list[str]]]:
    metadata = [c for c in frame.columns if c not in feature_columns]
    inspected: dict[str, list[str]] = {}

    for col in metadata:
        series = frame[col]
        if series.nunique(dropna=True) > 50:
            continue
        values = [str(v) for v in series.dropna().unique().tolist()]
        inspected[col] = values[:50]
        normalized = {norm(v): v for v in values}
        for token in ["clean", "original", "baseline", "none"]:
            if token in normalized:
                return col, normalized[token], inspected

    for col in metadata:
        if norm(col) in {"is_clean", "clean"}:
            values = frame[col].dropna().unique().tolist()
            if True in values or 1 in values:
                return col, True if True in values else 1, inspected

    raise RuntimeError(
        "Could not detect the clean-view column. Metadata unique values: "
        + json.dumps(inspected, ensure_ascii=False)
    )


def filter_full_oof(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(frame) == 18573:
        return frame.copy(), {"filter_column": "", "filter_value": ""}

    for col in frame.columns:
        if frame[col].nunique(dropna=True) > 20:
            continue
        values = frame[col].dropna().astype(str).unique().tolist()
        for value in values:
            token = norm(value)
            if token == "soft_full_all" or (
                "full" in token and "compact" not in token
            ):
                selected = frame[frame[col].astype(str) == value].copy()
                if len(selected) == 18573:
                    return selected, {
                        "filter_column": col,
                        "filter_value": value,
                    }

    raise RuntimeError(
        f"Could not select 18,573 Full OOF rows from {len(frame)} rows."
    )


def source_summary_from_oof(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    frame, filter_info = filter_full_oof(frame)
    participant_col = first_existing(frame.columns, PARTICIPANT_ALIASES)
    label_col = first_existing(frame.columns, LABEL_ALIASES)
    score_col = first_existing(frame.columns, SCORE_ALIASES)
    if not participant_col or not label_col or not score_col:
        raise RuntimeError(
            f"OOF metadata not found: participant={participant_col}, "
            f"label={label_col}, score={score_col}"
        )

    y = normalize_binary_label(frame[label_col], label_col)
    score = pd.to_numeric(frame[score_col], errors="raise").to_numpy(float)
    participant = frame[participant_col].astype(str).to_numpy()

    participant_means = (
        pd.DataFrame({"participant": participant, "score": score})
        .groupby("participant", sort=False)["score"]
        .mean()
    )
    participant_medians = (
        pd.DataFrame({"participant": participant, "score": score})
        .groupby("participant", sort=False)["score"]
        .median()
    )
    participant_mads = (
        pd.DataFrame({"participant": participant, "score": score})
        .groupby("participant", sort=False)["score"]
        .apply(lambda values: robust_mad(values.to_numpy(float)))
    )

    info = {
        **filter_info,
        "n_rows": int(len(frame)),
        "n_participants": int(pd.unique(participant).size),
        "median_of_participant_medians": float(np.median(participant_medians)),
        "median_of_participant_means": float(np.median(participant_means)),
        "median_participant_mad": float(np.median(participant_mads)),
    }
    return y, score, participant, info


def fit_content_only_model(
    source_clean: pd.DataFrame,
    target: pd.DataFrame,
    base_pipeline: Any,
    content_features: list[str],
    source_participant_col: str,
    source_label_col: str,
    target_label: np.ndarray,
    target_participant: np.ndarray,
    threshold_for_other_methods: float,
    output_dir: Path,
) -> tuple[np.ndarray, float, pd.DataFrame, dict[str, Any]]:
    y = normalize_binary_label(source_clean[source_label_col], source_label_col)
    groups = source_clean[source_participant_col].astype(str).to_numpy()
    X = source_clean[content_features]

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    candidates = [0.01, 0.1, 1.0, 10.0]
    selection_rows: list[dict[str, Any]] = []
    best: tuple[float, float, float, np.ndarray] | None = None

    for c_value in candidates:
        oof = np.full(len(source_clean), np.nan, dtype=float)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            estimator = clone(base_pipeline)
            estimator.set_params(model__C=c_value)
            estimator.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = estimator.decision_function(X.iloc[test_idx])

        if np.isnan(oof).any():
            raise RuntimeError(f"Missing OOF predictions for C={c_value}")

        selected_threshold, selected_macro_f1 = exact_best_macro_f1_threshold(
            y, oof
        )
        auc = float(roc_auc_score(y, oof))
        selection_rows.append({
            "C": c_value,
            "selected_threshold": selected_threshold,
            "oof_macro_f1": selected_macro_f1,
            "oof_roc_auc": auc,
        })
        ranking = (selected_macro_f1, auc, -c_value)
        if best is None or ranking > (best[0], best[1], -best[2]):
            best = (
                selected_macro_f1,
                auc,
                c_value,
                oof.copy(),
            )

    assert best is not None
    best_macro, best_auc, best_c, best_oof = best
    best_threshold, _ = exact_best_macro_f1_threshold(y, best_oof)

    final_model = clone(base_pipeline)
    final_model.set_params(model__C=best_c)
    final_model.fit(X, y)
    target_score = np.asarray(
        final_model.decision_function(target[content_features]), dtype=float
    )
    joblib.dump(
        {
            "artifact_version": "iui_p0_content_only_29_v1",
            "feature_columns": content_features,
            "C": best_c,
            "decision_threshold": best_threshold,
            "pipeline": final_model,
            "selection_protocol": (
                "5-fold StratifiedGroupKFold by OneStop participant; "
                "C and exact macro-F1 threshold selected using source OOF only"
            ),
        },
        output_dir / "content_only_29_source_model.joblib",
    )

    selection = pd.DataFrame(selection_rows)
    info = {
        "selected_C": best_c,
        "selected_threshold": best_threshold,
        "selected_oof_macro_f1": best_macro,
        "selected_oof_roc_auc": best_auc,
        "n_source_clean": int(len(source_clean)),
        "n_source_participants": int(pd.unique(groups).size),
    }
    return target_score, best_threshold, selection, info


def bootstrap_differences(
    y: np.ndarray,
    participant: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    methods: list[str],
    reference_methods: list[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unique_participants = pd.unique(participant)
    participant_indices = {
        pid: np.flatnonzero(participant == pid) for pid in unique_participants
    }

    replicate_rows: list[dict[str, Any]] = []
    metrics_to_store = ["macro_f1", "balanced_accuracy", "roc_auc", "pr_auc_nr"]

    for replicate in range(n_bootstrap):
        sampled = rng.choice(
            unique_participants,
            size=len(unique_participants),
            replace=True,
        )
        indices = np.concatenate([participant_indices[pid] for pid in sampled])
        y_b = y[indices]

        if np.unique(y_b).size < 2:
            continue

        for method in methods:
            score_b = scores[method][indices]
            pred_b = (score_b >= thresholds[method]).astype(int)
            row = {
                "replicate": replicate,
                "method": method,
                "macro_f1": f1_score(
                    y_b, pred_b, average="macro", zero_division=0
                ),
                "balanced_accuracy": balanced_accuracy_score(y_b, pred_b),
                "roc_auc": roc_auc_score(y_b, score_b),
                "pr_auc_nr": average_precision_nr(y_b, score_b),
            }
            replicate_rows.append(row)

    replicates = pd.DataFrame(replicate_rows)
    delta_rows: list[dict[str, Any]] = []

    for reference in reference_methods:
        for method in methods:
            if method == reference:
                continue
            for metric in metrics_to_store:
                left = (
                    replicates[replicates["method"] == method]
                    .set_index("replicate")[metric]
                )
                right = (
                    replicates[replicates["method"] == reference]
                    .set_index("replicate")[metric]
                )
                common = left.index.intersection(right.index)
                delta = (left.loc[common] - right.loc[common]).to_numpy(float)
                delta_rows.append({
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "mean_difference": float(np.mean(delta)),
                    "ci_low": float(np.quantile(delta, 0.025)),
                    "ci_high": float(np.quantile(delta, 0.975)),
                    "n_replicates": int(len(delta)),
                })

    return replicates, pd.DataFrame(delta_rows)


def random_group_controls(
    y: np.ndarray,
    target_participant: np.ndarray,
    raw_target: np.ndarray,
    target_contributions: np.ndarray,
    source_participant: np.ndarray,
    raw_source: np.ndarray,
    source_contributions: np.ndarray,
    threshold: float,
    participant_baseline_auc: float,
    quality_indices: list[int],
    n_groups: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    feature_count = target_contributions.shape[1]
    observed: set[tuple[int, ...]] = set()
    rows: list[dict[str, Any]] = []

    while len(observed) < n_groups:
        group = tuple(sorted(rng.choice(feature_count, 12, replace=False).tolist()))
        if group in observed or list(group) == sorted(quality_indices):
            continue
        observed.add(group)

    for group_index, group in enumerate(sorted(observed), start=1):
        group_array = np.asarray(group)
        target_contribution = target_contributions[:, group_array].sum(axis=1)
        source_contribution = source_contributions[:, group_array].sum(axis=1)

        source_contribution_frame = pd.DataFrame({
            "participant": source_participant,
            "contribution": source_contribution,
        })
        contribution_reference = float(np.median(
            source_contribution_frame
            .groupby("participant", sort=False)["contribution"]
            .median()
        ))

        source_neutral = raw_source - source_contribution + contribution_reference
        source_neutral_reference = float(np.median(
            pd.DataFrame({
                "participant": source_participant,
                "score": source_neutral,
            })
            .groupby("participant", sort=False)["score"]
            .median()
        ))

        target_neutral = (
            raw_target - target_contribution + contribution_reference
        )
        target_center, _ = leave_one_out_center(
            target_neutral, target_participant, "median"
        )
        target_neutral_pc = (
            target_neutral - target_center + source_neutral_reference
        )
        metric = calculate_metrics(
            f"random_group_{group_index:03d}",
            y,
            target_neutral_pc,
            target_participant,
            threshold,
        )
        rows.append({
            "random_group_index": group_index,
            "feature_indices": "|".join(map(str, group)),
            "contribution_reference": contribution_reference,
            "neutralized_participant_reference": source_neutral_reference,
            "macro_f1": metric.macro_f1,
            "roc_auc": metric.roc_auc,
            "pr_auc_nr": metric.pr_auc_nr,
            "auc_gain_over_participant_median": (
                metric.roc_auc - participant_baseline_auc
            ),
        })

    frame = pd.DataFrame(rows)
    summary = {
        "n_random_groups": int(len(frame)),
        "auc_gain_mean": float(frame["auc_gain_over_participant_median"].mean()),
        "auc_gain_q95": float(
            frame["auc_gain_over_participant_median"].quantile(0.95)
        ),
        "auc_gain_max": float(frame["auc_gain_over_participant_median"].max()),
    }
    return frame, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run IUI P0-2 standard calibration, content-only, and random-feature "
            "neutralization controls on English WebQAmGaze development data."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/iui_p0_standard_controls_v1",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-groups", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": project_root / MODEL_REL,
        "source": project_root / SOURCE_REL,
        "source_oof": project_root / SOURCE_OOF_REL,
        "target": project_root / TARGET_REL,
        "factor_manifest": project_root / FACTOR_MANIFEST_REL,
        "exact_components": project_root / EXACT_COMPONENTS_REL,
    }
    for name in ["model", "source", "source_oof", "target", "factor_manifest"]:
        if not paths[name].exists():
            raise FileNotFoundError(f"Required {name} file not found: {paths[name]}")

    artifact, pipeline, features, threshold = load_frozen_model(paths["model"])
    references = parse_factor_references(paths["factor_manifest"])
    full_reference = references["full_final_reference"]
    quality_reference = references["quality_contribution_reference"]
    qcn_participant_reference = references["quality_removed_final_reference"]

    print("Loading English WebQAmGaze Full-41 target features...")
    target = pd.read_csv(paths["target"])
    missing_target = [f for f in features if f not in target.columns]
    if missing_target:
        raise RuntimeError(f"Target is missing frozen features: {missing_target}")

    target_participant_col = first_existing(
        target.columns, PARTICIPANT_ALIASES
    )
    target_trial_col = first_existing(target.columns, TRIAL_ALIASES)
    target_label_col = first_existing(target.columns, LABEL_ALIASES)
    if not target_participant_col or not target_label_col:
        raise RuntimeError(
            f"Target metadata missing: participant={target_participant_col}, "
            f"label={target_label_col}"
        )

    target_participant = target[target_participant_col].astype(str).to_numpy()
    target_y = normalize_binary_label(target[target_label_col], target_label_col)
    target_trial = (
        target[target_trial_col].astype(str).to_numpy()
        if target_trial_col
        else np.arange(len(target)).astype(str)
    )

    raw, content, quality, contributions = exact_score_components(
        pipeline, target, features
    )

    print("Loading OneStop source OOF scores...")
    source_oof = pd.read_parquet(paths["source_oof"])
    _, source_oof_score, source_oof_participant, source_oof_info = (
        source_summary_from_oof(source_oof)
    )

    source_mean_reference = source_oof_info["median_of_participant_means"]
    source_mad_scale = source_oof_info["median_participant_mad"]
    if source_mad_scale <= 1e-12:
        raise RuntimeError("Source participant MAD scale is zero.")

    raw_median_center, median_info = leave_one_out_center(
        raw, target_participant, "median"
    )
    participant_median = raw - raw_median_center + full_reference

    raw_mean_center, mean_info = leave_one_out_center(
        raw, target_participant, "mean"
    )
    participant_mean = raw - raw_mean_center + source_mean_reference

    participant_mad, mad_info = participant_median_mad_scores(
        raw,
        target_participant,
        full_reference,
        source_mad_scale,
    )
    participant_rank, rank_info = participant_rank_scores(
        raw, target_participant, source_oof_score
    )

    qcn_only = raw - quality + quality_reference
    qcn_center, _ = leave_one_out_center(
        qcn_only, target_participant, "median"
    )
    qcn_pc = qcn_only - qcn_center + qcn_participant_reference

    print("Loading OneStop paired quality-view feature table...")
    source_all = pd.read_parquet(paths["source"])
    missing_source = [f for f in features if f not in source_all.columns]
    if missing_source:
        raise RuntimeError(f"Source is missing frozen features: {missing_source}")

    source_clean_col, source_clean_value, clean_scan = find_clean_column(
        source_all, set(features)
    )
    if source_clean_value is True or source_clean_value == 1:
        source_clean = source_all[
            source_all[source_clean_col] == source_clean_value
        ].copy()
    else:
        source_clean = source_all[
            source_all[source_clean_col].astype(str) == str(source_clean_value)
        ].copy()

    if len(source_clean) != 18573:
        raise RuntimeError(
            f"Expected 18,573 clean source rows, found {len(source_clean)} "
            f"using {source_clean_col}={source_clean_value!r}"
        )

    source_participant_col = first_existing(
        source_clean.columns, PARTICIPANT_ALIASES
    )
    source_label_col = first_existing(source_clean.columns, LABEL_ALIASES)
    if not source_participant_col or not source_label_col:
        raise RuntimeError(
            f"Source metadata missing: participant={source_participant_col}, "
            f"label={source_label_col}"
        )
    source_participant = (
        source_clean[source_participant_col].astype(str).to_numpy()
    )

    raw_source, _, _, source_contributions = exact_score_components(
        pipeline, source_clean, features
    )

    print("Training the OneStop-only content-only 29 control...")
    content_features = [f for f in features if f not in QUALITY_FEATURES]
    (
        content_only_score,
        content_only_threshold,
        content_selection,
        content_info,
    ) = fit_content_only_model(
        source_clean=source_clean,
        target=target,
        base_pipeline=pipeline,
        content_features=content_features,
        source_participant_col=source_participant_col,
        source_label_col=source_label_col,
        target_label=target_y,
        target_participant=target_participant,
        threshold_for_other_methods=threshold,
        output_dir=output_dir,
    )
    content_selection.to_csv(
        output_dir / "content_only_29_source_selection.csv",
        index=False,
    )

    scores = {
        "frozen_full41": raw,
        "participant_median_loo": participant_median,
        "participant_mean_loo": participant_mean,
        "participant_median_mad": participant_mad,
        "participant_rank_normalization": participant_rank,
        "content_only_29": content_only_score,
        "qcn_only": qcn_only,
        "qcn_pc": qcn_pc,
    }
    thresholds = {method: threshold for method in scores}
    thresholds["content_only_29"] = content_only_threshold

    metric_rows = [
        calculate_metrics(
            method,
            target_y,
            score,
            target_participant,
            thresholds[method],
        )
        for method, score in scores.items()
    ]
    metrics = pd.DataFrame([asdict(row) for row in metric_rows])
    metrics.to_csv(output_dir / "standard_control_metrics.csv", index=False)

    metric_map = {row.method: row for row in metric_rows}

    print("Running 100 random 12-feature neutralization controls...")
    quality_indices = [features.index(f) for f in QUALITY_FEATURES]
    random_groups, random_summary = random_group_controls(
        y=target_y,
        target_participant=target_participant,
        raw_target=raw,
        target_contributions=contributions,
        source_participant=source_participant,
        raw_source=raw_source,
        source_contributions=source_contributions,
        threshold=threshold,
        participant_baseline_auc=metric_map[
            "participant_median_loo"
        ].roc_auc,
        quality_indices=quality_indices,
        n_groups=args.random_groups,
        seed=args.seed,
    )
    random_groups.to_csv(
        output_dir / "random_12_feature_neutralization.csv",
        index=False,
    )
    qcn_auc_gain = (
        metric_map["qcn_pc"].roc_auc
        - metric_map["participant_median_loo"].roc_auc
    )
    random_summary["qcn_pc_auc_gain"] = qcn_auc_gain
    random_summary["qcn_pc_percentile_among_random_groups"] = float(
        np.mean(
            random_groups["auc_gain_over_participant_median"].to_numpy()
            <= qcn_auc_gain
        )
    )

    print(f"Running {args.bootstrap} participant-cluster bootstrap replicates...")
    bootstrap_replicates, bootstrap_deltas = bootstrap_differences(
        y=target_y,
        participant=target_participant,
        scores=scores,
        thresholds=thresholds,
        methods=BOOTSTRAP_METHODS,
        reference_methods=["participant_median_loo", "qcn_pc"],
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    bootstrap_replicates.to_parquet(
        output_dir / "standard_control_bootstrap_replicates.parquet",
        index=False,
    )
    bootstrap_deltas.to_csv(
        output_dir / "standard_control_bootstrap_deltas.csv",
        index=False,
    )

    prediction_frame = pd.DataFrame({
        "participant_id": target_participant,
        "trial_id": target_trial,
        "label_nr": target_y,
        "frozen_full41": raw,
        "content_contribution": content,
        "quality_contribution": quality,
        **scores,
    })
    # Remove duplicate frozen column added through **scores.
    prediction_frame = prediction_frame.loc[
        :, ~prediction_frame.columns.duplicated()
    ]
    prediction_frame.to_parquet(
        output_dir / "standard_control_trial_predictions.parquet",
        index=False,
    )

    expected_checks = {
        "raw_score_reconstruction_max_abs_error": float(
            np.max(np.abs(raw - (content + quality + pipeline.named_steps[
                "model"
            ].intercept_[0])))
        ),
        "frozen_threshold": threshold,
        "full_reference": full_reference,
        "quality_reference": quality_reference,
        "qcn_participant_reference": qcn_participant_reference,
        "source_oof": source_oof_info,
        "source_clean_detection": {
            "column": source_clean_col,
            "value": str(source_clean_value),
            "scanned_metadata": clean_scan,
        },
        "participant_controls": {
            "median": median_info,
            "mean": mean_info,
            "median_mad": mad_info,
            "rank": rank_info,
        },
        "content_only": content_info,
        "random_group_summary": random_summary,
    }

    if paths["exact_components"].exists():
        existing = pd.read_csv(paths["exact_components"])
        expected_checks["existing_exact_components_rows"] = int(len(existing))
        # The exact existing schema can differ; record rather than silently assume.
        expected_checks["existing_exact_components_columns"] = list(
            existing.columns
        )

    (output_dir / "standard_control_audit.json").write_text(
        json.dumps(expected_checks, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    input_hashes = {
        name: {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
        if path.exists()
    }
    release_manifest = {
        "artifact_version": "iui_p0_standard_controls_v1",
        "development_target": (
            "English WebQAmGaze; controls only, not confirmatory target"
        ),
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "random_feature_groups": args.random_groups,
        "input_hashes": input_hashes,
        "frozen_features": features,
        "quality_features": QUALITY_FEATURES,
        "methods": list(scores),
        "thresholds": thresholds,
        "references": references,
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary_lines = [
        "# IUI P0-2 Standard Control Results",
        "",
        "Development target: **English WebQAmGaze**.",
        "",
        "This analysis does not use the unused-language confirmatory target.",
        "",
        "## Pooled metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## QCN-PC versus random 12-feature groups",
        "",
        f"- QCN-PC AUC gain over participant median: `{qcn_auc_gain:.6f}`",
        f"- Random-group mean gain: "
        f"`{random_summary['auc_gain_mean']:.6f}`",
        f"- Random-group 95th percentile: "
        f"`{random_summary['auc_gain_q95']:.6f}`",
        f"- QCN-PC percentile among random controls: "
        f"`{random_summary['qcn_pc_percentile_among_random_groups']:.3f}`",
        "",
        "## Content-only source selection",
        "",
        content_selection.to_markdown(index=False),
        "",
        "## Interpretation constraints",
        "",
        "- Do not modify the locked QCN-PC definition using these results.",
        "- These are development-set controls, not independent confirmation.",
        "- Continue to the unused-language audit only after archiving this output.",
    ]
    (output_dir / "RESULTS_SUMMARY.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("")
    print("P0-2 standard controls completed.")
    print(f"Output: {output_dir}")
    print("")
    print(metrics[[
        "method", "macro_f1", "roc_auc", "pr_auc_nr",
        "nr_prediction_rate"
    ]].to_string(index=False))
    print("")
    print(
        "QCN-PC percentile among random groups: "
        f"{random_summary['qcn_pc_percentile_among_random_groups']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
