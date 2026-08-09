#!/usr/bin/env python
r"""
Post-hoc deployment sensitivity: causal / few-shot participant self-calibration.

Purpose
-------
The manuscript's participant self-calibration uses a leave-one-trial-out (LOO)
participant median. It is label-free, but the center for trial i can use trials
that occur later in the participant/session. This script stress-tests whether
similar behavior survives a strictly past-only deployment policy.

Important interpretation boundary
---------------------------------
This is a POST-HOC DEPLOYMENT SENSITIVITY ANALYSIS. It does not alter the locked
Turkish confirmatory analysis, its primary estimand, or its confirmation label.
The Turkish labels are already open; therefore no result from this script may be
reported as a new preregistered/locked confirmation.

The WebQAmGaze procedure is block-ordered (in the audited manifests, early trial
positions are NR and later positions are IS). Consequently, past-only calibration
also probes a real but task-mixture-sensitive deployment scenario. The script
therefore reports trial-position and task-block diagnostics explicitly.

Calibration policy
------------------
For each calibration session, trials are sorted by trial_index. For current trial t,
only strictly previous RAW FROZEN scores are allowed to form the participant center.
For a history cap k in {1,2,3,5,...}:

    center_t = median(last min(k, n_previous) previous raw scores)

For 'all':

    center_t = median(all previous raw scores)

Then

    causal_score_t = raw_score_t - center_t + r_full

where r_full is the frozen OneStop source reference. The first trial has no previous
history. For full-deployment metrics it falls back to the raw frozen score; for the
'has_prior_only' scope it is excluded.

Calibration sessions
--------------------
To avoid inventing chronology across reused WebQAmGaze participant IDs, the primary
causal sequence is reset per participant record. When available, the session key is:

    participant_id + set_name + source_file

This is intentionally more conservative and deployable than imposing an arbitrary
order across multiple records that share an ID. Participant-cluster bootstrapping is
still performed at participant_id level.

Outputs
-------
- causal_calibration_trial_predictions.csv
- causal_calibration_metrics.csv
- causal_calibration_trial_position_metrics.csv
- causal_calibration_task_metrics.csv
- causal_calibration_session_audit.csv
- causal_calibration_bootstrap_replicates.csv
- causal_calibration_bootstrap_deltas.csv
- causal_calibration_manifest.json
- RESULTS_SUMMARY.md
- file_hash_manifest.csv

Example
-------
python -u src/analysis/run_causal_participant_calibration_sensitivity.py ^
  --english-predictions outputs/linear_feature_group_factorization_v1/target_feature_group_predictions.csv ^
  --english-manifest outputs/webqamgaze_external_validation_stage2/trial_manifest_primary_candidates.csv ^
  --factor-manifest outputs/linear_feature_group_factorization_v1/linear_feature_group_factorization_manifest.json ^
  --turkish-snapshot outputs/iui_p0_confirmatory_TR_v1 ^
  --turkish-manifest outputs/webqamgaze_tr_confirmatory_stage2_v1/trial_manifest_primary_candidates.csv ^
  --output-dir outputs/iui_p5_causal_participant_calibration_v1 ^
  --history-caps 1,2,3,5,all --bootstrap 2000 --seed 20260810
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc_nr",
    "average_precision_nr",
)
SCOPES = ("all_trials", "has_prior_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc causal/few-shot participant self-calibration sensitivity "
            "for English and Turkish WebQAmGaze."
        )
    )
    parser.add_argument("--english-predictions", type=Path, required=True)
    parser.add_argument("--english-manifest", type=Path, required=True)
    parser.add_argument("--factor-manifest", type=Path, required=True)
    parser.add_argument("--turkish-snapshot", type=Path, required=True)
    parser.add_argument("--turkish-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--history-caps",
        default="1,2,3,5,all",
        help="Comma-separated positive integers and/or 'all'.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=1e-10,
        help="Tolerance when comparing frozen English/Turkish source references.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_dir(path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_history_caps(text: str) -> list[int | None]:
    result: list[int | None] = []
    seen: set[str] = set()
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"all", "none", "full"}:
            key = "all"
            value: int | None = None
        else:
            value = int(token)
            if value < 1:
                raise ValueError("History caps must be >=1 or 'all'.")
            key = str(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    if not result:
        raise ValueError("No history caps were supplied.")
    return result


def cap_name(cap: int | None) -> str:
    return "all" if cap is None else str(cap)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def pr_auc_trapezoid(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, score)
    order = np.argsort(recall)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(precision[order], recall[order]))


def safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def metric_dict(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    prediction = (score >= float(threshold)).astype(int)
    if len(y_true) == 0:
        return {name: float("nan") for name in METRICS} | {
            "predicted_nr_rate": float("nan"),
            "true_nr_rate": float("nan"),
        }

    # Use an explicit two-class macro-F1 so position/task diagnostics remain
    # deterministic even when one class is absent. Balanced accuracy, ROC-AUC,
    # and PR-AUC are deliberately NaN for single-class subsets because they do
    # not have their usual binary-discrimination interpretation there.
    both_classes = np.unique(y_true).size == 2
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, prediction))
            if both_classes
            else float("nan")
        ),
        "roc_auc": safe_auc(y_true, score),
        "pr_auc_nr": pr_auc_trapezoid(y_true, score),
        "average_precision_nr": safe_average_precision(y_true, score),
        "predicted_nr_rate": float(np.mean(prediction == 1)),
        "true_nr_rate": float(np.mean(y_true == 1)),
    }


def normalize_manifest(manifest: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"external_trial_id", "participant_id", "trial_index"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{name} manifest lacks columns: {missing}")

    result = manifest.copy()
    result["external_trial_id"] = result["external_trial_id"].astype(str).str.strip()
    result["participant_id"] = result["participant_id"].astype(str).str.strip()
    result["trial_index"] = pd.to_numeric(
        result["trial_index"], errors="raise"
    ).astype(int)

    for column in ("set_name", "source_file", "task"):
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip()
    if "task" in result.columns:
        result["task"] = result["task"].str.upper()
    if "label_nr" in result.columns:
        result["label_nr"] = pd.to_numeric(
            result["label_nr"], errors="raise"
        ).astype(int)

    if result["external_trial_id"].duplicated().any():
        duplicates = result.loc[
            result["external_trial_id"].duplicated(keep=False),
            ["external_trial_id", "participant_id", "trial_index"],
        ].head(20)
        raise ValueError(
            f"{name} manifest has duplicate external_trial_id rows:\n"
            + duplicates.to_string(index=False)
        )
    return result


def load_factor_reference(path: Path) -> tuple[float, dict[str, Any]]:
    path = require_file(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    references = data.get("source_references", {})
    if "full_final_reference" not in references:
        raise KeyError(
            "Factorization manifest lacks source_references.full_final_reference"
        )
    return float(references["full_final_reference"]), data


def choose_session_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["participant_id"]
    if "set_name" in frame.columns:
        columns.append("set_name")
    if "source_file" in frame.columns:
        columns.append("source_file")
    return columns


def add_session_id(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    columns = choose_session_columns(result)
    result["calibration_session_id"] = (
        result[columns].astype(str).agg("||".join, axis=1)
    )
    return result, columns


def validate_score_table(frame: pd.DataFrame, dataset: str) -> None:
    required = {
        "external_trial_id",
        "participant_id",
        "label_nr",
        "trial_index",
        "raw_frozen_score",
        "loo_paper_score",
        "loo_same_reference_score",
        "frozen_threshold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset} prepared table lacks: {missing}")

    if frame["external_trial_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicate trial IDs after merge.")
    if set(frame["label_nr"].unique()) != {0, 1}:
        raise ValueError(f"{dataset}: expected both labels 0 and 1.")
    numeric = frame[
        [
            "raw_frozen_score",
            "loo_paper_score",
            "loo_same_reference_score",
            "frozen_threshold",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{dataset}: non-finite score/threshold values.")
    if frame["frozen_threshold"].nunique(dropna=False) != 1:
        raise ValueError(f"{dataset}: threshold is not unique.")


def load_english(
    predictions_path: Path,
    manifest_path: Path,
    source_reference: float,
) -> pd.DataFrame:
    predictions_path = require_file(predictions_path)
    manifest = normalize_manifest(pd.read_csv(require_file(manifest_path)), "English")
    predictions = pd.read_csv(predictions_path, low_memory=False)

    required = {
        "external_trial_id",
        "participant_id",
        "label_nr",
        "method",
        "score",
        "frozen_threshold",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"English predictions lack columns: {missing}")

    for column in ("external_trial_id", "participant_id", "method"):
        predictions[column] = predictions[column].astype(str).str.strip()
    predictions["label_nr"] = pd.to_numeric(
        predictions["label_nr"], errors="raise"
    ).astype(int)
    predictions["score"] = pd.to_numeric(
        predictions["score"], errors="raise"
    ).astype(float)
    predictions["frozen_threshold"] = pd.to_numeric(
        predictions["frozen_threshold"], errors="raise"
    ).astype(float)

    def method_rows(method: str, rename: str) -> pd.DataFrame:
        part = predictions[predictions["method"].eq(method)].copy()
        if part.empty:
            raise ValueError(f"English predictions have no method={method!r}")
        if part["external_trial_id"].duplicated().any():
            raise ValueError(f"Duplicate English rows for method={method}")
        return part[
            [
                "external_trial_id",
                "participant_id",
                "label_nr",
                "score",
                "frozen_threshold",
            ]
        ].rename(columns={"score": rename})

    raw = method_rows("raw_frozen", "raw_frozen_score")
    loo_paper = method_rows("participant_only_oof", "loo_paper_score").drop(
        columns=["participant_id", "label_nr", "frozen_threshold"]
    )

    if "participant_only_final_reference" in set(predictions["method"]):
        loo_same = method_rows(
            "participant_only_final_reference", "loo_same_reference_score"
        ).drop(columns=["participant_id", "label_nr", "frozen_threshold"])
    else:
        loo_same = loo_paper.rename(
            columns={"loo_paper_score": "loo_same_reference_score"}
        )

    merged = raw.merge(loo_paper, on="external_trial_id", validate="one_to_one")
    merged = merged.merge(loo_same, on="external_trial_id", validate="one_to_one")

    manifest_columns = [
        c
        for c in (
            "external_trial_id",
            "participant_id",
            "trial_index",
            "set_name",
            "source_file",
            "task",
            "label_nr",
        )
        if c in manifest.columns
    ]
    manifest_part = manifest[manifest_columns].copy()
    merged = merged.merge(
        manifest_part,
        on="external_trial_id",
        how="inner",
        suffixes=("_pred", "_manifest"),
        validate="one_to_one",
    )

    # Reconcile participant/label fields explicitly.
    if "participant_id_manifest" in merged.columns:
        if not (
            merged["participant_id_pred"].astype(str)
            == merged["participant_id_manifest"].astype(str)
        ).all():
            raise ValueError("English participant IDs disagree between predictions/manifest.")
        merged["participant_id"] = merged["participant_id_pred"].astype(str)
    else:
        merged["participant_id"] = merged["participant_id"]

    if "label_nr_manifest" in merged.columns:
        if not np.array_equal(
            merged["label_nr_pred"].to_numpy(dtype=int),
            merged["label_nr_manifest"].to_numpy(dtype=int),
        ):
            raise ValueError("English labels disagree between predictions/manifest.")
        merged["label_nr"] = merged["label_nr_pred"].astype(int)
    elif "label_nr_pred" in merged.columns:
        merged["label_nr"] = merged["label_nr_pred"].astype(int)

    keep = [
        "external_trial_id",
        "participant_id",
        "label_nr",
        "trial_index",
        "raw_frozen_score",
        "loo_paper_score",
        "loo_same_reference_score",
        "frozen_threshold",
    ]
    for column in ("set_name", "source_file", "task"):
        if column in merged.columns:
            keep.append(column)

    result = merged[keep].copy()
    result["dataset"] = "English"
    result["source_reference"] = float(source_reference)
    result, _ = add_session_id(result)
    validate_score_table(result, "English")
    return result


def load_turkish(
    snapshot_dir: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    snapshot_dir = require_dir(snapshot_dir)
    predictions_path = require_file(
        snapshot_dir / "predictions" / "unlabeled_frozen_scores.csv"
    )
    labels_path = require_file(snapshot_dir / "sealed" / "labels_sealed.csv")
    release_path = require_file(snapshot_dir / "release_manifest.json")
    manifest = normalize_manifest(pd.read_csv(require_file(manifest_path)), "Turkish")

    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path)
    release = json.loads(release_path.read_text(encoding="utf-8-sig"))

    keys = ["participant_id", "external_trial_id"]
    for name, frame in (("predictions", predictions), ("labels", labels)):
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise ValueError(f"Turkish {name} lacks keys: {missing}")
        for key in keys:
            frame[key] = frame[key].astype(str).str.strip()
        if frame.duplicated(keys).any():
            raise ValueError(f"Turkish {name} has duplicate keys.")

    required_scores = {
        "frozen_full41_score",
        "participant_median_loo_score",
    }
    missing_scores = sorted(required_scores - set(predictions.columns))
    if missing_scores:
        raise ValueError(f"Turkish frozen prediction table lacks: {missing_scores}")
    if "label_nr" not in labels.columns:
        raise ValueError("Turkish sealed label table lacks label_nr.")
    labels["label_nr"] = pd.to_numeric(labels["label_nr"], errors="raise").astype(int)

    frame = predictions.merge(labels[keys + ["label_nr"]], on=keys, validate="one_to_one")
    manifest_columns = [
        c
        for c in (
            "external_trial_id",
            "participant_id",
            "trial_index",
            "set_name",
            "source_file",
            "task",
            "label_nr",
        )
        if c in manifest.columns
    ]
    frame = frame.merge(
        manifest[manifest_columns],
        on=keys,
        how="inner",
        suffixes=("_sealed", "_manifest"),
        validate="one_to_one",
    )

    if "label_nr_manifest" in frame.columns:
        if not np.array_equal(
            frame["label_nr_sealed"].to_numpy(dtype=int),
            frame["label_nr_manifest"].to_numpy(dtype=int),
        ):
            raise ValueError("Turkish sealed and manifest labels disagree.")
        frame["label_nr"] = frame["label_nr_sealed"].astype(int)
    elif "label_nr_sealed" in frame.columns:
        frame["label_nr"] = frame["label_nr_sealed"].astype(int)

    threshold = float(release["frozen_threshold"])
    references = release.get("source_references", {})
    if "full_final_reference" not in references:
        raise KeyError("Turkish release lacks source_references.full_final_reference")
    source_reference = float(references["full_final_reference"])

    frame["raw_frozen_score"] = pd.to_numeric(
        frame["frozen_full41_score"], errors="raise"
    ).astype(float)
    frame["loo_paper_score"] = pd.to_numeric(
        frame["participant_median_loo_score"], errors="raise"
    ).astype(float)
    frame["loo_same_reference_score"] = frame["loo_paper_score"]
    frame["frozen_threshold"] = threshold

    keep = [
        "external_trial_id",
        "participant_id",
        "label_nr",
        "trial_index",
        "raw_frozen_score",
        "loo_paper_score",
        "loo_same_reference_score",
        "frozen_threshold",
    ]
    for column in ("set_name", "source_file", "task"):
        if column in frame.columns:
            keep.append(column)

    result = frame[keep].copy()
    result["dataset"] = "Turkish"
    result["source_reference"] = source_reference
    result, _ = add_session_id(result)
    validate_score_table(result, "Turkish")
    return result, source_reference, release


def task_order_diagnostic(group: pd.DataFrame) -> str:
    if "task" not in group.columns:
        return "unknown"
    ordered = group.sort_values("trial_index", kind="stable")
    return "->".join(ordered["task"].astype(str).tolist())


def session_audit(frame: pd.DataFrame) -> pd.DataFrame:
    participant_session_counts = (
        frame[["participant_id", "calibration_session_id"]]
        .drop_duplicates()
        .groupby("participant_id", observed=True)
        .size()
    )
    reused = set(participant_session_counts[participant_session_counts > 1].index.astype(str))

    rows: list[dict[str, Any]] = []
    for session_id, group in frame.groupby(
        "calibration_session_id", observed=True, sort=False
    ):
        ordered = group.sort_values("trial_index", kind="stable")
        participant = str(ordered["participant_id"].iloc[0])
        rows.append(
            {
                "dataset": str(ordered["dataset"].iloc[0]),
                "participant_id": participant,
                "calibration_session_id": str(session_id),
                "n_trials": int(len(ordered)),
                "min_trial_index": int(ordered["trial_index"].min()),
                "max_trial_index": int(ordered["trial_index"].max()),
                "n_unique_trial_indices": int(ordered["trial_index"].nunique()),
                "duplicate_trial_index": bool(
                    ordered["trial_index"].duplicated().any()
                ),
                "complete_10_trial_record": bool(
                    len(ordered) == 10 and ordered["trial_index"].nunique() == 10
                ),
                "task_order": task_order_diagnostic(ordered),
                "participant_id_reused_across_sessions": participant in reused,
            }
        )
    return pd.DataFrame(rows)


def apply_causal_scores(
    frame: pd.DataFrame,
    history_caps: list[int | None],
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    result["n_prior_available"] = 0
    result["has_prior"] = False

    method_columns: list[str] = []
    for cap in history_caps:
        name = f"causal_cap_{cap_name(cap)}"
        result[f"{name}_score"] = result["raw_frozen_score"].astype(float)
        result[f"{name}_prior_center"] = np.nan
        result[f"{name}_history_used"] = 0
        method_columns.append(name)

    for _, group in result.groupby(
        "calibration_session_id", observed=True, sort=False
    ):
        ordered = group.sort_values(
            ["trial_index", "external_trial_id"],
            kind="stable",
        )
        rows = ordered.index.to_numpy()
        raw = ordered["raw_frozen_score"].to_numpy(dtype=float)
        source_reference = float(ordered["source_reference"].iloc[0])

        for local_index, row_index in enumerate(rows):
            n_prior = local_index
            result.loc[row_index, "n_prior_available"] = int(n_prior)
            result.loc[row_index, "has_prior"] = bool(n_prior > 0)
            if n_prior == 0:
                continue

            previous = raw[:local_index]
            for cap in history_caps:
                name = f"causal_cap_{cap_name(cap)}"
                history = previous if cap is None else previous[-min(cap, n_prior) :]
                center = float(np.median(history))
                score = float(raw[local_index] - center + source_reference)
                result.loc[row_index, f"{name}_score"] = score
                result.loc[row_index, f"{name}_prior_center"] = center
                result.loc[row_index, f"{name}_history_used"] = int(len(history))

    return result, method_columns


def method_score_columns(causal_methods: Iterable[str]) -> dict[str, str]:
    mapping = {
        "raw_frozen": "raw_frozen_score",
        "loo_paper": "loo_paper_score",
        "loo_same_reference": "loo_same_reference_score",
    }
    for method in causal_methods:
        mapping[method] = f"{method}_score"
    return mapping


def evaluation_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all_trials":
        return np.ones(len(frame), dtype=bool)
    if scope == "has_prior_only":
        return frame["has_prior"].to_numpy(dtype=bool)
    raise ValueError(scope)


def pooled_metrics_table(
    datasets: dict[str, pd.DataFrame],
    method_columns: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in datasets.items():
        threshold = float(frame["frozen_threshold"].iloc[0])
        for scope in SCOPES:
            mask = evaluation_mask(frame, scope)
            subset = frame.loc[mask]
            for method, score_column in method_columns.items():
                metrics = metric_dict(
                    subset["label_nr"].to_numpy(dtype=int),
                    subset[score_column].to_numpy(dtype=float),
                    threshold,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "method": method,
                        "n_trials": int(len(subset)),
                        "n_participants": int(subset["participant_id"].nunique()),
                        "n_sessions": int(subset["calibration_session_id"].nunique()),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def grouped_metrics_table(
    datasets: dict[str, pd.DataFrame],
    method_columns: dict[str, str],
    grouping: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in datasets.items():
        if grouping not in frame.columns:
            continue
        threshold = float(frame["frozen_threshold"].iloc[0])
        for group_value, group in frame.groupby(grouping, observed=True, sort=True):
            for method, score_column in method_columns.items():
                metrics = metric_dict(
                    group["label_nr"].to_numpy(dtype=int),
                    group[score_column].to_numpy(dtype=float),
                    threshold,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        grouping: group_value,
                        "method": method,
                        "n_trials": int(len(group)),
                        "n_participants": int(group["participant_id"].nunique()),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def long_prediction_table(
    datasets: dict[str, pd.DataFrame],
    method_columns: dict[str, str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for dataset, frame in datasets.items():
        threshold = float(frame["frozen_threshold"].iloc[0])
        base_columns = [
            "dataset",
            "participant_id",
            "calibration_session_id",
            "external_trial_id",
            "trial_index",
            "label_nr",
            "n_prior_available",
            "has_prior",
        ]
        for column in ("set_name", "source_file", "task"):
            if column in frame.columns:
                base_columns.append(column)

        for method, score_column in method_columns.items():
            part = frame[base_columns].copy()
            part["method"] = method
            part["score"] = frame[score_column].to_numpy(dtype=float)
            part["frozen_threshold"] = threshold
            part["prediction"] = (part["score"] >= threshold).astype(int)
            if method.startswith("causal_cap_"):
                part["prior_center"] = frame[
                    f"{method}_prior_center"
                ].to_numpy(dtype=float)
                part["history_used"] = frame[
                    f"{method}_history_used"
                ].to_numpy(dtype=int)
                part["first_trial_fallback_to_raw"] = ~frame[
                    "has_prior"
                ].to_numpy(dtype=bool)
            else:
                part["prior_center"] = np.nan
                part["history_used"] = 0
                part["first_trial_fallback_to_raw"] = False
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def bootstrap_indices_by_participant(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    participants = np.asarray(
        sorted(frame["participant_id"].astype(str).unique()), dtype=object
    )
    indices_by_participant = {
        participant: np.flatnonzero(
            frame["participant_id"].astype(str).to_numpy() == participant
        )
        for participant in participants
    }
    sampled = rng.choice(participants, size=len(participants), replace=True)
    return np.concatenate([indices_by_participant[p] for p in sampled])


def bootstrap_analysis(
    datasets: dict[str, pd.DataFrame],
    causal_methods: list[str],
    method_columns: dict[str, str],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100 for the paper analysis.")
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, Any]] = []

    for dataset, frame in datasets.items():
        threshold = float(frame["frozen_threshold"].iloc[0])
        loo_reference = "loo_same_reference"
        for replicate in range(n_bootstrap):
            sampled_indices = bootstrap_indices_by_participant(frame, rng)
            boot = frame.iloc[sampled_indices].reset_index(drop=True)

            for scope in SCOPES:
                mask = evaluation_mask(boot, scope)
                scoped = boot.loc[mask]
                if len(scoped) == 0:
                    continue

                cached: dict[str, dict[str, float]] = {}
                for method in ["raw_frozen", loo_reference, *causal_methods]:
                    cached[method] = metric_dict(
                        scoped["label_nr"].to_numpy(dtype=int),
                        scoped[method_columns[method]].to_numpy(dtype=float),
                        threshold,
                    )

                for causal_method in causal_methods:
                    for reference in ("raw_frozen", loo_reference):
                        for metric in METRICS:
                            comparison_value = cached[causal_method][metric]
                            reference_value = cached[reference][metric]
                            if not (
                                math.isfinite(comparison_value)
                                and math.isfinite(reference_value)
                            ):
                                delta = float("nan")
                            else:
                                delta = float(comparison_value - reference_value)
                            replicate_rows.append(
                                {
                                    "dataset": dataset,
                                    "replicate": replicate,
                                    "scope": scope,
                                    "comparison": causal_method,
                                    "reference": reference,
                                    "metric": metric,
                                    "delta": delta,
                                }
                            )

    replicates = pd.DataFrame(replicate_rows)
    summary_rows: list[dict[str, Any]] = []
    grouping = ["dataset", "scope", "comparison", "reference", "metric"]
    for keys, group in replicates.groupby(grouping, observed=True, sort=True):
        delta = group["delta"].to_numpy(dtype=float)
        delta = delta[np.isfinite(delta)]
        row = dict(zip(grouping, keys))
        if len(delta) == 0:
            row.update(
                {
                    "delta_mean": float("nan"),
                    "delta_median": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "probability_delta_gt_0": float("nan"),
                    "n_bootstrap": 0,
                }
            )
        else:
            row.update(
                {
                    "delta_mean": float(np.mean(delta)),
                    "delta_median": float(np.median(delta)),
                    "ci_low": float(np.quantile(delta, 0.025)),
                    "ci_high": float(np.quantile(delta, 0.975)),
                    "probability_delta_gt_0": float(np.mean(delta > 0)),
                    "n_bootstrap": int(len(delta)),
                }
            )
        summary_rows.append(row)
    return replicates, pd.DataFrame(summary_rows)


def source_reference_audit(
    english_reference: float,
    turkish_reference: float,
    tolerance: float,
) -> dict[str, Any]:
    absolute_difference = abs(english_reference - turkish_reference)
    return {
        "english_full_final_reference": english_reference,
        "turkish_full_final_reference": turkish_reference,
        "absolute_difference": absolute_difference,
        "tolerance": tolerance,
        "match_within_tolerance": bool(absolute_difference <= tolerance),
    }


def session_summary(audit: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_sessions": int(len(audit)),
        "n_participants": int(audit["participant_id"].nunique()),
        "n_complete_10_trial_sessions": int(audit["complete_10_trial_record"].sum()),
        "n_duplicate_trial_index_sessions": int(audit["duplicate_trial_index"].sum()),
        "n_participants_reused_across_sessions": int(
            audit.loc[
                audit["participant_id_reused_across_sessions"], "participant_id"
            ].nunique()
        ),
        "session_size_distribution": {
            str(int(key)): int(value)
            for key, value in audit["n_trials"].value_counts().sort_index().items()
        },
        "top_task_orders": {
            str(key): int(value)
            for key, value in audit["task_order"].value_counts().head(10).items()
        },
    }


def format_metric_row(row: pd.Series) -> str:
    return (
        f"Macro-F1={row['macro_f1']:.4f}, BA={row['balanced_accuracy']:.4f}, "
        f"ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc_nr']:.4f}"
    )


def build_summary(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    session_audits: dict[str, pd.DataFrame],
    causal_methods: list[str],
    reference_audit: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Causal / Few-shot Participant Calibration Sensitivity")
    lines.append("")
    lines.append(
        "**Status: post-hoc deployment sensitivity. This does not change the locked Turkish confirmatory result.**"
    )
    lines.append("")
    lines.append(
        "Calibration uses only strictly previous raw frozen scores within each participant record/session. "
        "The first trial falls back to the raw frozen score for `all_trials`; it is excluded from `has_prior_only`."
    )
    lines.append("")
    lines.append("## Source-reference audit")
    lines.append("")
    lines.append(
        f"English r_full={reference_audit['english_full_final_reference']:.12g}; "
        f"Turkish r_full={reference_audit['turkish_full_final_reference']:.12g}; "
        f"|difference|={reference_audit['absolute_difference']:.3e}; "
        f"match={reference_audit['match_within_tolerance']}."
    )
    lines.append("")

    lines.append("## Session/order audit")
    lines.append("")
    for dataset, audit in session_audits.items():
        summary = session_summary(audit)
        lines.append(
            f"- **{dataset}**: {summary['n_participants']} participant IDs, "
            f"{summary['n_sessions']} calibration sessions, "
            f"{summary['n_complete_10_trial_sessions']} complete 10-trial sessions, "
            f"{summary['n_participants_reused_across_sessions']} participant IDs reused across sessions."
        )
    lines.append("")

    for dataset in ("English", "Turkish"):
        lines.append(f"## {dataset}: pooled results")
        lines.append("")
        subset = metrics[
            (metrics["dataset"] == dataset) & (metrics["scope"] == "all_trials")
        ].set_index("method")
        for method in ["raw_frozen", "loo_paper", "loo_same_reference", *causal_methods]:
            if method in subset.index:
                lines.append(f"- `{method}`: {format_metric_row(subset.loc[method])}")
        lines.append("")

        lines.append("### Same-reference bootstrap contrasts (all trials)")
        lines.append("")
        boot = bootstrap[
            (bootstrap["dataset"] == dataset)
            & (bootstrap["scope"] == "all_trials")
            & (bootstrap["reference"] == "loo_same_reference")
            & (bootstrap["metric"].isin(["macro_f1", "roc_auc"]))
        ]
        if boot.empty:
            lines.append("No bootstrap contrasts available.")
        else:
            for method in causal_methods:
                method_rows = boot[boot["comparison"] == method]
                if method_rows.empty:
                    continue
                parts = []
                for metric_name in ("macro_f1", "roc_auc"):
                    row = method_rows[method_rows["metric"] == metric_name]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    parts.append(
                        f"{metric_name} Δ={r['delta_mean']:+.4f} "
                        f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                    )
                lines.append(f"- `{method}` vs `loo_same_reference`: " + "; ".join(parts))
        lines.append("")

    lines.append("## Reporting boundary")
    lines.append("")
    lines.append(
        "Do not call the Turkish results confirmatory. Report them as a post-hoc online/deployment sensitivity analysis "
        "performed after the locked Turkish labels had already been inspected. Because WebQAmGaze trial order is task-blocked, "
        "interpret causal past-only performance as sensitivity to the observed deployment order, not as a task-order-invariant estimate."
    )
    lines.append("")
    return "\n".join(lines)


def write_hash_manifest(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file() or path.name == "file_hash_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "file_hash_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> int:
    args = parse_args()
    history_caps = parse_history_caps(args.history_caps)
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    english_reference, factor_manifest = load_factor_reference(args.factor_manifest)
    english = load_english(
        args.english_predictions,
        args.english_manifest,
        english_reference,
    )
    turkish, turkish_reference, turkish_release = load_turkish(
        args.turkish_snapshot,
        args.turkish_manifest,
    )

    reference_audit = source_reference_audit(
        english_reference,
        turkish_reference,
        args.reference_tolerance,
    )
    if not reference_audit["match_within_tolerance"]:
        raise RuntimeError(
            "Frozen source reference differs between English factor manifest and "
            "Turkish locked release. Refusing to pool the sensitivity analysis. "
            f"Audit={reference_audit}"
        )

    english, causal_methods = apply_causal_scores(english, history_caps)
    turkish, causal_methods_tr = apply_causal_scores(turkish, history_caps)
    if causal_methods != causal_methods_tr:
        raise RuntimeError("Causal method lists unexpectedly differ by dataset.")

    datasets = {"English": english, "Turkish": turkish}
    score_columns = method_score_columns(causal_methods)

    session_audits = {
        dataset: session_audit(frame)
        for dataset, frame in datasets.items()
    }
    session_audit_all = pd.concat(session_audits.values(), ignore_index=True)

    if session_audit_all["duplicate_trial_index"].any():
        bad = session_audit_all[session_audit_all["duplicate_trial_index"]]
        raise RuntimeError(
            "Duplicate trial_index values exist within a calibration session; "
            "past-only order would be ambiguous. Examples:\n"
            + bad.head(20).to_string(index=False)
        )

    metrics = pooled_metrics_table(datasets, score_columns)
    position_metrics = grouped_metrics_table(
        datasets, score_columns, "trial_index"
    )
    task_metrics = grouped_metrics_table(datasets, score_columns, "task")
    predictions_long = long_prediction_table(datasets, score_columns)
    bootstrap_replicates, bootstrap_deltas = bootstrap_analysis(
        datasets=datasets,
        causal_methods=causal_methods,
        method_columns=score_columns,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    predictions_long.to_csv(
        output_dir / "causal_calibration_trial_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        output_dir / "causal_calibration_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    position_metrics.to_csv(
        output_dir / "causal_calibration_trial_position_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    task_metrics.to_csv(
        output_dir / "causal_calibration_task_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    session_audit_all.to_csv(
        output_dir / "causal_calibration_session_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_replicates.to_csv(
        output_dir / "causal_calibration_bootstrap_replicates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_deltas.to_csv(
        output_dir / "causal_calibration_bootstrap_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary(
        metrics,
        bootstrap_deltas,
        session_audits,
        causal_methods,
        reference_audit,
    )
    (output_dir / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")

    manifest = {
        "artifact_version": "iui_p5_causal_participant_calibration_v1",
        "analysis_status": "post_hoc_deployment_sensitivity",
        "changes_locked_turkish_confirmation": False,
        "target_labels_used_for_calibration": False,
        "target_labels_used_for_history_selection": False,
        "history_order_variable": "trial_index",
        "history_policy": (
            "strictly previous raw frozen scores within calibration_session_id; "
            "first trial falls back to raw frozen score for all_trials scope"
        ),
        "session_key_policy": (
            "participant_id + set_name + source_file when available; resets history "
            "at each participant record to avoid inventing chronology across reused IDs"
        ),
        "history_caps": [cap_name(cap) for cap in history_caps],
        "source_reference_audit": reference_audit,
        "bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.seed,
            "cluster_unit": "participant_id",
            "paired": True,
            "comparisons": [
                "each causal method minus raw_frozen",
                "each causal method minus loo_same_reference",
            ],
        },
        "blocked_order_caveat": (
            "WebQAmGaze trial order is task-blocked in the audited design; report "
            "trial-position/task diagnostics and do not interpret as task-order-invariant."
        ),
        "inputs": {
            "english_predictions": str(require_file(args.english_predictions)),
            "english_manifest": str(require_file(args.english_manifest)),
            "factor_manifest": str(require_file(args.factor_manifest)),
            "turkish_snapshot": str(require_dir(args.turkish_snapshot)),
            "turkish_manifest": str(require_file(args.turkish_manifest)),
        },
        "input_hashes": {
            "english_predictions_sha256": sha256(require_file(args.english_predictions)),
            "english_manifest_sha256": sha256(require_file(args.english_manifest)),
            "factor_manifest_sha256": sha256(require_file(args.factor_manifest)),
            "turkish_predictions_sha256": sha256(
                require_file(
                    args.turkish_snapshot
                    / "predictions"
                    / "unlabeled_frozen_scores.csv"
                )
            ),
            "turkish_labels_sha256": sha256(
                require_file(args.turkish_snapshot / "sealed" / "labels_sealed.csv")
            ),
            "turkish_release_sha256": sha256(
                require_file(args.turkish_snapshot / "release_manifest.json")
            ),
            "turkish_manifest_sha256": sha256(require_file(args.turkish_manifest)),
        },
        "dataset_audits": {
            dataset: session_summary(audit)
            for dataset, audit in session_audits.items()
        },
        "frozen_factor_manifest_primary_comparison": factor_manifest.get(
            "primary_comparison"
        ),
        "turkish_locked_confirmation_classification": turkish_release.get(
            "confirmation_classification", "stored_in_separate_evaluation_artifact"
        ),
    }
    (output_dir / "causal_calibration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_hash_manifest(output_dir)

    print("=== CAUSAL / FEW-SHOT PARTICIPANT CALIBRATION SENSITIVITY ===")
    print("Status: POST-HOC deployment sensitivity")
    print(f"Output: {output_dir}")
    print(f"History caps: {[cap_name(cap) for cap in history_caps]}")
    print(
        "Source reference match: "
        f"{reference_audit['match_within_tolerance']} "
        f"(|diff|={reference_audit['absolute_difference']:.3e})"
    )
    for dataset, frame in datasets.items():
        audit = session_summary(session_audits[dataset])
        print(
            f"{dataset}: trials={len(frame)}, participants={frame['participant_id'].nunique()}, "
            f"sessions={audit['n_sessions']}, complete10={audit['n_complete_10_trial_sessions']}, "
            f"reused_IDs={audit['n_participants_reused_across_sessions']}"
        )

    print("\n=== POOLED ALL-TRIAL METRICS ===")
    display = metrics[metrics["scope"].eq("all_trials")][
        [
            "dataset",
            "method",
            "n_trials",
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
        ]
    ]
    print(display.to_string(index=False))
    print("\nWritten:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
