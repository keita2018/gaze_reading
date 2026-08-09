#!/usr/bin/env python
r"""
Post-hoc fully future-free selective-policy sensitivity for WebQAmGaze.

Purpose
-------
The manuscript's participant-balanced selective analysis retains the highest-
confidence requested fraction within each participant. That is useful as an
offline diagnostic, but it retrospectively ranks a participant's full set of
trials and can therefore depend on future trials.

This script tests a deployable alternative:

1. Use the strictly past-only participant-calibrated score produced by
   run_causal_participant_calibration_sensitivity.py (default: causal_cap_all).
2. Define confidence for the current trial only as

       confidence_t = abs(score_t - frozen_threshold)

3. On ENGLISH DEVELOPMENT ONLY, select one absolute confidence threshold tau
   to approximately realize a pre-specified coverage (principal: 50%).
   Labels are NOT used to choose tau.
4. Freeze tau.
5. Apply exactly the same tau to Turkish without re-ranking, re-tuning, or
   enforcing a target coverage there.
6. Report realized coverage, selective metrics, participant representation,
   and the post-hoc interaction-cost sensitivity.

Important interpretation boundary
---------------------------------
This is a POST-HOC DEPLOYMENT SENSITIVITY ANALYSIS. It does not alter the locked
Turkish confirmatory result. Turkish labels have already been inspected, so no
result from this script may be described as preregistered, locked, or confirmatory.

The policy is "future-free" with respect to participant history and selection:
- score_t uses only strictly previous raw scores within the calibration session;
- the action/abstention decision at t uses only score_t, the frozen decision
  threshold, and an English-development confidence threshold tau;
- no future trial from the current participant is needed;
- Turkish target labels and Turkish target confidence quantiles are not used
  to select tau or to force a coverage.

The WebQAmGaze observed order is task-blocked, so this analysis remains a
sensitivity to that observed deployment order, not a task-order-invariant
prospective user study.

Expected input
--------------
causal_calibration_trial_predictions.csv created by
run_causal_participant_calibration_sensitivity.py.

Example
-------
python -u src/analysis/run_future_free_selective_policy_sensitivity.py ^
  --causal-predictions outputs/iui_p5_causal_participant_calibration_v1/causal_calibration_trial_predictions.csv ^
  --output-dir outputs/iui_p5_future_free_selective_policy_v1 ^
  --method causal_cap_all ^
  --english-target-coverages 0.8,0.5,0.2 ^
  --principal-coverage 0.5 ^
  --lambda-error-over-miss 1.5 ^
  --bootstrap 5000 ^
  --seed 20260810
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fully future-free fixed-margin selective-policy sensitivity."
    )
    p.add_argument("--causal-predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", default="causal_cap_all")
    p.add_argument(
        "--english-target-coverages",
        default="0.8,0.5,0.2",
        help="English-development coverages used only to select fixed confidence thresholds.",
    )
    p.add_argument("--principal-coverage", type=float, default=0.5)
    p.add_argument("--lambda-error-over-miss", type=float, default=1.5)
    p.add_argument(
        "--key-lambdas",
        default="0.5,1,1.5,2,3,5,10",
        help="Post-hoc cost-ratio grid for observed fixed-threshold policies.",
    )
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_float_list(text: str) -> list[float]:
    out = sorted(
        {float(x.strip()) for x in str(text).split(",") if x.strip()},
        reverse=True,
    )
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def validate_coverages(values: list[float]) -> None:
    bad = [x for x in values if not (0.0 < x < 1.0)]
    if bad:
        raise ValueError(f"Development target coverages must be in (0,1): {bad}")


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else float("nan")


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return (
        float(average_precision_score(y, score))
        if np.unique(y).size == 2
        else float("nan")
    )


def metrics(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc_nr": float("nan"),
            "predicted_nr_rate": float("nan"),
            "true_nr_rate": float("nan"),
        }
    y = df["label_nr"].to_numpy(int)
    pred = df["prediction"].to_numpy(int)
    score = df["score"].to_numpy(float)
    both_classes = np.unique(y).size == 2
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, pred))
            if both_classes
            else float("nan")
        ),
        "macro_f1": (
            float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0))
            if both_classes
            else float("nan")
        ),
        "roc_auc": safe_auc(y, score),
        "pr_auc_nr": safe_ap(y, score),
        "predicted_nr_rate": float(np.mean(pred == 1)),
        "true_nr_rate": float(np.mean(y == 1)),
    }


def load_method(path: Path, method: str) -> tuple[pd.DataFrame, float]:
    path = require_file(path)
    df = pd.read_csv(path, low_memory=False)

    required = {
        "dataset",
        "participant_id",
        "external_trial_id",
        "label_nr",
        "method",
        "score",
        "frozen_threshold",
        "prediction",
        "trial_index",
        "calibration_session_id",
        "has_prior",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Causal prediction table lacks columns: {missing}")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["external_trial_id"] = df["external_trial_id"].astype(str).str.strip()
    df["calibration_session_id"] = df["calibration_session_id"].astype(str).str.strip()
    df["method"] = df["method"].astype(str).str.strip()
    df["label_nr"] = pd.to_numeric(df["label_nr"], errors="raise").astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="raise").astype(float)
    df["frozen_threshold"] = pd.to_numeric(
        df["frozen_threshold"], errors="raise"
    ).astype(float)
    df["prediction"] = pd.to_numeric(df["prediction"], errors="raise").astype(int)
    df["trial_index"] = pd.to_numeric(df["trial_index"], errors="raise").astype(int)

    keep = df[df["method"].eq(method)].copy()
    if keep.empty:
        available = sorted(df["method"].dropna().unique().tolist())
        raise ValueError(f"Method {method!r} not found. Available={available}")

    expected_datasets = {"English", "Turkish"}
    if set(keep["dataset"].unique()) != expected_datasets:
        raise ValueError(
            f"Expected exactly English/Turkish datasets, found {sorted(keep['dataset'].unique())}"
        )

    if not keep["label_nr"].isin([0, 1]).all():
        raise ValueError("label_nr must be binary.")
    if not keep["prediction"].isin([0, 1]).all():
        raise ValueError("prediction must be binary.")
    if not np.isfinite(keep[["score", "frozen_threshold"]].to_numpy(float)).all():
        raise ValueError("Non-finite score/threshold.")
    if keep.duplicated(["dataset", "external_trial_id"]).any():
        raise ValueError("Duplicate dataset/external_trial_id rows for selected method.")

    thresholds = np.unique(np.round(keep["frozen_threshold"].to_numpy(float), 12))
    if len(thresholds) != 1:
        raise ValueError(f"Expected one frozen decision threshold, got {thresholds.tolist()}")
    frozen_threshold = float(keep["frozen_threshold"].iloc[0])

    recomputed = (keep["score"].to_numpy(float) >= frozen_threshold).astype(int)
    if not np.array_equal(recomputed, keep["prediction"].to_numpy(int)):
        raise ValueError("Stored predictions disagree with score >= frozen_threshold.")

    keep["confidence"] = np.abs(keep["score"] - frozen_threshold)
    keep["correct"] = (keep["prediction"] == keep["label_nr"]).astype(np.int8)
    return keep.reset_index(drop=True), frozen_threshold


def choose_fixed_tau(confidence: np.ndarray, target_coverage: float) -> dict[str, float]:
    """
    Choose tau from ENGLISH DEVELOPMENT confidence only.
    Retain iff confidence >= tau.

    Among observed confidence values, choose the threshold whose realized coverage
    is closest to the requested target. If equally close, choose the larger tau
    (more conservative / lower or equal realized coverage).
    """
    conf = np.asarray(confidence, float)
    if len(conf) == 0 or not np.isfinite(conf).all():
        raise ValueError("English confidence values are empty/non-finite.")
    candidates = np.unique(conf)
    best = None
    n = len(conf)
    for tau in candidates:
        realized = float(np.sum(conf >= tau) / n)
        key = (abs(realized - target_coverage), -float(tau))
        row = (key, float(tau), realized, int(np.sum(conf >= tau)))
        if best is None or row[0] < best[0]:
            best = row
    assert best is not None
    return {
        "target_coverage": float(target_coverage),
        "tau": best[1],
        "english_realized_coverage": best[2],
        "english_n_retained": best[3],
        "english_n_total": n,
    }


def policy_counts(full: pd.DataFrame, retained: pd.DataFrame) -> dict[str, Any]:
    fe = int((full["correct"] == 0).sum())
    fc = int((full["correct"] == 1).sum())
    re = int((retained["correct"] == 0).sum())
    rc = int((retained["correct"] == 1).sum())
    we = fe - re
    wc = fc - rc
    be = wc / we if we > 0 else (float("nan") if wc == 0 else float("inf"))
    return {
        "n_total": int(len(full)),
        "n_retained": int(len(retained)),
        "n_withheld": int(len(full) - len(retained)),
        "realized_coverage": float(len(retained) / len(full)),
        "full_errors": fe,
        "full_correct": fc,
        "retained_errors": re,
        "retained_correct": rc,
        "withheld_errors": we,
        "withheld_correct": wc,
        "withheld_error_fraction": float(we / fe) if fe else float("nan"),
        "withheld_correct_fraction": float(wc / fc) if fc else float("nan"),
        "break_even_lambda_error_over_miss": float(be),
    }


def participant_representation(full: pd.DataFrame, retained: pd.DataFrame) -> dict[str, Any]:
    total_counts = full.groupby("participant_id", observed=True).size().rename("n_total")
    retained_counts = (
        retained.groupby("participant_id", observed=True).size().rename("n_retained")
    )
    table = pd.concat([total_counts, retained_counts], axis=1).fillna(0)
    table["n_retained"] = table["n_retained"].astype(int)
    table["coverage"] = table["n_retained"] / table["n_total"]
    return {
        "n_participants_total": int(len(table)),
        "n_participants_retained_at_least_one": int((table["n_retained"] > 0).sum()),
        "n_participants_zero_retained": int((table["n_retained"] == 0).sum()),
        "participant_coverage_min": float(table["coverage"].min()),
        "participant_coverage_q25": float(table["coverage"].quantile(0.25)),
        "participant_coverage_median": float(table["coverage"].median()),
        "participant_coverage_q75": float(table["coverage"].quantile(0.75)),
        "participant_coverage_max": float(table["coverage"].max()),
    }


def observed_policy_tables(
    data: pd.DataFrame,
    policies: list[dict[str, float]],
    lambdas: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    curve_rows = []
    participant_rows = []
    for policy in policies:
        tau = float(policy["tau"])
        target_cov = float(policy["target_coverage"])
        for dataset, full in data.groupby("dataset", observed=True, sort=True):
            retained = full[full["confidence"] >= tau].copy()
            c = policy_counts(full, retained)
            rep = participant_representation(full, retained)
            rows.append(
                {
                    "dataset": dataset,
                    "english_target_coverage": target_cov,
                    "fixed_tau_from_english": tau,
                    **c,
                    **metrics(retained),
                    **rep,
                }
            )

            by_participant = (
                full.groupby("participant_id", observed=True)
                .size()
                .rename("n_total")
                .to_frame()
                .join(
                    retained.groupby("participant_id", observed=True)
                    .size()
                    .rename("n_retained"),
                    how="left",
                )
                .fillna(0)
                .reset_index()
            )
            by_participant["n_retained"] = by_participant["n_retained"].astype(int)
            by_participant["realized_coverage"] = (
                by_participant["n_retained"] / by_participant["n_total"]
            )
            by_participant["dataset"] = dataset
            by_participant["english_target_coverage"] = target_cov
            by_participant["fixed_tau_from_english"] = tau
            participant_rows.append(by_participant)

            for lam in lambdas:
                full_cost = float(lam * c["full_errors"])
                selective_cost = float(lam * c["retained_errors"] + c["withheld_correct"])
                saving = full_cost - selective_cost
                curve_rows.append(
                    {
                        "dataset": dataset,
                        "english_target_coverage": target_cov,
                        "fixed_tau_from_english": tau,
                        "lambda_error_over_miss": float(lam),
                        "full_cost": full_cost,
                        "selective_cost": selective_cost,
                        "absolute_cost_saving": saving,
                        "relative_cost_saving": (
                            float(saving / full_cost) if full_cost > 0 else float("nan")
                        ),
                        "policy_cheaper_than_full": bool(saving > 0),
                        "break_even_lambda_error_over_miss": c[
                            "break_even_lambda_error_over_miss"
                        ],
                    }
                )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(curve_rows),
        pd.concat(participant_rows, ignore_index=True),
    )


def subgroup_retention_table(
    data: pd.DataFrame,
    policies: list[dict[str, float]],
    group_col: str,
) -> pd.DataFrame:
    if group_col not in data.columns:
        return pd.DataFrame()
    rows = []
    for policy in policies:
        tau = float(policy["tau"])
        target_cov = float(policy["target_coverage"])
        for (dataset, group_value), g in data.groupby(
            ["dataset", group_col], observed=True, sort=True
        ):
            retained = g[g["confidence"] >= tau]
            rows.append(
                {
                    "dataset": dataset,
                    "english_target_coverage": target_cov,
                    "fixed_tau_from_english": tau,
                    group_col: group_value,
                    "n_total": int(len(g)),
                    "n_retained": int(len(retained)),
                    "realized_coverage": float(len(retained) / len(g)),
                    "n_participants": int(g["participant_id"].nunique()),
                    "mean_confidence_all": float(g["confidence"].mean()),
                    "median_confidence_all": float(g["confidence"].median()),
                    "mean_confidence_retained": (
                        float(retained["confidence"].mean())
                        if len(retained)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


class ParticipantBlock:
    def __init__(self, g: pd.DataFrame):
        self.confidence = g["confidence"].to_numpy(float)
        self.correct = g["correct"].to_numpy(np.int8)
        self.label = g["label_nr"].to_numpy(np.int8)
        self.prediction = g["prediction"].to_numpy(np.int8)
        self.score = g["score"].to_numpy(float)


def make_blocks(df: pd.DataFrame) -> list[ParticipantBlock]:
    return [
        ParticipantBlock(g)
        for _, g in df.groupby("participant_id", observed=True, sort=True)
    ]


def bootstrap_dataset(
    df: pd.DataFrame,
    policies: list[dict[str, float]],
    lam: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    dataset: str,
    progress_every: int,
) -> pd.DataFrame:
    blocks = make_blocks(df)
    n_blocks = len(blocks)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    for b in range(n_bootstrap):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        sample = [blocks[int(i)] for i in chosen]

        full_correct = np.concatenate([x.correct for x in sample])
        full_errors = int(np.sum(full_correct == 0))
        full_correct_n = int(np.sum(full_correct == 1))

        for policy in policies:
            tau = float(policy["tau"])
            target_cov = float(policy["target_coverage"])

            retained_correct_parts = []
            retained_label_parts = []
            retained_prediction_parts = []
            retained_score_parts = []
            retained_n = 0

            for block in sample:
                mask = block.confidence >= tau
                retained_n += int(mask.sum())
                if mask.any():
                    retained_correct_parts.append(block.correct[mask])
                    retained_label_parts.append(block.label[mask])
                    retained_prediction_parts.append(block.prediction[mask])
                    retained_score_parts.append(block.score[mask])

            n_total = len(full_correct)
            if retained_correct_parts:
                rc_vec = np.concatenate(retained_correct_parts)
                y = np.concatenate(retained_label_parts)
                pred = np.concatenate(retained_prediction_parts)
                score = np.concatenate(retained_score_parts)
            else:
                rc_vec = np.empty(0, dtype=np.int8)
                y = np.empty(0, dtype=np.int8)
                pred = np.empty(0, dtype=np.int8)
                score = np.empty(0, dtype=float)

            retained_errors = int(np.sum(rc_vec == 0))
            retained_correct = int(np.sum(rc_vec == 1))
            withheld_errors = full_errors - retained_errors
            withheld_correct = full_correct_n - retained_correct

            full_cost = float(lam * full_errors)
            selective_cost = float(lam * retained_errors + withheld_correct)
            saving = full_cost - selective_cost
            break_even = (
                withheld_correct / withheld_errors
                if withheld_errors > 0
                else (float("nan") if withheld_correct == 0 else float("inf"))
            )

            if len(y):
                acc = float(np.mean(y == pred))
                both_classes = np.unique(y).size == 2
                ba = (
                    float(balanced_accuracy_score(y, pred))
                    if both_classes
                    else float("nan")
                )
                mf1 = (
                    float(
                        f1_score(
                            y,
                            pred,
                            average="macro",
                            labels=[0, 1],
                            zero_division=0,
                        )
                    )
                    if both_classes
                    else float("nan")
                )
                auc = safe_auc(y, score)
            else:
                acc = ba = mf1 = auc = float("nan")

            rows.append(
                {
                    "dataset": dataset,
                    "replicate": b,
                    "english_target_coverage": target_cov,
                    "fixed_tau_from_english": tau,
                    "realized_coverage": float(retained_n / n_total),
                    "retained_accuracy": acc,
                    "retained_balanced_accuracy": ba,
                    "retained_macro_f1": mf1,
                    "retained_roc_auc": auc,
                    "full_errors": full_errors,
                    "retained_errors": retained_errors,
                    "withheld_errors": withheld_errors,
                    "withheld_correct": withheld_correct,
                    "lambda_error_over_miss": float(lam),
                    "relative_cost_saving": (
                        float(saving / full_cost) if full_cost > 0 else float("nan")
                    ),
                    "policy_cheaper_than_full": bool(saving > 0),
                    "break_even_lambda_error_over_miss": float(break_even),
                }
            )

        if progress_every and (
            (b + 1) % progress_every == 0 or b + 1 == n_bootstrap
        ):
            elapsed = time.perf_counter() - start
            rate = (b + 1) / elapsed if elapsed > 0 else float("nan")
            eta = (
                (n_bootstrap - b - 1) / rate
                if math.isfinite(rate) and rate > 0
                else float("nan")
            )
            print(
                f"[bootstrap] {dataset}: {b+1}/{n_bootstrap} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    return pd.DataFrame(rows)


def summarize_bootstrap(reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "english_target_coverage", "fixed_tau_from_english"]
    for key, g in reps.groupby(keys, observed=True, sort=True):
        def q(column: str, prob: float) -> float:
            a = pd.to_numeric(g[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna().to_numpy(float)
            return float(np.quantile(a, prob)) if len(a) else float("nan")

        be = pd.to_numeric(
            g["break_even_lambda_error_over_miss"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)

        rows.append(
            {
                "dataset": key[0],
                "english_target_coverage": float(key[1]),
                "fixed_tau_from_english": float(key[2]),
                "realized_coverage_mean": float(g["realized_coverage"].mean()),
                "realized_coverage_ci_low": q("realized_coverage", 0.025),
                "realized_coverage_ci_high": q("realized_coverage", 0.975),
                "relative_cost_saving_mean": float(g["relative_cost_saving"].mean()),
                "relative_cost_saving_ci_low": q("relative_cost_saving", 0.025),
                "relative_cost_saving_ci_high": q("relative_cost_saving", 0.975),
                "bootstrap_probability_policy_cheaper_than_full": float(
                    g["policy_cheaper_than_full"].mean()
                ),
                "break_even_lambda_median": (
                    float(np.median(be)) if len(be) else float("nan")
                ),
                "break_even_lambda_ci_low": (
                    float(np.quantile(be, 0.025)) if len(be) else float("nan")
                ),
                "break_even_lambda_ci_high": (
                    float(np.quantile(be, 0.975)) if len(be) else float("nan")
                ),
                "retained_macro_f1_mean": float(g["retained_macro_f1"].mean()),
                "retained_macro_f1_ci_low": q("retained_macro_f1", 0.025),
                "retained_macro_f1_ci_high": q("retained_macro_f1", 0.975),
                "retained_roc_auc_mean": float(g["retained_roc_auc"].mean()),
                "retained_roc_auc_ci_low": q("retained_roc_auc", 0.025),
                "retained_roc_auc_ci_high": q("retained_roc_auc", 0.975),
                "n_bootstrap": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def file_hash_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "file_hash_manifest.csv":
            continue
        rows.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    method: str,
    frozen_threshold: float,
    policies: list[dict[str, float]],
    observed: pd.DataFrame,
    cost_curves: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    principal_coverage: float,
    lam: float,
) -> str:
    lines = [
        "# Fully Future-Free Selective-Policy Sensitivity",
        "",
        "**Status: post-hoc deployment sensitivity. This does not change the locked Turkish confirmatory result.**",
        "",
        f"- Past-only score method: `{method}`",
        f"- Frozen class-decision threshold: {frozen_threshold:.12g}",
        "- Confidence: `abs(score - frozen_threshold)`",
        "- Selection threshold(s): chosen from English development confidence only; no labels used.",
        "- Turkish policy: exact English-fixed threshold, no Turkish re-ranking and no forced Turkish coverage.",
        f"- Cost ratio shown below: lambda = C_error / C_miss = {lam:g}",
        "",
        "## Frozen English-development confidence thresholds",
        "",
    ]
    for p in sorted(policies, key=lambda x: x["target_coverage"], reverse=True):
        lines.append(
            f"- target {p['target_coverage']:.0%}: tau={p['tau']:.8g}; "
            f"English realized coverage={p['english_realized_coverage']:.4f}"
        )

    lines += ["", f"## Principal policy: English target coverage {principal_coverage:.0%}", ""]
    obs = observed[
        np.isclose(observed["english_target_coverage"], principal_coverage)
    ].copy()
    bsum = bootstrap_summary[
        np.isclose(bootstrap_summary["english_target_coverage"], principal_coverage)
    ].copy()

    for dataset in ["English", "Turkish"]:
        o = obs[obs["dataset"].eq(dataset)]
        b = bsum[bsum["dataset"].eq(dataset)]
        if len(o) != 1 or len(b) != 1:
            continue
        o = o.iloc[0]
        b = b.iloc[0]
        c = cost_curves[
            cost_curves["dataset"].eq(dataset)
            & np.isclose(
                cost_curves["english_target_coverage"],
                principal_coverage,
            )
            & np.isclose(
                cost_curves["lambda_error_over_miss"],
                lam,
            )
        ]
        observed_saving = (
            float(c.iloc[0]["relative_cost_saving"])
            if len(c) == 1
            else float("nan")
        )
        lines += [
            f"### {dataset}",
            "",
            f"- realized coverage={o.realized_coverage:.4f} "
            f"(bootstrap mean {b.realized_coverage_mean:.4f}, "
            f"95% CI [{b.realized_coverage_ci_low:.4f}, {b.realized_coverage_ci_high:.4f}])",
            f"- retained Macro-F1={o.macro_f1:.4f}; ROC-AUC={o.roc_auc:.4f}",
            f"- participants retained at least once={int(o.n_participants_retained_at_least_one)}/"
            f"{int(o.n_participants_total)}; zero-retained={int(o.n_participants_zero_retained)}",
            f"- withheld errors={int(o.withheld_errors)} "
            f"({o.withheld_error_fraction:.1%} of full-coverage errors)",
            f"- withheld correct predictions={int(o.withheld_correct)} "
            f"({o.withheld_correct_fraction:.1%} of full-coverage correct predictions)",
            f"- relative cost saving at lambda={lam:g}: "
            f"observed {observed_saving:.2%}; bootstrap mean "
            f"{b.relative_cost_saving_mean:.2%} "
            f"[{b.relative_cost_saving_ci_low:.2%}, {b.relative_cost_saving_ci_high:.2%}]",
            f"- bootstrap fraction policy cheaper than full coverage="
            f"{b.bootstrap_probability_policy_cheaper_than_full:.4f}",
            f"- break-even lambda*: median {b.break_even_lambda_median:.3f} "
            f"[{b.break_even_lambda_ci_low:.3f}, {b.break_even_lambda_ci_high:.3f}]",
            "",
        ]

    lines += [
        "## Reporting boundary",
        "",
        "This analysis is fully future-free with respect to the participant history and "
        "the trial-level retain/abstain decision, but it is still post-hoc and uses the "
        "observed task-blocked WebQAmGaze order. It is not a prospective user study and "
        "does not establish task-order-invariant real-time utility.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    a = parse_args()
    if a.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100.")
    if a.lambda_error_over_miss <= 0:
        raise ValueError("--lambda-error-over-miss must be > 0.")

    coverages = parse_float_list(a.english_target_coverages)
    validate_coverages(coverages)
    if not any(np.isclose(a.principal_coverage, c) for c in coverages):
        raise ValueError("--principal-coverage must be included in --english-target-coverages.")
    lambdas = sorted(set(parse_float_list(a.key_lambdas) + [a.lambda_error_over_miss]))

    output_dir = a.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading strictly past-only trial scores...", flush=True)
    data, frozen_threshold = load_method(a.causal_predictions, a.method)
    en = data[data["dataset"].eq("English")].copy()

    print("[2/8] Freezing absolute-margin threshold(s) from English development...", flush=True)
    policies = [
        choose_fixed_tau(en["confidence"].to_numpy(float), coverage)
        for coverage in coverages
    ]
    policy_table = pd.DataFrame(policies)

    print("[3/8] Applying the same fixed threshold(s) to English and Turkish...", flush=True)
    observed, cost_curves, participant_cov = observed_policy_tables(
        data, policies, lambdas
    )

    print("[4/8] Auditing retention by trial position and task...", flush=True)
    by_position = subgroup_retention_table(data, policies, "trial_index")
    by_task = subgroup_retention_table(data, policies, "task")

    print(f"[5/8] Participant bootstrap: {a.bootstrap} replicates per dataset...", flush=True)
    rep_parts = []
    for index, dataset in enumerate(["English", "Turkish"]):
        d = data[data["dataset"].eq(dataset)].copy()
        rep_parts.append(
            bootstrap_dataset(
                d,
                policies,
                a.lambda_error_over_miss,
                a.bootstrap,
                np.random.default_rng(a.seed + index * 100003),
                dataset,
                a.progress_every,
            )
        )
    reps = pd.concat(rep_parts, ignore_index=True)

    print("[6/8] Summarizing bootstrap uncertainty...", flush=True)
    bsum = summarize_bootstrap(reps)

    print("[7/8] Writing audit artifacts...", flush=True)
    policy_table.to_csv(
        output_dir / "frozen_english_confidence_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    observed.to_csv(
        output_dir / "future_free_policy_observed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cost_curves.to_csv(
        output_dir / "future_free_policy_cost_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    participant_cov.to_csv(
        output_dir / "future_free_policy_participant_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_position.to_csv(
        output_dir / "future_free_policy_trial_position_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not by_task.empty:
        by_task.to_csv(
            output_dir / "future_free_policy_task_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
    reps.to_csv(
        output_dir / "future_free_policy_bootstrap_replicates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bsum.to_csv(
        output_dir / "future_free_policy_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "artifact_version": "iui_p5_future_free_selective_policy_v1",
        "analysis_status": "post_hoc_deployment_sensitivity",
        "changes_locked_turkish_confirmation": False,
        "input": str(require_file(a.causal_predictions)),
        "input_sha256": sha256(require_file(a.causal_predictions)),
        "past_only_score_method": a.method,
        "frozen_class_decision_threshold": frozen_threshold,
        "confidence": "abs(score - frozen_threshold)",
        "threshold_training_dataset": "English development only",
        "threshold_training_uses_labels": False,
        "threshold_selection": (
            "Choose observed English confidence value giving realized coverage closest "
            "to each pre-specified target; ties choose larger tau (more conservative)."
        ),
        "english_target_coverages": coverages,
        "principal_english_target_coverage": a.principal_coverage,
        "turkish_policy": (
            "Apply exact English-fixed tau; no Turkish ranking, re-tuning, "
            "or forced target coverage."
        ),
        "target_labels_used_for_trial_selection": False,
        "future_trials_used_for_trial_selection": False,
        "future_trials_used_for_past_only_score": False,
        "cost_model": (
            "L = C_error * retained_errors + C_miss * withheld_correct; "
            "lambda = C_error/C_miss, normalize C_miss=1"
        ),
        "principal_lambda_error_over_miss": a.lambda_error_over_miss,
        "key_lambdas": lambdas,
        "bootstrap": {
            "replicates": a.bootstrap,
            "seed": a.seed,
            "cluster_unit": "participant_id",
            "tau_refit_inside_bootstrap": False,
            "reason_tau_fixed": (
                "The deployable policy freezes tau once from the full English "
                "development set before target evaluation."
            ),
        },
        "blocked_order_caveat": (
            "Observed WebQAmGaze order is task-blocked; this is a deployment-order "
            "sensitivity, not a task-order-invariant prospective estimate."
        ),
        "boundary": (
            "Offline policy-cost proxy only; no claim about actual comprehension, "
            "frustration, trust, delay cost, or intervention utility."
        ),
    }
    (output_dir / "future_free_policy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = build_summary(
        a.method,
        frozen_threshold,
        policies,
        observed,
        cost_curves,
        bsum,
        a.principal_coverage,
        a.lambda_error_over_miss,
    )
    (output_dir / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")

    hashes = file_hash_manifest(output_dir)
    hashes.to_csv(
        output_dir / "file_hash_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[8/8] Principal result...", flush=True)
    principal_obs = observed[
        np.isclose(observed["english_target_coverage"], a.principal_coverage)
    ]
    principal_boot = bsum[
        np.isclose(bsum["english_target_coverage"], a.principal_coverage)
    ]
    principal_cost = cost_curves[
        np.isclose(cost_curves["english_target_coverage"], a.principal_coverage)
        & np.isclose(
            cost_curves["lambda_error_over_miss"],
            a.lambda_error_over_miss,
        )
    ][
        [
            "dataset",
            "english_target_coverage",
            "relative_cost_saving",
            "full_cost",
            "selective_cost",
        ]
    ].rename(
        columns={
            "relative_cost_saving": "observed_relative_cost_saving",
            "full_cost": "observed_full_cost",
            "selective_cost": "observed_selective_cost",
        }
    )
    display = principal_obs[
        [
            "dataset",
            "english_target_coverage",
            "fixed_tau_from_english",
            "realized_coverage",
            "n_retained",
            "n_total",
            "n_participants_retained_at_least_one",
            "n_participants_zero_retained",
            "macro_f1",
            "roc_auc",
            "withheld_errors",
            "withheld_correct",
            "break_even_lambda_error_over_miss",
        ]
    ].merge(
        principal_cost,
        on=["dataset", "english_target_coverage"],
        how="left",
    ).merge(
        principal_boot[
            [
                "dataset",
                "english_target_coverage",
                "relative_cost_saving_mean",
                "relative_cost_saving_ci_low",
                "relative_cost_saving_ci_high",
                "bootstrap_probability_policy_cheaper_than_full",
                "break_even_lambda_median",
                "break_even_lambda_ci_low",
                "break_even_lambda_ci_high",
            ]
        ],
        on=["dataset", "english_target_coverage"],
        how="left",
    )

    with pd.option_context("display.max_columns", None, "display.width", 320):
        print("\n=== FULLY FUTURE-FREE PRINCIPAL FIXED-MARGIN POLICY ===")
        print(display.to_string(index=False))

    print("\nFrozen English thresholds:")
    print(policy_table.to_string(index=False))
    print(f"\nSaved: {output_dir.resolve()}")
    print("Status: POST-HOC deployment sensitivity; locked Turkish result unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
