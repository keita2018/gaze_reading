#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

PRIMARY = "qcn_pc"
REFERENCE = "participant_median_loo"
COVERAGES_DEFAULT = "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1"

EN_METHOD_MAP = {
    "participant_only_oof": REFERENCE,
    "quality_removed_plus_participant": PRIMARY,
}

TR_METHOD_COLUMNS = {
    REFERENCE: "participant_median_loo_score",
    PRIMARY: "qcn_pc_score",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--english-predictions",
        type=Path,
        default=Path(
            "outputs/linear_feature_group_factorization_v1/"
            "target_feature_group_predictions.csv"
        ),
    )
    p.add_argument(
        "--turkish-predictions",
        type=Path,
        default=Path(
            "outputs/iui_p0_confirmatory_TR_evaluation_v1/"
            "confirmatory_joined_trial_predictions.parquet"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/selective_prediction_reviewer_diagnostic_v2"
        ),
    )
    p.add_argument("--coverages", default=COVERAGES_DEFAULT)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_coverages(text: str) -> list[float]:
    values = sorted(
        {float(x.strip()) for x in text.split(",") if x.strip()},
        reverse=True,
    )
    if 1.0 not in values:
        values.insert(0, 1.0)
    if any(x <= 0 or x > 1 for x in values):
        raise ValueError("Coverages must be in (0,1].")
    return values


def normalize_base(frame: pd.DataFrame) -> pd.DataFrame:
    needed = {"participant_id", "external_trial_id", "label_nr"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"Missing base columns: {missing}")
    out = frame.copy()
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["external_trial_id"] = out["external_trial_id"].astype(str).str.strip()
    out["label_nr"] = pd.to_numeric(out["label_nr"], errors="raise").astype(int)
    if not out["label_nr"].isin([0, 1]).all():
        raise ValueError("label_nr must be 0/1.")
    return out


def load_english(path: Path) -> tuple[pd.DataFrame, float]:
    f = normalize_base(pd.read_csv(require_file(path), low_memory=False))
    needed = {"method", "score", "frozen_threshold"}
    missing = sorted(needed - set(f.columns))
    if missing:
        raise ValueError(f"English table lacks columns: {missing}")

    f["method"] = f["method"].astype(str).str.strip()
    f = f[f["method"].isin(EN_METHOD_MAP)].copy()
    f["method"] = f["method"].map(EN_METHOD_MAP)
    f["score"] = pd.to_numeric(f["score"], errors="raise").astype(float)
    f["frozen_threshold"] = pd.to_numeric(
        f["frozen_threshold"], errors="raise"
    ).astype(float)

    thresholds = np.unique(np.round(f["frozen_threshold"].to_numpy(float), 12))
    if len(thresholds) != 1:
        raise RuntimeError(
            f"Expected one frozen threshold, found {thresholds.tolist()}"
        )
    threshold = float(f["frozen_threshold"].iloc[0])

    f["dataset"] = "English"
    f["prediction"] = (f["score"] >= f["frozen_threshold"]).astype(int)
    f["confidence"] = np.abs(f["score"] - f["frozen_threshold"])
    f["correct"] = (f["prediction"] == f["label_nr"]).astype(int)

    if f.duplicated(["method", "external_trial_id"]).any():
        raise ValueError("Duplicate English method/trial rows.")

    return (
        f[
            [
                "dataset",
                "participant_id",
                "external_trial_id",
                "label_nr",
                "method",
                "score",
                "frozen_threshold",
                "prediction",
                "confidence",
                "correct",
            ]
        ],
        threshold,
    )


def load_turkish(path: Path, frozen_threshold: float) -> pd.DataFrame:
    f = normalize_base(pd.read_parquet(require_file(path)))
    missing = sorted(set(TR_METHOD_COLUMNS.values()) - set(f.columns))
    if missing:
        raise ValueError(f"Turkish table lacks score columns: {missing}")

    parts = []
    for method, column in TR_METHOD_COLUMNS.items():
        p = f[
            ["participant_id", "external_trial_id", "label_nr"]
        ].copy()
        p["dataset"] = "Turkish"
        p["method"] = method
        p["score"] = pd.to_numeric(f[column], errors="raise").astype(float)
        p["frozen_threshold"] = float(frozen_threshold)
        p["prediction"] = (p["score"] >= frozen_threshold).astype(int)
        p["confidence"] = np.abs(p["score"] - frozen_threshold)
        p["correct"] = (p["prediction"] == p["label_nr"]).astype(int)
        parts.append(p)

    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(["method", "external_trial_id"]).any():
        raise ValueError("Duplicate Turkish method/trial rows.")
    return out


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["label_nr"].to_numpy(int)
    pred = frame["prediction"].to_numpy(int)
    score = frame["score"].to_numpy(float)
    acc = float(accuracy_score(y, pred))
    return {
        "accuracy": acc,
        "selective_risk": 1.0 - acc,
        "macro_f1": float(
            f1_score(y, pred, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": safe_auc(y, score),
        "true_nr_rate": float(np.mean(y == 1)),
        "predicted_nr_rate": float(np.mean(pred == 1)),
        "mean_confidence": float(frame["confidence"].mean()),
    }


def select_global(frame: pd.DataFrame, coverage: float) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["confidence", "external_trial_id"],
        ascending=[False, True],
        kind="stable",
    )
    keep = max(1, int(math.ceil(coverage * len(ordered))))
    return ordered.iloc[:keep].copy()


def select_pb(frame: pd.DataFrame, coverage: float) -> pd.DataFrame:
    parts = []
    for _, g in frame.groupby("participant_id", observed=True, sort=False):
        ordered = g.sort_values(
            ["confidence", "external_trial_id"],
            ascending=[False, True],
            kind="stable",
        )
        keep = max(1, int(math.ceil(coverage * len(ordered))))
        parts.append(ordered.iloc[:keep])
    return pd.concat(parts, ignore_index=True)


def select_rows(frame: pd.DataFrame, coverage: float, mode: str) -> pd.DataFrame:
    if mode == "global":
        return select_global(frame, coverage)
    if mode == "participant_balanced":
        return select_pb(frame, coverage)
    raise ValueError(mode)


def participant_concentration(frame: pd.DataFrame) -> dict[str, float]:
    counts = (
        frame.groupby("participant_id", observed=True)
        .size()
        .to_numpy(float)
    )
    shares = counts / counts.sum()
    effective_n = float(1.0 / np.sum(shares**2))
    max_share = float(np.max(shares))
    if len(counts) <= 1:
        gini = 0.0
    else:
        x = np.sort(counts)
        n = len(x)
        gini = float(
            (2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x)))
            - (n + 1) / n
        )
    return {
        "participants_retained": int(len(counts)),
        "effective_participants_simpson": effective_n,
        "max_participant_share": max_share,
        "participant_count_gini": gini,
    }


def status_label(dataset: str, coverage: float) -> str:
    if dataset == "English":
        return "development_posthoc_diagnostic"
    if any(abs(coverage - x) < 1e-12 for x in (1.0, 0.5, 0.2)):
        return "turkish_locked_coverage_reanalysis"
    return "turkish_posthoc_diagnostic_not_confirmatory"


def coverage_table(long: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in long.groupby(
        ["dataset", "method"], observed=True
    ):
        for mode in ("global", "participant_balanced"):
            for coverage in coverages:
                retained = select_rows(g, coverage, mode)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "selection_mode": mode,
                        "target_coverage": coverage,
                        "actual_coverage": len(retained) / len(g),
                        "n_retained": len(retained),
                        "n_total": len(g),
                        "analysis_status": status_label(dataset, coverage),
                        **participant_concentration(retained),
                        **metrics(retained),
                    }
                )
    return pd.DataFrame(rows)


def decile_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in long.groupby(
        ["dataset", "method"], observed=True
    ):
        ordered = g.sort_values(
            ["confidence", "external_trial_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        ranks = np.arange(len(ordered))
        ordered["confidence_decile"] = np.minimum(
            10, np.floor(ranks * 10 / len(ordered)).astype(int) + 1
        )

        for decile, d in ordered.groupby("confidence_decile", observed=True):
            y = d["label_nr"].to_numpy(int)
            pred = d["prediction"].to_numpy(int)
            fp = int(np.sum((y == 0) & (pred == 1)))
            fn = int(np.sum((y == 1) & (pred == 0)))
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "confidence_decile": int(decile),
                    "confidence_rank_range": (
                        f"{(decile-1)*10}-{decile*10}% "
                        "(1=highest confidence)"
                    ),
                    "n_trials": len(d),
                    **participant_concentration(d),
                    **metrics(d),
                    "false_nr_direction_count": fp,
                    "false_is_direction_count": fn,
                    "incorrect_count": fp + fn,
                    "correct_count": int(len(d) - fp - fn),
                }
            )
    return pd.DataFrame(rows)


def action_tradeoff_table(
    long: pd.DataFrame,
    coverages: list[float],
) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in long.groupby(
        ["dataset", "method"], observed=True
    ):
        total = len(g)
        total_errors = int(np.sum(g["correct"].to_numpy(int) == 0))
        total_correct = total - total_errors

        for mode in ("global", "participant_balanced"):
            for coverage in coverages:
                kept = select_rows(g, coverage, mode)
                kept_ids = set(
                    zip(
                        kept["participant_id"].astype(str),
                        kept["external_trial_id"].astype(str),
                    )
                )
                key = list(
                    zip(
                        g["participant_id"].astype(str),
                        g["external_trial_id"].astype(str),
                    )
                )
                retained_mask = np.asarray([k in kept_ids for k in key])
                correct = g["correct"].to_numpy(int).astype(bool)

                errors_retained = int(np.sum(retained_mask & ~correct))
                errors_withheld = int(np.sum(~retained_mask & ~correct))
                correct_retained = int(np.sum(retained_mask & correct))
                correct_withheld = int(np.sum(~retained_mask & correct))

                y = g["label_nr"].to_numpy(int)
                pred = g["prediction"].to_numpy(int)
                false_nr_retained = int(
                    np.sum(retained_mask & (y == 0) & (pred == 1))
                )
                false_is_retained = int(
                    np.sum(retained_mask & (y == 1) & (pred == 0))
                )

                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "selection_mode": mode,
                        "target_coverage": coverage,
                        "analysis_status": status_label(dataset, coverage),
                        "n_total": total,
                        "n_retained": int(np.sum(retained_mask)),
                        "n_withheld": int(np.sum(~retained_mask)),
                        "errors_retained_incorrect_actions": errors_retained,
                        "errors_withheld_prevented_incorrect_actions": errors_withheld,
                        "correct_retained_actions": correct_retained,
                        "correct_withheld_missed_opportunities": correct_withheld,
                        "wrong_action_rate_per_all_trials": errors_retained / total,
                        "error_prevention_fraction_of_full_errors": (
                            errors_withheld / total_errors
                            if total_errors else np.nan
                        ),
                        "correct_sacrifice_fraction_of_full_correct": (
                            correct_withheld / total_correct
                            if total_correct else np.nan
                        ),
                        "false_nr_direction_retained": false_nr_retained,
                        "false_is_direction_retained": false_is_retained,
                        "prevented_errors_per_correct_sacrificed": (
                            errors_withheld / correct_withheld
                            if correct_withheld else np.inf
                        ),
                    }
                )
    return pd.DataFrame(rows)


def aurc_stats(frame: pd.DataFrame) -> dict[str, float]:
    ordered = frame.sort_values(
        ["confidence", "external_trial_id"],
        ascending=[False, True],
        kind="stable",
    )
    errors = 1 - ordered["correct"].to_numpy(int)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    aurc = float(np.mean(cumulative_risk))
    total_errors = int(errors.sum())
    oracle = np.concatenate(
        [np.zeros(len(errors) - total_errors), np.ones(total_errors)]
    )
    oracle_risk = np.cumsum(oracle) / np.arange(1, len(errors) + 1)
    oracle_aurc = float(np.mean(oracle_risk))
    spearman = float(
        pd.Series(ordered["confidence"].to_numpy(float)).corr(
            pd.Series(ordered["correct"].to_numpy(float)),
            method="spearman",
        )
    )
    return {
        "aurc_error": aurc,
        "oracle_aurc_error": oracle_aurc,
        "excess_aurc_error": aurc - oracle_aurc,
        "confidence_correct_spearman": spearman,
    }


def confidence_diagnostics(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in long.groupby(
        ["dataset", "method"], observed=True
    ):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_trials": len(g),
                "n_participants": g["participant_id"].nunique(),
                **aurc_stats(g),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FAST BOOTSTRAP
# ---------------------------------------------------------------------------

def confusion_arrays(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Return rows [TP, TN, FP, FN] for each trial."""
    return np.column_stack(
        [
            ((y == 1) & (pred == 1)).astype(np.int32),
            ((y == 0) & (pred == 0)).astype(np.int32),
            ((y == 0) & (pred == 1)).astype(np.int32),
            ((y == 1) & (pred == 0)).astype(np.int32),
        ]
    )


def metrics_from_confusion(counts: np.ndarray) -> tuple[float, float, float]:
    tp, tn, fp, fn = [float(x) for x in counts]
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else np.nan
    risk = 1.0 - acc if np.isfinite(acc) else np.nan

    denom_pos = 2 * tp + fp + fn
    denom_neg = 2 * tn + fp + fn
    f1_pos = 2 * tp / denom_pos if denom_pos else 0.0
    f1_neg = 2 * tn / denom_neg if denom_neg else 0.0
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    return float(acc), float(risk), float(macro_f1)


class ParticipantBlock:
    __slots__ = (
        "confidence",
        "confusion",
        "n",
        "pb_cumulative",
    )

    def __init__(
        self,
        confidence: np.ndarray,
        y: np.ndarray,
        pred: np.ndarray,
    ):
        confidence = np.asarray(confidence, dtype=float)
        y = np.asarray(y, dtype=np.int8)
        pred = np.asarray(pred, dtype=np.int8)

        order = np.argsort(-confidence, kind="stable")
        self.confidence = confidence
        self.confusion = confusion_arrays(y, pred)
        self.n = int(len(y))

        sorted_confusion = self.confusion[order]
        # Row 0 = no retained trials; row k = top-k retained.
        self.pb_cumulative = np.vstack(
            [
                np.zeros((1, 4), dtype=np.int64),
                np.cumsum(sorted_confusion, axis=0, dtype=np.int64),
            ]
        )


def build_blocks(frame: pd.DataFrame) -> list[ParticipantBlock]:
    blocks = []
    for _, g in frame.groupby("participant_id", observed=True, sort=True):
        blocks.append(
            ParticipantBlock(
                g["confidence"].to_numpy(float),
                g["label_nr"].to_numpy(int),
                g["prediction"].to_numpy(int),
            )
        )
    if not blocks:
        raise RuntimeError("No participant blocks.")
    return blocks


def bootstrap_one_combo_fast(
    frame: pd.DataFrame,
    coverages: list[float],
    n_bootstrap: int,
    rng: np.random.Generator,
    dataset: str,
    method: str,
    progress_every: int,
) -> list[dict[str, Any]]:
    blocks = build_blocks(frame)
    n_participants = len(blocks)
    rows: list[dict[str, Any]] = []

    start = time.perf_counter()

    for b in range(n_bootstrap):
        sampled_idx = rng.integers(
            0, n_participants, size=n_participants, endpoint=False
        )
        sampled = [blocks[int(i)] for i in sampled_idx]

        # ----- GLOBAL -----
        # Concatenate the sampled clusters once, sort once, then reuse prefix
        # cumulative confusion counts for every coverage.
        confidences = np.concatenate([x.confidence for x in sampled])
        confusions = np.concatenate([x.confusion for x in sampled], axis=0)
        order = np.argsort(-confidences, kind="stable")
        cumulative = np.vstack(
            [
                np.zeros((1, 4), dtype=np.int64),
                np.cumsum(confusions[order], axis=0, dtype=np.int64),
            ]
        )
        n_trials = len(confidences)

        for coverage in coverages:
            k = max(1, int(math.ceil(coverage * n_trials)))
            acc, risk, macro_f1 = metrics_from_confusion(cumulative[k])
            rows.extend(
                [
                    {
                        "replicate": b,
                        "selection_mode": "global",
                        "target_coverage": coverage,
                        "metric": "accuracy",
                        "value": acc,
                    },
                    {
                        "replicate": b,
                        "selection_mode": "global",
                        "target_coverage": coverage,
                        "metric": "selective_risk",
                        "value": risk,
                    },
                    {
                        "replicate": b,
                        "selection_mode": "global",
                        "target_coverage": coverage,
                        "metric": "macro_f1",
                        "value": macro_f1,
                    },
                ]
            )

        # ----- PARTICIPANT-BALANCED -----
        # No sorting is needed here during bootstrap. Each original participant
        # was sorted once in build_blocks(), and only cumulative top-k counts
        # are summed for the sampled participant instances.
        for coverage in coverages:
            total_counts = np.zeros(4, dtype=np.int64)
            for block in sampled:
                k = max(1, int(math.ceil(coverage * block.n)))
                total_counts += block.pb_cumulative[k]
            acc, risk, macro_f1 = metrics_from_confusion(total_counts)
            rows.extend(
                [
                    {
                        "replicate": b,
                        "selection_mode": "participant_balanced",
                        "target_coverage": coverage,
                        "metric": "accuracy",
                        "value": acc,
                    },
                    {
                        "replicate": b,
                        "selection_mode": "participant_balanced",
                        "target_coverage": coverage,
                        "metric": "selective_risk",
                        "value": risk,
                    },
                    {
                        "replicate": b,
                        "selection_mode": "participant_balanced",
                        "target_coverage": coverage,
                        "metric": "macro_f1",
                        "value": macro_f1,
                    },
                ]
            )

        if progress_every > 0 and (
            (b + 1) % progress_every == 0 or b + 1 == n_bootstrap
        ):
            elapsed = time.perf_counter() - start
            rate = (b + 1) / elapsed if elapsed > 0 else np.nan
            remaining = (
                (n_bootstrap - (b + 1)) / rate
                if np.isfinite(rate) and rate > 0
                else np.nan
            )
            print(
                f"[bootstrap] {dataset}/{method}: "
                f"{b+1}/{n_bootstrap} "
                f"elapsed={elapsed:.1f}s "
                f"eta={remaining:.1f}s",
                flush=True,
            )

    return rows


def bootstrap_selective_fast(
    long: pd.DataFrame,
    coverages: list[float],
    n_bootstrap: int,
    seed: int,
    progress_every: int,
) -> pd.DataFrame:
    replicate_rows: list[dict[str, Any]] = []
    combo_index = 0

    for dataset in ("English", "Turkish"):
        for method in (REFERENCE, PRIMARY):
            frame = long[
                (long["dataset"] == dataset)
                & (long["method"] == method)
            ].copy()

            rng = np.random.default_rng(seed + combo_index * 100003)
            combo_index += 1

            combo_rows = bootstrap_one_combo_fast(
                frame=frame,
                coverages=coverages,
                n_bootstrap=n_bootstrap,
                rng=rng,
                dataset=dataset,
                method=method,
                progress_every=progress_every,
            )
            for row in combo_rows:
                row["dataset"] = dataset
                row["method"] = method
            replicate_rows.extend(combo_rows)

    reps = pd.DataFrame(replicate_rows)

    summary_rows = []
    for (dataset, method, mode, coverage, metric_name), g in reps.groupby(
        [
            "dataset",
            "method",
            "selection_mode",
            "target_coverage",
            "metric",
        ],
        observed=True,
        sort=True,
    ):
        values = g["value"].to_numpy(float)
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "selection_mode": mode,
                "target_coverage": float(coverage),
                "metric": metric_name,
                "bootstrap_mean": float(np.mean(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "n_bootstrap": int(len(values)),
                "analysis_status": status_label(dataset, float(coverage)),
            }
        )

    return pd.DataFrame(summary_rows)


def anomaly_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ("English", "Turkish"):
        cov = coverage[
            (coverage["dataset"] == dataset)
            & (coverage["method"] == PRIMARY)
            & (coverage["selection_mode"] == "global")
        ].set_index("target_coverage")
        if 0.2 in cov.index and 0.1 in cov.index:
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": "global_10pct_minus_20pct",
                    "accuracy_delta": float(
                        cov.loc[0.1, "accuracy"] - cov.loc[0.2, "accuracy"]
                    ),
                    "selective_risk_delta": float(
                        cov.loc[0.1, "selective_risk"]
                        - cov.loc[0.2, "selective_risk"]
                    ),
                    "macro_f1_delta": float(
                        cov.loc[0.1, "macro_f1"] - cov.loc[0.2, "macro_f1"]
                    ),
                    "participants_at_10pct": int(
                        cov.loc[0.1, "participants_retained"]
                    ),
                    "participants_at_20pct": int(
                        cov.loc[0.2, "participants_retained"]
                    ),
                    "true_nr_rate_10pct": float(cov.loc[0.1, "true_nr_rate"]),
                    "predicted_nr_rate_10pct": float(
                        cov.loc[0.1, "predicted_nr_rate"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    coverages = parse_coverages(args.coverages)

    print("[1/6] Loading frozen target scores...", flush=True)
    english, frozen_threshold = load_english(args.english_predictions)
    turkish = load_turkish(args.turkish_predictions, frozen_threshold)
    long = pd.concat([english, turkish], ignore_index=True)

    for dataset, g in long.groupby("dataset", observed=True):
        methods = set(g["method"].unique())
        if methods != {PRIMARY, REFERENCE}:
            raise RuntimeError(
                f"{dataset}: expected exactly {PRIMARY, REFERENCE}, found {methods}"
            )

    print("[2/6] Computing exact coverage tables...", flush=True)
    coverage = coverage_table(long, coverages)

    print("[3/6] Computing confidence deciles...", flush=True)
    deciles = decile_table(long)

    print("[4/6] Computing action/abstention trade-offs...", flush=True)
    tradeoff = action_tradeoff_table(long, coverages)
    confidence = confidence_diagnostics(long)

    print(
        f"[5/6] Running optimized participant bootstrap "
        f"({args.bootstrap} replicates)...",
        flush=True,
    )
    bootstrap = bootstrap_selective_fast(
        long=long,
        coverages=coverages,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        progress_every=args.progress_every,
    )

    print("[6/6] Writing outputs...", flush=True)
    anomaly = anomaly_summary(coverage)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(
        args.output_dir / "coverage_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    deciles.to_csv(
        args.output_dir / "confidence_deciles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tradeoff.to_csv(
        args.output_dir / "action_abstention_tradeoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    confidence.to_csv(
        args.output_dir / "confidence_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        args.output_dir / "bootstrap_selective_ci.csv",
        index=False,
        encoding="utf-8-sig",
    )
    anomaly.to_csv(
        args.output_dir / "extreme_coverage_anomaly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "analysis": "Selective-prediction reviewer diagnostic v2",
        "primary_method": PRIMARY,
        "reference_method": REFERENCE,
        "confidence": "absolute distance from frozen OneStop threshold",
        "frozen_threshold": frozen_threshold,
        "target_labels_used_for_ranking": False,
        "coverages": coverages,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "bootstrap_implementation": (
            "optimized participant-cluster bootstrap: one global sort per "
            "replicate and cached participant-wise cumulative confusion counts"
        ),
        "confirmatory_boundary": (
            "Turkish 100%, 50%, and 20% coverage points were locked previously. "
            "All finer Turkish coverage/decile analyses are post-hoc diagnostics."
        ),
    }
    (args.output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    qcn_cov = coverage[coverage["method"].eq(PRIMARY)].copy()
    qcn_dec = deciles[deciles["method"].eq(PRIMARY)].copy()

    print("\n=== QCN-PC SELECTIVE COVERAGE DIAGNOSTIC ===")
    print(
        qcn_cov[
            [
                "dataset",
                "selection_mode",
                "target_coverage",
                "actual_coverage",
                "n_retained",
                "participants_retained",
                "effective_participants_simpson",
                "accuracy",
                "selective_risk",
                "macro_f1",
                "roc_auc",
                "true_nr_rate",
                "predicted_nr_rate",
            ]
        ].to_string(index=False)
    )

    print("\n=== QCN-PC CONFIDENCE DECILES (1 = HIGHEST CONFIDENCE) ===")
    print(
        qcn_dec[
            [
                "dataset",
                "confidence_decile",
                "n_trials",
                "participants_retained",
                "accuracy",
                "selective_risk",
                "macro_f1",
                "true_nr_rate",
                "predicted_nr_rate",
                "mean_confidence",
                "incorrect_count",
            ]
        ].to_string(index=False)
    )

    print("\n=== CONFIDENCE DIAGNOSTICS ===")
    print(confidence.to_string(index=False))

    print("\n=== EXTREME-COVERAGE ANOMALY SUMMARY ===")
    print(anomaly.to_string(index=False))

    print("\n=== QCN-PC ACTION/ABSTENTION TRADEOFF: KEY COVERAGES ===")
    key = tradeoff[
        tradeoff["method"].eq(PRIMARY)
        & tradeoff["target_coverage"].isin([1.0, 0.5, 0.2, 0.1])
    ]
    print(
        key[
            [
                "dataset",
                "selection_mode",
                "target_coverage",
                "n_retained",
                "errors_retained_incorrect_actions",
                "errors_withheld_prevented_incorrect_actions",
                "correct_retained_actions",
                "correct_withheld_missed_opportunities",
                "wrong_action_rate_per_all_trials",
                "error_prevention_fraction_of_full_errors",
                "correct_sacrifice_fraction_of_full_correct",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
