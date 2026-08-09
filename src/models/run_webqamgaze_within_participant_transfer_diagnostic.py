#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ENGLISH_METHOD_MAP = {
    "raw_frozen": "frozen_full41",
    "participant_only_oof": "participant_median_loo",
    "quality_removed": "qcn_only",
    "quality_removed_plus_participant": "qcn_pc",
}

TURKISH_METHOD_COLUMNS = {
    "frozen_full41": "frozen_full41_score",
    "participant_median_loo": "participant_median_loo_score",
    "qcn_only": "qcn_only_score",
    "qcn_pc": "qcn_pc_score",
}

METHOD_ORDER = (
    "frozen_full41",
    "participant_median_loo",
    "qcn_only",
    "qcn_pc",
)


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
            "outputs/webqamgaze_within_participant_transfer_diagnostic_v1"
        ),
    )
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260809)
    return p.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def normalize_base(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "external_trial_id", "label_nr"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame.copy()
    result["participant_id"] = result["participant_id"].astype(str).str.strip()
    result["external_trial_id"] = result["external_trial_id"].astype(str).str.strip()
    result["label_nr"] = pd.to_numeric(result["label_nr"], errors="raise").astype(int)
    if not result["label_nr"].isin([0, 1]).all():
        raise ValueError("label_nr must be binary.")
    return result


def load_english(path: Path) -> pd.DataFrame:
    frame = normalize_base(pd.read_csv(require_file(path), low_memory=False))
    required = {"method", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"English table lacks columns: {missing}")
    frame["method"] = frame["method"].astype(str).str.strip()
    frame = frame[frame["method"].isin(ENGLISH_METHOD_MAP)].copy()
    frame["method"] = frame["method"].map(ENGLISH_METHOD_MAP)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    frame["dataset"] = "English"
    if frame.duplicated(["method", "participant_id", "external_trial_id"]).any():
        raise ValueError("Duplicate English method/trial rows.")
    return frame[
        ["dataset", "participant_id", "external_trial_id", "label_nr", "method", "score"]
    ]


def load_turkish(path: Path) -> pd.DataFrame:
    frame = normalize_base(pd.read_parquet(require_file(path)))
    missing = sorted(set(TURKISH_METHOD_COLUMNS.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"Turkish table lacks score columns: {missing}")
    parts = []
    for method, column in TURKISH_METHOD_COLUMNS.items():
        part = frame[["participant_id", "external_trial_id", "label_nr"]].copy()
        part["dataset"] = "Turkish"
        part["method"] = method
        part["score"] = pd.to_numeric(frame[column], errors="raise").astype(float)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[
        ["dataset", "participant_id", "external_trial_id", "label_nr", "method", "score"]
    ]


def participant_auc_table(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, method, participant_id), group in long.groupby(
        ["dataset", "method", "participant_id"],
        observed=True,
        sort=True,
    ):
        y = group["label_nr"].to_numpy(dtype=int)
        score = group["score"].to_numpy(dtype=float)
        n_nr = int(np.sum(y == 1))
        n_is = int(np.sum(y == 0))
        auc = (
            float(roc_auc_score(y, score))
            if n_nr > 0 and n_is > 0
            else float("nan")
        )
        nr_score = score[y == 1]
        is_score = score[y == 0]
        diff = (
            float(np.mean(nr_score) - np.mean(is_score))
            if len(nr_score) and len(is_score)
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "participant_id": participant_id,
                "n_trials": int(len(group)),
                "n_nr": n_nr,
                "n_is": n_is,
                "within_participant_auc": auc,
                "mean_score_nr": float(np.mean(nr_score)) if len(nr_score) else np.nan,
                "mean_score_is": float(np.mean(is_score)) if len(is_score) else np.nan,
                "mean_score_difference_nr_minus_is": diff,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(aucs: np.ndarray, n_bootstrap: int, rng: np.random.Generator):
    draw = rng.choice(aucs, size=(n_bootstrap, len(aucs)), replace=True)
    means = draw.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def score_average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def auc_from_positive_ranks(ranks: np.ndarray, positive_mask: np.ndarray) -> float:
    n_pos = int(np.sum(positive_mask))
    n_neg = int(len(positive_mask) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    rank_sum = float(np.sum(ranks[positive_mask]))
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def permutation_mean_auc(
    group: pd.DataFrame,
    n_permutations: int,
    rng: np.random.Generator,
):
    items = []
    observed = []
    for _, part in group.groupby("participant_id", observed=True, sort=False):
        y = part["label_nr"].to_numpy(dtype=int)
        if np.unique(y).size != 2:
            continue
        ranks = score_average_ranks(part["score"].to_numpy(dtype=float))
        n_pos = int(np.sum(y == 1))
        observed.append(auc_from_positive_ranks(ranks, y == 1))
        items.append((ranks, n_pos))
    observed_mean = float(np.mean(observed))
    null = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        total = 0.0
        for ranks, n_pos in items:
            idx = rng.choice(len(ranks), size=n_pos, replace=False)
            mask = np.zeros(len(ranks), dtype=bool)
            mask[idx] = True
            total += auc_from_positive_ranks(ranks, mask)
        null[k] = total / len(items)
    return observed_mean, null


def summarize(
    participant: pd.DataFrame,
    long: pd.DataFrame,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
):
    summary_rows = []
    perm_rows = []
    null_rows = []
    method_index = {m: i for i, m in enumerate(METHOD_ORDER)}

    for dataset in ("English", "Turkish"):
        for method in METHOD_ORDER:
            part = participant[
                (participant["dataset"] == dataset)
                & (participant["method"] == method)
            ].copy()
            aucs = part["within_participant_auc"].dropna().to_numpy(float)
            if len(aucs) == 0:
                continue
            rng_boot = np.random.default_rng(
                seed + (0 if dataset == "English" else 100000)
                + method_index[method] * 1000
            )
            ci_low, ci_high = bootstrap_ci(aucs, n_bootstrap, rng_boot)
            diffs = part["mean_score_difference_nr_minus_is"].dropna().to_numpy(float)

            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_participants_with_both_labels": int(len(aucs)),
                    "mean_within_participant_auc": float(np.mean(aucs)),
                    "median_within_participant_auc": float(np.median(aucs)),
                    "std_within_participant_auc": float(np.std(aucs, ddof=1)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "mean_auc_minus_chance": float(np.mean(aucs) - 0.5),
                    "fraction_auc_above_0_5": float(np.mean(aucs > 0.5)),
                    "fraction_auc_equal_0_5": float(np.mean(aucs == 0.5)),
                    "fraction_auc_below_0_5": float(np.mean(aucs < 0.5)),
                    "mean_score_difference_nr_minus_is": float(np.mean(diffs)),
                    "fraction_participants_nr_score_higher": float(np.mean(diffs > 0)),
                }
            )

            group = long[
                (long["dataset"] == dataset)
                & (long["method"] == method)
            ]
            rng_perm = np.random.default_rng(
                seed + 500000
                + (0 if dataset == "English" else 100000)
                + method_index[method] * 1000
            )
            observed, null = permutation_mean_auc(
                group, n_permutations, rng_perm
            )
            p_one_sided = float(
                (1 + np.sum(null >= observed)) / (len(null) + 1)
            )
            perm_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "primary_confound_diagnostic": method == "frozen_full41",
                    "observed_mean_within_participant_auc": observed,
                    "null_mean": float(np.mean(null)),
                    "null_q95": float(np.quantile(null, 0.95)),
                    "one_sided_p_auc_above_chance": p_one_sided,
                    "n_permutations": int(len(null)),
                }
            )
            for k, value in enumerate(null):
                null_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "permutation": k,
                        "mean_within_participant_auc": float(value),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(perm_rows), pd.DataFrame(null_rows)


def offset_audit(participant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ("English", "Turkish"):
        raw = participant[
            (participant["dataset"] == dataset)
            & (participant["method"] == "frozen_full41")
        ][["participant_id", "within_participant_auc"]].rename(
            columns={"within_participant_auc": "raw_auc"}
        )
        cal = participant[
            (participant["dataset"] == dataset)
            & (participant["method"] == "participant_median_loo")
        ][["participant_id", "within_participant_auc"]].rename(
            columns={"within_participant_auc": "calibrated_auc"}
        )
        merged = raw.merge(cal, on="participant_id", validate="one_to_one")
        delta = (merged["calibrated_auc"] - merged["raw_auc"]).to_numpy(float)
        rows.append(
            {
                "dataset": dataset,
                "n_participants": int(len(merged)),
                "max_abs_auc_difference_after_participant_offset": float(
                    np.nanmax(np.abs(delta))
                ),
                "mean_auc_difference_after_participant_offset": float(
                    np.nanmean(delta)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1 or args.permutations < 1:
        raise ValueError("bootstrap/permutations must be positive.")

    english = load_english(args.english_predictions)
    turkish = load_turkish(args.turkish_predictions)
    long = pd.concat([english, turkish], ignore_index=True)

    expected = set(METHOD_ORDER)
    for dataset, group in long.groupby("dataset", observed=True):
        methods = set(group["method"].unique())
        if methods != expected:
            raise RuntimeError(
                f"{dataset}: expected methods={sorted(expected)}, found={sorted(methods)}"
            )

    participant = participant_auc_table(long)
    summary, permutations, null = summarize(
        participant,
        long,
        args.bootstrap,
        args.permutations,
        args.seed,
    )
    audit = offset_audit(participant)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    participant.to_csv(
        args.output_dir / "within_participant_auc_by_participant.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "within_participant_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    permutations.to_csv(
        args.output_dir / "within_participant_permutation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    null.to_parquet(
        args.output_dir / "within_participant_permutation_replicates.parquet",
        index=False,
    )
    audit.to_csv(
        args.output_dir / "participant_offset_auc_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "analysis": "WebQAmGaze within-participant transfer diagnostic",
        "primary_diagnostic_method": "frozen_full41",
        "primary_statistic": "mean within-participant ROC-AUC",
        "null": 0.5,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "interpretation_boundary": (
            "Above-chance target within-participant discrimination rules out "
            "source participant-group memorization as a sufficient explanation "
            "of transfer, but does not remove all OneStop confounding or prove "
            "a causal within-person reading-goal effect."
        ),
    }
    (args.output_dir / "within_participant_diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    primary = permutations[
        permutations["method"] == "frozen_full41"
    ]
    lines = [
        "# WebQAmGaze Within-Participant Transfer Diagnostic",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Permutation inference",
        "",
        permutations.to_markdown(index=False),
        "",
        "## Participant-offset audit",
        "",
        audit.to_markdown(index=False),
        "",
        "## Primary interpretation",
        "",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"- {row['dataset']}: frozen Full-41 mean within-participant "
            f"AUC={row['observed_mean_within_participant_auc']:.6f}, "
            f"one-sided permutation p={row['one_sided_p_auc_above_chance']:.6g}."
        )
    lines.append(
        "\nAbove-chance within-participant target discrimination cannot be "
        "explained by source participant-group memorization alone. It does not "
        "remove all source confounding or establish a causal within-person effect."
    )
    (args.output_dir / "RESULTS_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("=== WEBQAMGAZE WITHIN-PARTICIPANT TRANSFER DIAGNOSTIC ===")
    print(summary.to_string(index=False))
    print("\n=== WITHIN-PARTICIPANT PERMUTATION TESTS ===")
    print(permutations.to_string(index=False))
    print("\n=== PARTICIPANT OFFSET AUC AUDIT ===")
    print(audit.to_string(index=False))
    print(f"\nSaved: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
