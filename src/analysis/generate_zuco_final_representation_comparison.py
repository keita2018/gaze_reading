#!/usr/bin/env python
"""Final subject-level comparison across Soft-AOI, Hard-AOI, fixed I-VT, and nested I-VT."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_zuco_soft_aoi_confound_controls import METRICS

CONDITIONS = ["clean", "rate10", "noise50", "random40", "burst40", "high_random", "high_burst"]
MODELS = ["soft_common25", "hard_argmax_common25", "fixed_ivt9", "nested_ivt9"]
PRIMARY_METRICS = ["macro_f1", "balanced_accuracy", "roc_auc", "pr_auc_nr"]
EXPECTED_SUBJECTS = 18


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p[index])
        out[index] = min(running, 1.0)
    return out.tolist()


def normalize(frame: pd.DataFrame, condition: str, model_column: str, mapping: dict[str, str]) -> pd.DataFrame:
    required = {model_column, "held_out_subject", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{condition}: missing columns {missing}")
    result = frame.copy()
    result["model"] = result[model_column].map(mapping)
    result = result[result["model"].notna()].copy()
    result["condition"] = condition
    result["held_out_subject"] = result["held_out_subject"].astype(str).str.upper()
    return result[["condition", "model", "held_out_subject", *METRICS]]


def load_all(args: argparse.Namespace) -> pd.DataFrame:
    mapping = {"soft_common25": "soft_common25", "hard_argmax_common25": "hard_argmax_common25"}
    parts: list[pd.DataFrame] = []

    parts.append(normalize(pd.read_csv(args.clean), "clean", "feature_set", mapping))
    parts.append(normalize(pd.read_csv(args.rate10), "rate10", "feature_set", mapping))

    noise = pd.read_csv(args.noise50)
    noise = noise[np.isclose(pd.to_numeric(noise["noise_sigma_px"], errors="coerce"), 50.0)]
    parts.append(normalize(noise, "noise50", "feature_set", mapping))

    missingness = pd.read_csv(args.missingness)
    missingness["missingness_mode"] = missingness["missingness_mode"].astype(str).str.lower()
    percentage = pd.to_numeric(missingness["additional_missing_percent"], errors="coerce")
    missingness = missingness[np.isclose(percentage, 40.0)]
    for mode, condition in [("random", "random40"), ("burst", "burst40")]:
        parts.append(normalize(missingness[missingness["missingness_mode"].eq(mode)], condition, "feature_set", mapping))

    combined = pd.read_csv(args.combined)
    for condition in ["high_random", "high_burst"]:
        parts.append(normalize(combined[combined["combined_condition"].eq(condition)], condition, "feature_set", mapping))

    for path, model in [(args.fixed_ivt, "fixed_ivt9"), (args.nested_ivt, "nested_ivt9")]:
        frame = pd.read_csv(path)
        required = {"condition", "held_out_subject", *METRICS}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{model}: missing columns {missing}")
        frame["condition"] = frame["condition"].astype(str).str.lower()
        frame["model"] = model
        frame["held_out_subject"] = frame["held_out_subject"].astype(str).str.upper()
        parts.append(frame[["condition", "model", "held_out_subject", *METRICS]])

    result = pd.concat(parts, ignore_index=True)
    result = result[result["condition"].isin(CONDITIONS) & result["model"].isin(MODELS)].copy()

    duplicated = result.duplicated(["condition", "model", "held_out_subject"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate condition/model/subject rows:\n" + result.loc[duplicated, ["condition", "model", "held_out_subject"]].head(20).to_string(index=False))

    counts = result.groupby(["condition", "model"])["held_out_subject"].nunique()
    expected_pairs = {(condition, model) for condition in CONDITIONS for model in MODELS}
    actual_pairs = set(counts.index.tolist())
    if expected_pairs != actual_pairs:
        raise ValueError(f"Condition/model mismatch. Missing={sorted(expected_pairs-actual_pairs)}, extra={sorted(actual_pairs-expected_pairs)}")
    if not counts.eq(EXPECTED_SUBJECTS).all():
        raise ValueError("Every condition/model must contain 18 subjects:\n" + counts[~counts.eq(EXPECTED_SUBJECTS)].to_string())
    if result[METRICS].isna().any().any():
        raise ValueError("Final subject metrics contain missing values.")
    return result


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, model), group in data.groupby(["condition", "model"], sort=False):
        row = {"condition": condition, "model": model, "n_subjects": group["held_out_subject"].nunique()}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(out["condition"], CONDITIONS, ordered=True)
    out["model"] = pd.Categorical(out["model"], MODELS, ordered=True)
    return out.sort_values(["condition", "model"]).reset_index(drop=True)


def paired_test(data: pd.DataFrame, condition: str, metric: str, reference: str, candidate: str) -> dict:
    subset = data[data["condition"].eq(condition)]
    a = subset[subset["model"].eq(reference)].set_index("held_out_subject")[metric].sort_index()
    b = subset[subset["model"].eq(candidate)].set_index("held_out_subject")[metric].sort_index()
    paired = pd.concat([a.rename("reference"), b.rename("candidate")], axis=1, join="inner").dropna()
    if len(paired) != EXPECTED_SUBJECTS:
        raise ValueError(f"Pairing failed: {condition}, {metric}, {reference}, {candidate}")
    diff = paired["candidate"] - paired["reference"]
    if np.allclose(diff.to_numpy(dtype=float), 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(paired["candidate"], paired["reference"], alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "condition": condition,
        "metric": metric,
        "reference_model": reference,
        "candidate_model": candidate,
        "reference_mean": float(paired["reference"].mean()),
        "candidate_mean": float(paired["candidate"].mean()),
        "mean_difference_candidate_minus_reference": float(diff.mean()),
        "candidate_wins": int((diff > 0).sum()),
        "ties": int(np.isclose(diff, 0.0).sum()),
        "candidate_losses": int((diff < 0).sum()),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def all_pairwise(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for metric in METRICS:
            family = [paired_test(data, condition, metric, a, b) for a, b in combinations(MODELS, 2)]
            adjusted = holm_adjust([row["p_value"] for row in family])
            for row, value in zip(family, adjusted):
                row["p_holm_six_pairs_within_condition_metric"] = value
            rows.extend(family)
    return pd.DataFrame(rows)


def planned_tests(data: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("hard_argmax_common25", "soft_common25"),
        ("fixed_ivt9", "soft_common25"),
        ("nested_ivt9", "soft_common25"),
        ("fixed_ivt9", "nested_ivt9"),
    ]
    rows = []
    for reference, candidate in comparisons:
        comparison = f"{candidate}_minus_{reference}"
        for metric in METRICS:
            family = []
            for condition in CONDITIONS:
                row = paired_test(data, condition, metric, reference, candidate)
                row["comparison"] = comparison
                family.append(row)
            adjusted = holm_adjust([row["p_value"] for row in family])
            for row, value in zip(family, adjusted):
                row["p_holm_seven_conditions_within_comparison_metric"] = value
            rows.extend(family)
    return pd.DataFrame(rows)


def wide_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        subset = summary[summary["condition"].astype(str).eq(condition)]
        for metric in PRIMARY_METRICS:
            row = {"condition": condition, "metric": metric}
            for model in MODELS:
                row[model] = float(subset[subset["model"].astype(str).eq(model)][f"{metric}_mean"].iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)


def winner_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        row = {"condition": condition}
        subset = wide[wide["condition"].eq(condition)]
        for metric in PRIMARY_METRICS:
            metric_row = subset[subset["metric"].eq(metric)].iloc[0]
            ranking = sorted(((model, float(metric_row[model])) for model in MODELS), key=lambda item: item[1], reverse=True)
            row[f"{metric}_winner"] = ranking[0][0]
            row[f"{metric}_winner_mean"] = ranking[0][1]
            row[f"{metric}_runner_up"] = ranking[1][0]
            row[f"{metric}_margin"] = ranking[0][1] - ranking[1][1]
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(wide: pd.DataFrame) -> str:
    labels = {"macro_f1": "Macro F1", "balanced_accuracy": "Balanced Accuracy", "roc_auc": "ROC-AUC", "pr_auc_nr": "PR-AUC (NR)"}
    model_labels = {"soft_common25": "Soft-AOI 25", "hard_argmax_common25": "Hard-AOI 25", "fixed_ivt9": "Fixed I-VT 9", "nested_ivt9": "Nested I-VT 9"}
    lines = [
        "# Final representation comparison",
        "",
        "Nested I-VT is a quality-adaptive auxiliary baseline: the measurement condition is assumed known and the threshold is selected without outer-test labels.",
        "",
    ]
    for metric in PRIMARY_METRICS:
        lines += [f"## {labels[metric]}", "", "| Condition | " + " | ".join(model_labels[m] for m in MODELS) + " |", "|---|" + "---:|" * len(MODELS)]
        subset = wide[wide["metric"].eq(metric)]
        for condition in CONDITIONS:
            row = subset[subset["condition"].eq(condition)].iloc[0]
            lines.append(f"| {condition} | " + " | ".join(f"{float(row[m]):.4f}" for m in MODELS) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="outputs/soft_vs_hard_aoi_clean_30hz/soft_vs_hard_subject_metrics.csv")
    parser.add_argument("--rate10", default="outputs/soft_vs_hard_aoi_train30_test10/soft_vs_hard_train30_test10_subject_metrics.csv")
    parser.add_argument("--noise50", default="outputs/soft_vs_hard_aoi_coordinate_noise_30hz/soft_vs_hard_noise_subject_metrics.csv")
    parser.add_argument("--missingness", default="outputs/soft_vs_hard_aoi_missingness_30hz/soft_vs_hard_missingness_subject_metrics.csv")
    parser.add_argument("--combined", default="outputs/soft_vs_hard_aoi_combined_high_30hz/soft_vs_hard_combined_high_subject_metrics.csv")
    parser.add_argument("--fixed-ivt", dest="fixed_ivt", default="outputs/ivt_event_baseline_fixed_benchmark_exact/ivt_fixed_subject_metrics.csv")
    parser.add_argument("--nested-ivt", dest="nested_ivt", default="outputs/ivt_event_baseline_nested_threshold/ivt_nested_subject_metrics.csv")
    parser.add_argument("--output-dir", default="outputs/final_representation_comparison")
    args = parser.parse_args()

    data = load_all(args)
    summary = summarize(data)
    wide = wide_table(summary)
    winners = winner_table(wide)
    pairwise = all_pairwise(data)
    planned = planned_tests(data)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "subject_metrics": output / "final_representation_subject_metrics.csv",
        "summary": output / "final_representation_summary.csv",
        "wide": output / "final_representation_primary_metrics_wide.csv",
        "winners": output / "final_representation_winners.csv",
        "pairwise": output / "final_representation_all_pairwise_tests.csv",
        "planned": output / "final_representation_planned_tests.csv",
        "markdown": output / "final_representation_tables.md",
        "manifest": output / "final_representation_manifest.json",
    }
    data.to_csv(paths["subject_metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    wide.to_csv(paths["wide"], index=False, encoding="utf-8-sig")
    winners.to_csv(paths["winners"], index=False, encoding="utf-8-sig")
    pairwise.to_csv(paths["pairwise"], index=False, encoding="utf-8-sig")
    planned.to_csv(paths["planned"], index=False, encoding="utf-8-sig")
    paths["markdown"].write_text(markdown_table(wide), encoding="utf-8")
    paths["manifest"].write_text(json.dumps({
        "models": MODELS,
        "conditions": CONDITIONS,
        "primary_metrics": PRIMARY_METRICS,
        "n_subjects": EXPECTED_SUBJECTS,
        "all_pairwise_holm_family": "six model pairs within each condition and metric",
        "planned_holm_family": "seven conditions within each planned comparison and metric",
        "nested_ivt_assumption": "quality condition known; threshold selected without outer-test labels",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== FINAL REPRESENTATION PRIMARY METRICS ===")
    print(wide.to_string(index=False))
    print("\n=== FINAL REPRESENTATION WINNERS ===")
    print(winners.to_string(index=False))
    key = planned[
        planned["metric"].isin(["macro_f1", "roc_auc", "pr_auc_nr"])
        & planned["comparison"].isin(["soft_common25_minus_fixed_ivt9", "soft_common25_minus_nested_ivt9"])
    ]
    print("\n=== KEY PLANNED TESTS: SOFT VS EVENT BASELINES ===")
    print(key[[
        "comparison", "condition", "metric", "reference_mean", "candidate_mean",
        "mean_difference_candidate_minus_reference", "candidate_wins", "ties",
        "candidate_losses", "p_value", "p_holm_seven_conditions_within_comparison_metric",
    ]].to_string(index=False))
    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
