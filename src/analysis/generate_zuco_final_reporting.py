#!/usr/bin/env python
"""Generate final result tables and figures for the ZuCo robustness study.

This script does not train or modify any model. It reads the already completed
experiment outputs and creates a fixed reporting snapshot.

Outputs
-------
outputs/final_reporting/
├─ table01_final_nested_summary.csv
├─ table02_final_nested_paired_comparisons.csv
├─ table03_key_robustness_results.csv
├─ table04_subject_level_final_macro_f1.csv
├─ final_results_summary.md
├─ fig01_final_common16_macro_f1.png
├─ fig02_downsampling_macro_f1.png
├─ fig03_coordinate_noise_augmentation_macro_f1.png
├─ fig04_missingness_content_only_macro_f1.png
├─ fig05_combined_degradation_content_only_macro_f1.png
└─ fig06_subject_level_final_macro_f1.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_ORDER = [
    "dummy_reference",
    "proposed_30hz_content_clean_nested",
    "proposed_30hz_content_mixed_weighted_nested",
    "official_sent_gaze_sacc_nested",
]

FINAL_LABELS = {
    "dummy_reference": "Dummy",
    "proposed_30hz_content_clean_nested": "30 Hz content\nclean training",
    "proposed_30hz_content_mixed_weighted_nested": (
        "30 Hz content\nmixed degradation training"
    ),
    "official_sent_gaze_sacc_nested": "Official\nfixation/saccade",
}

SUBJECT_LABELS = {
    "proposed_30hz_content_clean_nested": "30 Hz clean",
    "proposed_30hz_content_mixed_weighted_nested": "30 Hz mixed augmented",
    "official_sent_gaze_sacc_nested": "Official sent_gaze_sacc",
}

COMBINED_ORDER = [
    "moderate_random",
    "moderate_burst",
    "high_random",
    "high_burst",
    "extreme_random",
    "extreme_burst",
]

COMBINED_LABELS = {
    "moderate_random": "25 px + random 20%",
    "moderate_burst": "25 px + burst 20%",
    "high_random": "50 px + random 40%",
    "high_burst": "50 px + burst 40%",
    "extreme_random": "100 px + random 40%",
    "extreme_burst": "100 px + burst 40%",
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def build_final_tables(
    summary_path: Path,
    paired_path: Path,
    fold_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(require_file(summary_path))
    paired = pd.read_csv(require_file(paired_path))
    folds = pd.read_csv(require_file(fold_path))

    summary["display_label"] = summary["feature_set"].map(FINAL_LABELS)
    summary["order"] = summary["feature_set"].map(
        {name: index for index, name in enumerate(FINAL_ORDER)}
    )
    summary = summary.sort_values("order").drop(columns="order")

    summary_columns = [
        "feature_set",
        "display_label",
        "n_folds",
        "accuracy_subject_mean",
        "macro_f1_subject_mean",
        "macro_f1_subject_sd",
        "roc_auc_subject_mean",
        "pr_auc_nr_subject_mean",
        "macro_f1_pooled_oof",
    ]
    summary[summary_columns].to_csv(
        output_dir / "table01_final_nested_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        output_dir / "table02_final_nested_paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subject_table = folds[
        folds["feature_set"].isin(SUBJECT_LABELS)
    ][
        ["held_out_subject", "feature_set", "macro_f1"]
    ].copy()
    subject_table["display_label"] = subject_table["feature_set"].map(
        SUBJECT_LABELS
    )
    subject_wide = subject_table.pivot(
        index="held_out_subject",
        columns="display_label",
        values="macro_f1",
    ).reset_index()
    subject_wide.to_csv(
        output_dir / "table04_subject_level_final_macro_f1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary, paired, subject_wide


def plot_final_comparison(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_data = summary.set_index("feature_set").loc[FINAL_ORDER].reset_index()

    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(plot_data))
    axis.bar(
        positions,
        plot_data["macro_f1_subject_mean"],
        yerr=plot_data["macro_f1_subject_sd"],
        capsize=5,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [FINAL_LABELS[name] for name in plot_data["feature_set"]]
    )
    axis.set_ylabel("Subject-mean Macro F1")
    axis.set_title("Clean common-trial comparison: 16 subjects, 11,324 trials")
    axis.set_ylim(0.25, 0.80)

    for position, value in zip(
        positions,
        plot_data["macro_f1_subject_mean"],
    ):
        axis.text(
            position,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    save_figure(output_dir / "fig01_final_common16_macro_f1.png")


def plot_downsampling(
    path: Path,
    output_dir: Path,
) -> None:
    data = pd.read_csv(require_file(path))
    data = data[data["feature_set"].eq("strict_rate_invariant")].copy()

    figure, axis = plt.subplots(figsize=(8, 5))
    for protocol, group in data.groupby("protocol"):
        group = group.sort_values("sampling_rate_hz")
        axis.plot(
            group["sampling_rate_hz"],
            group["macro_f1_subject_mean"],
            marker="o",
            label=protocol,
        )

    axis.set_xscale("log")
    axis.set_xticks([15, 30, 60, 125, 250, 500])
    axis.set_xticklabels(["15", "30", "60", "125", "250", "500"])
    axis.set_xlabel("Sampling rate (Hz)")
    axis.set_ylabel("Subject-mean Macro F1")
    axis.set_title("Sampling-rate robustness")
    axis.legend()
    axis.grid(True, alpha=0.3)

    save_figure(output_dir / "fig02_downsampling_macro_f1.png")


def plot_noise_augmentation(
    path: Path,
    output_dir: Path,
) -> None:
    data = pd.read_csv(require_file(path))

    figure, axis = plt.subplots(figsize=(8, 5))
    for protocol, group in data.groupby("protocol"):
        group = group.sort_values("noise_sigma_px")
        axis.plot(
            group["noise_sigma_px"],
            group["macro_f1_subject_mean"],
            marker="o",
            label=protocol,
        )

    axis.set_xlabel("Gaussian coordinate noise σ (pixels)")
    axis.set_ylabel("Subject-mean Macro F1")
    axis.set_title("Coordinate-noise augmentation at 30 Hz")
    axis.legend()
    axis.grid(True, alpha=0.3)

    save_figure(
        output_dir / "fig03_coordinate_noise_augmentation_macro_f1.png"
    )


def plot_missingness(
    path: Path,
    output_dir: Path,
) -> None:
    data = pd.read_csv(require_file(path))
    data = data[
        data["feature_set"].eq("strict_content_only")
        & data["protocol"].eq("train_clean_test_missing")
    ].copy()

    clean_value = float(
        data.loc[
            data["missingness_mode"].eq("clean"),
            "macro_f1_subject_mean",
        ].iloc[0]
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    for mode in ["random", "burst"]:
        group = data[data["missingness_mode"].eq(mode)].copy()
        clean_row = pd.DataFrame(
            {
                "additional_missing_percent": [0.0],
                "macro_f1_subject_mean": [clean_value],
            }
        )
        group = pd.concat(
            [
                clean_row,
                group[
                    [
                        "additional_missing_percent",
                        "macro_f1_subject_mean",
                    ]
                ],
            ],
            ignore_index=True,
        ).sort_values("additional_missing_percent")

        axis.plot(
            group["additional_missing_percent"],
            group["macro_f1_subject_mean"],
            marker="o",
            label=mode,
        )

    axis.set_xlabel("Additional missingness (%)")
    axis.set_ylabel("Subject-mean Macro F1")
    axis.set_title("Missingness robustness without quality features")
    axis.legend()
    axis.grid(True, alpha=0.3)

    save_figure(
        output_dir / "fig04_missingness_content_only_macro_f1.png"
    )


def plot_combined_degradation(
    path: Path,
    output_dir: Path,
) -> None:
    data = pd.read_csv(require_file(path))
    data = data[
        data["feature_set"].eq("strict_content_only")
    ].copy()

    pivot = data.pivot(
        index="combined_condition",
        columns="protocol",
        values="macro_f1_subject_mean",
    ).loc[COMBINED_ORDER]

    protocols = [
        "clean_train",
        "same_condition_augmented",
        "mixed_combined_augmented",
    ]
    positions = np.arange(len(pivot))
    width = 0.25

    figure, axis = plt.subplots(figsize=(11, 5.5))
    for offset_index, protocol in enumerate(protocols):
        axis.bar(
            positions + (offset_index - 1) * width,
            pivot[protocol],
            width=width,
            label=protocol,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [COMBINED_LABELS[name] for name in pivot.index],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Subject-mean Macro F1")
    axis.set_title("Combined coordinate-noise and missingness degradation")
    axis.legend()
    axis.set_ylim(0.25, 0.62)

    save_figure(
        output_dir / "fig05_combined_degradation_content_only_macro_f1.png"
    )


def plot_subject_level(
    subject_wide: pd.DataFrame,
    output_dir: Path,
) -> None:
    model_columns = [
        "30 Hz clean",
        "30 Hz mixed augmented",
        "Official sent_gaze_sacc",
    ]
    ordered = subject_wide.sort_values(
        "Official sent_gaze_sacc"
    ).reset_index(drop=True)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(ordered))

    for column in model_columns:
        axis.plot(
            positions,
            ordered[column],
            marker="o",
            label=column,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        ordered["held_out_subject"],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Held-out subject")
    axis.set_ylabel("Macro F1")
    axis.set_title("Subject-level final comparison")
    axis.legend()
    axis.grid(True, alpha=0.3)

    save_figure(output_dir / "fig06_subject_level_final_macro_f1.png")


def build_key_results_table(
    rate_path: Path,
    noise_path: Path,
    missing_path: Path,
    combined_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    rate = pd.read_csv(require_file(rate_path))
    rate = rate[
        rate["feature_set"].eq("strict_rate_invariant")
    ]
    for _, record in rate.iterrows():
        rows.append(
            {
                "experiment": "sampling_rate",
                "condition": (
                    f"{record['protocol']} / "
                    f"{int(record['sampling_rate_hz'])} Hz"
                ),
                "macro_f1_subject_mean": record[
                    "macro_f1_subject_mean"
                ],
                "roc_auc_subject_mean": record[
                    "roc_auc_subject_mean"
                ],
                "pr_auc_nr_subject_mean": record[
                    "pr_auc_nr_subject_mean"
                ],
            }
        )

    noise = pd.read_csv(require_file(noise_path))
    for _, record in noise.iterrows():
        rows.append(
            {
                "experiment": "coordinate_noise",
                "condition": (
                    f"{record['protocol']} / "
                    f"σ={record['noise_sigma_px']:g}px"
                ),
                "macro_f1_subject_mean": record[
                    "macro_f1_subject_mean"
                ],
                "roc_auc_subject_mean": record[
                    "roc_auc_subject_mean"
                ],
                "pr_auc_nr_subject_mean": record[
                    "pr_auc_nr_subject_mean"
                ],
            }
        )

    missing = pd.read_csv(require_file(missing_path))
    missing = missing[
        missing["feature_set"].eq("strict_content_only")
    ]
    for _, record in missing.iterrows():
        rows.append(
            {
                "experiment": "missingness",
                "condition": (
                    f"{record['protocol']} / "
                    f"{record['missingness_mode']} / "
                    f"{record['additional_missing_percent']:g}%"
                ),
                "macro_f1_subject_mean": record[
                    "macro_f1_subject_mean"
                ],
                "roc_auc_subject_mean": record[
                    "roc_auc_subject_mean"
                ],
                "pr_auc_nr_subject_mean": record[
                    "pr_auc_nr_subject_mean"
                ],
            }
        )

    combined = pd.read_csv(require_file(combined_path))
    combined = combined[
        combined["feature_set"].eq("strict_content_only")
    ]
    for _, record in combined.iterrows():
        rows.append(
            {
                "experiment": "combined_degradation",
                "condition": (
                    f"{record['combined_condition']} / "
                    f"{record['protocol']}"
                ),
                "macro_f1_subject_mean": record[
                    "macro_f1_subject_mean"
                ],
                "roc_auc_subject_mean": record[
                    "roc_auc_subject_mean"
                ],
                "pr_auc_nr_subject_mean": record[
                    "pr_auc_nr_subject_mean"
                ],
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(
        output_dir / "table03_key_robustness_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def write_markdown_summary(
    final_summary: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> None:
    values = final_summary.set_index("feature_set")
    official = values.loc[
        "official_sent_gaze_sacc_nested",
        "macro_f1_subject_mean",
    ]
    clean = values.loc[
        "proposed_30hz_content_clean_nested",
        "macro_f1_subject_mean",
    ]
    augmented = values.loc[
        "proposed_30hz_content_mixed_weighted_nested",
        "macro_f1_subject_mean",
    ]
    augmented_sd = values.loc[
        "proposed_30hz_content_mixed_weighted_nested",
        "macro_f1_subject_sd",
    ]

    macro = paired[paired["metric"].eq("macro_f1")].set_index(
        ["candidate", "reference"]
    )
    comparison = macro.loc[
        (
            "proposed_30hz_content_mixed_weighted_nested",
            "official_sent_gaze_sacc_nested",
        )
    ]
    augmentation = macro.loc[
        (
            "proposed_30hz_content_mixed_weighted_nested",
            "proposed_30hz_content_clean_nested",
        )
    ]

    text = f"""# Final result snapshot

## Confirmatory clean common-trial comparison

- Official sent_gaze_sacc: Macro F1 = {official:.4f}
- Proposed 30 Hz content-only, clean training: Macro F1 = {clean:.4f}
- Proposed 30 Hz content-only, weighted mixed-degradation training:
  Macro F1 = {augmented:.4f}, subject SD = {augmented_sd:.4f}

The weighted mixed-degradation proposal was {comparison['mean_difference']:.4f}
below the official feature baseline on average. Its 95% bootstrap confidence
interval for the subject-mean difference was
[{comparison['mean_difference_bootstrap_ci95_low']:.4f},
 {comparison['mean_difference_bootstrap_ci95_high']:.4f}].
The paired Wilcoxon-Holm result was p =
{comparison['p_holm_within_metric']:.4f}. This does not establish equivalence.

Weighted mixed-degradation training changed clean Macro F1 by
{augmentation['mean_difference']:+.4f} relative to clean-only training, with a
95% bootstrap interval of
[{augmentation['mean_difference_bootstrap_ci95_low']:.4f},
 {augmentation['mean_difference_bootstrap_ci95_high']:.4f}].

## Final interpretation

The official fixation/saccade representation had the highest average clean
performance. The proposed representation used 30 Hz continuous gaze without
fixation or saccade detection. Mixed degradation training preserved clean
thresholded Macro F1 while providing robustness to coordinate noise and
missingness. ROC-AUC and PR-AUC should be reported separately because mixed
augmentation reduced the clean ranking metrics relative to clean-only training.
"""

    (output_dir / "final_results_summary.md").write_text(
        text,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nested-summary",
        default=(
            "outputs/nested_final_common16/"
            "nested_final_summary.csv"
        ),
    )
    parser.add_argument(
        "--nested-paired",
        default=(
            "outputs/nested_final_common16/"
            "nested_final_paired_comparisons.csv"
        ),
    )
    parser.add_argument(
        "--nested-folds",
        default=(
            "outputs/nested_final_common16/"
            "nested_final_fold_metrics.csv"
        ),
    )
    parser.add_argument(
        "--rate-summary",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_summary.csv"
        ),
    )
    parser.add_argument(
        "--noise-summary",
        default=(
            "outputs/noise_augmentation_30hz/"
            "noise_augmentation_summary.csv"
        ),
    )
    parser.add_argument(
        "--missingness-summary",
        default=(
            "outputs/missingness_30hz/"
            "missingness_summary.csv"
        ),
    )
    parser.add_argument(
        "--combined-summary",
        default=(
            "outputs/combined_degradation_30hz/"
            "combined_degradation_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final_reporting",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, paired, subject_wide = build_final_tables(
        summary_path=Path(args.nested_summary),
        paired_path=Path(args.nested_paired),
        fold_path=Path(args.nested_folds),
        output_dir=output_dir,
    )

    build_key_results_table(
        rate_path=Path(args.rate_summary),
        noise_path=Path(args.noise_summary),
        missing_path=Path(args.missingness_summary),
        combined_path=Path(args.combined_summary),
        output_dir=output_dir,
    )

    plot_final_comparison(summary, output_dir)
    plot_downsampling(Path(args.rate_summary), output_dir)
    plot_noise_augmentation(Path(args.noise_summary), output_dir)
    plot_missingness(Path(args.missingness_summary), output_dir)
    plot_combined_degradation(Path(args.combined_summary), output_dir)
    plot_subject_level(subject_wide, output_dir)
    write_markdown_summary(summary, paired, output_dir)

    print(f"Written reporting snapshot to: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
