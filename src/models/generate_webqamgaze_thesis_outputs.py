#!/usr/bin/env python
"""Generate thesis-ready external-validation tables, figures, and Japanese text.

Inputs
------
Stage 4:
- external_pooled_metrics.csv
- external_summary.csv
- compact_pairwise_subject_tests.csv
- compact_pairwise_bootstrap_differences.csv
- feature_transport_audit.csv

Stage 5:
- model_score_polarity_audit.csv
- model_feature_contribution_summary.csv
- transport_contribution_merged.csv

Outputs
-------
Tables:
- thesis_external_validation_results.csv
- thesis_external_validation_results.md
- thesis_compact_pairwise_tests.csv
- thesis_compact_pairwise_tests.md
- thesis_key_transport_shifts.csv
- thesis_key_feature_contributions.csv

Figures:
- figure_external_validation_metrics.png
- figure_score_orientation.png
- figure_transport_shift_top20.png
- figure_compact_feature_contributions.png
- figure_full_dynamic_feature_contributions.png
- figure_official_feature_contributions.png

Text:
- thesis_external_validation_results_ja.md
- stage6_manifest.json

This stage does not modify models, predictions, thresholds, or Stage 4/5 outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "dummy_reference",
    "rawstat_all",
    "soft_full_all",
    "soft_compact_distribution_membership",
    "official_sent_gaze_sacc",
]

DISPLAY_NAMES = {
    "dummy_reference": "Dummy",
    "rawstat_all": "RawStat",
    "soft_full_all": "Full Soft-AOI",
    "soft_compact_distribution_membership": "Compact Soft-AOI",
    "official_sent_gaze_sacc": "Transferred official 9",
}

METRIC_ORDER = [
    "macro_f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc_nr",
]

METRIC_LABELS = {
    "macro_f1": "Macro F1",
    "balanced_accuracy": "Balanced Accuracy",
    "roc_auc": "ROC-AUC",
    "pr_auc_nr": "PR-AUC (NR)",
}

FEATURE_LABELS = {
    "speed_median_px_s": "Median gaze speed",
    "soft_aoi_transition_rate_per_sec": "Transition rate",
    "soft_aoi_line_backward_fraction": "Line backward fraction",
    "soft_aoi_stationary_transition_fraction": "Stationary transition fraction",
    "soft_aoi_line_forward_fraction": "Line forward fraction",
    "soft_aoi_regression_events_per_sec": "Regression events/sec",
    "soft_aoi_forward_transition_fraction": "Forward transition fraction",
    "soft_aoi_backward_transition_fraction": "Backward transition fraction",
    "soft_aoi_sample_entropy_mean": "Sample entropy mean",
    "soft_aoi_sample_entropy_median": "Sample entropy median",
    "soft_aoi_occupancy_effective_word_fraction": "Effective word fraction",
    "soft_aoi_occupancy_entropy_norm": "Occupancy entropy",
    "soft_aoi_progress_std": "Progress SD",
    "soft_aoi_active_rate": "Active rate",
    "soft_aoi_occupancy_top1_share": "Top-1 occupancy share",
    "soft_aoi_qmax_median": "Median qmax",
    "soft_aoi_qmax_mean": "Mean qmax",
    "soft_aoi_dominant_word_coverage": "Dominant word coverage",
    "soft_aoi_dominant_line_coverage": "Dominant line coverage",
    "soft_aoi_progress_mean": "Progress mean",
    "soft_aoi_line_position_mean": "Line position mean",
    "soft_aoi_nearest_distance_q95_word_heights": "Nearest distance q95",
    "soft_aoi_valid_rate": "Valid rate",
    "official_feature_0": "Omission rate",
    "official_feature_1": "Fixation number",
    "official_feature_2": "Reading speed",
    "official_feature_3": "Mean transferred duration",
    "official_feature_4": "Max transferred velocity",
    "official_feature_5": "Mean transferred velocity",
    "official_feature_6": "Max transferred duration",
    "official_feature_7": "Mean amplitude",
    "official_feature_8": "Max amplitude",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def format_p(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            str(value).replace("|", r"\|")
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    order_map = {
        feature_set: index
        for index, feature_set in enumerate(MODEL_ORDER)
    }
    result["_order"] = result["feature_set"].map(order_map)
    return result.sort_values("_order").drop(columns="_order")


def build_results_table(
    pooled: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    pooled = ordered(pooled)
    summary = ordered(summary)

    merged = pooled.merge(
        summary[
            [
                "feature_set",
                "macro_f1_subject_mean",
                "balanced_accuracy_subject_mean",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ],
        on="feature_set",
        how="left",
        validate="one_to_one",
    )

    result = pd.DataFrame(
        {
            "Condition": merged["feature_set"].map(DISPLAY_NAMES),
            "Features": merged["n_input_features"].astype(int),
            "Macro F1": merged["macro_f1"],
            "Balanced Accuracy": merged["balanced_accuracy"],
            "ROC-AUC": merged["roc_auc"],
            "PR-AUC (NR)": merged["pr_auc_nr"],
            "Predicted NR rate": merged["predicted_nr_rate"],
            "Subject-mean Macro F1": merged[
                "macro_f1_subject_mean"
            ],
            "Subject-mean Balanced Accuracy": merged[
                "balanced_accuracy_subject_mean"
            ],
            "Subject-mean ROC-AUC": merged[
                "roc_auc_subject_mean"
            ],
            "Subject-mean PR-AUC (NR)": merged[
                "pr_auc_nr_subject_mean"
            ],
        }
    )
    return result


def make_results_markdown(result: pd.DataFrame) -> str:
    display = result[
        [
            "Condition",
            "Features",
            "Macro F1",
            "Balanced Accuracy",
            "ROC-AUC",
            "PR-AUC (NR)",
            "Predicted NR rate",
        ]
    ].copy()
    for column in display.columns[2:]:
        display[column] = display[column].map(
            lambda value: format_float(value, 3)
        )
    return markdown_table(display)


def build_pairwise_table(
    subject_tests: pd.DataFrame,
    bootstrap_differences: pd.DataFrame,
) -> pd.DataFrame:
    subject = subject_tests[
        subject_tests["metric"].isin(
            ["macro_f1", "balanced_accuracy"]
        )
    ].copy()

    bootstrap = bootstrap_differences[
        bootstrap_differences["metric"].isin(
            ["macro_f1", "balanced_accuracy"]
        )
    ].copy()

    merged = subject.merge(
        bootstrap[
            [
                "metric",
                "reference_feature_set",
                "candidate_feature_set",
                "mean_difference_candidate_minus_reference",
                "ci95_lower",
                "ci95_upper",
                "bootstrap_probability_candidate_gt_reference",
            ]
        ],
        on=[
            "metric",
            "reference_feature_set",
            "candidate_feature_set",
        ],
        how="left",
        suffixes=("_subject", "_bootstrap"),
        validate="one_to_one",
    )

    result = pd.DataFrame(
        {
            "Metric": merged["metric"].map(METRIC_LABELS),
            "Reference": merged[
                "reference_feature_set"
            ].map(DISPLAY_NAMES),
            "Candidate": merged[
                "candidate_feature_set"
            ].map(DISPLAY_NAMES),
            "Subject mean difference": merged[
                "mean_difference_candidate_minus_reference_subject"
            ],
            "Wins": merged["wins"].astype(int),
            "Ties": merged["ties"].astype(int),
            "Losses": merged["losses"].astype(int),
            "Wilcoxon p": merged["p_value"],
            "Holm p": merged["p_holm_within_metric"],
            "Bootstrap mean difference": merged[
                "mean_difference_candidate_minus_reference_bootstrap"
            ],
            "CI95 lower": merged["ci95_lower"],
            "CI95 upper": merged["ci95_upper"],
            "P(candidate > reference)": merged[
                "bootstrap_probability_candidate_gt_reference"
            ],
        }
    )
    return result


def make_pairwise_markdown(result: pd.DataFrame) -> str:
    display = result[
        [
            "Metric",
            "Reference",
            "Candidate",
            "Subject mean difference",
            "Holm p",
            "Bootstrap mean difference",
            "CI95 lower",
            "CI95 upper",
        ]
    ].copy()
    for column in (
        "Subject mean difference",
        "Bootstrap mean difference",
        "CI95 lower",
        "CI95 upper",
    ):
        display[column] = display[column].map(
            lambda value: format_float(value, 3)
        )
    display["Holm p"] = display["Holm p"].map(format_p)
    return markdown_table(display)


def make_external_metrics_figure(
    pooled: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = ordered(pooled)
    labels = frame["feature_set"].map(DISPLAY_NAMES).tolist()
    x = np.arange(len(labels))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for index, metric in enumerate(METRIC_ORDER):
        offsets = x + (index - 1.5) * width
        plt.bar(
            offsets,
            frame[metric].to_numpy(dtype=float),
            width,
            label=METRIC_LABELS[metric],
        )
    plt.axhline(0.5, linewidth=1, linestyle="--")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0.25, 0.70)
    plt.ylabel("Score")
    plt.title("WebQAmGaze external validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_score_orientation_figure(
    polarity: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = ordered(polarity)
    labels = frame["feature_set"].map(DISPLAY_NAMES).tolist()
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width / 2,
        frame["pooled_auc"].to_numpy(dtype=float),
        width,
        label="Ordinary NR-positive AUC",
    )
    plt.bar(
        x + width / 2,
        frame["pooled_reversed_auc"].to_numpy(dtype=float),
        width,
        label="Score-reversed AUC",
    )
    plt.axhline(0.5, linewidth=1, linestyle="--")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0.25, 0.80)
    plt.ylabel("ROC-AUC")
    plt.title("Decision-score orientation across datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_transport_shift_figure(
    transport: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> pd.DataFrame:
    frame = transport.copy()
    frame["abs_shift"] = pd.to_numeric(
        frame["external_median_standardized_by_zuco"],
        errors="coerce",
    ).abs()
    frame = (
        frame.sort_values("abs_shift", ascending=False)
        .head(top_n)
        .sort_values(
            "external_median_standardized_by_zuco",
            ascending=True,
        )
    )
    labels = [
        f"{DISPLAY_NAMES.get(fs, fs)}: "
        f"{FEATURE_LABELS.get(name, name)}"
        for fs, name in zip(
            frame["feature_set"],
            frame["feature_name"],
        )
    ]
    values = frame[
        "external_median_standardized_by_zuco"
    ].to_numpy(dtype=float)

    plt.figure(figsize=(12, 9))
    y = np.arange(len(frame))
    plt.barh(y, values)
    plt.axvline(0, linewidth=1)
    plt.yticks(y, labels)
    plt.xlabel("External median standardized by ZuCo training distribution")
    plt.title("Largest feature-transport shifts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return frame


def contribution_subset(
    contribution: pd.DataFrame,
    feature_set: str,
    names: list[str] | None = None,
    top_n: int = 12,
) -> pd.DataFrame:
    frame = contribution[
        contribution["feature_set"].eq(feature_set)
    ].copy()
    if names is not None:
        frame = frame[
            frame["source_feature_name"].isin(names)
        ]
    frame = frame.sort_values(
        "mean_abs_contribution",
        ascending=False,
    ).head(top_n)
    return frame.sort_values(
        "mean_contribution",
        ascending=True,
    )


def make_contribution_figure(
    frame: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    labels = [
        FEATURE_LABELS.get(name, name)
        for name in frame["source_feature_name"]
    ]
    values = frame["mean_contribution"].to_numpy(dtype=float)

    plt.figure(figsize=(10, max(5, 0.48 * len(frame))))
    y = np.arange(len(frame))
    plt.barh(y, values)
    plt.axvline(0, linewidth=1)
    plt.yticks(y, labels)
    plt.xlabel("Mean contribution to NR decision score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def key_contribution_table(
    contribution: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for feature_set in MODEL_ORDER[1:]:
        subset = contribution[
            contribution["feature_set"].eq(feature_set)
        ].sort_values(
            "mean_abs_contribution",
            ascending=False,
        ).head(10)
        for row in subset.itertuples(index=False):
            rows.append(
                {
                    "Condition": DISPLAY_NAMES[feature_set],
                    "Feature": FEATURE_LABELS.get(
                        row.source_feature_name,
                        row.source_feature_name,
                    ),
                    "Coefficient": row.coefficient,
                    "External standardized median": (
                        row.external_standardized_median
                    ),
                    "Mean contribution": row.mean_contribution,
                    "Mean absolute contribution": (
                        row.mean_abs_contribution
                    ),
                    "NR-IS contribution difference": (
                        row.nr_minus_is_mean_contribution
                    ),
                    "Contribution ROC-AUC": (
                        row.auc_nr_decision_contribution
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_japanese_results_text(
    result: pd.DataFrame,
    pairwise: pd.DataFrame,
    polarity: pd.DataFrame,
    transport: pd.DataFrame,
    contribution: pd.DataFrame,
) -> str:
    lookup = result.set_index("Condition")
    dummy = lookup.loc["Dummy"]
    raw = lookup.loc["RawStat"]
    full = lookup.loc["Full Soft-AOI"]
    compact = lookup.loc["Compact Soft-AOI"]
    official = lookup.loc["Transferred official 9"]

    raw_pair = pairwise[
        pairwise["Reference"].eq("RawStat")
        & pairwise["Metric"].eq("Macro F1")
    ].iloc[0]
    full_pair = pairwise[
        pairwise["Reference"].eq("Full Soft-AOI")
        & pairwise["Metric"].eq("Macro F1")
    ].iloc[0]
    official_pair = pairwise[
        pairwise["Reference"].eq("Transferred official 9")
        & pairwise["Metric"].eq("Macro F1")
    ].iloc[0]

    polarity_lookup = polarity.set_index(
        polarity["feature_set"].map(DISPLAY_NAMES)
    )

    top_shift = (
        transport.assign(
            abs_shift=pd.to_numeric(
                transport[
                    "external_median_standardized_by_zuco"
                ],
                errors="coerce",
            ).abs()
        )
        .sort_values("abs_shift", ascending=False)
        .iloc[0]
    )

    compact_top = (
        contribution[
            contribution["feature_set"].eq(
                "soft_compact_distribution_membership"
            )
        ]
        .sort_values(
            "mean_abs_contribution",
            ascending=False,
        )
        .iloc[0]
    )

    official_success = (
        contribution[
            contribution["feature_set"].eq(
                "official_sent_gaze_sacc"
            )
        ]
        .sort_values(
            "auc_nr_decision_contribution",
            ascending=False,
        )
        .head(3)
    )
    official_success_names = "、".join(
        FEATURE_LABELS.get(name, name)
        for name in official_success["source_feature_name"]
    )

    return f"""# WebQAmGaze外部検証：結果と考察

## 外部検証設定

ZuCo Common16で学習・固定した5条件を、WebQAmGazeの131名・1,349試行へ再学習なしで適用した。ラベルはZuCoでTSR=0、NR=1、WebQAmGazeでIS=0、NR=1とし、NRを共通の正例とした。固定モデルにはZuCo学習データで推定した中央値補完、StandardScaler、LinearSVC（C=0.01）、判定閾値0.0を使用した。WebQAmGazeによる再学習、特徴選択、閾値調整および較正は行っていない。

## 外部検証結果

全試行を統合した評価では、DummyのMacro F1は{dummy['Macro F1']:.3f}、Balanced Accuracyは{dummy['Balanced Accuracy']:.3f}であった。RawStatはMacro F1={raw['Macro F1']:.3f}、Balanced Accuracy={raw['Balanced Accuracy']:.3f}、ROC-AUC={raw['ROC-AUC']:.3f}、PR-AUC={raw['PR-AUC (NR)']:.3f}であった。Full Soft-AOIはそれぞれ{full['Macro F1']:.3f}、{full['Balanced Accuracy']:.3f}、{full['ROC-AUC']:.3f}、{full['PR-AUC (NR)']:.3f}、Compact Soft-AOIは{compact['Macro F1']:.3f}、{compact['Balanced Accuracy']:.3f}、{compact['ROC-AUC']:.3f}、{compact['PR-AUC (NR)']:.3f}であった。転用イベントベース公式9特徴条件はMacro F1={official['Macro F1']:.3f}、Balanced Accuracy={official['Balanced Accuracy']:.3f}、ROC-AUC={official['ROC-AUC']:.3f}、PR-AUC={official['PR-AUC (NR)']:.3f}となり、5条件中で最も高かった。

Compact Soft-AOIはRawStatに対してMacro F1で{raw_pair['Subject mean difference']:+.3f}の被験者平均差を示し、Holm補正後p値は{format_p(raw_pair['Holm p'])}であった。被験者クラスターブートストラップによるMacro F1差の95%信頼区間は[{raw_pair['CI95 lower']:.3f}, {raw_pair['CI95 upper']:.3f}]であり、RawStatに対する相対的優位性が確認された。一方、Full Soft-AOIとの差は{full_pair['Subject mean difference']:+.3f}であり、Holm補正後p値は{format_p(full_pair['Holm p'])}であった。転用公式9特徴との差は{official_pair['Subject mean difference']:+.3f}であり、Compact Soft-AOIは同条件を下回った。

## スコア方向の診断

RawStat、Full Soft-AOI、Compact Soft-AOIの通常方向ROC-AUCはそれぞれ{polarity_lookup.loc['RawStat', 'pooled_auc']:.3f}、{polarity_lookup.loc['Full Soft-AOI', 'pooled_auc']:.3f}、{polarity_lookup.loc['Compact Soft-AOI', 'pooled_auc']:.3f}であり、スコア反転後は{polarity_lookup.loc['RawStat', 'pooled_reversed_auc']:.3f}、{polarity_lookup.loc['Full Soft-AOI', 'pooled_reversed_auc']:.3f}、{polarity_lookup.loc['Compact Soft-AOI', 'pooled_reversed_auc']:.3f}であった。ラベル監査ではNR=1の方向が全処理段階で一致していることが確認されているため、この傾向は数値ラベルの反転ではなく、ZuCoとWebQAmGazeの間で特徴と読書目的との関係が変化したことを示す。

## 特徴分布の移送差

最大の標準化分布差は、{DISPLAY_NAMES.get(top_shift['feature_set'], top_shift['feature_set'])}の{FEATURE_LABELS.get(top_shift['feature_name'], top_shift['feature_name'])}であり、WebQAmGaze中央値はZuCo学習分布に対して{top_shift['external_median_standardized_by_zuco']:+.2f}標準偏差ずれていた。RawStatでは視線速度中央値が大きく上昇し、Full Soft-AOIでは単語間・行間遷移率、前方・後方遷移率および回帰イベント率が大きく乖離した。これはWebGazerのフレーム間揺れが、実際の読書行動とは異なる擬似的な速度・AOI遷移として計数された可能性を示す。

Compact Soft-AOIで平均絶対寄与が最も大きかった特徴は{FEATURE_LABELS.get(compact_top['source_feature_name'], compact_top['source_feature_name'])}であり、NRとISの寄与差は{compact_top['nr_minus_is_mean_contribution']:+.3f}、単独寄与のROC-AUCは{compact_top['auc_nr_decision_contribution']:.3f}であった。時間順序特徴を除いたCompact設計はFullで生じた極端な動的特徴寄与を抑制したが、占有分布やSoft-AOI不確実性に関する特徴でもタスクとの対応関係が完全には維持されなかった。

## 転用公式9特徴の解釈

転用公式9特徴条件では、{official_success_names}が比較的高いNR識別方向を示した。一方、WebQAmGazeではフィルタ済みfixation間の正の時間差を転用saccadeとして用いており、EyeLinkで検出されたZuCo公式saccadeイベントとは同一ではない。したがって、この条件の性能は「ZuCo公式特徴の完全再現」ではなく、「WebQAmGazeのfixation情報から構成した転用イベントベース特徴」の有効性として解釈する必要がある。

## 結論

Compact Soft-AOIはRawStatよりもセンサー差に対して相対的に頑健であり、時間順序特徴を除く設計の妥当性は部分的に支持された。しかし、ROC-AUCおよびPR-AUCはランダム基準を超えず、絶対的な外部汎化は達成されなかった。主因として、Webカメラ視線推定に伴う位置揺れ、ZuCoの文単位とWebQAmGazeの長文単位との差、および主要特徴と読書目的との関係変化が考えられる。転用イベントベース公式9特徴条件は最良性能を示したが、EyeLink公式特徴とは非同一であるため、独立した転用ベースラインとして報告する。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage4"
        ),
    )
    parser.add_argument(
        "--stage5-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage5_diagnostics"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage6_thesis"
        ),
    )
    args = parser.parse_args()

    stage4 = args.stage4_dir.resolve()
    stage5 = args.stage5_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "pooled": require_file(
            stage4 / "external_pooled_metrics.csv"
        ),
        "summary": require_file(
            stage4 / "external_summary.csv"
        ),
        "subject_tests": require_file(
            stage4 / "compact_pairwise_subject_tests.csv"
        ),
        "bootstrap_differences": require_file(
            stage4
            / "compact_pairwise_bootstrap_differences.csv"
        ),
        "transport": require_file(
            stage4 / "feature_transport_audit.csv"
        ),
        "polarity": require_file(
            stage5 / "model_score_polarity_audit.csv"
        ),
        "contribution": require_file(
            stage5
            / "model_feature_contribution_summary.csv"
        ),
        "transport_contribution": require_file(
            stage5 / "transport_contribution_merged.csv"
        ),
    }

    pooled = pd.read_csv(paths["pooled"])
    summary = pd.read_csv(paths["summary"])
    subject_tests = pd.read_csv(paths["subject_tests"])
    bootstrap_differences = pd.read_csv(
        paths["bootstrap_differences"]
    )
    transport = pd.read_csv(paths["transport"])
    polarity = pd.read_csv(paths["polarity"])
    contribution = pd.read_csv(paths["contribution"])

    results_table = build_results_table(
        pooled,
        summary,
    )
    pairwise_table = build_pairwise_table(
        subject_tests,
        bootstrap_differences,
    )
    key_contributions = key_contribution_table(
        contribution
    )

    output_paths = {
        "results_csv": (
            output / "thesis_external_validation_results.csv"
        ),
        "results_md": (
            output / "thesis_external_validation_results.md"
        ),
        "pairwise_csv": (
            output / "thesis_compact_pairwise_tests.csv"
        ),
        "pairwise_md": (
            output / "thesis_compact_pairwise_tests.md"
        ),
        "transport_csv": (
            output / "thesis_key_transport_shifts.csv"
        ),
        "contribution_csv": (
            output
            / "thesis_key_feature_contributions.csv"
        ),
        "figure_metrics": (
            output / "figure_external_validation_metrics.png"
        ),
        "figure_orientation": (
            output / "figure_score_orientation.png"
        ),
        "figure_transport": (
            output / "figure_transport_shift_top20.png"
        ),
        "figure_compact": (
            output
            / "figure_compact_feature_contributions.png"
        ),
        "figure_full": (
            output
            / "figure_full_dynamic_feature_contributions.png"
        ),
        "figure_official": (
            output
            / "figure_official_feature_contributions.png"
        ),
        "results_text": (
            output / "thesis_external_validation_results_ja.md"
        ),
        "manifest": output / "stage6_manifest.json",
    }

    results_table.to_csv(
        output_paths["results_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    output_paths["results_md"].write_text(
        make_results_markdown(results_table),
        encoding="utf-8",
    )

    pairwise_table.to_csv(
        output_paths["pairwise_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    output_paths["pairwise_md"].write_text(
        make_pairwise_markdown(pairwise_table),
        encoding="utf-8",
    )

    key_contributions.to_csv(
        output_paths["contribution_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    make_external_metrics_figure(
        pooled,
        output_paths["figure_metrics"],
    )
    make_score_orientation_figure(
        polarity,
        output_paths["figure_orientation"],
    )
    top_transport = make_transport_shift_figure(
        transport,
        output_paths["figure_transport"],
        top_n=20,
    )
    top_transport.to_csv(
        output_paths["transport_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    compact_frame = contribution_subset(
        contribution,
        "soft_compact_distribution_membership",
        top_n=12,
    )
    make_contribution_figure(
        compact_frame,
        "Compact Soft-AOI: dominant score contributions",
        output_paths["figure_compact"],
    )

    dynamic_names = [
        "soft_aoi_transition_rate_per_sec",
        "soft_aoi_line_backward_fraction",
        "soft_aoi_stationary_transition_fraction",
        "soft_aoi_line_forward_fraction",
        "soft_aoi_regression_events_per_sec",
        "soft_aoi_forward_transition_fraction",
        "soft_aoi_backward_transition_fraction",
    ]
    full_frame = contribution_subset(
        contribution,
        "soft_full_all",
        names=dynamic_names,
        top_n=len(dynamic_names),
    )
    make_contribution_figure(
        full_frame,
        "Full Soft-AOI: dynamic-feature contributions",
        output_paths["figure_full"],
    )

    official_frame = contribution_subset(
        contribution,
        "official_sent_gaze_sacc",
        top_n=9,
    )
    make_contribution_figure(
        official_frame,
        "Transferred event-based official features",
        output_paths["figure_official"],
    )

    japanese_text = build_japanese_results_text(
        results_table,
        pairwise_table,
        polarity,
        transport,
        contribution,
    )
    output_paths["results_text"].write_text(
        japanese_text,
        encoding="utf-8",
    )

    manifest = {
        "stage": (
            "webqamgaze_external_validation_stage6_thesis_outputs"
        ),
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "stage4_inputs": str(stage4),
        "stage5_inputs": str(stage5),
        "outputs": {
            name: str(path)
            for name, path in output_paths.items()
            if name != "manifest"
        },
        "notes": [
            "All figures and tables are descriptive outputs from frozen Stage 4/5 results.",
            "No model or metric was recomputed with altered labels or thresholds.",
            "The official 9-feature condition must be reported as transferred event-based and nonidentical to EyeLink official events.",
        ],
    }
    output_paths["manifest"].write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE STAGE 6 THESIS OUTPUTS ===")
    print("Primary results:")
    print(
        results_table[
            [
                "Condition",
                "Macro F1",
                "Balanced Accuracy",
                "ROC-AUC",
                "PR-AUC (NR)",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print("Written:")
    for path in output_paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
