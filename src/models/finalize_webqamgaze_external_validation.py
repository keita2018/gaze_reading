#!/usr/bin/env python
"""Finalize the WebQAmGaze external-validation thesis package.

This stage integrates:
- Stage 6 primary five-condition thesis outputs
- Stage 7 secondary Common16-vs-All18 Compact analysis

It preserves the reporting rule:
- the primary five-condition Common16 comparison remains the main result;
- All18 Compact is reported only as a separate secondary analysis.

Additional work performed here:
- Holm correction across the four Stage 7 paired subject tests;
- final primary and secondary markdown tables;
- a single overview figure;
- a final Japanese results/discussion section;
- a reporting checklist to prevent overclaiming.

No model, prediction, threshold, label, or metric is changed.
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


PRIMARY_DISPLAY = {
    "Dummy": "Dummy",
    "RawStat": "RawStat",
    "Full Soft-AOI": "Full Soft-AOI",
    "Compact Soft-AOI": "Compact Soft-AOI",
    "Transferred official 9": "Transferred official 9",
}

SECONDARY_DISPLAY = {
    "Common16 Compact": "Common16 Compact",
    "All18 Compact": "All18 Compact",
}

METRIC_COLUMNS = [
    "Macro F1",
    "Balanced Accuracy",
    "ROC-AUC",
    "PR-AUC (NR)",
]

STAGE7_METRIC_MAP = {
    "macro_f1": "Macro F1",
    "balanced_accuracy": "Balanced Accuracy",
    "roc_auc": "ROC-AUC",
    "pr_auc_nr": "PR-AUC (NR)",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def holm_adjust(p_values: list[float]) -> list[float]:
    array = np.asarray(p_values, dtype=float)
    n = len(array)
    order = np.argsort(array)
    adjusted = np.empty(n, dtype=float)

    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * array[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def format_number(value: Any, digits: int = 3) -> str:
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


def prepare_primary_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "Condition",
        "Features",
        "Macro F1",
        "Balanced Accuracy",
        "ROC-AUC",
        "PR-AUC (NR)",
        "Predicted NR rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Primary result table lacks columns: {missing}"
        )
    return frame[
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


def prepare_secondary_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "display_name",
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
        "predicted_nr_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Secondary pooled table lacks columns: {missing}"
        )
    return pd.DataFrame(
        {
            "Condition": frame["display_name"],
            "Training scope": [
                (
                    "ZuCo Common16: 16 subjects, "
                    "11,324 trials"
                )
                if name == "Common16 Compact"
                else (
                    "ZuCo All18: 18 subjects, "
                    "13,302 trials"
                )
                for name in frame["display_name"]
            ],
            "Macro F1": frame["macro_f1"],
            "Balanced Accuracy": frame[
                "balanced_accuracy"
            ],
            "ROC-AUC": frame["roc_auc"],
            "PR-AUC (NR)": frame["pr_auc_nr"],
            "Predicted NR rate": frame[
                "predicted_nr_rate"
            ],
        }
    )


def prepare_corrected_tests(
    paired_path: Path,
    bootstrap_path: Path,
) -> pd.DataFrame:
    paired = pd.read_csv(paired_path)
    bootstrap = pd.read_csv(bootstrap_path)

    paired["holm_p_across_4_metrics"] = holm_adjust(
        paired["p_value"].astype(float).tolist()
    )
    merged = paired.merge(
        bootstrap[
            [
                "metric",
                "mean_difference_all18_minus_common16",
                "ci95_lower",
                "ci95_upper",
                "bootstrap_probability_all18_gt_common16",
            ]
        ],
        on="metric",
        how="left",
        suffixes=(
            "_subject",
            "_bootstrap",
        ),
        validate="one_to_one",
    )

    result = pd.DataFrame(
        {
            "Metric": merged["metric"].map(
                STAGE7_METRIC_MAP
            ),
            "Subject mean difference": merged[
                "mean_difference_all18_minus_common16_subject"
            ],
            "Wins": merged["wins_all18"].astype(int),
            "Ties": merged["ties"].astype(int),
            "Losses": merged["losses_all18"].astype(int),
            "Wilcoxon p": merged["p_value"],
            "Holm p across 4 metrics": merged[
                "holm_p_across_4_metrics"
            ],
            "Bootstrap mean difference": merged[
                "mean_difference_all18_minus_common16_bootstrap"
            ],
            "CI95 lower": merged["ci95_lower"],
            "CI95 upper": merged["ci95_upper"],
            "P(All18 > Common16)": merged[
                "bootstrap_probability_all18_gt_common16"
            ],
        }
    )
    return result


def markdown_numeric_table(
    frame: pd.DataFrame,
    p_columns: set[str] | None = None,
) -> str:
    p_columns = p_columns or set()
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(
            display[column]
        ):
            if column in p_columns:
                display[column] = display[column].map(
                    format_p
                )
            elif column not in {
                "Features",
                "Wins",
                "Ties",
                "Losses",
            }:
                display[column] = display[column].map(
                    lambda value: format_number(
                        value,
                        3,
                    )
                )
    return markdown_table(display)


def make_overview_figure(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_primary = primary[
        primary["Condition"].isin(
            [
                "RawStat",
                "Full Soft-AOI",
                "Compact Soft-AOI",
                "Transferred official 9",
            ]
        )
    ].copy()
    secondary_all18 = secondary[
        secondary["Condition"].eq(
            "All18 Compact"
        )
    ].copy()

    overview = pd.concat(
        [
            selected_primary[
                ["Condition", *METRIC_COLUMNS]
            ],
            secondary_all18[
                ["Condition", *METRIC_COLUMNS]
            ],
        ],
        ignore_index=True,
    )

    labels = overview["Condition"].tolist()
    x = np.arange(len(labels))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for index, metric in enumerate(
        METRIC_COLUMNS
    ):
        positions = (
            x + (index - 1.5) * width
        )
        plt.bar(
            positions,
            overview[metric].to_numpy(
                dtype=float
            ),
            width,
            label=metric,
        )
    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )
    plt.xticks(
        x,
        labels,
        rotation=15,
        ha="right",
    )
    plt.ylabel("Score")
    plt.ylim(0.30, 0.62)
    plt.title(
        "Primary external validation and secondary All18 Compact"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def build_final_text(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    corrected_tests: pd.DataFrame,
) -> str:
    p_lookup = primary.set_index("Condition")
    s_lookup = secondary.set_index("Condition")
    t_lookup = corrected_tests.set_index("Metric")

    compact = p_lookup.loc[
        "Compact Soft-AOI"
    ]
    official = p_lookup.loc[
        "Transferred official 9"
    ]
    rawstat = p_lookup.loc["RawStat"]
    full = p_lookup.loc["Full Soft-AOI"]
    common16 = s_lookup.loc[
        "Common16 Compact"
    ]
    all18 = s_lookup.loc[
        "All18 Compact"
    ]

    macro_test = t_lookup.loc["Macro F1"]
    ba_test = t_lookup.loc[
        "Balanced Accuracy"
    ]
    roc_test = t_lookup.loc["ROC-AUC"]
    pr_test = t_lookup.loc["PR-AUC (NR)"]

    return f"""# 外部検証の最終結果と考察

## 一次外部検証

ZuCo Common16で固定した5条件をWebQAmGazeの131名・1,349試行へ適用した。一次比較では、学習対象者と学習試行数を揃えたCommon16条件のみを用いた。RawStatはMacro F1={rawstat['Macro F1']:.3f}、Balanced Accuracy={rawstat['Balanced Accuracy']:.3f}、ROC-AUC={rawstat['ROC-AUC']:.3f}、PR-AUC={rawstat['PR-AUC (NR)']:.3f}であった。Full Soft-AOIはそれぞれ{full['Macro F1']:.3f}、{full['Balanced Accuracy']:.3f}、{full['ROC-AUC']:.3f}、{full['PR-AUC (NR)']:.3f}、Compact Soft-AOIは{compact['Macro F1']:.3f}、{compact['Balanced Accuracy']:.3f}、{compact['ROC-AUC']:.3f}、{compact['PR-AUC (NR)']:.3f}であった。転用イベントベース公式9特徴条件はMacro F1={official['Macro F1']:.3f}、Balanced Accuracy={official['Balanced Accuracy']:.3f}、ROC-AUC={official['ROC-AUC']:.3f}、PR-AUC={official['PR-AUC (NR)']:.3f}となり、一次5条件中で最も高かった。

Compact Soft-AOIはRawStatよりMacro F1が高く、時間順序特徴を除いた設計の相対的な頑健性が示された。一方、Compact Soft-AOIのROC-AUC={compact['ROC-AUC']:.3f}およびPR-AUC={compact['PR-AUC (NR)']:.3f}はランダム基準を超えず、絶対的な外部汎化は確認されなかった。ラベル方向の完全監査ではZuCoとWebQAmGazeの双方でNR=1が維持されていたため、この低性能はラベル反転によるものではない。

## All18 Compactの副次解析

学習対象をZuCo 16名・11,324試行から18名・13,302試行へ増やしたAll18 Compactでは、Macro F1が{common16['Macro F1']:.3f}から{all18['Macro F1']:.3f}、Balanced Accuracyが{common16['Balanced Accuracy']:.3f}から{all18['Balanced Accuracy']:.3f}、ROC-AUCが{common16['ROC-AUC']:.3f}から{all18['ROC-AUC']:.3f}、PR-AUCが{common16['PR-AUC (NR)']:.3f}から{all18['PR-AUC (NR)']:.3f}へわずかに上昇した。

被験者別Macro F1差は{macro_test['Subject mean difference']:+.3f}であり、Holm補正後p値は{format_p(macro_test['Holm p across 4 metrics'])}であった。Balanced Accuracy差は{ba_test['Subject mean difference']:+.3f}、Holm補正後p値は{format_p(ba_test['Holm p across 4 metrics'])}であった。ROC-AUC差は{roc_test['Subject mean difference']:+.3f}、Holm補正後p値は{format_p(roc_test['Holm p across 4 metrics'])}、PR-AUC差は{pr_test['Subject mean difference']:+.3f}、Holm補正後p値は{format_p(pr_test['Holm p across 4 metrics'])}であった。したがって、名目上はROC-AUCとPR-AUCでp<0.05であったものの、4指標に対するHolm補正後には有意差は残らなかった。

被験者クラスターブートストラップでは、ROC-AUC差の95%信頼区間のみ[{roc_test['CI95 lower']:.3f}, {roc_test['CI95 upper']:.3f}]と0を上回った。Macro F1、Balanced AccuracyおよびPR-AUCの信頼区間は0を含んだ。よって、学習対象者数の増加は順位性能をわずかに改善した可能性があるが、その効果は小さく、分類性能全体の明確な改善とは結論できない。

## 総合考察

All18 CompactでもROC-AUC={all18['ROC-AUC']:.3f}、PR-AUC={all18['PR-AUC (NR)']:.3f}であり、絶対性能は依然として0.5を下回った。このことから、Compact Soft-AOIの外部性能制約は、ZuCo学習データを約18%増加させるだけでは解消されず、主として計測方式、視線推定ノイズ、文章長、分析単位およびタスク構造の違いに由来すると考えられる。

一方、転用イベントベース公式9特徴条件はROC-AUC={official['ROC-AUC']:.3f}を示した。ただし、この条件はWebQAmGazeのフィルタ済みfixationから近似イベントを構成したものであり、EyeLinkの公式saccadeイベントと同一ではない。そのため、公式特徴の完全再現ではなく、独立した転用イベントベースラインとして報告する必要がある。

## 最終結論

本外部検証から、Compact Soft-AOIはRawStatおよび時間順序特徴を含むFull Soft-AOIに対して、センサー差に対する相対的頑健性を示した。しかし、WebQAmGazeに対する絶対的な外部汎化は達成されなかった。学習対象者を16名から18名へ増やす副次解析では小幅な改善が観察されたものの、多重比較補正後の被験者別検定では有意差は認められず、ドメイン差が主要な制約であるという解釈を支持した。
"""


def build_reporting_checklist() -> str:
    return """# 外部検証の報告チェックリスト

## 必ず分離して報告する結果

- 一次解析：Common16の5条件比較
- 副次解析：Common16 CompactとAll18 Compactの比較
- All18 Compactを一次5条件表へ追加しない

## ラベル表記

- ZuCo：TSR=0、NR=1
- WebQAmGaze：IS=0、NR=1
- 正例は一貫してNR
- 反転AUCはラベル反転ではなく、特徴関係のドメイン間反転として記述する

## Compact Soft-AOIについて許容される主張

- RawStatより相対的に頑健だった
- Fullの動的特徴による極端な寄与を抑えた
- 学習対象者増加により小幅な改善傾向があった

## Compact Soft-AOIについて避ける主張

- WebQAmGazeへ良好に汎化した
- ROC-AUCまたはPR-AUCがランダム基準を上回った
- All18がCommon16より統計的に有意に優れていた
- 学習データ量を増やせば問題が解決する

## 転用公式9特徴の表記

推奨：
- 転用イベントベース公式特徴式条件
- transferred event-based official-feature-formula condition

避ける：
- ZuCo公式特徴を完全再現した
- EyeLink公式イベントと同一
- 公式ベースラインをそのままWebカメラへ適用した

## 統計結果

- Stage 7のWilcoxon検定は4指標についてHolm補正する
- 補正後は4指標すべてp>=0.05
- クラスターブートストラップでROC-AUC差のみ95%信頼区間が0を上回る
- ROC-AUC改善幅自体は約0.005と小さい

## 最終的な中心結論

Compact Soft-AOIはRawStatより相対的に頑健であったが、絶対的な外部汎化は達成されなかった。All18への学習データ増加でも改善は小さく、主要な制約は計測方式、ノイズ、文章単位およびタスク構造のドメイン差にある。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage6-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage6_thesis"
        ),
    )
    parser.add_argument(
        "--stage7-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage7_all18"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage8_final"
        ),
    )
    args = parser.parse_args()

    stage6 = args.stage6_dir.resolve()
    stage7 = args.stage7_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "primary_results": require_file(
            stage6
            / "thesis_external_validation_results.csv"
        ),
        "secondary_pooled": require_file(
            stage7
            / "secondary_compact_pooled_metrics.csv"
        ),
        "secondary_paired": require_file(
            stage7
            / "secondary_compact_paired_subject_tests.csv"
        ),
        "secondary_bootstrap": require_file(
            stage7
            / "secondary_compact_bootstrap_differences.csv"
        ),
    }

    primary = prepare_primary_table(
        paths["primary_results"]
    )
    secondary = prepare_secondary_table(
        paths["secondary_pooled"]
    )
    corrected_tests = prepare_corrected_tests(
        paths["secondary_paired"],
        paths["secondary_bootstrap"],
    )

    output_paths = {
        "primary_csv": (
            output / "final_primary_results.csv"
        ),
        "primary_md": (
            output / "final_primary_results.md"
        ),
        "secondary_csv": (
            output / "final_secondary_all18_results.csv"
        ),
        "secondary_md": (
            output / "final_secondary_all18_results.md"
        ),
        "corrected_tests_csv": (
            output
            / "final_secondary_all18_tests_holm.csv"
        ),
        "corrected_tests_md": (
            output
            / "final_secondary_all18_tests_holm.md"
        ),
        "overview_figure": (
            output
            / "figure_final_external_validation_overview.png"
        ),
        "final_text": (
            output
            / "final_external_validation_results_ja.md"
        ),
        "checklist": (
            output
            / "external_validation_reporting_checklist.md"
        ),
        "manifest": (
            output / "stage8_manifest.json"
        ),
    }

    primary.to_csv(
        output_paths["primary_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    output_paths["primary_md"].write_text(
        markdown_numeric_table(primary),
        encoding="utf-8",
    )

    secondary.to_csv(
        output_paths["secondary_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    output_paths["secondary_md"].write_text(
        markdown_numeric_table(secondary),
        encoding="utf-8",
    )

    corrected_tests.to_csv(
        output_paths["corrected_tests_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    output_paths["corrected_tests_md"].write_text(
        markdown_numeric_table(
            corrected_tests,
            p_columns={
                "Wilcoxon p",
                "Holm p across 4 metrics",
            },
        ),
        encoding="utf-8",
    )

    make_overview_figure(
        primary,
        secondary,
        output_paths["overview_figure"],
    )

    output_paths["final_text"].write_text(
        build_final_text(
            primary,
            secondary,
            corrected_tests,
        ),
        encoding="utf-8",
    )
    output_paths["checklist"].write_text(
        build_reporting_checklist(),
        encoding="utf-8",
    )

    manifest = {
        "stage": (
            "webqamgaze_external_validation_stage8_final"
        ),
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "primary_analysis": (
            "Stage 6 Common16 matched five-condition comparison"
        ),
        "secondary_analysis": (
            "Stage 7 Common16 vs All18 Compact"
        ),
        "multiple_testing": {
            "family": (
                "four paired subject metrics in Stage 7"
            ),
            "method": "Holm",
            "metrics": list(
                corrected_tests["Metric"]
            ),
        },
        "no_model_changes": True,
        "no_metric_changes": True,
        "outputs": {
            name: str(path)
            for name, path in output_paths.items()
            if name != "manifest"
        },
    }
    output_paths["manifest"].write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== WEBQAMGAZE STAGE 8 FINAL PACKAGE ==="
    )
    print(
        "Primary and secondary analyses remain separated."
    )
    print()
    print("Holm-corrected Stage 7 tests:")
    print(
        corrected_tests[
            [
                "Metric",
                "Subject mean difference",
                "Wilcoxon p",
                "Holm p across 4 metrics",
                "Bootstrap mean difference",
                "CI95 lower",
                "CI95 upper",
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
