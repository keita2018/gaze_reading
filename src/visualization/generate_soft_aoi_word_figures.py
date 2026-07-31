#!/usr/bin/env python
"""Word統合稿に掲載したSoft-AOI関連の7図をCSVから再生成する。

既定では、gaze_reading_goal リポジトリのルートで実行する。

出力図
------
1. fig01_common16.png
   同一16名・同一11,324試行での最終比較
2. fig02_rate.png
   30 Hz学習モデルの低レート直接適用
3. fig03_noise.png
   座標ノイズに対する頑健性
4. fig04_random_missing.png
   ランダム欠損に対する頑健性
5. fig05_burst_missing.png
   1秒連続欠損に対する頑健性
6. fig06_combined.png
   座標ノイズと欠損の同時劣化
7. fig07_quality.png
   座標ノイズに伴うSoft-AOI所属品質の変化

使用例
------
python src/visualization/generate_soft_aoi_word_figures.py

出力先変更
----------
python src/visualization/generate_soft_aoi_word_figures.py --output-dir outputs/soft_aoi_word_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


COMPACT = "soft_compact_distribution_membership"
FULL = "soft_full_all"
RAW = "rawstat_all"
OFFICIAL = "official_sent_gaze_sacc"
DUMMY = "dummy_reference"

MODEL_LABELS = {
    COMPACT: "Compact Soft-AOI",
    FULL: "Full Soft-AOI",
    RAW: "RawStat",
    OFFICIAL: "公式 sent_gaze_sacc",
    DUMMY: "Dummy",
}

THREE_MODEL_ORDER = [COMPACT, FULL, RAW]
COMMON16_ORDER = [COMPACT, FULL, OFFICIAL, RAW, DUMMY]

COMBINED_ORDER = [
    "moderate_random",
    "moderate_burst",
    "high_random",
    "high_burst",
    "extreme_random",
    "extreme_burst",
]
COMBINED_LABELS = {
    "moderate_random": "25px+Random20%",
    "moderate_burst": "25px+Burst20%",
    "high_random": "50px+Random40%",
    "high_burst": "50px+Burst40%",
    "extreme_random": "100px+Random40%",
    "extreme_burst": "100px+Burst40%",
}


def configure_japanese_font() -> None:
    """利用可能な日本語フォントを優先して設定する。"""
    preferred = [
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "TakaoGothic",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"必要なCSVが見つかりません: {path}\n"
            "実験出力ディレクトリと引数を確認してください。"
        )
    return path


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    source: Path,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} に必要列がありません: {missing}")


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Written: {output_path}")


def apply_common_axis_style(
    axis: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str = "被験者平均 Macro F1",
    ylim: tuple[float, float] | None = None,
) -> None:
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(True, alpha=0.25)


def plot_common16(summary_path: Path, output_path: Path) -> None:
    data = pd.read_csv(require_file(summary_path))
    require_columns(
        data,
        ["feature_set", "macro_f1_subject_mean"],
        summary_path,
    )

    indexed = data.set_index("feature_set")
    missing_models = [name for name in COMMON16_ORDER if name not in indexed.index]
    if missing_models:
        raise ValueError(
            f"{summary_path} に比較対象がありません: {missing_models}"
        )

    values = [
        float(indexed.loc[name, "macro_f1_subject_mean"])
        for name in COMMON16_ORDER
    ]
    labels = [MODEL_LABELS[name] for name in COMMON16_ORDER]

    figure, axis = plt.subplots(figsize=(12, 7))
    bars = axis.bar(labels, values)
    axis.set_title(
        "同一16名・同一11,324試行での最終比較",
        fontweight="bold",
    )
    axis.set_ylabel("被験者平均 Macro F1")
    axis.set_ylim(0.30, 0.67)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    save_figure(figure, output_path)


def plot_rate(summary_path: Path, output_path: Path) -> None:
    data = pd.read_csv(require_file(summary_path))
    require_columns(
        data,
        [
            "protocol",
            "sampling_rate_hz",
            "training_rate_hz",
            "feature_set",
            "macro_f1_subject_mean",
        ],
        summary_path,
    )

    selected = data[
        data["protocol"].eq("train30_test_rate")
        & data["training_rate_hz"].eq(30)
        & data["feature_set"].isin(THREE_MODEL_ORDER)
    ].copy()

    rates = sorted(selected["sampling_rate_hz"].unique())
    if rates != [10, 15, 30]:
        raise ValueError(
            f"{summary_path} の評価レートが想定と異なります: {rates}"
        )

    figure, axis = plt.subplots(figsize=(12, 7))
    for feature_set in THREE_MODEL_ORDER:
        group = (
            selected[selected["feature_set"].eq(feature_set)]
            .sort_values("sampling_rate_hz")
        )
        axis.plot(
            group["sampling_rate_hz"],
            group["macro_f1_subject_mean"],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[feature_set],
        )

    apply_common_axis_style(
        axis,
        title="30 Hz学習モデルの低レート直接適用",
        xlabel="評価時サンプリングレート（Hz）",
        ylim=(0.43, 0.66),
    )
    axis.set_xticks(rates)
    axis.legend()
    save_figure(figure, output_path)


def plot_noise(summary_path: Path, output_path: Path) -> None:
    data = pd.read_csv(require_file(summary_path))
    require_columns(
        data,
        [
            "noise_sigma_px",
            "feature_set",
            "macro_f1_subject_mean",
        ],
        summary_path,
    )

    selected = data[data["feature_set"].isin(THREE_MODEL_ORDER)].copy()
    sigmas = sorted(selected["noise_sigma_px"].unique())

    figure, axis = plt.subplots(figsize=(12, 7))
    for feature_set in THREE_MODEL_ORDER:
        group = (
            selected[selected["feature_set"].eq(feature_set)]
            .sort_values("noise_sigma_px")
        )
        axis.plot(
            group["noise_sigma_px"],
            group["macro_f1_subject_mean"],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[feature_set],
        )

    apply_common_axis_style(
        axis,
        title="座標ノイズに対する頑健性",
        xlabel="座標ノイズ標準偏差（px）",
        ylim=(0.28, 0.67),
    )
    axis.set_xticks(sigmas)
    axis.legend()
    save_figure(figure, output_path)


def missingness_with_clean(
    data: pd.DataFrame,
    mode: str,
    feature_set: str,
) -> pd.DataFrame:
    clean = data[
        data["missingness_mode"].eq("clean")
        & data["feature_set"].eq(feature_set)
    ][
        ["additional_missing_percent", "macro_f1_subject_mean"]
    ].copy()

    degraded = data[
        data["missingness_mode"].eq(mode)
        & data["feature_set"].eq(feature_set)
    ][
        ["additional_missing_percent", "macro_f1_subject_mean"]
    ].copy()

    return (
        pd.concat([clean, degraded], ignore_index=True)
        .sort_values("additional_missing_percent")
        .drop_duplicates("additional_missing_percent", keep="first")
    )


def plot_missingness(
    summary_path: Path,
    output_path: Path,
    mode: str,
) -> None:
    data = pd.read_csv(require_file(summary_path))
    require_columns(
        data,
        [
            "missingness_mode",
            "additional_missing_percent",
            "feature_set",
            "macro_f1_subject_mean",
        ],
        summary_path,
    )

    if mode == "random":
        title = "ランダム欠損に対する頑健性"
        xlabel = "追加ランダム欠損率（%）"
    elif mode == "burst":
        title = "1秒連続欠損に対する頑健性"
        xlabel = "追加連続欠損率（%）"
    else:
        raise ValueError(f"未対応の欠損モードです: {mode}")

    figure, axis = plt.subplots(figsize=(12, 7))
    for feature_set in THREE_MODEL_ORDER:
        group = missingness_with_clean(data, mode, feature_set)
        axis.plot(
            group["additional_missing_percent"],
            group["macro_f1_subject_mean"],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[feature_set],
        )

    apply_common_axis_style(
        axis,
        title=title,
        xlabel=xlabel,
        ylim=(0.39, 0.66),
    )
    axis.set_xticks([0, 5, 10, 20, 40])
    axis.legend()
    save_figure(figure, output_path)


def plot_combined(summary_path: Path, output_path: Path) -> None:
    data = pd.read_csv(require_file(summary_path))
    require_columns(
        data,
        [
            "combined_condition",
            "feature_set",
            "macro_f1_subject_mean",
        ],
        summary_path,
    )

    selected = data[
        data["combined_condition"].isin(COMBINED_ORDER)
        & data["feature_set"].isin(THREE_MODEL_ORDER)
    ].copy()

    pivot = selected.pivot(
        index="combined_condition",
        columns="feature_set",
        values="macro_f1_subject_mean",
    ).reindex(COMBINED_ORDER)

    if pivot.isna().any().any():
        raise ValueError(
            f"{summary_path} の複合劣化条件または特徴セットが不足しています。"
        )

    positions = np.arange(len(COMBINED_ORDER))
    width = 0.26

    figure, axis = plt.subplots(figsize=(13, 6))
    for offset, feature_set in enumerate(THREE_MODEL_ORDER):
        axis.bar(
            positions + (offset - 1) * width,
            pivot[feature_set].to_numpy(dtype=float),
            width=width,
            label=MODEL_LABELS[feature_set],
        )

    axis.set_title("座標ノイズと欠損の同時劣化", fontweight="bold")
    axis.set_ylabel("被験者平均 Macro F1")
    axis.set_ylim(0.28, 0.60)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [COMBINED_LABELS[name] for name in COMBINED_ORDER],
        rotation=17,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save_figure(figure, output_path)


def plot_quality(summary_path: Path, output_path: Path) -> None:
    data = pd.read_csv(require_file(summary_path))
    columns = {
        "soft_aoi_active_rate_subject_median_mean": "Active率",
        "soft_aoi_word_box_hit_rate_subject_median_mean": "単語矩形内率",
        "soft_aoi_qmax_mean_subject_median_mean": "平均最大所属度",
        "soft_aoi_sentence_envelope_hit_rate_subject_median_mean": "文領域内率",
    }
    require_columns(
        data,
        ["noise_sigma_px", *columns.keys()],
        summary_path,
    )

    data = data.sort_values("noise_sigma_px")
    figure, axis = plt.subplots(figsize=(12, 7))
    for column, label in columns.items():
        axis.plot(
            data["noise_sigma_px"],
            data[column],
            marker="o",
            linewidth=2,
            label=label,
        )

    apply_common_axis_style(
        axis,
        title="座標ノイズに伴うSoft-AOI所属品質の変化",
        xlabel="座標ノイズ標準偏差（px）",
        ylabel="品質指標",
        ylim=(0.0, 1.0),
    )
    axis.set_xticks(sorted(data["noise_sigma_px"].unique()))
    axis.legend()
    save_figure(figure, output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=".",
        help="gaze_reading_goal リポジトリのルート",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/soft_aoi_word_figures",
    )
    parser.add_argument(
        "--common16-summary",
        default=(
            "outputs/soft_aoi_official_common16_nested_dual/"
            "common16_nested_dual_summary.csv"
        ),
    )
    parser.add_argument(
        "--rate-summary",
        default=(
            "outputs/soft_aoi_rate_robustness_30_15_10hz/"
            "rate_robustness_summary.csv"
        ),
    )
    parser.add_argument(
        "--noise-summary",
        default=(
            "outputs/soft_aoi_coordinate_noise_30hz/"
            "soft_aoi_coordinate_noise_summary.csv"
        ),
    )
    parser.add_argument(
        "--missingness-summary",
        default=(
            "outputs/soft_aoi_missingness_30hz/"
            "soft_aoi_missingness_summary.csv"
        ),
    )
    parser.add_argument(
        "--combined-summary",
        default=(
            "outputs/soft_aoi_combined_degradation_30hz/"
            "soft_aoi_combined_summary.csv"
        ),
    )
    parser.add_argument(
        "--quality-summary",
        default=(
            "outputs/soft_aoi_coordinate_noise_30hz/"
            "soft_aoi_coordinate_noise_quality_summary.csv"
        ),
    )
    args = parser.parse_args()

    configure_japanese_font()

    root = Path(args.repo_root).resolve()
    output_dir = root / args.output_dir

    common16 = root / args.common16_summary
    rate = root / args.rate_summary
    noise = root / args.noise_summary
    missingness = root / args.missingness_summary
    combined = root / args.combined_summary
    quality = root / args.quality_summary

    plot_common16(common16, output_dir / "fig01_common16.png")
    plot_rate(rate, output_dir / "fig02_rate.png")
    plot_noise(noise, output_dir / "fig03_noise.png")
    plot_missingness(
        missingness,
        output_dir / "fig04_random_missing.png",
        mode="random",
    )
    plot_missingness(
        missingness,
        output_dir / "fig05_burst_missing.png",
        mode="burst",
    )
    plot_combined(combined, output_dir / "fig06_combined.png")
    plot_quality(quality, output_dir / "fig07_quality.png")

    print(f"\nCompleted: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
