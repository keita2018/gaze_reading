from __future__ import annotations

import argparse
import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUTS = {
    "soft_hard_clean": (
        r"outputs\soft_vs_hard_aoi_clean_30hz\soft_vs_hard_summary.csv",
        "b4db0fa849fee0b42d76cba11f050f903b2663e4f1dfe220564ef95dc5b12522",
    ),
    "soft_hard_rate10": (
        r"outputs\soft_vs_hard_aoi_train30_test10\soft_vs_hard_train30_test10_summary.csv",
        "27ea7813a8e389f19ee429ef672a20d3e03c8d5df3c1fa082bd668fe4a369c4e",
    ),
    "soft_hard_noise": (
        r"outputs\soft_vs_hard_aoi_coordinate_noise_30hz\soft_vs_hard_noise_summary.csv",
        "60ee3e232ed5d8652e04ae5c499347a507837d5af27ec71ccde67aa4d3a467c8",
    ),
    "soft_hard_missing": (
        r"outputs\soft_vs_hard_aoi_missingness_30hz\soft_vs_hard_missingness_summary.csv",
        "613c8fd8b0d248e4b0196d9749adafdc72d0b02ff3bb5be750d26157c3bc98f7",
    ),
    "soft_hard_compound": (
        r"outputs\soft_vs_hard_aoi_combined_high_30hz\soft_vs_hard_combined_high_summary.csv",
        "83b966c78d17ed92abdc4102191438a9eb5e586b2a4614abb9603e1a58f86a5e",
    ),
    "ivt_fixed": (
        r"outputs\ivt_event_baseline_fixed_benchmark_exact\ivt_fixed_summary.csv",
        "a847f3f7a805f165cb803991c012f74dcf36ef8aef6c46c18ed794fc9ffc0009",
    ),
    "ivt_tuned": (
        r"outputs\ivt_event_baseline_nested_threshold\ivt_nested_summary.csv",
        "ecd8b2261da36677300e3179e05f2903dcf57788672f986743e69e6173bfe995",
    ),
    "qcsr": (
        r"outputs\onestop_qcsr_nested_v1\qcsr_robust_summary.csv",
        "d14e372dd4ff6ca956a0b32124ef40d00c7e5600534127c2ed20404951d30995",
    ),
    "qcr": (
        r"outputs\onestop_qcr_nested_v1\qcr_robust_summary.csv",
        "206915582b06e7b44ad12da3c3ea29e2b10bf8dfed8ff09ce88a655bc26d0753",
    ),
    "selective_metrics": (
        r"outputs\final_factorized_selective_prediction_v1\final_selective_prediction_metrics.csv",
        "0b6451dbf3578b27f659384f681bd454f37b01a7eed39665b1b6d9b8241ecb3e",
    ),
    "selective_aurc": (
        r"outputs\final_factorized_selective_prediction_v1\final_selective_prediction_aurc.csv",
        "685902ade2f3fbf8ea6e3dcaeb6c5e0cc739f41c7fe66a24d5db06a811f09b4e",
    ),
}

STANDARD_CONTROL_RELATIVE = r"outputs\iui_p0_standard_controls_v1\standard_control_metrics.csv"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_relative(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


class SourceLoader:
    def __init__(
        self,
        project_root: Path,
        evidence_bundle: Path | None,
        standard_control_metrics: Path | None,
    ) -> None:
        self.project_root = project_root
        self.standard_control_metrics = standard_control_metrics
        self.bundle: dict[str, dict[str, Any]] = {}
        if evidence_bundle is not None:
            payload = json.loads(evidence_bundle.read_text(encoding="utf-8"))
            for item in payload.get("evidence", []):
                rel = normalize_relative(str(item["relative_path"]))
                self.bundle[rel] = item

    def read_csv(self, relative: str, expected_hash: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        rel = normalize_relative(relative)
        if rel in self.bundle:
            item = self.bundle[rel]
            content = item.get("content")
            if content is None:
                raise RuntimeError(f"Text content is not embedded for {rel}")
            actual_hash = str(item.get("actual_sha256", ""))
            if expected_hash and actual_hash.lower() != expected_hash.lower():
                raise RuntimeError(
                    f"Hash mismatch in evidence bundle for {rel}: "
                    f"expected={expected_hash}, actual={actual_hash}"
                )
            return pd.read_csv(StringIO(content)), {
                "relative_path": rel,
                "sha256": actual_hash,
                "bytes": int(item.get("actual_bytes", 0)),
                "source": "evidence_bundle",
            }

        path = self.project_root / Path(rel)
        if not path.exists():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if expected_hash and actual_hash.lower() != expected_hash.lower():
            raise RuntimeError(
                f"Hash mismatch for {path}: expected={expected_hash}, actual={actual_hash}"
            )
        return pd.read_csv(path), {
            "relative_path": rel,
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
            "source": "project_file",
        }

    def read_standard_controls(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        candidate = self.standard_control_metrics
        if candidate is None:
            candidate = self.project_root / Path(normalize_relative(STANDARD_CONTROL_RELATIVE))
        if not candidate.exists():
            raise FileNotFoundError(
                f"Standard-control metrics not found: {candidate}. "
                "Pass --standard-control-metrics explicitly."
            )
        return pd.read_csv(candidate), {
            "relative_path": str(candidate),
            "sha256": sha256_file(candidate),
            "bytes": candidate.stat().st_size,
            "source": "standard_control_metrics",
        }


def round3(value: float) -> str:
    return f"{float(value):.3f}"


def build_figure4(loader: SourceLoader, output_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    loaded: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame] = {}
    for key in [
        "soft_hard_clean",
        "soft_hard_rate10",
        "soft_hard_noise",
        "soft_hard_missing",
        "soft_hard_compound",
        "ivt_fixed",
        "ivt_tuned",
    ]:
        rel, expected = INPUTS[key]
        tables[key], meta = loader.read_csv(rel, expected)
        loaded.append({"input": key, **meta})

    conditions = [
        ("Clean", "clean"),
        ("10 Hz", "rate10"),
        ("50 px noise", "noise50"),
        ("40% random", "random40"),
        ("40% burst", "burst40"),
        ("Noise + random", "high_random"),
        ("Noise + burst", "high_burst"),
    ]

    rows: list[dict[str, Any]] = []
    for label, code in conditions:
        if code == "clean":
            source = tables["soft_hard_clean"]
            soft_row = source[source["feature_set"] == "soft_common25"].iloc[0]
            hard_row = source[source["feature_set"] == "hard_argmax_common25"].iloc[0]
        elif code == "rate10":
            source = tables["soft_hard_rate10"]
            soft_row = source[source["feature_set"] == "soft_common25"].iloc[0]
            hard_row = source[source["feature_set"] == "hard_argmax_common25"].iloc[0]
        elif code == "noise50":
            source = tables["soft_hard_noise"]
            soft_row = source[source["feature_set"] == "soft_common25"].iloc[0]
            hard_row = source[source["feature_set"] == "hard_argmax_common25"].iloc[0]
        elif code in {"random40", "burst40"}:
            source = tables["soft_hard_missing"]
            mode = "random" if code == "random40" else "burst"
            soft_row = source[(source["feature_set"] == "soft_common25") & (source["missingness_mode"] == mode)].iloc[0]
            hard_row = source[(source["feature_set"] == "hard_argmax_common25") & (source["missingness_mode"] == mode)].iloc[0]
        else:
            source = tables["soft_hard_compound"]
            soft_row = source[(source["feature_set"] == "soft_common25") & (source["combined_condition"] == code)].iloc[0]
            hard_row = source[(source["feature_set"] == "hard_argmax_common25") & (source["combined_condition"] == code)].iloc[0]

        fixed_row = tables["ivt_fixed"][tables["ivt_fixed"]["condition"] == code].iloc[0]
        tuned_row = tables["ivt_tuned"][tables["ivt_tuned"]["condition"] == code].iloc[0]

        rows.append(
            {
                "condition": label,
                "condition_code": code,
                "soft_macro_f1": float(soft_row["macro_f1_subject_mean"]),
                "soft_roc_auc": float(soft_row["roc_auc_subject_mean"]),
                "hard_macro_f1": float(hard_row["macro_f1_subject_mean"]),
                "hard_roc_auc": float(hard_row["roc_auc_subject_mean"]),
                "fixed_ivt_macro_f1": float(fixed_row["macro_f1_subject_mean"]),
                "fixed_ivt_roc_auc": float(fixed_row["roc_auc_subject_mean"]),
                "tuned_ivt_macro_f1": float(tuned_row["macro_f1_subject_mean"]),
                "tuned_ivt_roc_auc": float(tuned_row["roc_auc_subject_mean"]),
            }
        )

    source_table = pd.DataFrame(rows)
    source_path = output_dir / "figure4_source_table.csv"
    source_table.to_csv(source_path, index=False, encoding="utf-8-sig")

    methods = [
        ("Event-free Soft-AOI", "soft_macro_f1"),
        ("Event-free Hard-AOI", "hard_macro_f1"),
        ("Fixed I-VT", "fixed_ivt_macro_f1"),
        ("Quality-tuned I-VT", "tuned_ivt_macro_f1"),
    ]
    x = np.arange(len(source_table))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for index, (label, column) in enumerate(methods):
        offset = (index - 1.5) * width
        ax.bar(x + offset, source_table[column], width=width, label=label)
    ax.set_title("Event-free AOI aggregation retains useful structure under measurement degradation")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(source_table["condition"], rotation=18, ha="right")
    ax.set_ylim(0.30, 0.68)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    figure_path = output_dir / "figure4_robustness.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path, source_path, loaded


def build_figure5(loader: SourceLoader, output_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    qcsr, qcsr_meta = loader.read_csv(*INPUTS["qcsr"])
    qcr, qcr_meta = loader.read_csv(*INPUTS["qcr"])
    loaded = [{"input": "qcsr", **qcsr_meta}, {"input": "qcr", **qcr_meta}]

    qcsr_index = qcsr.set_index("method")
    qcr_index = qcr.set_index("method")
    rows = [
        {
            "method": "Clean-only",
            "clean_macro_f1": float(qcsr_index.loc["clean_only", "clean_macro_f1"]),
            "mean_degraded_macro_f1": float(qcsr_index.loc["clean_only", "mean_degraded_macro_f1"]),
        },
        {
            "method": "Augmentation",
            "clean_macro_f1": float(qcsr_index.loc["augmentation", "clean_macro_f1"]),
            "mean_degraded_macro_f1": float(qcsr_index.loc["augmentation", "mean_degraded_macro_f1"]),
        },
        {
            "method": "QCSR",
            "clean_macro_f1": float(qcsr_index.loc["qcsr", "clean_macro_f1"]),
            "mean_degraded_macro_f1": float(qcsr_index.loc["qcsr", "mean_degraded_macro_f1"]),
        },
        {
            "method": "QCR",
            "clean_macro_f1": float(qcr_index.loc["qcr", "clean_macro_f1"]),
            "mean_degraded_macro_f1": float(qcr_index.loc["qcr", "mean_degraded_macro_f1"]),
        },
    ]
    table = pd.DataFrame(rows)
    table["clean_to_degraded_drop"] = table["mean_degraded_macro_f1"] - table["clean_macro_f1"]
    source_path = output_dir / "figure5_source_table.csv"
    table.to_csv(source_path, index=False, encoding="utf-8-sig")

    x = np.arange(len(table))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars1 = ax.bar(x - width / 2, table["clean_macro_f1"], width=width, label="Clean test view")
    bars2 = ax.bar(x + width / 2, table["mean_degraded_macro_f1"], width=width, label="Mean of six degraded views")
    ax.set_title("Ordinary quality-view augmentation provides the main robustness gain")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(table["method"])
    ax.set_ylim(0.42, 0.67)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    figure_path = output_dir / "figure5_onestop_robust_training.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path, source_path, loaded


def build_figure6(loader: SourceLoader, output_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    controls, meta = loader.read_standard_controls()
    loaded = [{"input": "standard_control_metrics", **meta}]
    mapping = [
        ("Frozen Full 41", "frozen_full41"),
        ("Participant\ncalibration", "participant_median_loo"),
        ("QCN-PC", "qcn_pc"),
    ]
    rows: list[dict[str, Any]] = []
    indexed = controls.set_index("method")
    for label, method in mapping:
        rows.append(
            {
                "display_method": label,
                "method": method,
                "macro_f1": float(indexed.loc[method, "macro_f1"]),
                "roc_auc": float(indexed.loc[method, "roc_auc"]),
                "pr_auc_nr": float(indexed.loc[method, "pr_auc_nr"]),
            }
        )
    table = pd.DataFrame(rows)
    source_path = output_dir / "figure6_source_table.csv"
    table.to_csv(source_path, index=False, encoding="utf-8-sig")

    x = np.arange(len(table))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars1 = ax.bar(x - width / 2, table["macro_f1"], width=width, label="Macro F1")
    bars2 = ax.bar(x + width / 2, table["roc_auc"], width=width, label="ROC-AUC")
    ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Label-free score correction progressively improves webcam transfer")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(table["display_method"])
    ax.set_ylim(0.48, 0.68)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    figure_path = output_dir / "figure6_english_transfer.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path, source_path, loaded


def build_figure7(loader: SourceLoader, output_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    metrics, metrics_meta = loader.read_csv(*INPUTS["selective_metrics"])
    aurc, aurc_meta = loader.read_csv(*INPUTS["selective_aurc"])
    loaded = [
        {"input": "selective_metrics", **metrics_meta},
        {"input": "selective_aurc", **aurc_meta},
    ]

    coverage_order = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    qcn = metrics[
        (metrics["method"] == "quality_removed_plus_participant")
        & (metrics["selection_mode"] == "global")
        & (metrics["target_coverage"].isin(coverage_order))
    ].copy()
    qcn["coverage_rank"] = qcn["target_coverage"].map({value: index for index, value in enumerate(coverage_order)})
    qcn = qcn.sort_values("coverage_rank")

    source_columns = [
        "target_coverage",
        "actual_coverage",
        "n_retained",
        "participants_retained",
        "accuracy",
        "selective_risk",
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ]
    source_path = output_dir / "figure7_source_table.csv"
    qcn[source_columns].to_csv(source_path, index=False, encoding="utf-8-sig")

    aurc_index = aurc.set_index("method")
    participant_aurc = float(aurc_index.loc["participant_only_final_reference", "aurc_error"])
    qcn_aurc = float(aurc_index.loc["quality_removed_plus_participant", "aurc_error"])

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    ax.plot(qcn["actual_coverage"], qcn["macro_f1"], marker="o", linewidth=2)
    ax.set_title("Selective prediction improves retained-set Macro F1")
    ax.set_xlabel("Coverage (fraction of trials retained)")
    ax.set_ylabel("Macro F1")
    ax.set_xlim(1.04, 0.06)
    ax.set_ylim(0.58, 0.74)
    ax.grid(alpha=0.25)

    for coverage in (0.5, 0.2):
        row = qcn[np.isclose(qcn["target_coverage"], coverage)].iloc[0]
        ax.annotate(
            f"{int(round(coverage * 100))}% coverage\n"
            f"{int(row['n_retained'])} trials, {int(row['participants_retained'])} participants\n"
            f"Macro F1 = {row['macro_f1']:.3f}",
            xy=(row["actual_coverage"], row["macro_f1"]),
            xytext=(row["actual_coverage"] + (0.10 if coverage == 0.5 else 0.12), row["macro_f1"] + (0.018 if coverage == 0.5 else -0.020)),
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            fontsize=9,
        )

    ax.text(
        0.97,
        0.588,
        f"AURC (selective risk): {participant_aurc:.3f} participant calibration → {qcn_aurc:.3f} QCN-PC",
        fontsize=9,
        fontweight="bold",
    )
    fig.tight_layout()
    figure_path = output_dir / "figure7_english_selective_prediction.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path, source_path, loaded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild IUI manuscript Figures 4–7 from audited aggregate outputs."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="outputs/iui_p1_rebuilt_figures_v1")
    parser.add_argument("--evidence-bundle", default="")
    parser.add_argument("--standard-control-metrics", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_bundle = Path(args.evidence_bundle).resolve() if args.evidence_bundle else None
    standard_control = Path(args.standard_control_metrics).resolve() if args.standard_control_metrics else None
    loader = SourceLoader(project_root, evidence_bundle, standard_control)

    outputs: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for figure_number, builder in [
        (4, build_figure4),
        (5, build_figure5),
        (6, build_figure6),
        (7, build_figure7),
    ]:
        figure_path, source_path, loaded = builder(loader, output_dir)
        inputs.extend({"figure": figure_number, **item} for item in loaded)
        outputs.extend(
            [
                {
                    "figure": figure_number,
                    "artifact": figure_path.name,
                    "sha256": sha256_file(figure_path),
                    "bytes": figure_path.stat().st_size,
                },
                {
                    "figure": figure_number,
                    "artifact": source_path.name,
                    "sha256": sha256_file(source_path),
                    "bytes": source_path.stat().st_size,
                },
            ]
        )

    manifest = {
        "artifact_version": "iui_p1_rebuilt_figures_v1",
        "purpose": "Reproducible manuscript Figures 4–7 from audited aggregate outputs",
        "inputs": inputs,
        "outputs": outputs,
    }
    manifest_path = output_dir / "figure_rebuild_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Figures 4–7 rebuilt successfully.")
    print(f"Output: {output_dir}")
    for item in outputs:
        print(f"  Figure {item['figure']}: {item['artifact']}  {item['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
