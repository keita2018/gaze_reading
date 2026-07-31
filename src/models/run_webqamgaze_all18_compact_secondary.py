#!/usr/bin/env python
"""Secondary external validation of the all-18-subject Compact model.

Primary analysis remains the prespecified five-condition Common16 comparison.
This script performs only the prespecified secondary comparison:

    Common16 Compact:
        16 ZuCo subjects, 11,324 matched trials

    All18 Compact:
        18 ZuCo subjects, 13,302 trials

Both use the same 27 Compact Soft-AOI features, fixed C=0.01, frozen ZuCo
preprocessing, threshold=0.0, and the same 1,349 WebQAmGaze external trials.

No external retraining, feature selection, threshold tuning, calibration, or
label-dependent preprocessing is performed.

Outputs
-------
secondary_compact_predictions.csv
secondary_compact_pooled_metrics.csv
secondary_compact_subject_metrics.csv
secondary_compact_summary.csv
secondary_compact_paired_subject_tests.csv
secondary_compact_bootstrap_replicates.csv
secondary_compact_bootstrap_differences.csv
secondary_compact_feature_transport_audit.csv
figure_compact_common16_vs_all18.png
secondary_compact_results_ja.md
stage7_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


COMMON16_NAME = "compact_common16_primary"
ALL18_NAME = "compact_all18_secondary"
CONDITION_ORDER = [COMMON16_NAME, ALL18_NAME]

DISPLAY_NAMES = {
    COMMON16_NAME: "Common16 Compact",
    ALL18_NAME: "All18 Compact",
}

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc_nr",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        return math.nan
    return float(roc_auc_score(y, score))


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        return math.nan
    return float(average_precision_score(y, score))


def compute_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "macro_f1": float(
            f1_score(
                y,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_nr": float(
            precision_score(
                y,
                prediction,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_nr": float(
            recall_score(
                y,
                prediction,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_nr": float(
            f1_score(
                y,
                prediction,
                pos_label=1,
                zero_division=0,
            )
        ),
        "precision_is": float(
            precision_score(
                y,
                prediction,
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_is": float(
            recall_score(
                y,
                prediction,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_is": float(
            f1_score(
                y,
                prediction,
                pos_label=0,
                zero_division=0,
            )
        ),
        "roc_auc": safe_auc(y, score),
        "pr_auc_nr": safe_pr_auc(y, score),
        "positive_rate_nr": float(np.mean(y == 1)),
        "predicted_nr_rate": float(
            np.mean(prediction == 1)
        ),
    }


def normalize_mapping(mapping: Any) -> dict[str, int]:
    if not isinstance(mapping, dict):
        return {}
    return {
        str(key).upper(): int(value)
        for key, value in mapping.items()
    }


def align_feature_table(
    path: Path,
    trial_ids: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "external_trial_id" not in frame.columns:
        raise ValueError(
            f"{path} lacks external_trial_id."
        )
    frame["external_trial_id"] = frame[
        "external_trial_id"
    ].astype(str)
    if frame["external_trial_id"].duplicated().any():
        raise ValueError(
            f"{path} contains duplicate trial IDs."
        )
    indexed = frame.set_index("external_trial_id")
    missing = sorted(set(trial_ids) - set(indexed.index))
    extra = sorted(set(indexed.index) - set(trial_ids))
    if missing or extra:
        raise ValueError(
            f"{path} trial-ID mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return indexed.loc[trial_ids].reset_index()


def find_common16_artifact(frozen_dir: Path) -> Path:
    exact = (
        frozen_dir
        / "models"
        / "soft_compact_distribution_membership.joblib"
    )
    if exact.is_file():
        return exact
    matches = list(
        frozen_dir.rglob(
            "soft_compact_distribution_membership.joblib"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Could not uniquely identify the Common16 Compact artifact: "
            f"{matches}"
        )
    return matches[0]


def find_all18_artifact(frozen_dir: Path) -> Path:
    exact = (
        frozen_dir
        / "models"
        / "soft_compact_all18_secondary.joblib"
    )
    if exact.is_file():
        return exact
    matches = list(
        frozen_dir.rglob(
            "soft_compact_all18_secondary.joblib"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Could not uniquely identify the all18 Compact artifact: "
            f"{matches}"
        )
    return matches[0]


def validate_artifact(
    artifact: dict[str, Any],
    *,
    expected_scope: str,
    name: str,
) -> None:
    mapping = normalize_mapping(
        artifact.get("label_to_int", {})
    )
    if mapping != {"TSR": 0, "NR": 1}:
        raise ValueError(
            f"{name} label mapping is unexpected: {mapping}"
        )
    if artifact.get("training_scope") != expected_scope:
        raise ValueError(
            f"{name} training scope="
            f"{artifact.get('training_scope')!r}, "
            f"expected={expected_scope!r}."
        )
    if float(artifact.get("decision_threshold")) != 0.0:
        raise ValueError(
            f"{name} threshold is not 0.0."
        )
    if float(artifact.get("c_value")) != 0.01:
        raise ValueError(
            f"{name} C is not 0.01."
        )
    pipeline = artifact["pipeline"]
    classes = np.asarray(
        pipeline.named_steps["model"].classes_,
        dtype=int,
    ).tolist()
    if classes != [0, 1]:
        raise ValueError(
            f"{name} model classes are {classes}, expected [0, 1]."
        )


def apply_model(
    artifact: dict[str, Any],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    columns = list(artifact["feature_columns"])
    missing = sorted(set(columns) - set(features.columns))
    if missing:
        raise ValueError(
            f"External Compact table lacks features: {missing}"
        )
    x = features[columns].copy()
    for column in columns:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )
    values = x.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(
            "External Compact features contain infinity."
        )

    pipeline = artifact["pipeline"]
    score = np.asarray(
        pipeline.decision_function(x),
        dtype=float,
    ).reshape(-1)
    threshold = float(
        artifact["decision_threshold"]
    )
    prediction = (score >= threshold).astype(int)

    direct_prediction = np.asarray(
        pipeline.predict(x),
        dtype=int,
    ).reshape(-1)
    if not np.array_equal(
        prediction,
        direct_prediction,
    ):
        raise RuntimeError(
            "Threshold predictions differ from pipeline.predict."
        )
    if not np.isfinite(score).all():
        raise RuntimeError(
            "Model produced non-finite scores."
        )
    return prediction, score


def build_prediction_frame(
    *,
    condition: str,
    trials: pd.DataFrame,
    prediction: np.ndarray,
    score: np.ndarray,
) -> pd.DataFrame:
    frame = trials[
        [
            "external_trial_id",
            "participant_id",
            "set_name",
            "trial_index",
            "trial_name",
            "task",
            "label_nr",
        ]
    ].copy()
    frame.insert(0, "condition", condition)
    frame["decision_score_nr"] = score
    frame["decision_threshold"] = 0.0
    frame["predicted_label_nr"] = prediction
    frame["predicted_task"] = np.where(
        prediction == 1,
        "NR",
        "IS",
    )
    frame["correct"] = (
        prediction
        == trials["label_nr"].to_numpy(dtype=int)
    ).astype(int)
    return frame


def build_subject_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, participant_id), group in predictions.groupby(
        ["condition", "participant_id"],
        sort=True,
    ):
        y = group["label_nr"].to_numpy(dtype=int)
        prediction = group[
            "predicted_label_nr"
        ].to_numpy(dtype=int)
        score = group[
            "decision_score_nr"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "condition": condition,
                "participant_id": participant_id,
                "n_trials": len(group),
                "n_nr": int(np.sum(y == 1)),
                "n_is": int(np.sum(y == 0)),
                **compute_metrics(
                    y,
                    prediction,
                    score,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        group = predictions[
            predictions["condition"].eq(condition)
        ]
        y = group["label_nr"].to_numpy(dtype=int)
        prediction = group[
            "predicted_label_nr"
        ].to_numpy(dtype=int)
        score = group[
            "decision_score_nr"
        ].to_numpy(dtype=float)
        cm = confusion_matrix(
            y,
            prediction,
            labels=[0, 1],
        )
        rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY_NAMES[condition],
                "n_trials": len(group),
                "n_participants": int(
                    group["participant_id"].nunique()
                ),
                **compute_metrics(
                    y,
                    prediction,
                    score,
                ),
                "tn_is_correct": int(cm[0, 0]),
                "fp_is_as_nr": int(cm[0, 1]),
                "fn_nr_as_is": int(cm[1, 0]),
                "tp_nr_correct": int(cm[1, 1]),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    pooled: pd.DataFrame,
    subject: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pooled_lookup = pooled.set_index("condition")
    for condition in CONDITION_ORDER:
        group = subject[
            subject["condition"].eq(condition)
        ]
        row: dict[str, Any] = {
            "condition": condition,
            "display_name": DISPLAY_NAMES[condition],
            "n_trials": int(
                pooled_lookup.loc[condition, "n_trials"]
            ),
            "n_participants": int(
                pooled_lookup.loc[
                    condition,
                    "n_participants",
                ]
            ),
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "pr_auc_nr",
            "f1_nr",
            "f1_is",
        ):
            row[f"{metric}_pooled"] = float(
                pooled_lookup.loc[condition, metric]
            )
            row[f"{metric}_subject_mean"] = float(
                group[metric].mean()
            )
            row[f"{metric}_subject_std"] = float(
                group[metric].std(ddof=1)
            )
        row["predicted_nr_rate_pooled"] = float(
            pooled_lookup.loc[
                condition,
                "predicted_nr_rate",
            ]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_subject_tests(
    subject: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in (
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ):
        common = (
            subject[
                subject["condition"].eq(COMMON16_NAME)
            ][["participant_id", metric]]
            .rename(columns={metric: "common16"})
        )
        all18 = (
            subject[
                subject["condition"].eq(ALL18_NAME)
            ][["participant_id", metric]]
            .rename(columns={metric: "all18"})
        )
        paired = common.merge(
            all18,
            on="participant_id",
            how="inner",
            validate="one_to_one",
        ).dropna()
        difference = (
            paired["all18"]
            - paired["common16"]
        ).to_numpy(dtype=float)

        if not len(difference):
            statistic = math.nan
            p_value = math.nan
        elif np.allclose(difference, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                difference,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append(
            {
                "metric": metric,
                "reference_condition": COMMON16_NAME,
                "candidate_condition": ALL18_NAME,
                "n_paired_participants": len(paired),
                "reference_mean": float(
                    paired["common16"].mean()
                ),
                "candidate_mean": float(
                    paired["all18"].mean()
                ),
                "mean_difference_all18_minus_common16": float(
                    np.mean(difference)
                ),
                "median_difference_all18_minus_common16": float(
                    np.median(difference)
                ),
                "wins_all18": int(
                    np.sum(difference > 0)
                ),
                "ties": int(
                    np.sum(difference == 0)
                ),
                "losses_all18": int(
                    np.sum(difference < 0)
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    participants = sorted(
        predictions["participant_id"]
        .astype(str)
        .unique()
    )

    lookup: dict[
        str,
        dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ],
    ] = {}
    for condition in CONDITION_ORDER:
        lookup[condition] = {}
        condition_frame = predictions[
            predictions["condition"].eq(condition)
        ]
        for participant_id, group in condition_frame.groupby(
            "participant_id",
            sort=False,
        ):
            lookup[condition][str(participant_id)] = (
                group["label_nr"].to_numpy(dtype=int),
                group[
                    "predicted_label_nr"
                ].to_numpy(dtype=int),
                group[
                    "decision_score_nr"
                ].to_numpy(dtype=float),
            )

    replicate_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []

    for replicate in range(n_bootstraps):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        replicate_metrics: dict[
            str,
            dict[str, float],
        ] = {}
        for condition in CONDITION_ORDER:
            y_parts = []
            prediction_parts = []
            score_parts = []
            for participant_id in sampled:
                y, prediction, score = lookup[
                    condition
                ][str(participant_id)]
                y_parts.append(y)
                prediction_parts.append(prediction)
                score_parts.append(score)

            y_boot = np.concatenate(y_parts)
            prediction_boot = np.concatenate(
                prediction_parts
            )
            score_boot = np.concatenate(score_parts)
            metrics = compute_metrics(
                y_boot,
                prediction_boot,
                score_boot,
            )
            replicate_metrics[condition] = metrics
            replicate_rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "condition": condition,
                    "n_sampled_participant_clusters": len(
                        sampled
                    ),
                    "n_trials_with_cluster_multiplicity": len(
                        y_boot
                    ),
                    **{
                        metric: metrics[metric]
                        for metric in METRICS
                    },
                }
            )

        for metric in (
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
        ):
            difference_rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "metric": metric,
                    "reference_condition": COMMON16_NAME,
                    "candidate_condition": ALL18_NAME,
                    "difference_all18_minus_common16": (
                        replicate_metrics[ALL18_NAME][
                            metric
                        ]
                        - replicate_metrics[COMMON16_NAME][
                            metric
                        ]
                    ),
                }
            )

    replicates = pd.DataFrame(replicate_rows)
    differences = pd.DataFrame(difference_rows)

    summary_rows: list[dict[str, Any]] = []
    for metric, group in differences.groupby(
        "metric",
        sort=False,
    ):
        values = group[
            "difference_all18_minus_common16"
        ].dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "reference_condition": COMMON16_NAME,
                "candidate_condition": ALL18_NAME,
                "n_bootstrap_values": len(values),
                "mean_difference_all18_minus_common16": float(
                    np.mean(values)
                ),
                "ci95_lower": float(
                    np.quantile(values, 0.025)
                ),
                "ci95_upper": float(
                    np.quantile(values, 0.975)
                ),
                "bootstrap_probability_all18_gt_common16": float(
                    np.mean(values > 0)
                ),
            }
        )
    return replicates, pd.DataFrame(summary_rows)


def feature_transport_audit(
    *,
    condition: str,
    artifact: dict[str, Any],
    features: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(artifact["feature_columns"])
    pipeline = artifact["pipeline"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    statistics = np.asarray(
        imputer.statistics_,
        dtype=float,
    )

    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(columns):
        external = pd.to_numeric(
            features[feature],
            errors="coerce",
        )
        imputed = external.fillna(
            statistics[index]
        )
        external_median = float(imputed.median())
        training_mean = float(scaler.mean_[index])
        training_scale = float(scaler.scale_[index])
        standardized_median = (
            (external_median - training_mean)
            / training_scale
            if training_scale > 0
            else math.nan
        )
        rows.append(
            {
                "condition": condition,
                "feature_index": index,
                "feature_name": feature,
                "n_external_rows": len(external),
                "n_external_missing": int(
                    external.isna().sum()
                ),
                "external_missing_rate": float(
                    external.isna().mean()
                ),
                "external_median_raw": (
                    float(external.median())
                    if external.notna().any()
                    else math.nan
                ),
                "training_imputer_median": float(
                    statistics[index]
                ),
                "training_scaler_mean": training_mean,
                "training_scaler_scale": training_scale,
                "external_median_standardized_by_training": (
                    standardized_median
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    pooled: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ]
    metric_names = [
        "Macro F1",
        "Balanced Accuracy",
        "ROC-AUC",
        "PR-AUC (NR)",
    ]
    frame = pooled.set_index("condition").loc[
        CONDITION_ORDER
    ]
    x = np.arange(len(CONDITION_ORDER))
    width = 0.18

    plt.figure(figsize=(9, 6))
    for index, (metric, metric_name) in enumerate(
        zip(metrics, metric_names)
    ):
        positions = (
            x + (index - 1.5) * width
        )
        plt.bar(
            positions,
            frame[metric].to_numpy(dtype=float),
            width,
            label=metric_name,
        )
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xticks(
        x,
        [
            DISPLAY_NAMES[condition]
            for condition in CONDITION_ORDER
        ],
    )
    plt.ylabel("Score")
    plt.ylim(0.30, 0.62)
    plt.title(
        "Secondary external validation: Compact training scope"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def format_p(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    if float(value) < 0.001:
        return "<0.001"
    return f"{float(value):.3f}"


def build_japanese_text(
    *,
    pooled: pd.DataFrame,
    subject_tests: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    lookup = pooled.set_index("condition")
    common = lookup.loc[COMMON16_NAME]
    all18 = lookup.loc[ALL18_NAME]

    subject_lookup = subject_tests.set_index(
        "metric"
    )
    bootstrap_lookup = bootstrap.set_index(
        "metric"
    )

    macro_subject = subject_lookup.loc[
        "macro_f1"
    ]
    macro_boot = bootstrap_lookup.loc[
        "macro_f1"
    ]
    ba_subject = subject_lookup.loc[
        "balanced_accuracy"
    ]
    ba_boot = bootstrap_lookup.loc[
        "balanced_accuracy"
    ]

    return f"""# Compact Soft-AOIの副次外部検証

## 解析目的

一次解析では、5条件の学習対象者数と試行数を揃えるため、16名・11,324試行のCommon16 Compactモデルを使用した。本副次解析では、利用可能な18名・13,302試行すべてで学習したAll18 Compactモデルを、同一のWebQAmGaze 131名・1,349試行へ適用し、学習データ増加が外部汎化性能に与える影響を確認した。特徴量は両モデルとも同一の27次元Compact Soft-AOIであり、WebQAmGazeによる再学習、閾値調整、較正は行っていない。

## 結果

Common16 Compactのpooled評価は、Macro F1={common['macro_f1']:.3f}、Balanced Accuracy={common['balanced_accuracy']:.3f}、ROC-AUC={common['roc_auc']:.3f}、PR-AUC={common['pr_auc_nr']:.3f}であった。All18 Compactでは、Macro F1={all18['macro_f1']:.3f}、Balanced Accuracy={all18['balanced_accuracy']:.3f}、ROC-AUC={all18['roc_auc']:.3f}、PR-AUC={all18['pr_auc_nr']:.3f}であった。

被験者別Macro F1の平均差（All18 - Common16）は{macro_subject['mean_difference_all18_minus_common16']:+.3f}であり、Wilcoxon符号付順位検定のp値は{format_p(macro_subject['p_value'])}であった。被験者クラスターブートストラップによるpooled Macro F1差の平均は{macro_boot['mean_difference_all18_minus_common16']:+.3f}、95%信頼区間は[{macro_boot['ci95_lower']:.3f}, {macro_boot['ci95_upper']:.3f}]であった。

Balanced Accuracyの被験者平均差は{ba_subject['mean_difference_all18_minus_common16']:+.3f}、p値は{format_p(ba_subject['p_value'])}であった。ブートストラップ差の平均は{ba_boot['mean_difference_all18_minus_common16']:+.3f}、95%信頼区間は[{ba_boot['ci95_lower']:.3f}, {ba_boot['ci95_upper']:.3f}]であった。

## 解釈

All18モデルはCommon16モデルと同一の特徴定義および固定ハイパーパラメータを用いており、差は主としてZuCo学習対象者数と試行数の増加による。ただし、All18条件は公式9特徴条件と学習対象者を揃えられないため、一次5条件比較へ混在させず、副次解析として報告する。外部性能が改善した場合は、Compact表現において学習対象者の多様性を増やすことがセンサー・データセット間移送に寄与した可能性を示す。改善が確認されない場合は、性能制約が学習データ量よりも計測方式、文章単位、タスク設計および特徴関係のドメイン差に由来する可能性を示す。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage2"
        ),
    )
    parser.add_argument(
        "--stage3c-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage3c"
        ),
    )
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage4"
        ),
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path(
            "outputs/frozen_zuco_external_validation_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage7_all18"
        ),
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=2000,
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.n_bootstraps < 100:
        raise ValueError(
            "--n-bootstraps must be at least 100."
        )

    stage2 = args.stage2_dir.resolve()
    stage3c = args.stage3c_dir.resolve()
    stage4 = args.stage4_dir.resolve()
    frozen = args.frozen_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "trials": require_file(
            stage2
            / "trial_manifest_primary_candidates.csv"
        ),
        "compact_features": require_file(
            stage3c
            / "soft_compact_features_primary.csv"
        ),
        "stage4_predictions": require_file(
            stage4 / "external_predictions.csv"
        ),
        "stage4_pooled": require_file(
            stage4 / "external_pooled_metrics.csv"
        ),
        "common16_artifact": find_common16_artifact(
            frozen
        ),
        "all18_artifact": find_all18_artifact(
            frozen
        ),
    }

    trials = pd.read_csv(paths["trials"])
    trials["external_trial_id"] = trials[
        "external_trial_id"
    ].astype(str)
    trials["participant_id"] = trials[
        "participant_id"
    ].astype(str)
    trials["task"] = trials[
        "task"
    ].astype(str).str.upper()
    trials["label_nr"] = pd.to_numeric(
        trials["label_nr"],
        errors="raise",
    ).astype(int)
    trials = trials.sort_values(
        ["participant_id", "trial_index"]
    ).reset_index(drop=True)

    expected_label = trials["task"].map(
        {"IS": 0, "NR": 1}
    )
    if expected_label.isna().any():
        raise ValueError(
            "External trials contain unexpected tasks."
        )
    if not np.array_equal(
        trials["label_nr"].to_numpy(dtype=int),
        expected_label.to_numpy(dtype=int),
    ):
        raise ValueError(
            "External task/label mapping is inconsistent."
        )

    trial_ids = trials[
        "external_trial_id"
    ].tolist()
    compact_features = align_feature_table(
        paths["compact_features"],
        trial_ids,
    )

    common16_artifact = joblib.load(
        paths["common16_artifact"]
    )
    all18_artifact = joblib.load(
        paths["all18_artifact"]
    )

    validate_artifact(
        common16_artifact,
        expected_scope="common16_matched",
        name="Common16 Compact",
    )
    validate_artifact(
        all18_artifact,
        expected_scope="all18_maxdata_secondary",
        name="All18 Compact",
    )

    if list(
        common16_artifact["feature_columns"]
    ) != list(
        all18_artifact["feature_columns"]
    ):
        raise ValueError(
            "Common16 and All18 Compact feature schemas differ."
        )
    if len(
        all18_artifact["feature_columns"]
    ) != 27:
        raise ValueError(
            "All18 Compact does not have 27 features."
        )
    if all18_artifact.get(
        "primary_external_comparison_eligible"
    ) is not False:
        raise ValueError(
            "All18 artifact is not marked secondary-only."
        )

    all18_prediction, all18_score = apply_model(
        all18_artifact,
        compact_features,
    )
    all18_frame = build_prediction_frame(
        condition=ALL18_NAME,
        trials=trials,
        prediction=all18_prediction,
        score=all18_score,
    )

    stage4_predictions = pd.read_csv(
        paths["stage4_predictions"]
    )
    stage4_predictions["external_trial_id"] = (
        stage4_predictions[
            "external_trial_id"
        ].astype(str)
    )
    common16_frame = stage4_predictions[
        stage4_predictions["feature_set"].eq(
            "soft_compact_distribution_membership"
        )
    ].copy()
    common16_frame = (
        common16_frame.set_index(
            "external_trial_id"
        )
        .loc[trial_ids]
        .reset_index()
    )
    common16_frame = common16_frame.rename(
        columns={"feature_set": "source_feature_set"}
    )
    common16_frame.insert(
        0,
        "condition",
        COMMON16_NAME,
    )
    common16_frame = common16_frame[
        all18_frame.columns
    ].copy()

    # Recompute Common16 from artifact as a consistency guard.
    common_prediction_fresh, common_score_fresh = (
        apply_model(
            common16_artifact,
            compact_features,
        )
    )
    if not np.array_equal(
        common_prediction_fresh,
        common16_frame[
            "predicted_label_nr"
        ].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "Fresh Common16 predictions differ from Stage 4."
        )
    if not np.allclose(
        common_score_fresh,
        common16_frame[
            "decision_score_nr"
        ].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Fresh Common16 scores differ from Stage 4."
        )

    predictions = pd.concat(
        [common16_frame, all18_frame],
        ignore_index=True,
    )

    pooled = build_pooled_metrics(predictions)
    subject = build_subject_metrics(predictions)
    summary = build_summary(pooled, subject)
    paired = paired_subject_tests(subject)
    bootstrap_replicates, bootstrap_difference = (
        cluster_bootstrap(
            predictions,
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
        )
    )

    common_transport = feature_transport_audit(
        condition=COMMON16_NAME,
        artifact=common16_artifact,
        features=compact_features,
    )
    all18_transport = feature_transport_audit(
        condition=ALL18_NAME,
        artifact=all18_artifact,
        features=compact_features,
    )
    transport = pd.concat(
        [common_transport, all18_transport],
        ignore_index=True,
    )

    output_paths = {
        "predictions": (
            output / "secondary_compact_predictions.csv"
        ),
        "pooled": (
            output
            / "secondary_compact_pooled_metrics.csv"
        ),
        "subject": (
            output
            / "secondary_compact_subject_metrics.csv"
        ),
        "summary": (
            output / "secondary_compact_summary.csv"
        ),
        "paired": (
            output
            / "secondary_compact_paired_subject_tests.csv"
        ),
        "bootstrap_replicates": (
            output
            / "secondary_compact_bootstrap_replicates.csv"
        ),
        "bootstrap_difference": (
            output
            / "secondary_compact_bootstrap_differences.csv"
        ),
        "transport": (
            output
            / "secondary_compact_feature_transport_audit.csv"
        ),
        "figure": (
            output
            / "figure_compact_common16_vs_all18.png"
        ),
        "text": (
            output / "secondary_compact_results_ja.md"
        ),
        "manifest": output / "stage7_manifest.json",
    }

    predictions.to_csv(
        output_paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        output_paths["pooled"],
        index=False,
        encoding="utf-8-sig",
    )
    subject.to_csv(
        output_paths["subject"],
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_paths["summary"],
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_paths["paired"],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_replicates.to_csv(
        output_paths["bootstrap_replicates"],
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_difference.to_csv(
        output_paths["bootstrap_difference"],
        index=False,
        encoding="utf-8-sig",
    )
    transport.to_csv(
        output_paths["transport"],
        index=False,
        encoding="utf-8-sig",
    )

    make_figure(
        pooled,
        output_paths["figure"],
    )
    output_paths["text"].write_text(
        build_japanese_text(
            pooled=pooled,
            subject_tests=paired,
            bootstrap=bootstrap_difference,
        ),
        encoding="utf-8",
    )

    manifest = {
        "stage": (
            "webqamgaze_external_validation_stage7_all18_secondary"
        ),
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "analysis_status": "secondary_only",
        "primary_stage4_modified": False,
        "external_data": {
            "n_participants": int(
                trials["participant_id"].nunique()
            ),
            "n_trials": int(len(trials)),
            "label_mapping": {
                "IS": 0,
                "NR": 1,
            },
        },
        "conditions": {
            COMMON16_NAME: {
                "training_scope": "common16_matched",
                "n_zuco_subjects": 16,
                "n_zuco_trials": 11324,
                "artifact": str(
                    paths["common16_artifact"]
                ),
                "artifact_sha256": sha256_file(
                    paths["common16_artifact"]
                ),
            },
            ALL18_NAME: {
                "training_scope": "all18_maxdata_secondary",
                "n_zuco_subjects": 18,
                "n_zuco_trials": 13302,
                "artifact": str(
                    paths["all18_artifact"]
                ),
                "artifact_sha256": sha256_file(
                    paths["all18_artifact"]
                ),
            },
        },
        "fixed_protocol": {
            "n_features": 27,
            "c_value": 0.01,
            "decision_threshold": 0.0,
            "no_external_retraining": True,
            "no_external_threshold_tuning": True,
            "no_external_calibration": True,
            "n_bootstraps": args.n_bootstraps,
            "bootstrap_unit": "participant",
            "seed": args.seed,
        },
        "input_files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "outputs": {
            name: str(path)
            for name, path in output_paths.items()
            if name != "manifest"
        },
        "reporting_constraint": (
            "All18 Compact is a secondary analysis and must not be "
            "inserted into the primary matched five-condition table."
        ),
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
        "=== WEBQAMGAZE STAGE 7 ALL18 SECONDARY ==="
    )
    print(
        "Primary Stage 4 results were not modified."
    )
    print(
        "External data: "
        f"{trials['participant_id'].nunique()} participants, "
        f"{len(trials)} trials"
    )
    print(
        "Compact schema: 27 features, exact match"
    )
    print(
        "Training scopes: Common16=16 subjects/11324 trials; "
        "All18=18 subjects/13302 trials"
    )
    print()
    print("Pooled metrics:")
    print(
        pooled[
            [
                "display_name",
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc_nr",
                "predicted_nr_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print("Paired subject tests:")
    print(
        paired.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )
    print()
    print("Cluster-bootstrap differences:")
    print(
        bootstrap_difference.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )
    print()
    print("Written:")
    for path in output_paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
