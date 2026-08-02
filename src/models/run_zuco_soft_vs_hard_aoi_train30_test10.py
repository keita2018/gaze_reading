#!/usr/bin/env python
"""Compare 30 Hz-trained Soft-AOI and Hard-AOI models on 10 Hz features.

Protocol
--------
- Train on clean 30 Hz ZuCo features.
- Test on the corresponding 10 Hz trials.
- Hold out both the test subject and the test sentence fold.
- Use the same 25 comparable features for Soft-AOI and Hard-AOI.
- Fixed LinearSVC C=0.01.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_zuco_soft_aoi_ablation import (
    holm_adjust,
    load_sentence_folds,
)
from run_zuco_soft_aoi_confound_controls import (
    METRICS,
    decision_scores,
    load_merged_data,
    make_svm,
    score_predictions,
)
from run_zuco_soft_vs_hard_aoi_clean_30hz import (
    HARD_COMMON_DISTRIBUTION,
    HARD_NAME,
    HARD_QUALITY,
    SOFT_COMMON_DISTRIBUTION,
    SOFT_NAME,
    SOFT_QUALITY,
    add_hard_features,
    audit_shared_quality,
    build_feature_sets,
    normalize_hard_table,
)


EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_PER_SUBJECT = 739
TRAIN_RATE = 30
TEST_RATE = 10


def load_soft_rate_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {
        "trial_id",
        "subject_id",
        "task",
        "task_sentence_index",
        *SOFT_COMMON_DISTRIBUTION,
        *SOFT_QUALITY,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Soft-AOI rate table lacks columns: {missing}")

    frame["trial_id"] = frame["trial_id"].astype(str)
    frame["subject_id"] = (
        frame["subject_id"].astype(str).str.upper()
    )
    frame["task"] = frame["task"].astype(str).str.upper()

    if len(frame) != EXPECTED_TRIALS:
        raise ValueError(
            f"Soft-AOI rows={len(frame)}, expected={EXPECTED_TRIALS}."
        )
    if frame["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Soft-AOI subjects={frame['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )
    if frame["trial_id"].duplicated().any():
        raise ValueError("Soft-AOI rate table has duplicate trial IDs.")

    numeric = SOFT_COMMON_DISTRIBUTION + SOFT_QUALITY
    if np.isinf(frame[numeric].to_numpy(dtype=float)).any():
        raise ValueError("Soft-AOI rate table contains infinity.")
    return frame


def build_test_table(
    metadata: pd.DataFrame,
    soft_rate: pd.DataFrame,
    hard_rate: pd.DataFrame,
) -> pd.DataFrame:
    soft_columns = SOFT_COMMON_DISTRIBUTION + SOFT_QUALITY
    hard_columns = HARD_COMMON_DISTRIBUTION + HARD_QUALITY

    soft_small = soft_rate[
        ["trial_id", "subject_id", "task"] + soft_columns
    ].rename(
        columns={
            column: f"soft__{column}"
            for column in soft_columns
        }
    )
    hard_small = hard_rate[
        ["trial_id", "subject_id", "task"] + hard_columns
    ].rename(
        columns={
            column: f"hard__{column}"
            for column in hard_columns
        }
    )

    merged = metadata.merge(
        soft_small,
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    ).merge(
        hard_small,
        on=["trial_id", "subject_id", "task"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_TRIALS:
        raise ValueError(
            f"10 Hz merged rows={len(merged)}, "
            f"expected={EXPECTED_TRIALS}."
        )
    audit_shared_quality(merged)
    return merged


def evaluate(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    c_value: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(train_table["subject_id"].unique())
    sentence_folds = sorted(
        train_table["sentence_fold"].unique()
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for subject_index, held_out in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"10 Hz direct evaluation subject "
            f"{subject_index:02d}/{len(subjects)}: {held_out}"
        )

        for sentence_fold in sentence_folds:
            train = train_table[
                (train_table["subject_id"] != held_out)
                & (
                    train_table["sentence_fold"]
                    != sentence_fold
                )
            ].copy()
            test = test_table[
                (test_table["subject_id"] == held_out)
                & (
                    test_table["sentence_fold"]
                    == sentence_fold
                )
            ].copy().reset_index(drop=True)

            if test.empty:
                raise ValueError(
                    f"Empty test split: {held_out}, "
                    f"fold={sentence_fold}."
                )
            if held_out in set(train["subject_id"]):
                raise ValueError("Subject leakage detected.")
            if set(train["sentence_key"]) & set(
                test["sentence_key"]
            ):
                raise ValueError("Sentence leakage detected.")

            y_train = train["label_nr"].to_numpy(dtype=int)
            y_test = test["label_nr"].to_numpy(dtype=int)
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError("Training split lacks both classes.")
            if set(np.unique(y_test)) != {0, 1}:
                raise ValueError("Test split lacks both classes.")

            for feature_set, columns in feature_sets.items():
                model = make_svm(c_value, seed)
                model.fit(train[columns], y_train)
                y_pred = model.predict(test[columns]).astype(int)
                scores = decision_scores(model, test[columns])
                metrics = score_predictions(
                    y_test,
                    y_pred,
                    scores,
                )

                fold_rows.append(
                    {
                        "protocol": "train30_test10_dual_holdout",
                        "training_rate_hz": TRAIN_RATE,
                        "sampling_rate_hz": TEST_RATE,
                        "feature_set": feature_set,
                        "held_out_subject": held_out,
                        "sentence_fold": int(sentence_fold),
                        "n_input_features": int(len(columns)),
                        "c_value": float(c_value),
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        **metrics,
                    }
                )

                for row_index, row in test.iterrows():
                    prediction_rows.append(
                        {
                            "protocol": (
                                "train30_test10_dual_holdout"
                            ),
                            "training_rate_hz": TRAIN_RATE,
                            "sampling_rate_hz": TEST_RATE,
                            "feature_set": feature_set,
                            "held_out_subject": held_out,
                            "sentence_fold": int(sentence_fold),
                            "trial_id": str(row["trial_id"]),
                            "subject_id": str(row["subject_id"]),
                            "task": str(row["task"]),
                            "task_sentence_index": int(
                                row["task_sentence_index"]
                            ),
                            "sentence_key": str(
                                row["sentence_key"]
                            ),
                            "label_nr": int(row["label_nr"]),
                            "prediction_nr": int(
                                y_pred[row_index]
                            ),
                            "decision_score_nr": float(
                                scores[row_index]
                            ),
                        }
                    )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    counts = predictions.groupby(
        ["feature_set"]
    ).size()
    if not counts.eq(EXPECTED_TRIALS).all():
        raise ValueError(
            f"Prediction counts are incomplete: {counts.to_dict()}"
        )
    return folds, predictions


def subject_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature_set, subject), group in predictions.groupby(
        ["feature_set", "held_out_subject"],
        sort=True,
    ):
        rows.append(
            {
                "protocol": "train30_test10_dual_holdout",
                "training_rate_hz": TRAIN_RATE,
                "sampling_rate_hz": TEST_RATE,
                "feature_set": feature_set,
                "held_out_subject": subject,
                "n_predictions": int(len(group)),
                **score_predictions(
                    group["label_nr"].to_numpy(dtype=int),
                    group["prediction_nr"].to_numpy(dtype=int),
                    group["decision_score_nr"].to_numpy(
                        dtype=float
                    ),
                ),
            }
        )

    result = pd.DataFrame(rows)
    if not result["n_predictions"].eq(
        EXPECTED_PER_SUBJECT
    ).all():
        raise ValueError(
            "Not every subject has 739 predictions."
        )
    return result


def summarize(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in metrics.groupby(
        "feature_set",
        sort=True,
    ):
        row: dict[str, Any] = {
            "protocol": "train30_test10_dual_holdout",
            "training_rate_hz": TRAIN_RATE,
            "sampling_rate_hz": TEST_RATE,
            "feature_set": feature_set,
            "n_subjects": int(
                group["held_out_subject"].nunique()
            ),
            "n_input_features": 25,
        }
        for metric in METRICS:
            row[f"{metric}_subject_mean"] = float(
                group[metric].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_tests(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in METRICS:
        soft = (
            metrics[
                metrics["feature_set"].eq(SOFT_NAME)
            ]
            .set_index("held_out_subject")[metric]
            .sort_index()
        )
        hard = (
            metrics[
                metrics["feature_set"].eq(HARD_NAME)
            ]
            .set_index("held_out_subject")[metric]
            .sort_index()
        )
        paired = pd.concat(
            [
                hard.rename("hard"),
                soft.rename("soft"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        difference = paired["soft"] - paired["hard"]

        if np.allclose(
            difference.to_numpy(dtype=float),
            0.0,
        ):
            statistic = 0.0
            p_value = 1.0
        else:
            test = wilcoxon(
                paired["soft"],
                paired["hard"],
                alternative="two-sided",
                zero_method="wilcox",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)

        rows.append(
            {
                "protocol": "train30_test10_dual_holdout",
                "training_rate_hz": TRAIN_RATE,
                "sampling_rate_hz": TEST_RATE,
                "metric": metric,
                "hard_mean": float(paired["hard"].mean()),
                "soft_mean": float(paired["soft"].mean()),
                "mean_difference_soft_minus_hard": float(
                    difference.mean()
                ),
                "soft_wins": int(np.sum(difference > 0)),
                "ties": int(
                    np.sum(np.isclose(difference, 0.0))
                ),
                "soft_losses": int(np.sum(difference < 0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "p_holm_across_metrics": math.nan,
            }
        )

    adjusted = holm_adjust(
        [float(row["p_value"]) for row in rows]
    )
    for row, adjusted_value in zip(rows, adjusted):
        row["p_holm_across_metrics"] = adjusted_value
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-raw-features",
        default=(
            "outputs/downsampling_robustness/"
            "complete_raw_18_0030hz_features.csv"
        ),
    )
    parser.add_argument(
        "--raw-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--clean-soft-features",
        default=(
            "data/processed/zuco2/soft_aoi_all_subjects/"
            "zuco2_soft_aoi_features_30hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--soft-10hz-features",
        default=(
            "data/processed/zuco2/soft_aoi_all_subjects/"
            "zuco2_soft_aoi_features_10hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--soft-feature-manifest",
        default=(
            "data/processed/zuco2/YAC/soft_aoi/"
            "YAC_soft_aoi_feature_manifest_30hz.json"
        ),
    )
    parser.add_argument(
        "--clean-hard-features",
        default=(
            "data/processed/zuco2/hard_aoi_all_subjects/"
            "zuco2_hard_aoi_features_30hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--hard-10hz-features",
        default=(
            "data/processed/zuco2/hard_aoi_all_subjects/"
            "zuco2_hard_aoi_features_10hz_all_subjects.csv"
        ),
    )
    parser.add_argument(
        "--word-template",
        default=(
            "data/processed/zuco2/soft_aoi_template/"
            "zuco_soft_aoi_word_template.csv"
        ),
    )
    parser.add_argument(
        "--sentence-folds-file",
        default=(
            "outputs/soft_aoi_confound_controls_30hz/"
            "dual_holdout_sentence_folds.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/soft_vs_hard_aoi_train30_test10",
    )
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    clean, _, _, _ = load_merged_data(
        raw_path=Path(args.clean_raw_features),
        raw_sets_path=Path(args.raw_feature_sets),
        soft_path=Path(args.clean_soft_features),
        soft_manifest_path=Path(
            args.soft_feature_manifest
        ),
        word_template_path=Path(args.word_template),
    )
    clean = add_hard_features(
        clean,
        normalize_hard_table(
            Path(args.clean_hard_features)
        ),
    )
    audit_shared_quality(clean)

    sentence_folds = load_sentence_folds(
        Path(args.sentence_folds_file)
    )
    clean = clean.merge(
        sentence_folds[
            ["sentence_key", "sentence_fold"]
        ],
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if clean["sentence_fold"].isna().any():
        raise ValueError("Clean data lacks sentence folds.")
    clean["sentence_fold"] = (
        clean["sentence_fold"].astype(int)
    )

    soft_10hz = load_soft_rate_table(
        Path(args.soft_10hz_features)
    )
    hard_10hz = normalize_hard_table(
        Path(args.hard_10hz_features)
    )

    metadata = clean[
        [
            "trial_id",
            "subject_id",
            "task",
            "task_sentence_index",
            "sentence_key",
            "label_nr",
            "sentence_fold",
        ]
    ].copy()
    test_10hz = build_test_table(
        metadata,
        soft_10hz,
        hard_10hz,
    )

    feature_sets = build_feature_sets()

    print("=== SOFT VS HARD TRAIN30 TEST10 INPUT ===")
    print(f"subjects: {clean['subject_id'].nunique()}")
    print(f"trials: {len(clean)}")
    print(f"training rate: {TRAIN_RATE} Hz")
    print(f"test rate: {TEST_RATE} Hz")
    print(f"fixed C: {args.c_value}")
    print("shared quality audit: PASS at 30 Hz and 10 Hz")

    folds, predictions = evaluate(
        train_table=clean,
        test_table=test_10hz,
        feature_sets=feature_sets,
        c_value=args.c_value,
        seed=args.seed,
    )
    metrics = subject_metrics(predictions)
    summary = summarize(metrics)
    tests = build_tests(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "folds": output_dir / "soft_vs_hard_train30_test10_folds.csv",
        "predictions": (
            output_dir / "soft_vs_hard_train30_test10_predictions.csv"
        ),
        "subject_metrics": (
            output_dir
            / "soft_vs_hard_train30_test10_subject_metrics.csv"
        ),
        "summary": (
            output_dir / "soft_vs_hard_train30_test10_summary.csv"
        ),
        "tests": (
            output_dir / "soft_vs_hard_train30_test10_tests.csv"
        ),
        "manifest": (
            output_dir / "soft_vs_hard_train30_test10_manifest.json"
        ),
    }

    folds.to_csv(
        paths["folds"],
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        paths["subject_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        paths["summary"],
        index=False,
        encoding="utf-8-sig",
    )
    tests.to_csv(
        paths["tests"],
        index=False,
        encoding="utf-8-sig",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "protocol": (
                    "train30_test10_dual_holdout"
                ),
                "training_rate_hz": TRAIN_RATE,
                "sampling_rate_hz": TEST_RATE,
                "c_value": float(args.c_value),
                "seed": int(args.seed),
                "feature_sets": feature_sets,
                "hard_assignment": (
                    "argmax_q_with_qmax_mass"
                ),
                "shared_quality_audit": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== SOFT-AOI VS HARD-AOI TRAIN30 TEST10 SUMMARY ===")
    print(
        summary[
            [
                "feature_set",
                "n_input_features",
                "macro_f1_subject_mean",
                "balanced_accuracy_subject_mean",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ].to_string(index=False)
    )

    print("\n=== SOFT-AOI VS HARD-AOI TRAIN30 TEST10 TESTS ===")
    print(
        tests[
            tests["metric"].isin(
                ["macro_f1", "roc_auc", "pr_auc_nr"]
            )
        ][
            [
                "metric",
                "hard_mean",
                "soft_mean",
                "mean_difference_soft_minus_hard",
                "soft_wins",
                "ties",
                "soft_losses",
                "p_value",
                "p_holm_across_metrics",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
