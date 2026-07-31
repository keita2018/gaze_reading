#!/usr/bin/env python
"""Final matched comparison of the robust 30 Hz proposal and official features.

This script evaluates the final candidate representation on the same 16
subjects and the same 11,324 trials used by the official sent_gaze_sacc
benchmark comparison.

Compared models
---------------
dummy_reference
    Most-frequent classifier on the matched trials.

proposed_30hz_content_clean
    30 Hz strict content-only features, trained only on clean matched trials.

proposed_30hz_content_mixed_augmented
    The same content-only representation, trained on clean matched trials plus
    all six combined degradation conditions and three corruption repeats.
    No degraded copy of the held-out subject enters training.

official_sent_gaze_sacc
    The official fixation/saccade feature vector, evaluated with the same
    LinearSVC pipeline and LOSO folds.

The test set is always the clean matched 30 Hz trial set. Thus this script
answers both:
1. How does the final proposal compare with the official feature baseline?
2. Does mixed degradation augmentation preserve clean-condition performance?

Inputs are existing feature CSV caches; raw Parquet files are not reread.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_zuco_raw_gaze_loso import make_model, score_predictions


EXPECTED_SUBJECTS = 16
EXPECTED_COMMON_TRIALS = 11324
QUALITY_COLUMNS = [
    "valid_xy_rate",
    "valid_pupil_rate",
    "valid_basic_rate",
    "longest_invalid_run_ratio",
]
COMBINED_CONDITIONS = [
    "moderate_random",
    "moderate_burst",
    "high_random",
    "high_burst",
    "extreme_random",
    "extreme_burst",
]


def prepare_labels(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()
    result["label_nr"] = result["task"].map(
        {"NR": 1, "TSR": 0}
    )
    if result["label_nr"].isna().any():
        bad = sorted(result.loc[
            result["label_nr"].isna(), "task"
        ].unique())
        raise ValueError(f"Unexpected task labels: {bad}")
    result["label_nr"] = result["label_nr"].astype(int)
    return result


def load_common_map(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    common = prepare_labels(pd.read_csv(path))
    official_columns = sorted(
        [
            column
            for column in common.columns
            if column.startswith("official_feature_")
            and column.rsplit("_", 1)[-1].isdigit()
        ],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )

    if len(common) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Common rows={len(common)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if common["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Common subjects={common['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )
    if common["trial_id"].duplicated().any():
        raise ValueError("Common map contains duplicate trial IDs.")
    if not official_columns:
        raise ValueError("Official feature columns were not found.")

    return common, official_columns


def load_clean_30hz(
    path: Path,
    common: pd.DataFrame,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    clean = prepare_labels(pd.read_csv(path))
    common_ids = set(common["trial_id"])
    clean = clean[clean["trial_id"].isin(common_ids)].copy()

    if len(clean) != EXPECTED_COMMON_TRIALS:
        raise ValueError(
            f"Matched clean rows={len(clean)}, "
            f"expected={EXPECTED_COMMON_TRIALS}."
        )
    if clean["trial_id"].duplicated().any():
        raise ValueError("Matched clean table has duplicate trial IDs.")
    if set(clean["trial_id"]) != common_ids:
        raise ValueError("Matched clean trial IDs differ from common map.")

    return clean


def load_augmented_tables(
    cache_root: Path,
    subjects: list[str],
    repeats: int,
    common_trial_ids: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for condition in COMBINED_CONDITIONS:
        for repeat in range(1, repeats + 1):
            subject_frames: list[pd.DataFrame] = []

            for subject in subjects:
                path = (
                    cache_root
                    / condition
                    / f"repeat_{repeat:02d}"
                    / (
                        f"{subject}_{condition}"
                        f"_repeat_{repeat:02d}.csv"
                    )
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                subject_frames.append(pd.read_csv(path))

            condition_table = prepare_labels(
                pd.concat(subject_frames, ignore_index=True)
            )
            condition_table = condition_table[
                condition_table["trial_id"].isin(common_trial_ids)
            ].copy()

            if len(condition_table) != EXPECTED_COMMON_TRIALS:
                raise ValueError(
                    f"{condition} repeat={repeat}: "
                    f"matched rows={len(condition_table)}, "
                    f"expected={EXPECTED_COMMON_TRIALS}."
                )
            if condition_table["trial_id"].duplicated().any():
                raise ValueError(
                    f"{condition} repeat={repeat} "
                    "has duplicate matched trial IDs."
                )

            condition_table["augmentation_condition"] = condition
            condition_table["augmentation_repeat"] = repeat
            frames.append(condition_table)

    augmented = pd.concat(frames, ignore_index=True)
    expected_rows = (
        EXPECTED_COMMON_TRIALS
        * len(COMBINED_CONDITIONS)
        * repeats
    )
    if len(augmented) != expected_rows:
        raise ValueError(
            f"Augmented rows={len(augmented)}, "
            f"expected={expected_rows}."
        )

    return augmented


def fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    y_train = train["label_nr"].astype(int).to_numpy()
    y_test = test["label_nr"].astype(int).to_numpy()

    model = make_model(model_name, seed)
    model.fit(train[feature_columns], y_train)
    y_pred = model.predict(test[feature_columns]).astype(int)

    if hasattr(model, "decision_function"):
        scores = np.asarray(
            model.decision_function(test[feature_columns]),
            dtype=float,
        )
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(
            model.predict_proba(test[feature_columns])[:, 1],
            dtype=float,
        )
    else:
        scores = y_pred.astype(float)

    metrics = score_predictions(y_test, y_pred, scores)
    return metrics, y_pred, scores


def evaluate_loso(
    clean: pd.DataFrame,
    augmented: pd.DataFrame,
    official: pd.DataFrame,
    content_columns: list[str],
    official_columns: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(clean["subject_id"].unique())
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    configurations = [
        (
            "dummy_reference",
            "dummy_most_frequent",
            "clean",
            [content_columns[0]],
        ),
        (
            "proposed_30hz_content_clean",
            "linear_svm",
            "clean",
            content_columns,
        ),
        (
            "proposed_30hz_content_mixed_augmented",
            "linear_svm",
            "mixed_augmented",
            content_columns,
        ),
        (
            "official_sent_gaze_sacc",
            "linear_svm",
            "official",
            official_columns,
        ),
    ]

    for fold_index, held_out_subject in enumerate(subjects, start=1):
        clean_train = clean[
            clean["subject_id"] != held_out_subject
        ]
        clean_test = clean[
            clean["subject_id"] == held_out_subject
        ]

        augmented_train = augmented[
            augmented["subject_id"] != held_out_subject
        ]
        mixed_train = pd.concat(
            [clean_train, augmented_train],
            ignore_index=True,
        )

        official_train = official[
            official["subject_id"] != held_out_subject
        ]
        official_test = official[
            official["subject_id"] == held_out_subject
        ]

        if set(clean_test["trial_id"]) != set(
            official_test["trial_id"]
        ):
            raise ValueError(
                f"{held_out_subject}: proposal and official "
                "test trial IDs differ."
            )

        clean_test = clean_test.sort_values("trial_id").reset_index(drop=True)
        official_test = (
            official_test.sort_values("trial_id").reset_index(drop=True)
        )
        if not clean_test["label_nr"].equals(
            official_test["label_nr"]
        ):
            raise ValueError(
                f"{held_out_subject}: test labels differ."
            )

        for (
            feature_set,
            model_name,
            train_source,
            feature_columns,
        ) in configurations:
            if train_source == "clean":
                train = clean_train
                test = clean_test
            elif train_source == "mixed_augmented":
                train = mixed_train
                test = clean_test
            elif train_source == "official":
                train = official_train
                test = official_test
            else:
                raise ValueError(train_source)

            metrics, y_pred, scores = fit_and_score(
                train=train,
                test=test,
                feature_columns=feature_columns,
                model_name=model_name,
                seed=seed,
            )

            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "fold_index": fold_index,
                    "held_out_subject": held_out_subject,
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics,
                }
            )

            for index, test_row in test.iterrows():
                predictions.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold_index": fold_index,
                        "held_out_subject": held_out_subject,
                        "trial_id": test_row["trial_id"],
                        "true_label_nr": int(
                            test_row["label_nr"]
                        ),
                        "predicted_label_nr": int(
                            y_pred[index]
                        ),
                        "score_nr": float(scores[index]),
                    }
                )

        print(
            f"[{fold_index}/{len(subjects)}] "
            f"Completed {held_out_subject}."
        )

    return pd.DataFrame(rows), pd.DataFrame(predictions)


def summarize(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "roc_auc",
        "pr_auc_nr",
    ]
    rows: list[dict[str, Any]] = []

    for (feature_set, model), group in folds.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        pred = predictions[
            predictions["feature_set"].eq(feature_set)
            & predictions["model"].eq(model)
        ]
        pooled = score_predictions(
            pred["true_label_nr"].astype(int).to_numpy(),
            pred["predicted_label_nr"].astype(int).to_numpy(),
            pred["score_nr"].astype(float).to_numpy(),
        )

        row: dict[str, Any] = {
            "feature_set": feature_set,
            "model": model,
            "n_folds": len(group),
            "n_oof_predictions": len(pred),
        }

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_sd"] = float(values.std(ddof=1))
            row[f"{metric}_pooled_oof"] = float(pooled[metric])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "macro_f1_subject_mean",
        ascending=False,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0

    for rank, original_index in enumerate(order):
        value = min(
            1.0,
            (n - rank) * p_values[original_index],
        )
        running = max(running, value)
        adjusted[original_index] = running

    return adjusted.tolist()


def paired_comparisons(
    folds: pd.DataFrame,
) -> pd.DataFrame:
    comparisons = [
        (
            "proposed_30hz_content_clean",
            "official_sent_gaze_sacc",
        ),
        (
            "proposed_30hz_content_mixed_augmented",
            "official_sent_gaze_sacc",
        ),
        (
            "proposed_30hz_content_mixed_augmented",
            "proposed_30hz_content_clean",
        ),
    ]
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    rows: list[dict[str, Any]] = []

    for metric in metrics:
        metric_indices: list[int] = []
        metric_p_values: list[float] = []

        for candidate_name, reference_name in comparisons:
            candidate = (
                folds[
                    folds["feature_set"].eq(candidate_name)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            reference = (
                folds[
                    folds["feature_set"].eq(reference_name)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )

            paired = pd.concat(
                [
                    candidate.rename("candidate"),
                    reference.rename("reference"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            differences = (
                paired["candidate"] - paired["reference"]
            )

            if np.allclose(differences.to_numpy(), 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    paired["candidate"],
                    paired["reference"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            rows.append(
                {
                    "metric": metric,
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "n_subjects": len(paired),
                    "candidate_mean": float(
                        paired["candidate"].mean()
                    ),
                    "reference_mean": float(
                        paired["reference"].mean()
                    ),
                    "mean_difference": float(
                        differences.mean()
                    ),
                    "median_difference": float(
                        differences.median()
                    ),
                    "wins": int(np.sum(differences > 0)),
                    "ties": int(
                        np.sum(np.isclose(differences, 0))
                    ),
                    "losses": int(np.sum(differences < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_within_metric": math.nan,
                }
            )
            metric_indices.append(len(rows) - 1)
            metric_p_values.append(p_value)

        adjusted = holm_adjust(metric_p_values)
        for row_index, adjusted_value in zip(
            metric_indices,
            adjusted,
        ):
            rows[row_index][
                "p_holm_within_metric"
            ] = adjusted_value

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common-map",
        default=(
            "outputs/common16_comparison/"
            "common16_trial_map.csv"
        ),
    )
    parser.add_argument(
        "--clean-30hz-features",
        default=(
            "outputs/downsampling_robustness/"
            "complete_raw_18_0030hz_features.csv"
        ),
    )
    parser.add_argument(
        "--combined-cache-root",
        default=(
            "outputs/combined_degradation_30hz/"
            "subject_combined_features"
        ),
    )
    parser.add_argument(
        "--rate-invariant-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="outputs/final_common16_comparison",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    common, official_columns = load_common_map(
        Path(args.common_map)
    )
    clean = load_clean_30hz(
        Path(args.clean_30hz_features),
        common,
    )

    feature_sets = pd.read_json(
        args.rate_invariant_feature_sets,
        typ="series",
    ).to_dict()
    strict_rate_invariant = feature_sets[
        "strict_rate_invariant"
    ]
    content_columns = [
        column
        for column in strict_rate_invariant
        if column not in QUALITY_COLUMNS
    ]

    missing = sorted(
        set(content_columns) - set(clean.columns)
    )
    if missing:
        raise ValueError(
            f"Content feature columns missing: {missing}"
        )

    subjects = sorted(common["subject_id"].unique())
    augmented = load_augmented_tables(
        cache_root=Path(args.combined_cache_root),
        subjects=subjects,
        repeats=args.repeats,
        common_trial_ids=set(common["trial_id"]),
    )

    folds, predictions = evaluate_loso(
        clean=clean,
        augmented=augmented,
        official=common,
        content_columns=content_columns,
        official_columns=official_columns,
        seed=args.seed,
    )
    summary = summarize(folds, predictions)
    paired = paired_comparisons(folds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_path = output_dir / "final_common16_fold_metrics.csv"
    predictions_path = (
        output_dir / "final_common16_predictions.csv"
    )
    summary_path = output_dir / "final_common16_summary.csv"
    paired_path = (
        output_dir / "final_common16_paired_comparisons.csv"
    )

    folds.to_csv(
        folds_path,
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        paired_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== FINAL MATCHED COMMON-TRIAL SUMMARY ===")
    print(
        summary[
            [
                "feature_set",
                "model",
                "n_folds",
                "accuracy_subject_mean",
                "macro_f1_subject_mean",
                "macro_f1_subject_sd",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
                "macro_f1_pooled_oof",
            ]
        ].to_string(index=False)
    )

    print("\n=== PAIRED COMPARISONS ===")
    print(
        paired[
            [
                "metric",
                "candidate",
                "reference",
                "candidate_mean",
                "reference_mean",
                "mean_difference",
                "wins",
                "ties",
                "losses",
                "p_value",
                "p_holm_within_metric",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    print(f"  {folds_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    print(f"  {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
