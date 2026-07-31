#!/usr/bin/env python
"""Evaluate Soft-AOI sampling-rate robustness under dual holdout.

Protocols
---------
rate_matched
    Train and test at the same sampling rate.

train30_test_rate
    Train on 30 Hz trials and test the held-out subject at the target rate.

Every split simultaneously excludes:
- the held-out subject from training;
- the test sentence fold from training.

Feature sets
------------
rawstat_all
soft_full_all
soft_compact_distribution_membership
combined_full
combined_compact
dummy_reference

The compact Soft-AOI set contains 15 distribution features and 12
membership/quality features. C is fixed in advance to avoid additional
selection on degraded-rate results.
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
from sklearn.dummy import DummyClassifier

from run_zuco_soft_aoi_confound_controls import (
    METRICS,
    decision_scores,
    load_merged_data,
    make_svm,
    score_predictions,
)
from run_zuco_soft_aoi_ablation import (
    DISTRIBUTION_BASE,
    load_sentence_folds,
    prefixed,
)


EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_PER_SUBJECT = 739
BASE_RATE = 30

COMPACT_NAME = "soft_compact_distribution_membership"

FEATURE_COMPARISONS = [
    ("rawstat_all", "soft_full_all", "soft_full_vs_rawstat"),
    ("rawstat_all", COMPACT_NAME, "soft_compact_vs_rawstat"),
    ("soft_full_all", COMPACT_NAME, "compact_vs_full"),
    ("soft_full_all", "combined_full", "rawstat_added_to_full"),
    (COMPACT_NAME, "combined_compact", "rawstat_added_to_compact"),
]


def parse_rates(text: str) -> list[int]:
    values = sorted(
        {
            int(item.strip())
            for item in text.split(",")
            if item.strip()
        },
        reverse=True,
    )
    if not values:
        raise ValueError("--rates is empty.")
    if BASE_RATE not in values:
        values.append(BASE_RATE)
        values = sorted(set(values), reverse=True)
    if any(rate <= 0 or rate > 500 for rate in values):
        raise ValueError("Rates must be within 1..500 Hz.")
    return values


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return output.tolist()

    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (m - rank) * values[index])
        running = max(running, adjusted)
        output[index] = running
    return output.tolist()


def load_rate_data(
    rates: list[int],
    raw_feature_dir: Path,
    soft_feature_dir: Path,
    raw_sets_path: Path,
    soft_manifest_path: Path,
    word_template_path: Path,
    sentence_folds_path: Path,
) -> tuple[
    dict[int, pd.DataFrame],
    dict[str, list[str]],
    dict[str, Any],
    pd.DataFrame,
]:
    sentence_folds = load_sentence_folds(sentence_folds_path)
    tables: dict[int, pd.DataFrame] = {}
    reference_ids: set[str] | None = None
    reference_feature_sets: dict[str, list[str]] | None = None
    reference_metadata: dict[str, Any] | None = None

    for rate in rates:
        raw_path = raw_feature_dir / (
            f"complete_raw_18_{rate:04d}hz_features.csv"
        )
        soft_path = soft_feature_dir / (
            f"zuco2_soft_aoi_features_{rate}hz_all_subjects.csv"
        )
        data, existing_sets, _, metadata = load_merged_data(
            raw_path=raw_path,
            raw_sets_path=raw_sets_path,
            soft_path=soft_path,
            soft_manifest_path=soft_manifest_path,
            word_template_path=word_template_path,
        )
        data = data.merge(
            sentence_folds[["sentence_key", "sentence_fold"]],
            on="sentence_key",
            how="left",
            validate="many_to_one",
        )
        if data["sentence_fold"].isna().any():
            raise ValueError(f"{rate} Hz has trials without sentence folds.")
        data["sentence_fold"] = data["sentence_fold"].astype(int)

        if len(data) != EXPECTED_TRIALS:
            raise ValueError(
                f"{rate} Hz rows={len(data)}, expected={EXPECTED_TRIALS}."
            )
        if data["subject_id"].nunique() != EXPECTED_SUBJECTS:
            raise ValueError(
                f"{rate} Hz subjects={data['subject_id'].nunique()}, "
                f"expected={EXPECTED_SUBJECTS}."
            )
        if data["trial_id"].duplicated().any():
            raise ValueError(f"{rate} Hz contains duplicate trial IDs.")

        trial_ids = set(data["trial_id"])
        if reference_ids is None:
            reference_ids = trial_ids
            reference_feature_sets = existing_sets
            reference_metadata = metadata
        elif trial_ids != reference_ids:
            raise ValueError(
                f"{rate} Hz trial IDs differ from the reference rate."
            )

        tables[rate] = data.sort_values("trial_id").reset_index(drop=True)

    assert reference_feature_sets is not None
    assert reference_metadata is not None
    return (
        tables,
        reference_feature_sets,
        reference_metadata,
        sentence_folds,
    )


def build_feature_sets(
    existing_sets: dict[str, list[str]],
    metadata: dict[str, Any],
) -> dict[str, list[str]]:
    rawstat = list(existing_sets["rawstat_all"])
    soft_full = prefixed(
        list(metadata["soft_content_columns"])
        + list(metadata["soft_quality_columns"])
    )
    soft_compact = (
        prefixed(DISTRIBUTION_BASE)
        + prefixed(list(metadata["soft_quality_columns"]))
    )

    if len(rawstat) != 18:
        raise ValueError(
            f"RawStat all feature count={len(rawstat)}, expected=18."
        )
    if len(soft_full) != 41:
        raise ValueError(
            f"Soft full feature count={len(soft_full)}, expected=41."
        )
    if len(soft_compact) != 27:
        raise ValueError(
            f"Soft compact feature count={len(soft_compact)}, expected=27."
        )

    sets = {
        "dummy_reference": [],
        "rawstat_all": rawstat,
        "soft_full_all": soft_full,
        COMPACT_NAME: soft_compact,
        "combined_full": list(dict.fromkeys(rawstat + soft_full)),
        "combined_compact": list(
            dict.fromkeys(rawstat + soft_compact)
        ),
    }
    for name, columns in sets.items():
        if len(columns) != len(set(columns)):
            raise ValueError(f"{name} contains duplicate columns.")
    return sets


def evaluate(
    tables: dict[int, pd.DataFrame],
    feature_sets: dict[str, list[str]],
    rates: list[int],
    c_value: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(tables[BASE_RATE]["subject_id"].unique())
    sentence_fold_values = sorted(
        tables[BASE_RATE]["sentence_fold"].unique()
    )
    protocols = ["rate_matched", "train30_test_rate"]

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for protocol in protocols:
        print("\n" + "#" * 78)
        print(f"Protocol: {protocol}")
        for rate in rates:
            train_rate = rate if protocol == "rate_matched" else BASE_RATE
            train_table = tables[train_rate]
            test_table = tables[rate]

            print("\n" + "=" * 78)
            print(
                f"Rate {rate} Hz: train={train_rate} Hz, test={rate} Hz"
            )

            for subject_index, held_out in enumerate(subjects, start=1):
                print(
                    f"  Subject {subject_index:02d}/{len(subjects)}: "
                    f"{held_out}"
                )
                for sentence_fold in sentence_fold_values:
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
                    ].copy()

                    if test.empty:
                        raise ValueError(
                            f"Empty test: {protocol}, {rate} Hz, "
                            f"{held_out}, fold={sentence_fold}."
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
                        if columns:
                            missing_train = sorted(
                                set(columns) - set(train.columns)
                            )
                            missing_test = sorted(
                                set(columns) - set(test.columns)
                            )
                            if missing_train or missing_test:
                                raise ValueError(
                                    f"{feature_set} missing columns: "
                                    f"train={missing_train}, "
                                    f"test={missing_test}"
                                )
                            if np.isinf(
                                train[columns].to_numpy(dtype=float)
                            ).any():
                                raise ValueError(
                                    f"{feature_set} train contains infinity."
                                )
                            if np.isinf(
                                test[columns].to_numpy(dtype=float)
                            ).any():
                                raise ValueError(
                                    f"{feature_set} test contains infinity."
                                )

                            model = make_svm(c_value, seed)
                            model.fit(train[columns], y_train)
                            y_pred = model.predict(
                                test[columns]
                            ).astype(int)
                            scores = decision_scores(
                                model, test[columns]
                            )
                            used_c = float(c_value)
                        else:
                            model = DummyClassifier(
                                strategy="most_frequent"
                            )
                            x_train = np.zeros(
                                (len(train), 1), dtype=float
                            )
                            x_test = np.zeros(
                                (len(test), 1), dtype=float
                            )
                            model.fit(x_train, y_train)
                            y_pred = model.predict(x_test).astype(int)
                            scores = decision_scores(model, x_test)
                            used_c = math.nan

                        metrics = score_predictions(
                            y_test, y_pred, scores
                        )
                        fold_rows.append(
                            {
                                "protocol": protocol,
                                "sampling_rate_hz": int(rate),
                                "training_rate_hz": int(train_rate),
                                "feature_set": feature_set,
                                "held_out_subject": held_out,
                                "sentence_fold": int(sentence_fold),
                                "n_input_features": int(len(columns)),
                                "c_value": used_c,
                                "n_train": int(len(train)),
                                "n_test": int(len(test)),
                                **metrics,
                            }
                        )

                        ordered = test.reset_index(drop=True)
                        for index, row in ordered.iterrows():
                            prediction_rows.append(
                                {
                                    "protocol": protocol,
                                    "sampling_rate_hz": int(rate),
                                    "training_rate_hz": int(train_rate),
                                    "feature_set": feature_set,
                                    "held_out_subject": held_out,
                                    "sentence_fold": int(sentence_fold),
                                    "n_input_features": int(len(columns)),
                                    "trial_id": row["trial_id"],
                                    "subject_id": row["subject_id"],
                                    "task": row["task"],
                                    "task_sentence_index": int(
                                        row["task_sentence_index"]
                                    ),
                                    "sentence_key": row["sentence_key"],
                                    "label_nr": int(row["label_nr"]),
                                    "prediction_nr": int(y_pred[index]),
                                    "decision_score_nr": float(scores[index]),
                                }
                            )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    expected_per_condition = EXPECTED_TRIALS
    counts = predictions.groupby(
        ["protocol", "sampling_rate_hz", "feature_set"]
    ).size()
    if not counts.eq(expected_per_condition).all():
        raise ValueError(
            f"Incomplete prediction counts: "
            f"{counts[counts != expected_per_condition].to_dict()}"
        )
    if predictions[
        [
            "protocol",
            "sampling_rate_hz",
            "feature_set",
            "trial_id",
        ]
    ].duplicated().any():
        raise ValueError("Duplicate prediction condition/trial rows.")

    return folds, predictions


def subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "protocol",
        "sampling_rate_hz",
        "training_rate_hz",
        "feature_set",
        "held_out_subject",
    ]
    for values, group in predictions.groupby(keys, sort=False):
        (
            protocol,
            rate,
            training_rate,
            feature_set,
            subject,
        ) = values
        metrics = score_predictions(
            group["label_nr"].to_numpy(dtype=int),
            group["prediction_nr"].to_numpy(dtype=int),
            group["decision_score_nr"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "protocol": protocol,
                "sampling_rate_hz": int(rate),
                "training_rate_hz": int(training_rate),
                "feature_set": feature_set,
                "held_out_subject": subject,
                "n_input_features": int(
                    group["n_input_features"].iloc[0]
                ),
                "n_predictions": int(len(group)),
                **metrics,
            }
        )

    result = pd.DataFrame(rows)
    counts = result.groupby(
        ["protocol", "sampling_rate_hz", "feature_set"]
    )["held_out_subject"].nunique()
    if not counts.eq(EXPECTED_SUBJECTS).all():
        raise ValueError(
            f"Subject metric counts are invalid: "
            f"{counts[counts != EXPECTED_SUBJECTS].to_dict()}"
        )
    if not result["n_predictions"].eq(EXPECTED_PER_SUBJECT).all():
        raise ValueError("Not every subject has 739 predictions.")
    return result


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "protocol",
        "sampling_rate_hz",
        "training_rate_hz",
        "feature_set",
    ]
    for values, group in metrics.groupby(keys, sort=False):
        protocol, rate, training_rate, feature_set = values
        row: dict[str, Any] = {
            "protocol": protocol,
            "sampling_rate_hz": int(rate),
            "training_rate_hz": int(training_rate),
            "feature_set": feature_set,
            "n_subjects": int(len(group)),
            "n_input_features": int(
                group["n_input_features"].iloc[0]
            ),
        }
        for metric in METRICS:
            values_numeric = pd.to_numeric(
                group[metric], errors="coerce"
            )
            row[f"{metric}_subject_mean"] = float(
                values_numeric.mean()
            )
            row[f"{metric}_subject_std"] = float(
                values_numeric.std(ddof=1)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "protocol",
            "sampling_rate_hz",
            "macro_f1_subject_mean",
        ],
        ascending=[True, False, False],
    )


def paired_test(
    a: pd.Series,
    b: pd.Series,
) -> tuple[int, float, int, int, int, float]:
    paired = pd.concat(
        [a.rename("reference"), b.rename("candidate")],
        axis=1,
        join="inner",
    ).dropna()
    diff = paired["candidate"] - paired["reference"]
    if len(paired) == 0:
        p_value = math.nan
        statistic = math.nan
    elif np.allclose(diff.to_numpy(dtype=float), 0.0):
        p_value = 1.0
        statistic = 0.0
    else:
        result = wilcoxon(
            paired["candidate"],
            paired["reference"],
            alternative="two-sided",
            zero_method="wilcox",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    return (
        int(len(paired)),
        float(diff.mean()),
        int(np.sum(diff > 0)),
        int(np.sum(np.isclose(diff, 0.0))),
        int(np.sum(diff < 0)),
        p_value,
    )


def build_tests(
    metrics: pd.DataFrame,
    rates: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Feature comparisons at each protocol/rate.
    for protocol in sorted(metrics["protocol"].unique()):
        for rate in rates:
            condition = metrics[
                metrics["protocol"].eq(protocol)
                & metrics["sampling_rate_hz"].eq(rate)
            ]
            for metric in [
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc_nr",
            ]:
                for reference, candidate, comparison in FEATURE_COMPARISONS:
                    ref = (
                        condition[
                            condition["feature_set"].eq(reference)
                        ]
                        .set_index("held_out_subject")[metric]
                        .sort_index()
                    )
                    cand = (
                        condition[
                            condition["feature_set"].eq(candidate)
                        ]
                        .set_index("held_out_subject")[metric]
                        .sort_index()
                    )
                    n, mean_diff, wins, ties, losses, p_value = (
                        paired_test(ref, cand)
                    )
                    rows.append(
                        {
                            "test_family": "feature_comparison",
                            "protocol": protocol,
                            "sampling_rate_hz": int(rate),
                            "feature_set": "",
                            "metric": metric,
                            "comparison": comparison,
                            "reference": reference,
                            "candidate": candidate,
                            "n_subjects": n,
                            "mean_difference_candidate_minus_reference": (
                                mean_diff
                            ),
                            "wins": wins,
                            "ties": ties,
                            "losses": losses,
                            "p_value": p_value,
                            "p_holm_within_family_metric": math.nan,
                        }
                    )

    # Rate degradation compared with 30 Hz.
    lower_rates = [rate for rate in rates if rate != BASE_RATE]
    for protocol in sorted(metrics["protocol"].unique()):
        for feature_set in sorted(metrics["feature_set"].unique()):
            for metric in [
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc_nr",
            ]:
                baseline = (
                    metrics[
                        metrics["protocol"].eq(protocol)
                        & metrics["sampling_rate_hz"].eq(BASE_RATE)
                        & metrics["feature_set"].eq(feature_set)
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )
                for rate in lower_rates:
                    current = (
                        metrics[
                            metrics["protocol"].eq(protocol)
                            & metrics["sampling_rate_hz"].eq(rate)
                            & metrics["feature_set"].eq(feature_set)
                        ]
                        .set_index("held_out_subject")[metric]
                        .sort_index()
                    )
                    n, mean_diff, wins, ties, losses, p_value = (
                        paired_test(baseline, current)
                    )
                    rows.append(
                        {
                            "test_family": "rate_vs_30hz",
                            "protocol": protocol,
                            "sampling_rate_hz": int(rate),
                            "feature_set": feature_set,
                            "metric": metric,
                            "comparison": f"{rate}hz_vs_30hz",
                            "reference": "30hz",
                            "candidate": f"{rate}hz",
                            "n_subjects": n,
                            "mean_difference_candidate_minus_reference": (
                                mean_diff
                            ),
                            "wins": wins,
                            "ties": ties,
                            "losses": losses,
                            "p_value": p_value,
                            "p_holm_within_family_metric": math.nan,
                        }
                    )

    tests = pd.DataFrame(rows)
    for (family, metric), index in tests.groupby(
        ["test_family", "metric"]
    ).groups.items():
        positions = list(index)
        adjusted = holm_adjust(
            tests.loc[positions, "p_value"].tolist()
        )
        tests.loc[
            positions, "p_holm_within_family_metric"
        ] = adjusted
    return tests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", default="30,15")
    parser.add_argument(
        "--raw-feature-dir",
        default="outputs/downsampling_robustness",
    )
    parser.add_argument(
        "--soft-feature-dir",
        default="data/processed/zuco2/soft_aoi_all_subjects",
    )
    parser.add_argument(
        "--raw-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
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
        default="outputs/soft_aoi_rate_robustness",
    )
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.c_value <= 0:
        raise ValueError("--c-value must be positive.")

    rates = parse_rates(args.rates)
    tables, existing_sets, metadata, _ = load_rate_data(
        rates=rates,
        raw_feature_dir=Path(args.raw_feature_dir),
        soft_feature_dir=Path(args.soft_feature_dir),
        raw_sets_path=Path(args.raw_feature_sets),
        soft_manifest_path=Path(args.soft_feature_manifest),
        word_template_path=Path(args.word_template),
        sentence_folds_path=Path(args.sentence_folds_file),
    )
    feature_sets = build_feature_sets(existing_sets, metadata)

    print("=== RATE ROBUSTNESS INPUT ===")
    print(f"rates: {rates}")
    print(f"subjects: {tables[BASE_RATE]['subject_id'].nunique()}")
    print(f"trials per rate: {len(tables[BASE_RATE])}")
    print(f"fixed C: {args.c_value}")
    for name, columns in feature_sets.items():
        print(f"  {name}: {len(columns)} features")

    folds, predictions = evaluate(
        tables=tables,
        feature_sets=feature_sets,
        rates=rates,
        c_value=args.c_value,
        seed=args.seed,
    )
    metrics = subject_metrics(predictions)
    summary = summarize(metrics)
    tests = build_tests(metrics, rates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "folds": output_dir / "rate_robustness_fold_metrics.csv",
        "predictions": output_dir / "rate_robustness_predictions.csv",
        "subject_metrics": (
            output_dir / "rate_robustness_subject_metrics.csv"
        ),
        "summary": output_dir / "rate_robustness_summary.csv",
        "tests": output_dir / "rate_robustness_pairwise_tests.csv",
        "feature_sets": output_dir / "rate_robustness_feature_sets.json",
        "manifest": output_dir / "rate_robustness_manifest.json",
    }
    folds.to_csv(paths["folds"], index=False, encoding="utf-8-sig")
    predictions.to_csv(
        paths["predictions"], index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        paths["subject_metrics"], index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        paths["summary"], index=False, encoding="utf-8-sig"
    )
    tests.to_csv(paths["tests"], index=False, encoding="utf-8-sig")
    paths["feature_sets"].write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "rates_hz": rates,
                "base_rate_hz": BASE_RATE,
                "protocols": [
                    "rate_matched",
                    "train30_test_rate",
                ],
                "split": "unseen_subject_and_unseen_sentence",
                "c_value": float(args.c_value),
                "seed": int(args.seed),
                "n_subjects": EXPECTED_SUBJECTS,
                "n_trials_per_rate": EXPECTED_TRIALS,
                "feature_counts": {
                    name: len(columns)
                    for name, columns in feature_sets.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    display = summary[
        summary["feature_set"].isin(
            [
                "rawstat_all",
                "soft_full_all",
                COMPACT_NAME,
                "combined_full",
                "combined_compact",
            ]
        )
    ][
        [
            "protocol",
            "sampling_rate_hz",
            "training_rate_hz",
            "feature_set",
            "n_input_features",
            "macro_f1_subject_mean",
            "balanced_accuracy_subject_mean",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
        ]
    ]
    print("\n=== SOFT-AOI RATE ROBUSTNESS SUMMARY ===")
    print(display.to_string(index=False))

    key_tests = tests[
        tests["metric"].eq("macro_f1")
        & (
            tests["test_family"].eq("rate_vs_30hz")
            | tests["comparison"].isin(
                [
                    "soft_full_vs_rawstat",
                    "soft_compact_vs_rawstat",
                    "compact_vs_full",
                ]
            )
        )
    ][
        [
            "test_family",
            "protocol",
            "sampling_rate_hz",
            "feature_set",
            "comparison",
            "reference",
            "candidate",
            "mean_difference_candidate_minus_reference",
            "wins",
            "ties",
            "losses",
            "p_value",
            "p_holm_within_family_metric",
        ]
    ]
    print("\n=== KEY MACRO-F1 RATE TESTS ===")
    print(key_tests.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
