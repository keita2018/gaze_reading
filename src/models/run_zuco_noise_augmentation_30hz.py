#!/usr/bin/env python
"""Evaluate noise augmentation for 30 Hz raw-gaze classification.

This script reuses the clean and noisy feature caches created by
run_zuco_coordinate_noise_30hz.py. It does not reread the 500 Hz Parquet data.

Feature representation:
    strict_rate_invariant

Test conditions:
    Gaussian coordinate noise with sigma = 10, 25, 50, 100 px
    and three deterministic corruption repeats per sigma.

Training protocols
------------------
clean_train:
    Train only on clean 30 Hz features.

same_sigma_augmented:
    Train on clean features plus every available training-subject corruption
    repeat at the same sigma as the test condition.

mixed_noise_augmented:
    Train on clean features plus all supplied sigma levels and repeats.

All protocols use leave-one-subject-out evaluation. The held-out subject is
never included in training, including its augmented copies.

The training model for clean_train and mixed_noise_augmented is fit once per
held-out subject. The same-sigma model is fit once per subject and sigma, then
evaluated on all corruption repeats for that sigma.
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

from run_zuco_raw_gaze_loso import make_model, score_predictions


EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_TRIALS_PER_SUBJECT = 739
RATE_HZ = 30


def parse_sigmas(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError(
            "All --sigmas values must be positive."
        )
    return values


def load_clean_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)
    data["subject_id"] = (
        data["subject_id"].astype(str).str.upper()
    )
    data["task"] = data["task"].astype(str).str.upper()
    data["label_nr"] = data["task"].map(
        {"NR": 1, "TSR": 0}
    ).astype(int)

    validate_table(
        data=data,
        condition_name="clean",
        allow_duplicate_trial_ids=False,
    )
    return data


def load_noisy_table(
    cache_root: Path,
    subjects: list[str],
    sigma_px: float,
    repeat: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for subject in subjects:
        path = (
            cache_root
            / f"sigma_{sigma_px:06.1f}px"
            / f"repeat_{repeat:02d}"
            / (
                f"{subject}_sigma_{sigma_px:06.1f}"
                f"_repeat_{repeat:02d}.csv"
            )
        )
        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["subject_id"] = (
        data["subject_id"].astype(str).str.upper()
    )
    data["task"] = data["task"].astype(str).str.upper()
    data["label_nr"] = data["task"].map(
        {"NR": 1, "TSR": 0}
    ).astype(int)

    validate_table(
        data=data,
        condition_name=(
            f"sigma={sigma_px:g}, repeat={repeat}"
        ),
        allow_duplicate_trial_ids=False,
    )
    return data


def validate_table(
    data: pd.DataFrame,
    condition_name: str,
    allow_duplicate_trial_ids: bool,
) -> None:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "label_nr",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            f"{condition_name} missing columns: {missing}"
        )

    if len(data) != EXPECTED_TRIALS:
        raise ValueError(
            f"{condition_name} rows={len(data)}, "
            f"expected={EXPECTED_TRIALS}."
        )

    if data["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"{condition_name} subjects="
            f"{data['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    if (
        not allow_duplicate_trial_ids
        and data["trial_id"].duplicated().any()
    ):
        raise ValueError(
            f"{condition_name} contains duplicate trial IDs."
        )

    counts = data.groupby("subject_id").size()
    if not counts.eq(EXPECTED_TRIALS_PER_SUBJECT).all():
        raise ValueError(
            f"{condition_name} has invalid per-subject counts:\n"
            f"{counts.to_string()}"
        )


def fit_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
):
    model = make_model("linear_svm", seed)
    model.fit(
        train[feature_columns],
        train["label_nr"].astype(int).to_numpy(),
    )
    return model


def evaluate_model(
    model,
    test: pd.DataFrame,
    protocol: str,
    sigma_px: float,
    repeat: int,
    held_out_subject: str,
    feature_columns: list[str],
    n_train_rows: int,
) -> dict[str, Any]:
    y_true = test["label_nr"].astype(int).to_numpy()
    y_pred = model.predict(
        test[feature_columns]
    ).astype(int)
    scores = np.asarray(
        model.decision_function(
            test[feature_columns]
        ),
        dtype=float,
    )
    metrics = score_predictions(
        y_true,
        y_pred,
        scores,
    )

    return {
        "protocol": protocol,
        "sampling_rate_hz": RATE_HZ,
        "noise_sigma_px": sigma_px,
        "corruption_repeat": repeat,
        "feature_set": "strict_rate_invariant",
        "held_out_subject": held_out_subject,
        "n_train": n_train_rows,
        "n_test": len(test),
        **metrics,
    }


def summarize_subject_repeats(
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "roc_auc",
        "pr_auc_nr",
    ]

    subject_condition = (
        folds.groupby(
            [
                "protocol",
                "noise_sigma_px",
                "held_out_subject",
            ],
            as_index=False,
        )[metrics]
        .mean()
    )

    rows: list[dict[str, Any]] = []
    for keys, group in subject_condition.groupby(
        ["protocol", "noise_sigma_px"],
        sort=False,
    ):
        protocol, sigma_px = keys
        row: dict[str, Any] = {
            "protocol": protocol,
            "sampling_rate_hz": RATE_HZ,
            "noise_sigma_px": sigma_px,
            "n_subjects": group[
                "held_out_subject"
            ].nunique(),
        }

        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_subject_mean"] = float(
                values.mean()
            )
            row[f"{metric}_subject_sd"] = float(
                values.std(ddof=1)
            )

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["noise_sigma_px", "protocol"]
    )
    return subject_condition, summary


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


def paired_protocol_tests(
    subject_condition: pd.DataFrame,
    reference_protocol: str = "clean_train",
) -> pd.DataFrame:
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    candidate_protocols = [
        protocol
        for protocol in sorted(
            subject_condition["protocol"].unique()
        )
        if protocol != reference_protocol
    ]

    rows: list[dict[str, Any]] = []

    for metric in metrics:
        for candidate_protocol in candidate_protocols:
            row_indices: list[int] = []
            p_values: list[float] = []

            for sigma_px in sorted(
                subject_condition["noise_sigma_px"].unique()
            ):
                reference = (
                    subject_condition[
                        subject_condition["protocol"].eq(
                            reference_protocol
                        )
                        & subject_condition[
                            "noise_sigma_px"
                        ].eq(sigma_px)
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )
                candidate = (
                    subject_condition[
                        subject_condition["protocol"].eq(
                            candidate_protocol
                        )
                        & subject_condition[
                            "noise_sigma_px"
                        ].eq(sigma_px)
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
                    paired["candidate"]
                    - paired["reference"]
                )

                if len(paired) == 0:
                    statistic = math.nan
                    p_value = math.nan
                elif np.allclose(
                    differences.to_numpy(),
                    0.0,
                ):
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
                        "noise_sigma_px": sigma_px,
                        "candidate_protocol": (
                            candidate_protocol
                        ),
                        "reference_protocol": (
                            reference_protocol
                        ),
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
                        "wins": int(
                            np.sum(differences > 0)
                        ),
                        "ties": int(
                            np.sum(
                                np.isclose(
                                    differences,
                                    0,
                                )
                            )
                        ),
                        "losses": int(
                            np.sum(differences < 0)
                        ),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "p_holm_within_metric_candidate": (
                            math.nan
                        ),
                    }
                )
                row_indices.append(len(rows) - 1)
                p_values.append(p_value)

            finite_positions = [
                index
                for index, value in enumerate(p_values)
                if math.isfinite(value)
            ]
            if finite_positions:
                adjusted = holm_adjust(
                    [
                        p_values[index]
                        for index in finite_positions
                    ]
                )
                for local_index, adjusted_value in zip(
                    finite_positions,
                    adjusted,
                ):
                    rows[
                        row_indices[local_index]
                    ][
                        "p_holm_within_metric_candidate"
                    ] = adjusted_value

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-features",
        default=(
            "outputs/downsampling_robustness/"
            "complete_raw_18_0030hz_features.csv"
        ),
    )
    parser.add_argument(
        "--noise-cache-root",
        default=(
            "outputs/coordinate_noise_30hz/"
            "subject_noise_features"
        ),
    )
    parser.add_argument(
        "--feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--sigmas",
        default="10,25,50,100",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/noise_augmentation_30hz",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    sigmas = parse_sigmas(args.sigmas)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    clean = load_clean_table(
        Path(args.clean_features)
    )
    subjects = sorted(clean["subject_id"].unique())

    feature_sets = json.loads(
        Path(args.feature_sets).read_text(
            encoding="utf-8"
        )
    )
    feature_columns = feature_sets[
        "strict_rate_invariant"
    ]
    missing_columns = sorted(
        set(feature_columns) - set(clean.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    noise_tables: dict[
        tuple[float, int],
        pd.DataFrame,
    ] = {}

    cache_root = Path(args.noise_cache_root)
    for sigma_px in sigmas:
        for repeat in range(1, args.repeats + 1):
            table = load_noisy_table(
                cache_root=cache_root,
                subjects=subjects,
                sigma_px=sigma_px,
                repeat=repeat,
            )
            missing = sorted(
                set(feature_columns) - set(table.columns)
            )
            if missing:
                raise ValueError(
                    f"sigma={sigma_px}, repeat={repeat} "
                    f"missing features: {missing}"
                )
            noise_tables[(sigma_px, repeat)] = table

    fold_rows: list[dict[str, Any]] = []

    for fold_index, held_out_subject in enumerate(
        subjects,
        start=1,
    ):
        print(
            f"[{fold_index}/{len(subjects)}] "
            f"Held out: {held_out_subject}"
        )

        clean_train = clean[
            clean["subject_id"] != held_out_subject
        ].copy()

        clean_model = fit_model(
            train=clean_train,
            feature_columns=feature_columns,
            seed=args.seed,
        )

        mixed_frames = [clean_train]
        for table in noise_tables.values():
            mixed_frames.append(
                table[
                    table["subject_id"]
                    != held_out_subject
                ]
            )
        mixed_train = pd.concat(
            mixed_frames,
            ignore_index=True,
        )
        mixed_model = fit_model(
            train=mixed_train,
            feature_columns=feature_columns,
            seed=args.seed,
        )

        same_sigma_models: dict[float, Any] = {}
        same_sigma_sizes: dict[float, int] = {}

        for sigma_px in sigmas:
            frames = [clean_train]
            for repeat in range(
                1,
                args.repeats + 1,
            ):
                table = noise_tables[
                    (sigma_px, repeat)
                ]
                frames.append(
                    table[
                        table["subject_id"]
                        != held_out_subject
                    ]
                )
            same_train = pd.concat(
                frames,
                ignore_index=True,
            )
            same_sigma_models[sigma_px] = fit_model(
                train=same_train,
                feature_columns=feature_columns,
                seed=args.seed,
            )
            same_sigma_sizes[sigma_px] = len(
                same_train
            )

        for sigma_px in sigmas:
            for repeat in range(
                1,
                args.repeats + 1,
            ):
                test = noise_tables[
                    (sigma_px, repeat)
                ]
                test = test[
                    test["subject_id"]
                    == held_out_subject
                ]

                fold_rows.append(
                    evaluate_model(
                        model=clean_model,
                        test=test,
                        protocol="clean_train",
                        sigma_px=sigma_px,
                        repeat=repeat,
                        held_out_subject=held_out_subject,
                        feature_columns=feature_columns,
                        n_train_rows=len(clean_train),
                    )
                )
                fold_rows.append(
                    evaluate_model(
                        model=same_sigma_models[
                            sigma_px
                        ],
                        test=test,
                        protocol=(
                            "same_sigma_augmented"
                        ),
                        sigma_px=sigma_px,
                        repeat=repeat,
                        held_out_subject=held_out_subject,
                        feature_columns=feature_columns,
                        n_train_rows=(
                            same_sigma_sizes[sigma_px]
                        ),
                    )
                )
                fold_rows.append(
                    evaluate_model(
                        model=mixed_model,
                        test=test,
                        protocol=(
                            "mixed_noise_augmented"
                        ),
                        sigma_px=sigma_px,
                        repeat=repeat,
                        held_out_subject=held_out_subject,
                        feature_columns=feature_columns,
                        n_train_rows=len(mixed_train),
                    )
                )

    folds = pd.DataFrame(fold_rows)
    subject_condition, summary = (
        summarize_subject_repeats(folds)
    )
    paired = paired_protocol_tests(
        subject_condition=subject_condition,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_path = output_dir / "noise_augmentation_fold_metrics.csv"
    subject_path = (
        output_dir
        / "noise_augmentation_subject_condition_metrics.csv"
    )
    summary_path = (
        output_dir / "noise_augmentation_summary.csv"
    )
    paired_path = (
        output_dir
        / "noise_augmentation_paired_vs_clean_train.csv"
    )

    folds.to_csv(
        folds_path,
        index=False,
        encoding="utf-8-sig",
    )
    subject_condition.to_csv(
        subject_path,
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

    print("\n=== NOISE AUGMENTATION SUMMARY ===")
    print(
        summary[
            [
                "noise_sigma_px",
                "protocol",
                "macro_f1_subject_mean",
                "macro_f1_subject_sd",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== PAIRED MACRO F1 VS CLEAN TRAIN ==="
    )
    print(
        paired[
            paired["metric"].eq("macro_f1")
        ][
            [
                "noise_sigma_px",
                "candidate_protocol",
                "candidate_mean",
                "reference_mean",
                "mean_difference",
                "wins",
                "ties",
                "losses",
                "p_value",
                "p_holm_within_metric_candidate",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    print(f"  {folds_path}")
    print(f"  {subject_path}")
    print(f"  {summary_path}")
    print(f"  {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
