#!/usr/bin/env python
"""Evaluate robustness to static calibration offsets at 30 Hz.

This experiment uses the already extracted 30 Hz trial feature table and does
not reread raw Parquet files.

A static calibration offset is modeled as an additive translation applied to
all valid gaze coordinates from one subject. Because a constant translation
does not alter dispersion, correlation, speed, pupil, or quality features, the
translation can be applied exactly at feature level by shifting only the
absolute x/y location statistics.

Each nonzero magnitude is evaluated with deterministic subject-specific offset
directions. One direction is fixed for all 739 trials from a subject within a
repeat, representing a subject/session-level calibration bias.

Feature sets
------------
- dummy_reference
- position_location
- strict_absolute
- gaze_rate_invariant
- strict_rate_invariant

Protocols
---------
offset_matched:
    Training and test subjects are transformed using subject-specific offsets
    of the same magnitude.

train_clean_test_offset:
    Train on clean 30 Hz data and test the held-out subject with an offset.

The strict_rate_invariant representation should be mathematically unchanged by
a pure translation. This condition therefore acts as an implementation sanity
check as well as a robustness result.

This simulation does not model nonlinear calibration warping, scale error,
rotation, or offset-induced tracking loss.
"""

from __future__ import annotations

import argparse
import hashlib
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

X_LOCATION_COLUMNS = [
    "x_mean",
    "x_q05",
    "x_q25",
    "x_median",
    "x_q75",
    "x_q95",
]
Y_LOCATION_COLUMNS = [
    "y_mean",
    "y_q05",
    "y_q25",
    "y_median",
    "y_q75",
    "y_q95",
]


def parse_magnitudes(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if any(value < 0 for value in values):
        raise ValueError("Offset magnitudes must be nonnegative.")
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def stable_angle(
    base_seed: int,
    subject: str,
    repeat: int,
) -> float:
    payload = f"{base_seed}|{subject}|{repeat}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(digest[:8], "little")
    fraction = integer / float(2**64)
    return 2.0 * math.pi * fraction


def load_clean_features(path: Path) -> pd.DataFrame:
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

    required = {
        "trial_id",
        "subject_id",
        "task",
        "label_nr",
        *X_LOCATION_COLUMNS,
        *Y_LOCATION_COLUMNS,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Clean feature columns missing: {missing}")

    if len(data) != EXPECTED_TRIALS:
        raise ValueError(
            f"Rows={len(data)}, expected={EXPECTED_TRIALS}."
        )
    if data["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Subjects={data['subject_id'].nunique()}, "
            f"expected={EXPECTED_SUBJECTS}."
        )
    if data["trial_id"].duplicated().any():
        raise ValueError("Clean features contain duplicate trial IDs.")

    counts = data.groupby("subject_id").size()
    if not counts.eq(EXPECTED_TRIALS_PER_SUBJECT).all():
        raise ValueError(
            "Invalid per-subject trial counts:\n"
            + counts.to_string()
        )

    return data


def build_offset_table(
    clean: pd.DataFrame,
    magnitude_px: float,
    repeat: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformed = clean.copy()
    offset_rows: list[dict[str, Any]] = []

    for subject in sorted(clean["subject_id"].unique()):
        if magnitude_px == 0:
            angle_rad = 0.0
            offset_x_px = 0.0
            offset_y_px = 0.0
        else:
            angle_rad = stable_angle(
                base_seed=seed,
                subject=subject,
                repeat=repeat,
            )
            offset_x_px = magnitude_px * math.cos(angle_rad)
            offset_y_px = magnitude_px * math.sin(angle_rad)

        mask = transformed["subject_id"].eq(subject)
        transformed.loc[mask, X_LOCATION_COLUMNS] = (
            transformed.loc[mask, X_LOCATION_COLUMNS]
            + offset_x_px
        )
        transformed.loc[mask, Y_LOCATION_COLUMNS] = (
            transformed.loc[mask, Y_LOCATION_COLUMNS]
            + offset_y_px
        )

        offset_rows.append(
            {
                "subject_id": subject,
                "offset_magnitude_px": magnitude_px,
                "offset_repeat": repeat,
                "offset_angle_rad": angle_rad,
                "offset_x_px": offset_x_px,
                "offset_y_px": offset_y_px,
            }
        )

    transformed["offset_magnitude_px"] = magnitude_px
    transformed["offset_repeat"] = repeat
    return transformed, pd.DataFrame(offset_rows)


def evaluate_condition(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    protocol: str,
    magnitude_px: float,
    repeat: int,
    feature_set_name: str,
    feature_columns: list[str],
    model_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    subjects = sorted(train_table["subject_id"].unique())
    if set(subjects) != set(test_table["subject_id"].unique()):
        raise ValueError("Train and test subjects differ.")

    rows: list[dict[str, Any]] = []

    for held_out_subject in subjects:
        train = train_table[
            train_table["subject_id"] != held_out_subject
        ]
        test = test_table[
            test_table["subject_id"] == held_out_subject
        ]

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
        rows.append(
            {
                "protocol": protocol,
                "sampling_rate_hz": RATE_HZ,
                "offset_magnitude_px": magnitude_px,
                "offset_repeat": repeat,
                "feature_set": feature_set_name,
                "model": model_name,
                "held_out_subject": held_out_subject,
                "n_train": len(train),
                "n_test": len(test),
                **metrics,
            }
        )

    return rows


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
                "offset_magnitude_px",
                "feature_set",
                "model",
                "held_out_subject",
            ],
            as_index=False,
        )[metrics]
        .mean()
    )

    rows: list[dict[str, Any]] = []
    group_columns = [
        "protocol",
        "offset_magnitude_px",
        "feature_set",
        "model",
    ]

    for keys, group in subject_condition.groupby(
        group_columns,
        sort=False,
    ):
        protocol, magnitude_px, feature_set, model = keys
        row: dict[str, Any] = {
            "protocol": protocol,
            "sampling_rate_hz": RATE_HZ,
            "offset_magnitude_px": magnitude_px,
            "feature_set": feature_set,
            "model": model,
            "n_subjects": group["held_out_subject"].nunique(),
        }

        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_subject_mean"] = float(values.mean())
            row[f"{metric}_subject_sd"] = float(values.std(ddof=1))

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        [
            "protocol",
            "feature_set",
            "offset_magnitude_px",
        ]
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


def paired_tests_vs_zero(
    subject_condition: pd.DataFrame,
) -> pd.DataFrame:
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    rows: list[dict[str, Any]] = []

    for protocol in sorted(
        subject_condition["protocol"].unique()
    ):
        for feature_set in sorted(
            subject_condition["feature_set"].unique()
        ):
            subset = subject_condition[
                subject_condition["protocol"].eq(protocol)
                & subject_condition["feature_set"].eq(feature_set)
            ]

            for metric in metrics:
                baseline = (
                    subset[
                        subset["offset_magnitude_px"].eq(0)
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )

                magnitudes = sorted(
                    [
                        float(value)
                        for value in subset[
                            "offset_magnitude_px"
                        ].unique()
                        if float(value) > 0
                    ]
                )

                row_indices: list[int] = []
                p_values: list[float] = []

                for magnitude_px in magnitudes:
                    current = (
                        subset[
                            subset[
                                "offset_magnitude_px"
                            ].eq(magnitude_px)
                        ]
                        .set_index("held_out_subject")[metric]
                        .sort_index()
                    )

                    paired = pd.concat(
                        [
                            current.rename("offset"),
                            baseline.rename("clean"),
                        ],
                        axis=1,
                        join="inner",
                    ).dropna()
                    differences = paired["offset"] - paired["clean"]

                    if np.allclose(differences.to_numpy(), 0.0):
                        statistic = 0.0
                        p_value = 1.0
                    else:
                        result = wilcoxon(
                            paired["offset"],
                            paired["clean"],
                            alternative="two-sided",
                            zero_method="wilcox",
                        )
                        statistic = float(result.statistic)
                        p_value = float(result.pvalue)

                    rows.append(
                        {
                            "protocol": protocol,
                            "feature_set": feature_set,
                            "metric": metric,
                            "offset_magnitude_px": magnitude_px,
                            "reference_magnitude_px": 0.0,
                            "n_subjects": len(paired),
                            "offset_mean": float(
                                paired["offset"].mean()
                            ),
                            "clean_mean": float(
                                paired["clean"].mean()
                            ),
                            "mean_difference": float(
                                differences.mean()
                            ),
                            "wins": int(np.sum(differences > 0)),
                            "ties": int(
                                np.sum(np.isclose(differences, 0))
                            ),
                            "losses": int(np.sum(differences < 0)),
                            "wilcoxon_statistic": statistic,
                            "p_value": p_value,
                            "p_holm_within_protocol_feature_metric": (
                                math.nan
                            ),
                        }
                    )
                    row_indices.append(len(rows) - 1)
                    p_values.append(p_value)

                if p_values:
                    adjusted = holm_adjust(p_values)
                    for row_index, adjusted_value in zip(
                        row_indices,
                        adjusted,
                    ):
                        rows[row_index][
                            "p_holm_within_protocol_feature_metric"
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
        "--raw-feature-sets",
        default="outputs/raw_loso/feature_sets.json",
    )
    parser.add_argument(
        "--rate-invariant-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--magnitudes",
        default="0,25,50,100,200",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/static_offset_30hz",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    magnitudes = parse_magnitudes(args.magnitudes)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    clean = load_clean_features(Path(args.clean_features))

    raw_sets = json.loads(
        Path(args.raw_feature_sets).read_text(encoding="utf-8")
    )
    invariant_sets = json.loads(
        Path(args.rate_invariant_feature_sets).read_text(
            encoding="utf-8"
        )
    )

    feature_sets = {
        "dummy_reference": ["valid_basic_rate"],
        "position_location": raw_sets["position_location"],
        "strict_absolute": raw_sets["strict_absolute"],
        "gaze_rate_invariant": invariant_sets[
            "gaze_rate_invariant"
        ],
        "strict_rate_invariant": invariant_sets[
            "strict_rate_invariant"
        ],
    }

    missing = {
        name: sorted(set(columns) - set(clean.columns))
        for name, columns in feature_sets.items()
        if set(columns) - set(clean.columns)
    }
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    condition_tables: dict[tuple[float, int], pd.DataFrame] = {}
    offset_rows: list[pd.DataFrame] = []

    for magnitude_px in magnitudes:
        repeats = [0] if magnitude_px == 0 else list(
            range(1, args.repeats + 1)
        )
        for repeat in repeats:
            table, offsets = build_offset_table(
                clean=clean,
                magnitude_px=magnitude_px,
                repeat=repeat,
                seed=args.seed,
            )
            condition_tables[(magnitude_px, repeat)] = table
            offset_rows.append(offsets)

    folds_rows: list[dict[str, Any]] = []

    for condition, offset_table in condition_tables.items():
        magnitude_px, repeat = condition

        for feature_set_name, columns in feature_sets.items():
            model_name = (
                "dummy_most_frequent"
                if feature_set_name == "dummy_reference"
                else "linear_svm"
            )

            folds_rows.extend(
                evaluate_condition(
                    train_table=offset_table,
                    test_table=offset_table,
                    protocol="offset_matched",
                    magnitude_px=magnitude_px,
                    repeat=repeat,
                    feature_set_name=feature_set_name,
                    feature_columns=columns,
                    model_name=model_name,
                    seed=args.seed,
                )
            )

            folds_rows.extend(
                evaluate_condition(
                    train_table=clean,
                    test_table=offset_table,
                    protocol="train_clean_test_offset",
                    magnitude_px=magnitude_px,
                    repeat=repeat,
                    feature_set_name=feature_set_name,
                    feature_columns=columns,
                    model_name=model_name,
                    seed=args.seed,
                )
            )

        print(
            f"Evaluated offset={magnitude_px:g}px, "
            f"repeat={repeat}."
        )

    folds = pd.DataFrame(folds_rows)
    subject_condition, summary = summarize_subject_repeats(folds)
    paired = paired_tests_vs_zero(subject_condition)
    offsets = pd.concat(offset_rows, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    offsets_path = output_dir / "static_offset_vectors.csv"
    folds_path = output_dir / "static_offset_fold_metrics.csv"
    subject_path = (
        output_dir / "static_offset_subject_condition_metrics.csv"
    )
    summary_path = output_dir / "static_offset_summary.csv"
    paired_path = output_dir / "static_offset_paired_tests_vs_clean.csv"

    offsets.to_csv(
        offsets_path,
        index=False,
        encoding="utf-8-sig",
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

    print("\n=== STATIC OFFSET SUMMARY ===")
    print(
        summary[
            [
                "protocol",
                "offset_magnitude_px",
                "feature_set",
                "macro_f1_subject_mean",
                "macro_f1_subject_sd",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    print(f"  {offsets_path}")
    print(f"  {folds_path}")
    print(f"  {subject_path}")
    print(f"  {summary_path}")
    print(f"  {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
