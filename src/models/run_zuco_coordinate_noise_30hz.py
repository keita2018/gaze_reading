#!/usr/bin/env python
"""Evaluate robustness to Gaussian coordinate noise at 30 Hz.

This experiment follows the sampling-rate diagnostics:

- Base device rate: 30 Hz.
- Feature representation: strict_rate_invariant.
- Coordinate noise is added independently to valid x/y samples.
- Existing missing samples remain missing.
- Noise does not create extra missingness: perturbed valid coordinates are
  floored at a small positive value when necessary.
- No fixation, saccade, blink-event, word-AOI, or target-position annotation
  is used.

Noise levels by default:
    sigma = 0, 10, 25, 50, 100 pixels

For sigma > 0, three deterministic corruption repeats are generated.

Protocols
---------
noise_matched:
    Training and test data use the same noise level and repeat.

train_clean_test_noisy:
    Training uses clean 30 Hz data; the held-out subject is tested with noisy
    30 Hz data.

Statistical comparison
----------------------
For every subject and noise level, metrics are averaged across corruption
repeats. Paired Wilcoxon tests compare these subject-level averages with the
clean 30 Hz baseline. Holm correction is applied across the nonzero noise
levels within each protocol and metric.

The script caches one feature CSV per subject, sigma, and repeat.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_zuco_downsampling_robustness import downsample_trial
from run_zuco_raw_gaze_loso import (
    extract_trial_features,
    make_model,
    score_predictions,
    validate_subject_feature_table,
    validate_trial_metadata,
)


BASE_RATE_HZ = 30
EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302


def parse_sigmas(text: str) -> list[float]:
    sigmas = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if any(value < 0 for value in sigmas):
        raise ValueError("Noise sigma must be nonnegative.")
    if 0.0 not in sigmas:
        sigmas.insert(0, 0.0)
    return sigmas


def stable_seed(
    base_seed: int,
    trial_id: str,
    sigma_px: float,
    repeat: int,
) -> int:
    payload = (
        f"{base_seed}|{trial_id}|{sigma_px:.8f}|{repeat}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def add_coordinate_noise(
    sampled: pd.DataFrame,
    sigma_px: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = sampled.copy()
    if sigma_px <= 0:
        return result

    x = result["x_px"].to_numpy(dtype=float)
    y = result["y_px"].to_numpy(dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )

    if np.any(valid):
        x_noise = rng.normal(
            loc=0.0,
            scale=sigma_px,
            size=int(np.sum(valid)),
        )
        y_noise = rng.normal(
            loc=0.0,
            scale=sigma_px,
            size=int(np.sum(valid)),
        )

        x[valid] = np.maximum(x[valid] + x_noise, 1e-6)
        y[valid] = np.maximum(y[valid] + y_noise, 1e-6)

    result["x_px"] = x
    result["y_px"] = y
    return result


def validate_condition_table(
    features: pd.DataFrame,
    expected_trials: pd.DataFrame,
    subject: str,
    sigma_px: float,
    repeat: int,
) -> None:
    validate_subject_feature_table(
        features=features,
        expected_trials=expected_trials,
        subject=subject,
    )

    required = {
        "sampling_rate_hz",
        "noise_sigma_px",
        "corruption_repeat",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(
            f"{subject} condition cache missing columns: {missing}"
        )

    if not pd.to_numeric(
        features["sampling_rate_hz"],
        errors="coerce",
    ).eq(BASE_RATE_HZ).all():
        raise ValueError(
            f"{subject} cache is not {BASE_RATE_HZ} Hz."
        )

    if not np.allclose(
        pd.to_numeric(
            features["noise_sigma_px"],
            errors="coerce",
        ),
        sigma_px,
        equal_nan=False,
    ):
        raise ValueError(
            f"{subject} cache has the wrong noise sigma."
        )

    if not pd.to_numeric(
        features["corruption_repeat"],
        errors="coerce",
    ).eq(repeat).all():
        raise ValueError(
            f"{subject} cache has the wrong repeat index."
        )


def extract_subject_noise_conditions(
    subject: str,
    subject_manifest: pd.DataFrame,
    subject_trials: pd.DataFrame,
    conditions: list[tuple[float, int]],
    cache_paths: dict[tuple[float, int], Path],
    base_seed: int,
) -> dict[tuple[float, int], pd.DataFrame]:
    metadata_lookup = subject_trials.set_index(
        "trial_id",
        verify_integrity=True,
    )

    rows_by_condition: dict[
        tuple[float, int],
        list[dict[str, Any]],
    ] = {
        condition: [] for condition in conditions
    }

    manifest = subject_manifest.sort_values(
        ["task", "block_id"]
    )
    if len(manifest) != 14:
        raise ValueError(
            f"{subject} manifest rows={len(manifest)}, expected 14."
        )

    for block_index, manifest_row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        parquet_path = Path(str(manifest_row.sample_parquet))
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)

        print(
            f"[{subject} {block_index}/14] "
            f"Reading for {len(conditions)} noise conditions: "
            f"{parquet_path}"
        )
        samples = pd.read_parquet(parquet_path)

        for trial_id, trial_group in samples.groupby(
            "trial_id",
            sort=False,
        ):
            if trial_id not in metadata_lookup.index:
                raise ValueError(
                    f"Parquet trial missing from metadata: {trial_id}"
                )

            sampled = downsample_trial(
                trial_group,
                BASE_RATE_HZ,
            )

            original_metadata = (
                metadata_lookup.loc[trial_id].copy()
            )
            original_metadata["trial_id"] = trial_id
            original_metadata["n_samples"] = len(sampled)

            for sigma_px, repeat in conditions:
                rng = np.random.default_rng(
                    stable_seed(
                        base_seed=base_seed,
                        trial_id=str(trial_id),
                        sigma_px=sigma_px,
                        repeat=repeat,
                    )
                )
                degraded = add_coordinate_noise(
                    sampled=sampled,
                    sigma_px=sigma_px,
                    rng=rng,
                )
                row = extract_trial_features(
                    degraded,
                    original_metadata,
                )
                row["sampling_rate_hz"] = BASE_RATE_HZ
                row["noise_sigma_px"] = sigma_px
                row["corruption_repeat"] = repeat
                rows_by_condition[(sigma_px, repeat)].append(
                    row
                )

        del samples

    results: dict[tuple[float, int], pd.DataFrame] = {}

    for condition in conditions:
        sigma_px, repeat = condition
        features = (
            pd.DataFrame(rows_by_condition[condition])
            .sort_values(
                ["task", "block_id", "trial_index_in_block"]
            )
            .reset_index(drop=True)
        )

        validate_condition_table(
            features=features,
            expected_trials=subject_trials,
            subject=subject,
            sigma_px=sigma_px,
            repeat=repeat,
        )

        output_path = cache_paths[condition]
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        features.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        results[condition] = features

    return results


def evaluate_condition(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    protocol: str,
    sigma_px: float,
    repeat: int,
    feature_columns: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    subjects = sorted(train_table["subject_id"].unique())
    if set(subjects) != set(
        test_table["subject_id"].unique()
    ):
        raise ValueError(
            "Train and test tables contain different subjects."
        )

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

        model = make_model("linear_svm", seed)
        model.fit(train[feature_columns], y_train)
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
            y_test,
            y_pred,
            scores,
        )

        rows.append(
            {
                "protocol": protocol,
                "sampling_rate_hz": BASE_RATE_HZ,
                "noise_sigma_px": sigma_px,
                "corruption_repeat": repeat,
                "feature_set": "strict_rate_invariant",
                "held_out_subject": held_out_subject,
                "n_train": len(train),
                "n_test": len(test),
                **metrics,
            }
        )

    return rows


def summarize_subject_repeat_averages(
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
            "sampling_rate_hz": BASE_RATE_HZ,
            "noise_sigma_px": sigma_px,
            "n_subjects": group["held_out_subject"].nunique(),
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
        ["protocol", "noise_sigma_px"]
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


def paired_tests_vs_clean(
    subject_condition: pd.DataFrame,
) -> pd.DataFrame:
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    rows: list[dict[str, Any]] = []

    for protocol in sorted(
        subject_condition["protocol"].unique()
    ):
        protocol_data = subject_condition[
            subject_condition["protocol"].eq(protocol)
        ]

        for metric in metrics:
            clean = (
                protocol_data[
                    protocol_data["noise_sigma_px"].eq(0)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )

            sigmas = sorted(
                [
                    float(value)
                    for value in protocol_data[
                        "noise_sigma_px"
                    ].unique()
                    if float(value) > 0
                ]
            )

            indices: list[int] = []
            p_values: list[float] = []

            for sigma_px in sigmas:
                noisy = (
                    protocol_data[
                        protocol_data[
                            "noise_sigma_px"
                        ].eq(sigma_px)
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )

                paired = pd.concat(
                    [
                        noisy.rename("noisy"),
                        clean.rename("clean"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                differences = paired["noisy"] - paired["clean"]

                if np.allclose(
                    differences.to_numpy(),
                    0.0,
                ):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    test = wilcoxon(
                        paired["noisy"],
                        paired["clean"],
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)

                rows.append(
                    {
                        "protocol": protocol,
                        "metric": metric,
                        "sampling_rate_hz": BASE_RATE_HZ,
                        "noise_sigma_px": sigma_px,
                        "reference_sigma_px": 0.0,
                        "n_subjects": len(paired),
                        "noisy_mean": float(
                            paired["noisy"].mean()
                        ),
                        "clean_mean": float(
                            paired["clean"].mean()
                        ),
                        "mean_difference": float(
                            differences.mean()
                        ),
                        "median_difference": float(
                            differences.median()
                        ),
                        "wins": int(np.sum(differences > 0)),
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
                        "p_holm_within_protocol_metric": math.nan,
                    }
                )
                indices.append(len(rows) - 1)
                p_values.append(p_value)

            if p_values:
                adjusted = holm_adjust(p_values)
                for row_index, adjusted_value in zip(
                    indices,
                    adjusted,
                ):
                    rows[row_index][
                        "p_holm_within_protocol_metric"
                    ] = adjusted_value

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "data/interim/zuco2/all_subjects/"
            "complete_raw_sample_manifest.csv"
        ),
    )
    parser.add_argument(
        "--all-trials",
        default=(
            "data/interim/zuco2/all_subjects/"
            "complete_raw_all_trials.csv"
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
        "--rate-invariant-feature-sets",
        default=(
            "outputs/downsampling_rate_invariant/"
            "rate_invariant_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--sigmas",
        default="0,10,25,50,100",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/coordinate_noise_30hz",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
    )
    args = parser.parse_args()

    sigmas = parse_sigmas(args.sigmas)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    manifest = pd.read_csv(args.manifest)
    trials = validate_trial_metadata(
        pd.read_csv(args.all_trials)
    )

    manifest["subject_id"] = (
        manifest["subject_id"]
        .astype(str)
        .str.upper()
    )
    trials["subject_id"] = (
        trials["subject_id"]
        .astype(str)
        .str.upper()
    )

    subjects = sorted(trials["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} subjects, "
            f"found {len(subjects)}."
        )

    clean = pd.read_csv(args.clean_30hz_features)
    clean["subject_id"] = (
        clean["subject_id"].astype(str).str.upper()
    )
    clean["task"] = clean["task"].astype(str).str.upper()
    clean["label_nr"] = clean["task"].map(
        {"NR": 1, "TSR": 0}
    )

    if len(clean) != EXPECTED_TRIALS:
        raise ValueError(
            f"Clean 30 Hz rows={len(clean)}, "
            f"expected={EXPECTED_TRIALS}."
        )

    feature_sets = pd.read_json(
        args.rate_invariant_feature_sets,
        typ="series",
    ).to_dict()
    feature_columns = feature_sets[
        "strict_rate_invariant"
    ]

    missing_columns = sorted(
        set(feature_columns) - set(clean.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Clean feature columns missing: {missing_columns}"
        )

    output_dir = Path(args.output_dir)
    cache_root = output_dir / "subject_noise_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean is a single deterministic condition.
    tables: dict[tuple[float, int], pd.DataFrame] = {
        (0.0, 0): clean
    }

    noisy_conditions = [
        (sigma_px, repeat)
        for sigma_px in sigmas
        if sigma_px > 0
        for repeat in range(1, args.repeats + 1)
    ]

    condition_frames: dict[
        tuple[float, int],
        list[pd.DataFrame],
    ] = {
        condition: [] for condition in noisy_conditions
    }

    for subject_index, subject in enumerate(
        subjects,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Subject {subject_index}/{len(subjects)}: "
            f"{subject}"
        )

        subject_trials = trials[
            trials["subject_id"].eq(subject)
        ].copy()
        subject_manifest = manifest[
            manifest["subject_id"].eq(subject)
        ].copy()

        cache_paths = {
            condition: (
                cache_root
                / f"sigma_{condition[0]:06.1f}px"
                / f"repeat_{condition[1]:02d}"
                / (
                    f"{subject}_sigma_{condition[0]:06.1f}"
                    f"_repeat_{condition[1]:02d}.csv"
                )
            )
            for condition in noisy_conditions
        }

        missing_conditions: list[
            tuple[float, int]
        ] = []

        for condition in noisy_conditions:
            sigma_px, repeat = condition
            cache_path = cache_paths[condition]

            if (
                cache_path.is_file()
                and not args.overwrite_features
            ):
                cached = pd.read_csv(cache_path)
                validate_condition_table(
                    features=cached,
                    expected_trials=subject_trials,
                    subject=subject,
                    sigma_px=sigma_px,
                    repeat=repeat,
                )
                condition_frames[condition].append(cached)
                print(f"Reusing cache: {cache_path}")
            else:
                missing_conditions.append(condition)

        if missing_conditions:
            extracted = extract_subject_noise_conditions(
                subject=subject,
                subject_manifest=subject_manifest,
                subject_trials=subject_trials,
                conditions=missing_conditions,
                cache_paths=cache_paths,
                base_seed=args.seed,
            )
            for condition in missing_conditions:
                condition_frames[condition].append(
                    extracted[condition]
                )

    for condition in noisy_conditions:
        combined = pd.concat(
            condition_frames[condition],
            ignore_index=True,
        )
        if len(combined) != EXPECTED_TRIALS:
            raise ValueError(
                f"Condition {condition} rows={len(combined)}, "
                f"expected={EXPECTED_TRIALS}."
            )
        if combined["trial_id"].duplicated().any():
            raise ValueError(
                f"Condition {condition} has duplicate IDs."
            )

        combined["subject_id"] = (
            combined["subject_id"]
            .astype(str)
            .str.upper()
        )
        combined["task"] = (
            combined["task"].astype(str).str.upper()
        )
        combined["label_nr"] = combined["task"].map(
            {"NR": 1, "TSR": 0}
        )
        tables[condition] = combined

    fold_rows: list[dict[str, Any]] = []

    # Clean baseline, represented once for both protocols.
    for protocol in (
        "noise_matched",
        "train_clean_test_noisy",
    ):
        fold_rows.extend(
            evaluate_condition(
                train_table=clean,
                test_table=clean,
                protocol=protocol,
                sigma_px=0.0,
                repeat=0,
                feature_columns=feature_columns,
                seed=args.seed,
            )
        )

    for condition in noisy_conditions:
        sigma_px, repeat = condition
        noisy = tables[condition]

        fold_rows.extend(
            evaluate_condition(
                train_table=noisy,
                test_table=noisy,
                protocol="noise_matched",
                sigma_px=sigma_px,
                repeat=repeat,
                feature_columns=feature_columns,
                seed=args.seed,
            )
        )

        fold_rows.extend(
            evaluate_condition(
                train_table=clean,
                test_table=noisy,
                protocol="train_clean_test_noisy",
                sigma_px=sigma_px,
                repeat=repeat,
                feature_columns=feature_columns,
                seed=args.seed,
            )
        )

        print(
            f"Evaluated sigma={sigma_px:g}px, "
            f"repeat={repeat}."
        )

    folds = pd.DataFrame(fold_rows)
    subject_condition, summary = (
        summarize_subject_repeat_averages(folds)
    )
    paired = paired_tests_vs_clean(
        subject_condition
    )

    folds_path = output_dir / "coordinate_noise_fold_metrics.csv"
    subject_path = (
        output_dir
        / "coordinate_noise_subject_condition_metrics.csv"
    )
    summary_path = (
        output_dir / "coordinate_noise_summary.csv"
    )
    paired_path = (
        output_dir
        / "coordinate_noise_paired_tests_vs_clean.csv"
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

    print("\n=== COORDINATE NOISE SUMMARY ===")
    print(
        summary[
            [
                "protocol",
                "noise_sigma_px",
                "macro_f1_subject_mean",
                "macro_f1_subject_sd",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ].to_string(index=False)
    )

    print("\n=== PAIRED TESTS: MACRO F1 VS CLEAN ===")
    print(
        paired[
            paired["metric"].eq("macro_f1")
        ][
            [
                "protocol",
                "noise_sigma_px",
                "noisy_mean",
                "clean_mean",
                "mean_difference",
                "wins",
                "ties",
                "losses",
                "p_value",
                "p_holm_within_protocol_metric",
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
