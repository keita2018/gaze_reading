#!/usr/bin/env python
"""Evaluate robustness to random and burst missingness at 30 Hz.

This experiment downsamples the continuous raw eye-tracking stream to 30 Hz,
then injects additional missing samples before recomputing fixation-free
features.

Missingness modes
-----------------
random:
    Independently removes valid samples until the requested additional missing
    fraction is reached.

burst:
    Removes the same target fraction in contiguous one-second windows. This
    models temporary tracking loss while holding the total missing fraction
    approximately equal to the random condition.

Rates:
    0, 5, 10, 20, and 40 percent additional missingness by default.

For every nonzero mode/rate combination, three deterministic corruption
repeats are generated. Existing invalid samples remain invalid and are not
counted as newly removed samples.

Protocols
---------
missingness_matched:
    Training and test subjects use the same missingness mode, rate, and repeat.

train_clean_test_missing:
    Training uses clean 30 Hz features and the held-out subject is evaluated
    under missingness.

Feature sets
------------
quality_only:
    Validity rates and longest invalid run.

gaze_content_only:
    Position dispersion and speed; excludes quality statistics.

strict_content_only:
    Gaze content plus rate-invariant relative pupil variation; excludes quality.

strict_rate_invariant:
    The preceding content features plus quality statistics.

The with/without-quality comparison determines whether performance changes are
caused by loss of gaze information or by a shift in validity-rate features.

Outputs are cached per subject, mode, rate, and repeat.
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

from run_zuco_downsampling_robustness import downsample_trial
from run_zuco_raw_gaze_loso import (
    extract_trial_features,
    make_model,
    score_predictions,
    validate_subject_feature_table,
    validate_trial_metadata,
)


RATE_HZ = 30
EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_TRIALS_PER_SUBJECT = 739
DEFAULT_BURST_SECONDS = 1.0


def parse_percentages(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if any(value < 0 or value >= 100 for value in values):
        raise ValueError(
            "Missingness percentages must be within [0, 100)."
        )
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def stable_seed(
    base_seed: int,
    trial_id: str,
    mode: str,
    percentage: float,
    repeat: int,
) -> int:
    payload = (
        f"{base_seed}|{trial_id}|{mode}|"
        f"{percentage:.8f}|{repeat}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def current_valid_basic_mask(samples: pd.DataFrame) -> np.ndarray:
    x = samples["x_px"].to_numpy(dtype=float)
    y = samples["y_px"].to_numpy(dtype=float)
    pupil = samples["pupil_area"].to_numpy(dtype=float)
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(pupil)
        & (x > 0)
        & (y > 0)
        & (pupil > 0)
    )


def apply_missing_indices(
    samples: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    result = samples.copy()
    if indices.size == 0:
        return result

    columns = ["x_px", "y_px", "pupil_area"]
    column_indices = [
        result.columns.get_loc(column)
        for column in columns
    ]
    result.iloc[indices, column_indices] = 0.0
    return result


def add_random_missingness(
    samples: pd.DataFrame,
    percentage: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    valid_indices = np.flatnonzero(
        current_valid_basic_mask(samples)
    )
    target = int(
        round(valid_indices.size * percentage / 100.0)
    )
    target = min(target, valid_indices.size)

    if target <= 0:
        return samples.copy(), 0

    selected = np.asarray(
        rng.choice(
            valid_indices,
            size=target,
            replace=False,
        ),
        dtype=np.int64,
    )
    return apply_missing_indices(samples, selected), target


def add_burst_missingness(
    samples: pd.DataFrame,
    percentage: float,
    burst_seconds: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    valid_before = current_valid_basic_mask(samples)
    valid_count = int(np.sum(valid_before))
    target = int(round(valid_count * percentage / 100.0))
    target = min(target, valid_count)

    if target <= 0:
        return samples.copy(), 0

    n_samples = len(samples)
    burst_samples = max(
        1,
        int(round(burst_seconds * RATE_HZ)),
    )
    selected = np.zeros(n_samples, dtype=bool)
    newly_selected_valid = 0
    attempts = 0
    max_attempts = max(1000, n_samples * 20)

    while (
        newly_selected_valid < target
        and attempts < max_attempts
    ):
        attempts += 1
        start = int(rng.integers(0, n_samples))
        end = min(n_samples, start + burst_samples)

        candidates = np.arange(start, end, dtype=np.int64)
        candidates = candidates[
            valid_before[candidates]
            & ~selected[candidates]
        ]
        remaining = target - newly_selected_valid
        if candidates.size > remaining:
            candidates = candidates[:remaining]

        if candidates.size:
            selected[candidates] = True
            newly_selected_valid += int(candidates.size)

    # Guaranteed deterministic completion if overlapping bursts could not
    # reach the exact target.
    if newly_selected_valid < target:
        remaining_indices = np.flatnonzero(
            valid_before & ~selected
        )
        needed = target - newly_selected_valid
        selected[remaining_indices[:needed]] = True
        newly_selected_valid += needed

    selected_indices = np.flatnonzero(selected)
    return (
        apply_missing_indices(samples, selected_indices),
        newly_selected_valid,
    )


def degrade_trial(
    sampled: pd.DataFrame,
    mode: str,
    percentage: float,
    burst_seconds: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    if percentage <= 0 or mode == "clean":
        return sampled.copy(), 0
    if mode == "random":
        return add_random_missingness(
            samples=sampled,
            percentage=percentage,
            rng=rng,
        )
    if mode == "burst":
        return add_burst_missingness(
            samples=sampled,
            percentage=percentage,
            burst_seconds=burst_seconds,
            rng=rng,
        )
    raise ValueError(f"Unknown missingness mode: {mode}")


def validate_condition_table(
    features: pd.DataFrame,
    expected_trials: pd.DataFrame,
    subject: str,
    mode: str,
    percentage: float,
    repeat: int,
) -> None:
    validate_subject_feature_table(
        features=features,
        expected_trials=expected_trials,
        subject=subject,
    )

    required = {
        "sampling_rate_hz",
        "missingness_mode",
        "additional_missing_percent",
        "corruption_repeat",
        "additional_missing_samples",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(
            f"{subject} cache missing columns: {missing}"
        )

    if not pd.to_numeric(
        features["sampling_rate_hz"],
        errors="coerce",
    ).eq(RATE_HZ).all():
        raise ValueError(
            f"{subject} cache is not {RATE_HZ} Hz."
        )

    if not features["missingness_mode"].astype(str).eq(
        mode
    ).all():
        raise ValueError(
            f"{subject} cache has the wrong mode."
        )

    if not np.allclose(
        pd.to_numeric(
            features["additional_missing_percent"],
            errors="coerce",
        ),
        percentage,
    ):
        raise ValueError(
            f"{subject} cache has the wrong percentage."
        )

    if not pd.to_numeric(
        features["corruption_repeat"],
        errors="coerce",
    ).eq(repeat).all():
        raise ValueError(
            f"{subject} cache has the wrong repeat."
        )


def extract_subject_conditions(
    subject: str,
    subject_manifest: pd.DataFrame,
    subject_trials: pd.DataFrame,
    conditions: list[tuple[str, float, int]],
    cache_paths: dict[tuple[str, float, int], Path],
    base_seed: int,
    burst_seconds: float,
) -> dict[tuple[str, float, int], pd.DataFrame]:
    metadata_lookup = subject_trials.set_index(
        "trial_id",
        verify_integrity=True,
    )
    rows_by_condition: dict[
        tuple[str, float, int],
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
            f"Reading for {len(conditions)} missingness conditions: "
            f"{parquet_path}"
        )
        samples = pd.read_parquet(parquet_path)

        for trial_id, trial_group in samples.groupby(
            "trial_id",
            sort=False,
        ):
            sampled = downsample_trial(
                trial_group,
                RATE_HZ,
            )

            if trial_id not in metadata_lookup.index:
                raise ValueError(
                    f"Trial missing from metadata: {trial_id}"
                )

            metadata = metadata_lookup.loc[trial_id].copy()
            metadata["trial_id"] = trial_id

            original_valid = int(
                np.sum(current_valid_basic_mask(sampled))
            )

            for mode, percentage, repeat in conditions:
                rng = np.random.default_rng(
                    stable_seed(
                        base_seed=base_seed,
                        trial_id=str(trial_id),
                        mode=mode,
                        percentage=percentage,
                        repeat=repeat,
                    )
                )
                degraded, removed = degrade_trial(
                    sampled=sampled,
                    mode=mode,
                    percentage=percentage,
                    burst_seconds=burst_seconds,
                    rng=rng,
                )

                condition_metadata = metadata.copy()
                condition_metadata["n_samples"] = len(degraded)

                row = extract_trial_features(
                    degraded,
                    condition_metadata,
                )
                row["sampling_rate_hz"] = RATE_HZ
                row["missingness_mode"] = mode
                row["additional_missing_percent"] = percentage
                row["corruption_repeat"] = repeat
                row["additional_missing_samples"] = removed
                row["original_valid_basic_samples"] = original_valid
                row["realized_additional_missing_percent"] = (
                    100.0 * removed / original_valid
                    if original_valid > 0
                    else math.nan
                )
                rows_by_condition[
                    (mode, percentage, repeat)
                ].append(row)

        del samples

    results: dict[
        tuple[str, float, int],
        pd.DataFrame,
    ] = {}

    for condition in conditions:
        mode, percentage, repeat = condition
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
            mode=mode,
            percentage=percentage,
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
    mode: str,
    percentage: float,
    repeat: int,
    feature_set_name: str,
    feature_columns: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    subjects = sorted(train_table["subject_id"].unique())
    if set(subjects) != set(
        test_table["subject_id"].unique()
    ):
        raise ValueError("Train/test subjects differ.")

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
                "sampling_rate_hz": RATE_HZ,
                "missingness_mode": mode,
                "additional_missing_percent": percentage,
                "corruption_repeat": repeat,
                "feature_set": feature_set_name,
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
                "missingness_mode",
                "additional_missing_percent",
                "feature_set",
                "held_out_subject",
            ],
            as_index=False,
        )[metrics]
        .mean()
    )

    rows: list[dict[str, Any]] = []
    group_columns = [
        "protocol",
        "missingness_mode",
        "additional_missing_percent",
        "feature_set",
    ]

    for keys, group in subject_condition.groupby(
        group_columns,
        sort=False,
    ):
        protocol, mode, percentage, feature_set = keys
        row: dict[str, Any] = {
            "protocol": protocol,
            "sampling_rate_hz": RATE_HZ,
            "missingness_mode": mode,
            "additional_missing_percent": percentage,
            "feature_set": feature_set,
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
        [
            "protocol",
            "feature_set",
            "missingness_mode",
            "additional_missing_percent",
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


def paired_tests_vs_clean(
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
            feature_data = subject_condition[
                subject_condition["protocol"].eq(protocol)
                & subject_condition["feature_set"].eq(feature_set)
            ]

            clean = (
                feature_data[
                    feature_data["missingness_mode"].eq("clean")
                    & feature_data[
                        "additional_missing_percent"
                    ].eq(0)
                ]
                .set_index("held_out_subject")
            )

            for metric in metrics:
                row_indices: list[int] = []
                p_values: list[float] = []

                degraded_conditions = (
                    feature_data[
                        ~feature_data[
                            "missingness_mode"
                        ].eq("clean")
                    ][
                        [
                            "missingness_mode",
                            "additional_missing_percent",
                        ]
                    ]
                    .drop_duplicates()
                    .sort_values(
                        [
                            "missingness_mode",
                            "additional_missing_percent",
                        ]
                    )
                )

                for condition in degraded_conditions.itertuples(
                    index=False
                ):
                    mode = str(condition.missingness_mode)
                    percentage = float(
                        condition.additional_missing_percent
                    )

                    current = (
                        feature_data[
                            feature_data[
                                "missingness_mode"
                            ].eq(mode)
                            & feature_data[
                                "additional_missing_percent"
                            ].eq(percentage)
                        ]
                        .set_index("held_out_subject")[metric]
                        .sort_index()
                    )
                    baseline = clean[metric].sort_index()

                    paired = pd.concat(
                        [
                            current.rename("missing"),
                            baseline.rename("clean"),
                        ],
                        axis=1,
                        join="inner",
                    ).dropna()
                    differences = (
                        paired["missing"] - paired["clean"]
                    )

                    if np.allclose(
                        differences.to_numpy(),
                        0.0,
                    ):
                        statistic = 0.0
                        p_value = 1.0
                    else:
                        test = wilcoxon(
                            paired["missing"],
                            paired["clean"],
                            alternative="two-sided",
                            zero_method="wilcox",
                        )
                        statistic = float(test.statistic)
                        p_value = float(test.pvalue)

                    rows.append(
                        {
                            "protocol": protocol,
                            "feature_set": feature_set,
                            "metric": metric,
                            "missingness_mode": mode,
                            "additional_missing_percent": percentage,
                            "n_subjects": len(paired),
                            "missing_mean": float(
                                paired["missing"].mean()
                            ),
                            "clean_mean": float(
                                paired["clean"].mean()
                            ),
                            "mean_difference": float(
                                differences.mean()
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
        "--percentages",
        default="0,5,10,20,40",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--burst-seconds",
        type=float,
        default=DEFAULT_BURST_SECONDS,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/missingness_30hz",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
    )
    args = parser.parse_args()

    percentages = parse_percentages(args.percentages)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.burst_seconds <= 0:
        raise ValueError("--burst-seconds must be positive.")

    manifest = pd.read_csv(args.manifest)
    trials = validate_trial_metadata(
        pd.read_csv(args.all_trials)
    )
    manifest["subject_id"] = (
        manifest["subject_id"].astype(str).str.upper()
    )
    trials["subject_id"] = (
        trials["subject_id"].astype(str).str.upper()
    )

    clean = pd.read_csv(args.clean_30hz_features)
    clean["subject_id"] = (
        clean["subject_id"].astype(str).str.upper()
    )
    clean["task"] = clean["task"].astype(str).str.upper()
    clean["label_nr"] = clean["task"].map(
        {"NR": 1, "TSR": 0}
    ).astype(int)

    if len(clean) != EXPECTED_TRIALS:
        raise ValueError(
            f"Clean rows={len(clean)}, expected={EXPECTED_TRIALS}."
        )

    invariant_sets = json.loads(
        Path(args.rate_invariant_feature_sets).read_text(
            encoding="utf-8"
        )
    )

    quality = [
        "valid_xy_rate",
        "valid_pupil_rate",
        "valid_basic_rate",
        "longest_invalid_run_ratio",
    ]
    gaze_content = [
        column
        for column in invariant_sets[
            "gaze_rate_invariant"
        ]
        if column not in quality
    ]
    pupil_content = invariant_sets[
        "pupil_rate_invariant"
    ]

    feature_sets = {
        "quality_only": quality,
        "gaze_content_only": gaze_content,
        "strict_content_only": list(
            dict.fromkeys(gaze_content + pupil_content)
        ),
        "strict_rate_invariant": invariant_sets[
            "strict_rate_invariant"
        ],
    }

    missing_columns = {
        name: sorted(set(columns) - set(clean.columns))
        for name, columns in feature_sets.items()
        if set(columns) - set(clean.columns)
    }
    if missing_columns:
        raise ValueError(
            f"Clean feature columns missing: {missing_columns}"
        )

    subjects = sorted(trials["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} subjects, "
            f"found {len(subjects)}."
        )

    nonzero_percentages = [
        value for value in percentages if value > 0
    ]
    conditions = [
        (mode, percentage, repeat)
        for mode in ("random", "burst")
        for percentage in nonzero_percentages
        for repeat in range(1, args.repeats + 1)
    ]

    output_dir = Path(args.output_dir)
    cache_root = output_dir / "subject_missingness_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_frames: dict[
        tuple[str, float, int],
        list[pd.DataFrame],
    ] = {
        condition: [] for condition in conditions
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
                / condition[0]
                / f"missing_{condition[1]:05.1f}pct"
                / f"repeat_{condition[2]:02d}"
                / (
                    f"{subject}_{condition[0]}"
                    f"_{condition[1]:05.1f}pct"
                    f"_repeat_{condition[2]:02d}.csv"
                )
            )
            for condition in conditions
        }

        missing_conditions: list[
            tuple[str, float, int]
        ] = []

        for condition in conditions:
            mode, percentage, repeat = condition
            path = cache_paths[condition]

            if (
                path.is_file()
                and not args.overwrite_features
            ):
                cached = pd.read_csv(path)
                validate_condition_table(
                    features=cached,
                    expected_trials=subject_trials,
                    subject=subject,
                    mode=mode,
                    percentage=percentage,
                    repeat=repeat,
                )
                condition_frames[condition].append(cached)
                print(f"Reusing cache: {path}")
            else:
                missing_conditions.append(condition)

        if missing_conditions:
            extracted = extract_subject_conditions(
                subject=subject,
                subject_manifest=subject_manifest,
                subject_trials=subject_trials,
                conditions=missing_conditions,
                cache_paths=cache_paths,
                base_seed=args.seed,
                burst_seconds=args.burst_seconds,
            )
            for condition in missing_conditions:
                condition_frames[condition].append(
                    extracted[condition]
                )

    condition_tables: dict[
        tuple[str, float, int],
        pd.DataFrame,
    ] = {}

    for condition in conditions:
        combined = pd.concat(
            condition_frames[condition],
            ignore_index=True,
        )
        if len(combined) != EXPECTED_TRIALS:
            raise ValueError(
                f"{condition} rows={len(combined)}, "
                f"expected={EXPECTED_TRIALS}."
            )
        if combined["trial_id"].duplicated().any():
            raise ValueError(
                f"{condition} has duplicate trial IDs."
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
        ).astype(int)
        condition_tables[condition] = combined

    fold_rows: list[dict[str, Any]] = []

    for protocol in (
        "missingness_matched",
        "train_clean_test_missing",
    ):
        for feature_set_name, columns in feature_sets.items():
            fold_rows.extend(
                evaluate_condition(
                    train_table=clean,
                    test_table=clean,
                    protocol=protocol,
                    mode="clean",
                    percentage=0.0,
                    repeat=0,
                    feature_set_name=feature_set_name,
                    feature_columns=columns,
                    seed=args.seed,
                )
            )

    for condition, degraded in condition_tables.items():
        mode, percentage, repeat = condition

        for feature_set_name, columns in feature_sets.items():
            fold_rows.extend(
                evaluate_condition(
                    train_table=degraded,
                    test_table=degraded,
                    protocol="missingness_matched",
                    mode=mode,
                    percentage=percentage,
                    repeat=repeat,
                    feature_set_name=feature_set_name,
                    feature_columns=columns,
                    seed=args.seed,
                )
            )
            fold_rows.extend(
                evaluate_condition(
                    train_table=clean,
                    test_table=degraded,
                    protocol="train_clean_test_missing",
                    mode=mode,
                    percentage=percentage,
                    repeat=repeat,
                    feature_set_name=feature_set_name,
                    feature_columns=columns,
                    seed=args.seed,
                )
            )

        print(
            f"Evaluated {mode}, missing={percentage:g}%, "
            f"repeat={repeat}."
        )

    folds = pd.DataFrame(fold_rows)
    subject_condition, summary = (
        summarize_subject_repeats(folds)
    )
    paired = paired_tests_vs_clean(
        subject_condition
    )

    audit_rows: list[dict[str, Any]] = []
    for condition, table in condition_tables.items():
        mode, percentage, repeat = condition
        audit_rows.append(
            {
                "missingness_mode": mode,
                "requested_additional_missing_percent": percentage,
                "corruption_repeat": repeat,
                "n_trials": len(table),
                "mean_realized_additional_missing_percent": float(
                    table[
                        "realized_additional_missing_percent"
                    ].mean()
                ),
                "median_realized_additional_missing_percent": float(
                    table[
                        "realized_additional_missing_percent"
                    ].median()
                ),
                "mean_valid_basic_rate_after": float(
                    table["valid_basic_rate"].mean()
                ),
                "mean_longest_invalid_run_ratio_after": float(
                    table[
                        "longest_invalid_run_ratio"
                    ].mean()
                ),
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values(
        [
            "missingness_mode",
            "requested_additional_missing_percent",
            "corruption_repeat",
        ]
    )

    audit_path = output_dir / "missingness_realization_audit.csv"
    folds_path = output_dir / "missingness_fold_metrics.csv"
    subject_path = (
        output_dir / "missingness_subject_condition_metrics.csv"
    )
    summary_path = output_dir / "missingness_summary.csv"
    paired_path = output_dir / "missingness_paired_tests_vs_clean.csv"

    audit.to_csv(
        audit_path,
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

    print("\n=== MISSINGNESS SUMMARY ===")
    print(
        summary[
            [
                "protocol",
                "missingness_mode",
                "additional_missing_percent",
                "feature_set",
                "macro_f1_subject_mean",
                "macro_f1_subject_sd",
                "roc_auc_subject_mean",
                "pr_auc_nr_subject_mean",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    print(f"  {audit_path}")
    print(f"  {folds_path}")
    print(f"  {subject_path}")
    print(f"  {summary_path}")
    print(f"  {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
