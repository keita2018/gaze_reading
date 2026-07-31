#!/usr/bin/env python
"""Evaluate combined coordinate noise and missingness at 30 Hz.

The experiment applies coordinate noise and additional missingness to the same
30 Hz sample stream before recomputing fixation-free features.

Combined test conditions
------------------------
moderate_random:
    sigma=25 px, random missingness=20%

moderate_burst:
    sigma=25 px, burst missingness=20%

high_random:
    sigma=50 px, random missingness=40%

high_burst:
    sigma=50 px, burst missingness=40%

extreme_random:
    sigma=100 px, random missingness=40%

extreme_burst:
    sigma=100 px, burst missingness=40%

Each condition has three deterministic corruption repeats by default.

Feature sets
------------
strict_content_only:
    Rate-invariant gaze and pupil content features, excluding quality features.

strict_rate_invariant:
    The same content features plus validity and longest-gap features.

Training protocols
------------------
clean_train:
    Train only on clean 30 Hz trials.

same_condition_augmented:
    Train on clean trials plus every corruption repeat of the same combined
    condition used at test time.

mixed_combined_augmented:
    Train on clean trials plus every repeat of all combined conditions.

All evaluation uses leave-one-subject-out. No clean or corrupted copy of the
held-out subject enters training.

Coordinate noise is added first. Missingness is then applied to the noisy
sample stream. Existing invalid samples remain invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_zuco_coordinate_noise_30hz import add_coordinate_noise
from run_zuco_downsampling_robustness import downsample_trial
from run_zuco_missingness_30hz import (
    add_burst_missingness,
    add_random_missingness,
    current_valid_basic_mask,
)
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


@dataclass(frozen=True)
class Condition:
    name: str
    sigma_px: float
    missingness_mode: str
    missing_percent: float


CONDITIONS = [
    Condition("moderate_random", 25.0, "random", 20.0),
    Condition("moderate_burst", 25.0, "burst", 20.0),
    Condition("high_random", 50.0, "random", 40.0),
    Condition("high_burst", 50.0, "burst", 40.0),
    Condition("extreme_random", 100.0, "random", 40.0),
    Condition("extreme_burst", 100.0, "burst", 40.0),
]


def stable_seed(
    base_seed: int,
    trial_id: str,
    condition_name: str,
    repeat: int,
) -> int:
    payload = (
        f"{base_seed}|{trial_id}|{condition_name}|{repeat}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def degrade_trial(
    sampled: pd.DataFrame,
    condition: Condition,
    repeat: int,
    base_seed: int,
    trial_id: str,
    burst_seconds: float,
) -> tuple[pd.DataFrame, int, int]:
    rng = np.random.default_rng(
        stable_seed(
            base_seed=base_seed,
            trial_id=trial_id,
            condition_name=condition.name,
            repeat=repeat,
        )
    )

    noisy = add_coordinate_noise(
        sampled=sampled,
        sigma_px=condition.sigma_px,
        rng=rng,
    )

    valid_before_missing = int(
        np.sum(current_valid_basic_mask(noisy))
    )

    if condition.missingness_mode == "random":
        degraded, removed = add_random_missingness(
            samples=noisy,
            percentage=condition.missing_percent,
            rng=rng,
        )
    elif condition.missingness_mode == "burst":
        degraded, removed = add_burst_missingness(
            samples=noisy,
            percentage=condition.missing_percent,
            burst_seconds=burst_seconds,
            rng=rng,
        )
    else:
        raise ValueError(
            f"Unknown missingness mode: "
            f"{condition.missingness_mode}"
        )

    return degraded, valid_before_missing, removed


def validate_condition_table(
    features: pd.DataFrame,
    expected_trials: pd.DataFrame,
    subject: str,
    condition: Condition,
    repeat: int,
) -> None:
    validate_subject_feature_table(
        features=features,
        expected_trials=expected_trials,
        subject=subject,
    )

    required = {
        "combined_condition",
        "noise_sigma_px",
        "missingness_mode",
        "additional_missing_percent",
        "corruption_repeat",
        "realized_additional_missing_percent",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(
            f"{subject} condition cache missing columns: "
            f"{missing}"
        )

    if not features["combined_condition"].astype(str).eq(
        condition.name
    ).all():
        raise ValueError(
            f"{subject} cache has the wrong condition."
        )

    if not np.allclose(
        pd.to_numeric(
            features["noise_sigma_px"],
            errors="coerce",
        ),
        condition.sigma_px,
    ):
        raise ValueError(
            f"{subject} cache has the wrong sigma."
        )

    if not features["missingness_mode"].astype(str).eq(
        condition.missingness_mode
    ).all():
        raise ValueError(
            f"{subject} cache has the wrong missingness mode."
        )

    if not np.allclose(
        pd.to_numeric(
            features["additional_missing_percent"],
            errors="coerce",
        ),
        condition.missing_percent,
    ):
        raise ValueError(
            f"{subject} cache has the wrong missingness rate."
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
    conditions: list[tuple[Condition, int]],
    cache_paths: dict[tuple[str, int], Path],
    base_seed: int,
    burst_seconds: float,
) -> dict[tuple[str, int], pd.DataFrame]:
    metadata_lookup = subject_trials.set_index(
        "trial_id",
        verify_integrity=True,
    )
    rows_by_condition: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = {
        (condition.name, repeat): []
        for condition, repeat in conditions
    }

    manifest = subject_manifest.sort_values(
        ["task", "block_id"]
    )
    if len(manifest) != 14:
        raise ValueError(
            f"{subject} manifest rows={len(manifest)}, "
            "expected 14."
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
            f"Reading for {len(conditions)} combined conditions: "
            f"{parquet_path}"
        )
        samples = pd.read_parquet(parquet_path)

        for trial_id, trial_group in samples.groupby(
            "trial_id",
            sort=False,
        ):
            if trial_id not in metadata_lookup.index:
                raise ValueError(
                    f"Trial missing from metadata: {trial_id}"
                )

            sampled = downsample_trial(
                trial_group,
                RATE_HZ,
            )
            metadata = metadata_lookup.loc[trial_id].copy()
            metadata["trial_id"] = trial_id
            metadata["n_samples"] = len(sampled)

            for condition, repeat in conditions:
                degraded, valid_before, removed = degrade_trial(
                    sampled=sampled,
                    condition=condition,
                    repeat=repeat,
                    base_seed=base_seed,
                    trial_id=str(trial_id),
                    burst_seconds=burst_seconds,
                )

                row = extract_trial_features(
                    degraded,
                    metadata,
                )
                row["sampling_rate_hz"] = RATE_HZ
                row["combined_condition"] = condition.name
                row["noise_sigma_px"] = condition.sigma_px
                row["missingness_mode"] = (
                    condition.missingness_mode
                )
                row["additional_missing_percent"] = (
                    condition.missing_percent
                )
                row["corruption_repeat"] = repeat
                row["valid_before_added_missing"] = (
                    valid_before
                )
                row["additional_missing_samples"] = removed
                row[
                    "realized_additional_missing_percent"
                ] = (
                    100.0 * removed / valid_before
                    if valid_before > 0
                    else math.nan
                )
                rows_by_condition[
                    (condition.name, repeat)
                ].append(row)

        del samples

    results: dict[tuple[str, int], pd.DataFrame] = {}

    for condition, repeat in conditions:
        key = (condition.name, repeat)
        features = (
            pd.DataFrame(rows_by_condition[key])
            .sort_values(
                ["task", "block_id", "trial_index_in_block"]
            )
            .reset_index(drop=True)
        )

        validate_condition_table(
            features=features,
            expected_trials=subject_trials,
            subject=subject,
            condition=condition,
            repeat=repeat,
        )

        output_path = cache_paths[key]
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        features.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        results[key] = features

    return results


def prepare_table(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()
    result["label_nr"] = result["task"].map(
        {"NR": 1, "TSR": 0}
    ).astype(int)
    return result


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
    condition: Condition,
    repeat: int,
    feature_set_name: str,
    feature_columns: list[str],
    held_out_subject: str,
    n_train: int,
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
        "combined_condition": condition.name,
        "noise_sigma_px": condition.sigma_px,
        "missingness_mode": condition.missingness_mode,
        "additional_missing_percent": (
            condition.missing_percent
        ),
        "corruption_repeat": repeat,
        "feature_set": feature_set_name,
        "held_out_subject": held_out_subject,
        "n_train": n_train,
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
                "combined_condition",
                "noise_sigma_px",
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
        "combined_condition",
        "noise_sigma_px",
        "missingness_mode",
        "additional_missing_percent",
        "feature_set",
    ]

    for keys, group in subject_condition.groupby(
        group_columns,
        sort=False,
    ):
        (
            protocol,
            condition_name,
            sigma_px,
            mode,
            missing_percent,
            feature_set,
        ) = keys

        row: dict[str, Any] = {
            "protocol": protocol,
            "sampling_rate_hz": RATE_HZ,
            "combined_condition": condition_name,
            "noise_sigma_px": sigma_px,
            "missingness_mode": mode,
            "additional_missing_percent": missing_percent,
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
            "feature_set",
            "combined_condition",
            "protocol",
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


def paired_protocol_tests(
    subject_condition: pd.DataFrame,
    reference_protocol: str = "clean_train",
) -> pd.DataFrame:
    metrics = ["macro_f1", "roc_auc", "pr_auc_nr"]
    candidates = [
        value
        for value in sorted(
            subject_condition["protocol"].unique()
        )
        if value != reference_protocol
    ]

    rows: list[dict[str, Any]] = []

    for feature_set in sorted(
        subject_condition["feature_set"].unique()
    ):
        for candidate_protocol in candidates:
            for metric in metrics:
                row_indices: list[int] = []
                p_values: list[float] = []

                for condition_name in sorted(
                    subject_condition[
                        "combined_condition"
                    ].unique()
                ):
                    reference = (
                        subject_condition[
                            subject_condition[
                                "feature_set"
                            ].eq(feature_set)
                            & subject_condition[
                                "protocol"
                            ].eq(reference_protocol)
                            & subject_condition[
                                "combined_condition"
                            ].eq(condition_name)
                        ]
                        .set_index("held_out_subject")[metric]
                        .sort_index()
                    )
                    candidate = (
                        subject_condition[
                            subject_condition[
                                "feature_set"
                            ].eq(feature_set)
                            & subject_condition[
                                "protocol"
                            ].eq(candidate_protocol)
                            & subject_condition[
                                "combined_condition"
                            ].eq(condition_name)
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
                            "feature_set": feature_set,
                            "metric": metric,
                            "combined_condition": condition_name,
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
                            "p_holm_within_feature_metric_candidate": (
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
                            "p_holm_within_feature_metric_candidate"
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
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--burst-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/combined_degradation_30hz",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
    )
    args = parser.parse_args()

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

    clean = prepare_table(
        pd.read_csv(args.clean_30hz_features)
    )
    if len(clean) != EXPECTED_TRIALS:
        raise ValueError(
            f"Clean rows={len(clean)}, expected={EXPECTED_TRIALS}."
        )

    invariant_sets = json.loads(
        Path(args.rate_invariant_feature_sets).read_text(
            encoding="utf-8"
        )
    )
    quality_columns = [
        "valid_xy_rate",
        "valid_pupil_rate",
        "valid_basic_rate",
        "longest_invalid_run_ratio",
    ]
    strict_rate_invariant = invariant_sets[
        "strict_rate_invariant"
    ]
    strict_content_only = [
        column
        for column in strict_rate_invariant
        if column not in quality_columns
    ]

    feature_sets = {
        "strict_content_only": strict_content_only,
        "strict_rate_invariant": strict_rate_invariant,
    }

    missing_features = {
        name: sorted(set(columns) - set(clean.columns))
        for name, columns in feature_sets.items()
        if set(columns) - set(clean.columns)
    }
    if missing_features:
        raise ValueError(
            f"Clean feature columns missing: {missing_features}"
        )

    subjects = sorted(trials["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} subjects, "
            f"found {len(subjects)}."
        )

    condition_repeats = [
        (condition, repeat)
        for condition in CONDITIONS
        for repeat in range(1, args.repeats + 1)
    ]

    output_dir = Path(args.output_dir)
    cache_root = output_dir / "subject_combined_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_frames: dict[
        tuple[str, int],
        list[pd.DataFrame],
    ] = {
        (condition.name, repeat): []
        for condition, repeat in condition_repeats
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
            (condition.name, repeat): (
                cache_root
                / condition.name
                / f"repeat_{repeat:02d}"
                / (
                    f"{subject}_{condition.name}"
                    f"_repeat_{repeat:02d}.csv"
                )
            )
            for condition, repeat in condition_repeats
        }

        missing_conditions: list[
            tuple[Condition, int]
        ] = []

        for condition, repeat in condition_repeats:
            key = (condition.name, repeat)
            path = cache_paths[key]

            if (
                path.is_file()
                and not args.overwrite_features
            ):
                cached = pd.read_csv(path)
                validate_condition_table(
                    features=cached,
                    expected_trials=subject_trials,
                    subject=subject,
                    condition=condition,
                    repeat=repeat,
                )
                condition_frames[key].append(cached)
                print(f"Reusing cache: {path}")
            else:
                missing_conditions.append(
                    (condition, repeat)
                )

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
            for condition, repeat in missing_conditions:
                key = (condition.name, repeat)
                condition_frames[key].append(
                    extracted[key]
                )

    condition_tables: dict[
        tuple[str, int],
        pd.DataFrame,
    ] = {}

    for condition, repeat in condition_repeats:
        key = (condition.name, repeat)
        combined = prepare_table(
            pd.concat(
                condition_frames[key],
                ignore_index=True,
            )
        )
        if len(combined) != EXPECTED_TRIALS:
            raise ValueError(
                f"{key} rows={len(combined)}, "
                f"expected={EXPECTED_TRIALS}."
            )
        if combined["trial_id"].duplicated().any():
            raise ValueError(
                f"{key} has duplicate trial IDs."
            )
        condition_tables[key] = combined

    fold_rows: list[dict[str, Any]] = []

    for fold_index, held_out_subject in enumerate(
        subjects,
        start=1,
    ):
        print(
            f"[LOSO {fold_index}/{len(subjects)}] "
            f"Held out: {held_out_subject}"
        )

        clean_train = clean[
            clean["subject_id"] != held_out_subject
        ].copy()

        mixed_frames = [clean_train]
        for table in condition_tables.values():
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

        for feature_set_name, columns in feature_sets.items():
            clean_model = fit_model(
                train=clean_train,
                feature_columns=columns,
                seed=args.seed,
            )
            mixed_model = fit_model(
                train=mixed_train,
                feature_columns=columns,
                seed=args.seed,
            )

            same_models: dict[str, Any] = {}
            same_sizes: dict[str, int] = {}

            for condition in CONDITIONS:
                same_frames = [clean_train]
                for repeat in range(
                    1,
                    args.repeats + 1,
                ):
                    table = condition_tables[
                        (condition.name, repeat)
                    ]
                    same_frames.append(
                        table[
                            table["subject_id"]
                            != held_out_subject
                        ]
                    )

                same_train = pd.concat(
                    same_frames,
                    ignore_index=True,
                )
                same_models[condition.name] = fit_model(
                    train=same_train,
                    feature_columns=columns,
                    seed=args.seed,
                )
                same_sizes[condition.name] = len(
                    same_train
                )

            for condition in CONDITIONS:
                for repeat in range(
                    1,
                    args.repeats + 1,
                ):
                    test = condition_tables[
                        (condition.name, repeat)
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
                            condition=condition,
                            repeat=repeat,
                            feature_set_name=feature_set_name,
                            feature_columns=columns,
                            held_out_subject=held_out_subject,
                            n_train=len(clean_train),
                        )
                    )
                    fold_rows.append(
                        evaluate_model(
                            model=same_models[
                                condition.name
                            ],
                            test=test,
                            protocol=(
                                "same_condition_augmented"
                            ),
                            condition=condition,
                            repeat=repeat,
                            feature_set_name=feature_set_name,
                            feature_columns=columns,
                            held_out_subject=held_out_subject,
                            n_train=same_sizes[
                                condition.name
                            ],
                        )
                    )
                    fold_rows.append(
                        evaluate_model(
                            model=mixed_model,
                            test=test,
                            protocol=(
                                "mixed_combined_augmented"
                            ),
                            condition=condition,
                            repeat=repeat,
                            feature_set_name=feature_set_name,
                            feature_columns=columns,
                            held_out_subject=held_out_subject,
                            n_train=len(mixed_train),
                        )
                    )

    folds = pd.DataFrame(fold_rows)
    subject_condition, summary = (
        summarize_subject_repeats(folds)
    )
    paired = paired_protocol_tests(
        subject_condition
    )

    audit_rows: list[dict[str, Any]] = []
    for condition, repeat in condition_repeats:
        table = condition_tables[
            (condition.name, repeat)
        ]
        audit_rows.append(
            {
                "combined_condition": condition.name,
                "noise_sigma_px": condition.sigma_px,
                "missingness_mode": (
                    condition.missingness_mode
                ),
                "requested_additional_missing_percent": (
                    condition.missing_percent
                ),
                "corruption_repeat": repeat,
                "n_trials": len(table),
                "mean_realized_additional_missing_percent": (
                    float(
                        table[
                            "realized_additional_missing_percent"
                        ].mean()
                    )
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
        ["combined_condition", "corruption_repeat"]
    )

    audit_path = output_dir / "combined_degradation_audit.csv"
    folds_path = (
        output_dir / "combined_degradation_fold_metrics.csv"
    )
    subject_path = (
        output_dir
        / "combined_degradation_subject_condition_metrics.csv"
    )
    summary_path = (
        output_dir / "combined_degradation_summary.csv"
    )
    paired_path = (
        output_dir
        / "combined_degradation_paired_vs_clean_train.csv"
    )

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

    print("\n=== COMBINED DEGRADATION SUMMARY ===")
    print(
        summary[
            [
                "feature_set",
                "combined_condition",
                "protocol",
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
