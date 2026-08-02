#!/usr/bin/env python
"""Compare clean-trained Soft-AOI and confidence-weighted Hard-AOI at 50 px noise.

This is the degradation follow-up to the clean 25-vs-25 comparison.

- Train only on clean 30 Hz ZuCo trials.
- Test on deterministic Gaussian coordinate-noise trials.
- Hold out both the test subject and the test sentence fold.
- Use the same 13 distribution and 12 membership/quality features.
- Reuse existing Soft-AOI noise caches.
- Extract resumable Hard-AOI noise caches from the identical corrupted samples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from extract_zuco_soft_aoi_features import (  # noqa: E402
    downsample_trial as soft_downsample_trial,
)
from run_zuco_hard_aoi_all_subjects import (  # noqa: E402
    HARD_COMMON25,
    HARD_DISTRIBUTION_COMMON,
    HARD_QUALITY,
    extract_hard_trial_features,
    load_trial_map,
)
from run_zuco_soft_aoi_ablation import (  # noqa: E402
    holm_adjust,
    load_sentence_folds,
)
from run_zuco_soft_aoi_confound_controls import (  # noqa: E402
    METRICS,
    decision_scores,
    load_merged_data,
    make_svm,
    score_predictions,
)
from run_zuco_soft_aoi_coordinate_noise_30hz import (  # noqa: E402
    add_coordinate_noise,
    coordinate_noise_seed,
    load_word_template,
    soft_noise_cache_path,
    validate_subject_soft_cache,
)
from run_zuco_soft_vs_hard_aoi_clean_30hz import (  # noqa: E402
    HARD_NAME,
    SOFT_COMMON_DISTRIBUTION,
    SOFT_NAME,
    SOFT_QUALITY,
    add_hard_features,
    audit_shared_quality,
    build_feature_sets,
    normalize_hard_table,
)


RATE_HZ = 30
EXPECTED_SUBJECTS = 18
EXPECTED_TRIALS = 13302
EXPECTED_PER_SUBJECT = 739
EXPECTED_TASK_COUNTS = {"NR": 6282, "TSR": 7020}


def parse_sigmas(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("--sigmas must contain positive values.")
    return values


def hard_noise_cache_path(
    root: Path,
    subject: str,
    sigma_px: float,
    repeat: int,
) -> Path:
    return (
        root
        / f"sigma_{sigma_px:06.1f}px"
        / f"repeat_{repeat:02d}"
        / (
            f"{subject}_hard_aoi_sigma_{sigma_px:06.1f}"
            f"_repeat_{repeat:02d}.csv"
        )
    )


def validate_hard_noise_cache(
    frame: pd.DataFrame,
    subject: str,
    expected_ids: set[str],
    sigma_px: float,
    repeat: int,
) -> pd.DataFrame:
    required = {
        "trial_id",
        "subject_id",
        "task",
        "task_sentence_index",
        "noise_sigma_px",
        "corruption_repeat",
        *HARD_COMMON25,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{subject} Hard-AOI noise cache lacks columns: {missing}"
        )

    result = frame.copy()
    result["trial_id"] = result["trial_id"].astype(str)
    result["subject_id"] = (
        result["subject_id"].astype(str).str.upper()
    )
    result["task"] = result["task"].astype(str).str.upper()

    if len(result) != EXPECTED_PER_SUBJECT:
        raise ValueError(
            f"{subject} Hard cache rows={len(result)}, "
            f"expected={EXPECTED_PER_SUBJECT}."
        )
    if result["trial_id"].duplicated().any():
        raise ValueError(f"{subject} Hard cache has duplicate trial IDs.")
    if set(result["trial_id"]) != expected_ids:
        raise ValueError(f"{subject} Hard cache trial IDs do not match.")
    if set(result["subject_id"]) != {subject}:
        raise ValueError(f"{subject} Hard cache subject is invalid.")
    if not np.allclose(
        pd.to_numeric(result["noise_sigma_px"], errors="coerce"),
        sigma_px,
    ):
        raise ValueError(f"{subject} Hard cache sigma is invalid.")
    if not pd.to_numeric(
        result["corruption_repeat"], errors="coerce"
    ).eq(repeat).all():
        raise ValueError(f"{subject} Hard cache repeat is invalid.")
    if np.isinf(result[HARD_COMMON25].to_numpy(dtype=float)).any():
        raise ValueError(f"{subject} Hard cache contains infinity.")
    return result


def extract_or_load_hard_conditions(
    manifest: pd.DataFrame,
    subjects: list[str],
    processed_root: Path,
    cache_root: Path,
    aoi_lookup: dict[tuple[str, int], pd.DataFrame],
    sigmas: list[float],
    repeats: int,
    seed: int,
    sigma_word_height_ratio: float,
    active_threshold: float,
    overwrite: bool,
) -> dict[tuple[float, int], pd.DataFrame]:
    combined: dict[tuple[float, int], list[pd.DataFrame]] = {
        (sigma, repeat): []
        for sigma in sigmas
        for repeat in range(1, repeats + 1)
    }

    for subject_index, subject in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"Hard-AOI noise subject "
            f"{subject_index:02d}/{len(subjects)}: {subject}"
        )

        trial_map = load_trial_map(
            processed_root
            / subject
            / f"{subject}_all_trials_with_sentence_index.csv",
            subject,
        )
        expected_ids = set(trial_map["trial_id"].astype(str))
        trial_lookup = trial_map.set_index(
            "trial_id", drop=False, verify_integrity=True
        )

        paths = {
            condition: hard_noise_cache_path(
                cache_root,
                subject,
                condition[0],
                condition[1],
            )
            for condition in combined
        }

        if not overwrite and all(path.is_file() for path in paths.values()):
            for condition, path in paths.items():
                frame = validate_hard_noise_cache(
                    pd.read_csv(path),
                    subject,
                    expected_ids,
                    condition[0],
                    condition[1],
                )
                combined[condition].append(frame)
            print("  Loaded all existing Hard-AOI noise caches.")
            continue

        subject_manifest = manifest[
            manifest["subject_id"].eq(subject)
        ].sort_values(["task", "block_id"])
        if len(subject_manifest) != 14:
            raise ValueError(
                f"{subject} manifest rows={len(subject_manifest)}, expected=14."
            )

        rows_by_condition: dict[
            tuple[float, int], list[dict[str, Any]]
        ] = {
            condition: []
            for condition in combined
            if overwrite or not paths[condition].is_file()
        }

        if rows_by_condition:
            for block_index, manifest_row in enumerate(
                subject_manifest.itertuples(index=False),
                start=1,
            ):
                sample_path = Path(str(manifest_row.sample_parquet))
                if not sample_path.is_file():
                    raise FileNotFoundError(sample_path)
                print(
                    f"  [{block_index:02d}/14] "
                    f"{sample_path.name}; "
                    f"conditions={len(rows_by_condition)}"
                )

                samples = pd.read_parquet(sample_path)
                required_samples = {
                    "trial_id",
                    "timestamp_ms_abs",
                    "x_px",
                    "y_px",
                    "valid_basic",
                }
                missing_samples = sorted(
                    required_samples - set(samples.columns)
                )
                if missing_samples:
                    raise ValueError(
                        f"{sample_path} lacks columns: {missing_samples}"
                    )

                samples["trial_id"] = samples["trial_id"].astype(str)
                for trial_id, trial_samples in samples.groupby(
                    "trial_id", sort=False
                ):
                    trial_id = str(trial_id)
                    if trial_id not in trial_lookup.index:
                        raise ValueError(
                            f"Trial missing from map: {trial_id}"
                        )
                    trial_row = trial_lookup.loc[trial_id].copy()
                    aoi_key = (
                        str(trial_row["task"]).upper(),
                        int(trial_row["task_sentence_index"]),
                    )
                    if aoi_key not in aoi_lookup:
                        raise ValueError(
                            f"AOI template missing for key={aoi_key}"
                        )

                    sampled = soft_downsample_trial(
                        trial_samples,
                        RATE_HZ,
                    )
                    for sigma_px, repeat in rows_by_condition:
                        rng = np.random.default_rng(
                            coordinate_noise_seed(
                                base_seed=seed,
                                trial_id=trial_id,
                                sigma_px=sigma_px,
                                repeat=repeat,
                            )
                        )
                        degraded = add_coordinate_noise(
                            sampled=sampled,
                            sigma_px=sigma_px,
                            rng=rng,
                        )
                        row = extract_hard_trial_features(
                            samples=degraded,
                            trial_row=trial_row,
                            aoi=aoi_lookup[aoi_key],
                            rate_hz=500,
                            sigma_word_height_ratio=(
                                sigma_word_height_ratio
                            ),
                            active_threshold=active_threshold,
                        )
                        row["hard_aoi_rate_hz"] = RATE_HZ
                        row["noise_sigma_px"] = float(sigma_px)
                        row["corruption_repeat"] = int(repeat)
                        rows_by_condition[
                            (sigma_px, repeat)
                        ].append(row)

                del samples

        for condition, path in paths.items():
            sigma_px, repeat = condition
            if condition in rows_by_condition:
                frame = pd.DataFrame(
                    rows_by_condition[condition]
                ).sort_values("trial_id").reset_index(drop=True)
                frame = validate_hard_noise_cache(
                    frame,
                    subject,
                    expected_ids,
                    sigma_px,
                    repeat,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(
                    path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"  Written: {path}")
            else:
                frame = validate_hard_noise_cache(
                    pd.read_csv(path),
                    subject,
                    expected_ids,
                    sigma_px,
                    repeat,
                )
            combined[condition].append(frame)

    results: dict[tuple[float, int], pd.DataFrame] = {}
    for condition, frames in combined.items():
        frame = pd.concat(frames, ignore_index=True)
        if len(frame) != EXPECTED_TRIALS:
            raise ValueError(
                f"Hard condition {condition} rows={len(frame)}, "
                f"expected={EXPECTED_TRIALS}."
            )
        if frame["subject_id"].nunique() != EXPECTED_SUBJECTS:
            raise ValueError(
                f"Hard condition {condition} subject count is invalid."
            )
        counts = frame.groupby("task").size().to_dict()
        if counts != EXPECTED_TASK_COUNTS:
            raise ValueError(
                f"Hard condition {condition} task counts={counts}."
            )
        results[condition] = frame
    return results


def load_soft_conditions(
    cache_root: Path,
    processed_root: Path,
    subjects: list[str],
    sigmas: list[float],
    repeats: int,
) -> dict[tuple[float, int], pd.DataFrame]:
    frames_by_condition: dict[
        tuple[float, int], list[pd.DataFrame]
    ] = {
        (sigma, repeat): []
        for sigma in sigmas
        for repeat in range(1, repeats + 1)
    }

    missing_paths: list[Path] = []
    for subject in subjects:
        trial_map = load_trial_map(
            processed_root
            / subject
            / f"{subject}_all_trials_with_sentence_index.csv",
            subject,
        )
        expected_ids = set(trial_map["trial_id"].astype(str))
        for condition in frames_by_condition:
            sigma_px, repeat = condition
            path = soft_noise_cache_path(
                cache_root,
                subject,
                sigma_px,
                repeat,
            )
            if not path.is_file():
                missing_paths.append(path)
                continue
            frame = validate_subject_soft_cache(
                pd.read_csv(path),
                subject,
                expected_ids,
                sigma_px,
                repeat,
            )
            frames_by_condition[condition].append(frame)

    if missing_paths:
        examples = "\n".join(
            f"  {path}" for path in missing_paths[:12]
        )
        raise FileNotFoundError(
            "Existing Soft-AOI coordinate-noise caches are missing. "
            "Run run_zuco_soft_aoi_coordinate_noise_30hz.py for the "
            "same sigma/repeat settings first. "
            f"Missing={len(missing_paths)}; examples:\n{examples}"
        )

    output: dict[tuple[float, int], pd.DataFrame] = {}
    for condition, frames in frames_by_condition.items():
        frame = pd.concat(frames, ignore_index=True)
        if len(frame) != EXPECTED_TRIALS:
            raise ValueError(
                f"Soft condition {condition} rows={len(frame)}, "
                f"expected={EXPECTED_TRIALS}."
            )
        output[condition] = frame
    return output


def make_condition_table(
    metadata: pd.DataFrame,
    soft: pd.DataFrame,
    hard: pd.DataFrame,
) -> pd.DataFrame:
    soft_columns = SOFT_COMMON_DISTRIBUTION + SOFT_QUALITY
    soft_small = soft[
        ["trial_id", "subject_id", "task"] + soft_columns
    ].copy()
    soft_small = soft_small.rename(
        columns={
            name: f"soft__{name}"
            for name in soft_columns
        }
    )

    hard_small = hard[
        ["trial_id", "subject_id", "task"]
        + HARD_DISTRIBUTION_COMMON
        + HARD_QUALITY
    ].copy()
    hard_small = hard_small.rename(
        columns={
            name: f"hard__{name}"
            for name in (
                HARD_DISTRIBUTION_COMMON + HARD_QUALITY
            )
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
            f"Condition merge rows={len(merged)}, "
            f"expected={EXPECTED_TRIALS}."
        )
    audit_shared_quality(merged)
    return merged


def evaluate_clean_train_noise_test(
    clean: pd.DataFrame,
    conditions: dict[tuple[float, int], pd.DataFrame],
    feature_sets: dict[str, list[str]],
    c_value: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(clean["subject_id"].unique())
    folds = sorted(clean["sentence_fold"].unique())
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for subject_index, held_out in enumerate(subjects, start=1):
        print("\n" + "=" * 78)
        print(
            f"Noise evaluation subject "
            f"{subject_index:02d}/{len(subjects)}: {held_out}"
        )
        for fold_index in folds:
            train = clean[
                (clean["subject_id"] != held_out)
                & (clean["sentence_fold"] != fold_index)
            ].copy()
            if held_out in set(train["subject_id"]):
                raise ValueError("Subject leakage detected.")

            y_train = train["label_nr"].to_numpy(dtype=int)
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError("Training split lacks both classes.")

            models = {}
            for feature_set, columns in feature_sets.items():
                model = make_svm(c_value, seed)
                model.fit(train[columns], y_train)
                models[feature_set] = model

            print(
                f"  Fold {fold_index + 1}/{len(folds)}: "
                f"train={len(train)}"
            )
            for (sigma_px, repeat), condition in conditions.items():
                test = condition[
                    (condition["subject_id"] == held_out)
                    & (condition["sentence_fold"] == fold_index)
                ].copy()
                if test.empty:
                    raise ValueError(
                        f"Empty noise test: subject={held_out}, "
                        f"fold={fold_index}, sigma={sigma_px}, "
                        f"repeat={repeat}."
                    )
                if set(train["sentence_key"]) & set(test["sentence_key"]):
                    raise ValueError("Sentence leakage detected.")

                y_test = test["label_nr"].to_numpy(dtype=int)
                if set(np.unique(y_test)) != {0, 1}:
                    raise ValueError("Noise test split lacks both classes.")

                ordered = test.reset_index(drop=True)
                for feature_set, columns in feature_sets.items():
                    model = models[feature_set]
                    y_pred = model.predict(
                        ordered[columns]
                    ).astype(int)
                    scores = decision_scores(
                        model,
                        ordered[columns],
                    )
                    metrics = score_predictions(
                        y_test,
                        y_pred,
                        scores,
                    )
                    fold_rows.append(
                        {
                            "protocol": (
                                "clean30_train_noise30_test_dual_holdout"
                            ),
                            "noise_sigma_px": float(sigma_px),
                            "corruption_repeat": int(repeat),
                            "feature_set": feature_set,
                            "held_out_subject": held_out,
                            "sentence_fold": int(fold_index),
                            "n_input_features": int(len(columns)),
                            "c_value": float(c_value),
                            "n_train": int(len(train)),
                            "n_test": int(len(ordered)),
                            **metrics,
                        }
                    )
                    for row_index, row in ordered.iterrows():
                        prediction_rows.append(
                            {
                                "protocol": (
                                    "clean30_train_noise30_test_dual_holdout"
                                ),
                                "noise_sigma_px": float(sigma_px),
                                "corruption_repeat": int(repeat),
                                "feature_set": feature_set,
                                "held_out_subject": held_out,
                                "sentence_fold": int(fold_index),
                                "trial_id": str(row["trial_id"]),
                                "subject_id": str(row["subject_id"]),
                                "task": str(row["task"]),
                                "task_sentence_index": int(
                                    row["task_sentence_index"]
                                ),
                                "sentence_key": str(row["sentence_key"]),
                                "label_nr": int(row["label_nr"]),
                                "prediction_nr": int(y_pred[row_index]),
                                "decision_score_nr": float(
                                    scores[row_index]
                                ),
                            }
                        )

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def subject_repeat_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "noise_sigma_px",
        "corruption_repeat",
        "feature_set",
        "held_out_subject",
    ]
    for values, group in predictions.groupby(keys, sort=True):
        sigma_px, repeat, feature_set, subject = values
        y_true = group["label_nr"].to_numpy(dtype=int)
        y_pred = group["prediction_nr"].to_numpy(dtype=int)
        scores = group["decision_score_nr"].to_numpy(dtype=float)
        rows.append(
            {
                "noise_sigma_px": float(sigma_px),
                "corruption_repeat": int(repeat),
                "feature_set": feature_set,
                "held_out_subject": subject,
                "n_predictions": int(len(group)),
                **score_predictions(y_true, y_pred, scores),
            }
        )
    return pd.DataFrame(rows)


def subject_mean_metrics(
    repeat_metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        repeat_metrics.groupby(
            [
                "noise_sigma_px",
                "feature_set",
                "held_out_subject",
            ],
            as_index=False,
        )[METRICS]
        .mean()
    )


def summarize_noise(
    subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, group in subject_metrics.groupby(
        ["noise_sigma_px", "feature_set"],
        sort=True,
    ):
        sigma_px, feature_set = values
        row: dict[str, Any] = {
            "noise_sigma_px": float(sigma_px),
            "feature_set": feature_set,
            "n_subjects": int(group["held_out_subject"].nunique()),
            "n_input_features": 25,
        }
        for metric in METRICS:
            row[f"{metric}_subject_mean"] = float(
                group[metric].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_tests(
    noise_subject: pd.DataFrame,
    clean_subject_path: Path,
) -> pd.DataFrame:
    if not clean_subject_path.is_file():
        raise FileNotFoundError(clean_subject_path)
    clean = pd.read_csv(clean_subject_path)
    required_clean = {
        "feature_set",
        "held_out_subject",
        *METRICS,
    }
    missing = sorted(required_clean - set(clean.columns))
    if missing:
        raise ValueError(
            f"Clean subject metrics lack columns: {missing}"
        )
    clean = clean[
        clean["feature_set"].isin([SOFT_NAME, HARD_NAME])
    ].copy()

    rows: list[dict[str, Any]] = []
    for sigma_px in sorted(noise_subject["noise_sigma_px"].unique()):
        sigma_data = noise_subject[
            noise_subject["noise_sigma_px"].eq(sigma_px)
        ]

        for metric in METRICS:
            soft = (
                sigma_data[
                    sigma_data["feature_set"].eq(SOFT_NAME)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            hard = (
                sigma_data[
                    sigma_data["feature_set"].eq(HARD_NAME)
                ]
                .set_index("held_out_subject")[metric]
                .sort_index()
            )
            paired = pd.concat(
                [hard.rename("reference"), soft.rename("candidate")],
                axis=1,
                join="inner",
            ).dropna()
            diff = paired["candidate"] - paired["reference"]
            if np.allclose(diff.to_numpy(dtype=float), 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    paired["candidate"],
                    paired["reference"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            rows.append(
                {
                    "test_family": "soft_vs_hard_at_noise",
                    "noise_sigma_px": float(sigma_px),
                    "metric": metric,
                    "comparison": "soft_common25_vs_hard_argmax_common25",
                    "reference_feature_set": HARD_NAME,
                    "candidate_feature_set": SOFT_NAME,
                    "n_subjects": int(len(paired)),
                    "reference_mean": float(
                        paired["reference"].mean()
                    ),
                    "candidate_mean": float(
                        paired["candidate"].mean()
                    ),
                    "mean_difference_candidate_minus_reference": float(
                        diff.mean()
                    ),
                    "wins": int(np.sum(diff > 0)),
                    "ties": int(np.sum(np.isclose(diff, 0.0))),
                    "losses": int(np.sum(diff < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_holm_within_family_metric": math.nan,
                }
            )

            for feature_set in [SOFT_NAME, HARD_NAME]:
                noise_values = (
                    sigma_data[
                        sigma_data["feature_set"].eq(feature_set)
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )
                clean_values = (
                    clean[
                        clean["feature_set"].eq(feature_set)
                    ]
                    .set_index("held_out_subject")[metric]
                    .sort_index()
                )
                paired_drop = pd.concat(
                    [
                        clean_values.rename("reference"),
                        noise_values.rename("candidate"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                drop = (
                    paired_drop["candidate"]
                    - paired_drop["reference"]
                )
                if np.allclose(
                    drop.to_numpy(dtype=float), 0.0
                ):
                    statistic_drop, p_drop = 0.0, 1.0
                else:
                    test_drop = wilcoxon(
                        paired_drop["candidate"],
                        paired_drop["reference"],
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                    statistic_drop = float(test_drop.statistic)
                    p_drop = float(test_drop.pvalue)
                rows.append(
                    {
                        "test_family": "noise_vs_clean",
                        "noise_sigma_px": float(sigma_px),
                        "metric": metric,
                        "comparison": f"{feature_set}_noise_vs_clean",
                        "reference_feature_set": f"{feature_set}_clean",
                        "candidate_feature_set": f"{feature_set}_noise",
                        "n_subjects": int(len(paired_drop)),
                        "reference_mean": float(
                            paired_drop["reference"].mean()
                        ),
                        "candidate_mean": float(
                            paired_drop["candidate"].mean()
                        ),
                        "mean_difference_candidate_minus_reference": float(
                            drop.mean()
                        ),
                        "wins": int(np.sum(drop > 0)),
                        "ties": int(np.sum(np.isclose(drop, 0.0))),
                        "losses": int(np.sum(drop < 0)),
                        "wilcoxon_statistic": statistic_drop,
                        "p_value": p_drop,
                        "p_holm_within_family_metric": math.nan,
                    }
                )

    tests = pd.DataFrame(rows)
    for (_, metric), indices in tests.groupby(
        ["test_family", "metric"]
    ).groups.items():
        index_list = list(indices)
        adjusted = holm_adjust(
            tests.loc[index_list, "p_value"].astype(float).tolist()
        )
        tests.loc[
            index_list,
            "p_holm_within_family_metric",
        ] = adjusted
    return tests


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
        "--processed-root",
        default="data/processed/zuco2",
    )
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
        "--soft-noise-cache-root",
        default=(
            "outputs/soft_aoi_coordinate_noise_30hz/"
            "subject_soft_aoi_noise_features"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/soft_vs_hard_aoi_coordinate_noise_30hz"
        ),
    )
    parser.add_argument(
        "--clean-subject-metrics",
        default=(
            "outputs/soft_vs_hard_aoi_clean_30hz/"
            "soft_vs_hard_subject_metrics.csv"
        ),
    )
    parser.add_argument("--sigmas", default="50")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sigma-word-height-ratio",
        type=float,
        default=2.0 / 3.0,
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--overwrite-hard-features",
        action="store_true",
    )
    args = parser.parse_args()

    sigmas = parse_sigmas(args.sigmas)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.c_value <= 0:
        raise ValueError("--c-value must be positive.")
    if args.sigma_word_height_ratio <= 0:
        raise ValueError(
            "--sigma-word-height-ratio must be positive."
        )
    if not (0 < args.active_threshold <= 1):
        raise ValueError(
            "--active-threshold must be within (0, 1]."
        )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required_manifest = {
        "subject_id",
        "task",
        "block_id",
        "sample_parquet",
    }
    missing_manifest = sorted(
        required_manifest - set(manifest.columns)
    )
    if missing_manifest:
        raise ValueError(
            f"Manifest lacks columns: {missing_manifest}"
        )
    manifest["subject_id"] = (
        manifest["subject_id"].astype(str).str.upper()
    )
    manifest["task"] = manifest["task"].astype(str).str.upper()
    subjects = sorted(manifest["subject_id"].unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Manifest subjects={len(subjects)}, "
            f"expected={EXPECTED_SUBJECTS}."
        )

    processed_root = Path(args.processed_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_cache_root = (
        output_dir / "subject_hard_aoi_noise_features"
    )

    clean, _, _, _ = load_merged_data(
        raw_path=Path(args.clean_raw_features),
        raw_sets_path=Path(args.raw_feature_sets),
        soft_path=Path(args.clean_soft_features),
        soft_manifest_path=Path(args.soft_feature_manifest),
        word_template_path=Path(args.word_template),
    )
    clean_hard = normalize_hard_table(
        Path(args.clean_hard_features)
    )
    clean = add_hard_features(clean, clean_hard)
    audit_shared_quality(clean)

    sentence_folds = load_sentence_folds(
        Path(args.sentence_folds_file)
    )
    clean = clean.merge(
        sentence_folds[["sentence_key", "sentence_fold"]],
        on="sentence_key",
        how="left",
        validate="many_to_one",
    )
    if clean["sentence_fold"].isna().any():
        raise ValueError("Clean trials lack sentence folds.")
    clean["sentence_fold"] = clean["sentence_fold"].astype(int)

    aoi_lookup = load_word_template(Path(args.word_template))
    soft_conditions = load_soft_conditions(
        cache_root=Path(args.soft_noise_cache_root),
        processed_root=processed_root,
        subjects=subjects,
        sigmas=sigmas,
        repeats=args.repeats,
    )
    hard_conditions = extract_or_load_hard_conditions(
        manifest=manifest,
        subjects=subjects,
        processed_root=processed_root,
        cache_root=hard_cache_root,
        aoi_lookup=aoi_lookup,
        sigmas=sigmas,
        repeats=args.repeats,
        seed=args.seed,
        sigma_word_height_ratio=(
            args.sigma_word_height_ratio
        ),
        active_threshold=args.active_threshold,
        overwrite=args.overwrite_hard_features,
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
    conditions = {
        condition: make_condition_table(
            metadata,
            soft_conditions[condition],
            hard_conditions[condition],
        )
        for condition in soft_conditions
    }
    feature_sets = build_feature_sets()

    print("\n=== SOFT VS HARD COORDINATE-NOISE INPUT ===")
    print(f"subjects: {len(subjects)}")
    print(f"trials: {len(clean)}")
    print(f"sigmas: {sigmas}")
    print(f"repeats: {args.repeats}")
    print(f"fixed C: {args.c_value}")
    print("shared quality audit: PASS for clean and all conditions")

    folds, predictions = evaluate_clean_train_noise_test(
        clean=clean,
        conditions=conditions,
        feature_sets=feature_sets,
        c_value=args.c_value,
        seed=args.seed,
    )
    repeat_metrics = subject_repeat_metrics(predictions)
    subject_metrics = subject_mean_metrics(repeat_metrics)
    summary = summarize_noise(subject_metrics)
    tests = build_tests(
        noise_subject=subject_metrics,
        clean_subject_path=Path(args.clean_subject_metrics),
    )

    paths = {
        "folds": output_dir / "soft_vs_hard_noise_fold_metrics.csv",
        "predictions": output_dir / "soft_vs_hard_noise_predictions.csv",
        "repeat_metrics": (
            output_dir / "soft_vs_hard_noise_subject_repeat_metrics.csv"
        ),
        "subject_metrics": (
            output_dir / "soft_vs_hard_noise_subject_metrics.csv"
        ),
        "summary": output_dir / "soft_vs_hard_noise_summary.csv",
        "tests": output_dir / "soft_vs_hard_noise_tests.csv",
        "manifest": output_dir / "soft_vs_hard_noise_manifest.json",
    }
    folds.to_csv(paths["folds"], index=False, encoding="utf-8-sig")
    predictions.to_csv(
        paths["predictions"], index=False, encoding="utf-8-sig"
    )
    repeat_metrics.to_csv(
        paths["repeat_metrics"], index=False, encoding="utf-8-sig"
    )
    subject_metrics.to_csv(
        paths["subject_metrics"], index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        paths["summary"], index=False, encoding="utf-8-sig"
    )
    tests.to_csv(
        paths["tests"], index=False, encoding="utf-8-sig"
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "protocol": (
                    "clean30_train_noise30_test_dual_holdout"
                ),
                "sigmas_px": sigmas,
                "repeats": int(args.repeats),
                "c_value": float(args.c_value),
                "seed": int(args.seed),
                "n_subjects": EXPECTED_SUBJECTS,
                "n_trials": EXPECTED_TRIALS,
                "feature_sets": feature_sets,
                "hard_assignment": "argmax_q_with_qmax_mass",
                "excluded_for_fairness": [
                    "soft_aoi_sample_entropy_mean",
                    "soft_aoi_sample_entropy_median",
                ],
                "soft_noise_cache_root": str(
                    args.soft_noise_cache_root
                ),
                "hard_noise_cache_root": str(hard_cache_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    display = summary[
        [
            "noise_sigma_px",
            "feature_set",
            "n_input_features",
            "macro_f1_subject_mean",
            "balanced_accuracy_subject_mean",
            "roc_auc_subject_mean",
            "pr_auc_nr_subject_mean",
        ]
    ]
    print("\n=== SOFT-AOI VS HARD-AOI NOISE SUMMARY ===")
    print(display.to_string(index=False))

    key_tests = tests[
        tests["metric"].isin(
            ["macro_f1", "roc_auc", "pr_auc_nr"]
        )
    ][
        [
            "test_family",
            "noise_sigma_px",
            "metric",
            "comparison",
            "reference_mean",
            "candidate_mean",
            "mean_difference_candidate_minus_reference",
            "wins",
            "ties",
            "losses",
            "p_value",
            "p_holm_within_family_metric",
        ]
    ]
    print("\n=== KEY SOFT-AOI VS HARD-AOI NOISE TESTS ===")
    print(key_tests.to_string(index=False))

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
