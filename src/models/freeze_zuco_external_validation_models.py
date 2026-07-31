#!/usr/bin/env python
"""Freeze ZuCo-trained models for no-retraining webcam external validation.

Primary external-validation bundle
----------------------------------
All five conditions are fitted on exactly the same 16 subjects and the same
11,324 trials for which official ``sent_gaze_sacc`` features are available:

1. dummy_reference
2. rawstat_all                         (18 features)
3. soft_full_all                       (41 features)
4. soft_compact_distribution_membership (27 features)
5. official_sent_gaze_sacc             (9 features)

This matched bundle must be used for the primary five-condition external
comparison so that training-subject and training-trial counts are identical.

Secondary deployment bundle
---------------------------
A Compact Soft-AOI model is additionally fitted on all 18 ZuCo subjects and
13,302 raw trials. It is saved separately and must not replace the matched
Common16 Compact model in the primary five-condition comparison.

Frozen settings
---------------
- labels: TSR=0, NR=1
- median imputation with missing indicators
- StandardScaler
- LinearSVC
- C=0.01 by default
- decision threshold=0.0
- no webcam labels are used

The output contains joblib artifacts, canonical feature schemas, preprocessing
statistics, input/artifact SHA-256 hashes, and a model manifest.
Only load the generated joblib files from a trusted location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import run_zuco_soft_aoi_official_common16_nested_dual as common16
from run_zuco_soft_aoi_ablation import DISTRIBUTION_BASE


EXPECTED_COMMON16_SUBJECTS = 16
EXPECTED_COMMON16_TRIALS = 11324
EXPECTED_ALL18_SUBJECTS = 18
EXPECTED_ALL18_TRIALS = 13302

COMPACT_NAME = "soft_compact_distribution_membership"
PRIMARY_MODEL_ORDER = [
    "dummy_reference",
    "rawstat_all",
    "soft_full_all",
    COMPACT_NAME,
    "official_sent_gaze_sacc",
]

LABEL_TO_INT = {"TSR": 0, "NR": 1}
INT_TO_LABEL = {0: "TSR", 1: "NR"}


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


def canonical_name(training_column: str) -> str:
    for prefix in ("raw__", "soft__", "official__"):
        if training_column.startswith(prefix):
            return training_column[len(prefix) :]
    return training_column


def validate_numeric_frame(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks feature columns: {missing}")

    result = frame[columns].copy()
    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )
    values = result.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{name} contains infinite feature values.")
    return result


def build_linear_pipeline(
    c_value: float,
    seed: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=float(c_value),
                    dual=False,
                    max_iter=30000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def fit_primary_models(
    data: pd.DataFrame,
    prefixed_feature_sets: dict[str, list[str]],
    c_value: float,
    decision_threshold: float,
    seed: int,
    models_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    y = data["label_nr"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Matched Common16 training data lacks both classes.")

    models_dir.mkdir(parents=True, exist_ok=True)

    manifest_models: dict[str, dict[str, Any]] = {}
    schema_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []

    # Dummy has no gaze-feature input.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(
        np.zeros((len(data), 1), dtype=float),
        y,
    )
    dummy_artifact = {
        "artifact_version": "1.0",
        "feature_set": "dummy_reference",
        "training_scope": "common16_matched",
        "model": dummy,
        "feature_columns": [],
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "decision_threshold": None,
        "expected_input_type": "no_features",
        "trained_at_utc": utc_now_iso(),
    }
    dummy_path = models_dir / "dummy_reference.joblib"
    joblib.dump(dummy_artifact, dummy_path, compress=3)
    reloaded_dummy = joblib.load(dummy_path)
    if int(reloaded_dummy["model"].predict([[0.0]])[0]) not in {0, 1}:
        raise RuntimeError("Dummy artifact round-trip verification failed.")

    manifest_models["dummy_reference"] = {
        "artifact_path": str(dummy_path),
        "artifact_sha256": sha256_file(dummy_path),
        "training_scope": "common16_matched",
        "n_training_subjects": int(
            data["subject_id"].nunique()
        ),
        "n_training_trials": int(len(data)),
        "n_input_features": 0,
        "c_value": None,
        "decision_threshold": None,
        "canonical_feature_columns": [],
    }

    for feature_set in PRIMARY_MODEL_ORDER[1:]:
        training_columns = list(
            prefixed_feature_sets[feature_set]
        )
        canonical_columns = [
            canonical_name(column)
            for column in training_columns
        ]
        if len(canonical_columns) != len(set(canonical_columns)):
            raise ValueError(
                f"{feature_set} canonical feature names are duplicated."
            )

        x_prefixed = validate_numeric_frame(
            data,
            training_columns,
            feature_set,
        )
        x = x_prefixed.rename(
            columns=dict(
                zip(training_columns, canonical_columns)
            )
        )

        pipeline = build_linear_pipeline(
            c_value=c_value,
            seed=seed,
        )
        pipeline.fit(x, y)

        training_predictions = pipeline.predict(x)
        training_scores = np.asarray(
            pipeline.decision_function(x),
            dtype=float,
        ).reshape(-1)
        if len(training_predictions) != len(data):
            raise RuntimeError(
                f"{feature_set} prediction-length verification failed."
            )
        if not np.isfinite(training_scores).all():
            raise RuntimeError(
                f"{feature_set} produced non-finite training scores."
            )

        artifact = {
            "artifact_version": "1.0",
            "feature_set": feature_set,
            "training_scope": "common16_matched",
            "pipeline": pipeline,
            "feature_columns": canonical_columns,
            "training_feature_columns": training_columns,
            "label_to_int": LABEL_TO_INT,
            "int_to_label": INT_TO_LABEL,
            "decision_threshold": float(decision_threshold),
            "prediction_rule": (
                "NR(1) if decision_function >= threshold; "
                "otherwise TSR(0)"
            ),
            "c_value": float(c_value),
            "seed": int(seed),
            "trained_at_utc": utc_now_iso(),
        }
        artifact_path = models_dir / f"{feature_set}.joblib"
        joblib.dump(artifact, artifact_path, compress=3)

        reloaded = joblib.load(artifact_path)
        reloaded_scores = np.asarray(
            reloaded["pipeline"].decision_function(x),
            dtype=float,
        ).reshape(-1)
        if not np.allclose(
            training_scores,
            reloaded_scores,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{feature_set} artifact round-trip changed scores."
            )

        imputer: SimpleImputer = pipeline.named_steps["imputer"]
        scaler: StandardScaler = pipeline.named_steps["scaler"]
        indicator_indices = (
            list(
                np.asarray(
                    imputer.indicator_.features_,
                    dtype=int,
                )
            )
            if imputer.indicator_ is not None
            else []
        )
        indicator_names = [
            canonical_columns[index]
            for index in indicator_indices
        ]

        for feature_index, feature_name in enumerate(
            canonical_columns
        ):
            schema_rows.append(
                {
                    "training_scope": "common16_matched",
                    "feature_set": feature_set,
                    "feature_index": int(feature_index),
                    "feature_name": feature_name,
                    "source_training_column": (
                        training_columns[feature_index]
                    ),
                    "dtype": "float64",
                    "allows_missing": True,
                    "is_missing_indicator_source": (
                        feature_index in indicator_indices
                    ),
                }
            )
            preprocessing_rows.append(
                {
                    "training_scope": "common16_matched",
                    "feature_set": feature_set,
                    "feature_index": int(feature_index),
                    "feature_name": feature_name,
                    "imputer_median": float(
                        imputer.statistics_[feature_index]
                    ),
                    "training_missing_count": int(
                        x[feature_name].isna().sum()
                    ),
                    "training_missing_rate": float(
                        x[feature_name].isna().mean()
                    ),
                }
            )

        manifest_models[feature_set] = {
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "training_scope": "common16_matched",
            "n_training_subjects": int(
                data["subject_id"].nunique()
            ),
            "n_training_trials": int(len(data)),
            "n_input_features": int(len(canonical_columns)),
            "n_post_imputation_features": int(
                len(scaler.mean_)
            ),
            "n_missing_indicator_features": int(
                len(indicator_indices)
            ),
            "missing_indicator_source_features": indicator_names,
            "c_value": float(c_value),
            "decision_threshold": float(
                decision_threshold
            ),
            "canonical_feature_columns": canonical_columns,
            "class_counts": {
                INT_TO_LABEL[int(label)]: int(
                    np.sum(y == label)
                )
                for label in sorted(np.unique(y))
            },
        }

    return (
        manifest_models,
        schema_rows,
        preprocessing_rows,
    )


def fit_compact_all18(
    soft_features_path: Path,
    soft_manifest_path: Path,
    c_value: float,
    decision_threshold: float,
    seed: int,
    output_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    soft = pd.read_csv(require_file(soft_features_path))
    required = {"trial_id", "subject_id", "task"}
    missing = sorted(required - set(soft.columns))
    if missing:
        raise ValueError(
            f"All18 Soft-AOI table lacks columns: {missing}"
        )

    soft["trial_id"] = soft["trial_id"].astype(str)
    soft["subject_id"] = (
        soft["subject_id"].astype(str).str.upper()
    )
    soft["task"] = soft["task"].astype(str).str.upper()
    soft["label_nr"] = soft["task"].map(LABEL_TO_INT)
    if soft["label_nr"].isna().any():
        bad = sorted(
            soft.loc[
                soft["label_nr"].isna(),
                "task",
            ].unique()
        )
        raise ValueError(
            f"All18 Soft-AOI has unexpected tasks: {bad}"
        )
    soft["label_nr"] = soft["label_nr"].astype(int)

    if len(soft) != EXPECTED_ALL18_TRIALS:
        raise ValueError(
            f"All18 Soft-AOI rows={len(soft)}, "
            f"expected={EXPECTED_ALL18_TRIALS}."
        )
    if soft["subject_id"].nunique() != EXPECTED_ALL18_SUBJECTS:
        raise ValueError(
            f"All18 subjects={soft['subject_id'].nunique()}, "
            f"expected={EXPECTED_ALL18_SUBJECTS}."
        )
    if soft["trial_id"].duplicated().any():
        raise ValueError(
            "All18 Soft-AOI table contains duplicate trial IDs."
        )

    _, quality_columns = common16.load_soft_manifest(
        soft_manifest_path
    )
    compact_columns = (
        list(DISTRIBUTION_BASE)
        + list(quality_columns)
    )
    if len(compact_columns) != 27:
        raise ValueError(
            f"All18 Compact count={len(compact_columns)}, "
            "expected=27."
        )

    x = validate_numeric_frame(
        soft,
        compact_columns,
        "All18 Compact Soft-AOI",
    )
    y = soft["label_nr"].to_numpy(dtype=int)

    pipeline = build_linear_pipeline(
        c_value=c_value,
        seed=seed,
    )
    pipeline.fit(x, y)

    scores = np.asarray(
        pipeline.decision_function(x),
        dtype=float,
    ).reshape(-1)
    if not np.isfinite(scores).all():
        raise RuntimeError(
            "All18 Compact model produced non-finite scores."
        )

    artifact = {
        "artifact_version": "1.0",
        "feature_set": COMPACT_NAME,
        "training_scope": "all18_maxdata_secondary",
        "pipeline": pipeline,
        "feature_columns": compact_columns,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "decision_threshold": float(decision_threshold),
        "prediction_rule": (
            "NR(1) if decision_function >= threshold; "
            "otherwise TSR(0)"
        ),
        "c_value": float(c_value),
        "seed": int(seed),
        "trained_at_utc": utc_now_iso(),
        "primary_external_comparison_eligible": False,
        "reason": (
            "Uses 18 subjects whereas official sent_gaze_sacc "
            "is available only for the matched Common16 subset."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)

    reloaded = joblib.load(output_path)
    reloaded_scores = np.asarray(
        reloaded["pipeline"].decision_function(x),
        dtype=float,
    ).reshape(-1)
    if not np.allclose(
        scores,
        reloaded_scores,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "All18 Compact artifact round-trip changed scores."
        )

    imputer: SimpleImputer = pipeline.named_steps["imputer"]
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    indicator_indices = list(
        np.asarray(
            imputer.indicator_.features_,
            dtype=int,
        )
    )

    schema_rows = []
    preprocessing_rows = []
    for feature_index, feature_name in enumerate(
        compact_columns
    ):
        schema_rows.append(
            {
                "training_scope": "all18_maxdata_secondary",
                "feature_set": COMPACT_NAME,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "source_training_column": feature_name,
                "dtype": "float64",
                "allows_missing": True,
                "is_missing_indicator_source": (
                    feature_index in indicator_indices
                ),
            }
        )
        preprocessing_rows.append(
            {
                "training_scope": "all18_maxdata_secondary",
                "feature_set": COMPACT_NAME,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "imputer_median": float(
                    imputer.statistics_[feature_index]
                ),
                "training_missing_count": int(
                    x[feature_name].isna().sum()
                ),
                "training_missing_rate": float(
                    x[feature_name].isna().mean()
                ),
            }
        )

    model_manifest = {
        "artifact_path": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "training_scope": "all18_maxdata_secondary",
        "primary_external_comparison_eligible": False,
        "n_training_subjects": int(
            soft["subject_id"].nunique()
        ),
        "n_training_trials": int(len(soft)),
        "n_input_features": int(len(compact_columns)),
        "n_post_imputation_features": int(
            len(scaler.mean_)
        ),
        "n_missing_indicator_features": int(
            len(indicator_indices)
        ),
        "missing_indicator_source_features": [
            compact_columns[index]
            for index in indicator_indices
        ],
        "c_value": float(c_value),
        "decision_threshold": float(
            decision_threshold
        ),
        "canonical_feature_columns": compact_columns,
        "class_counts": {
            INT_TO_LABEL[int(label)]: int(
                np.sum(y == label)
            )
            for label in sorted(np.unique(y))
        },
    }
    return (
        model_manifest,
        schema_rows,
        preprocessing_rows,
    )


def write_subject_audit(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = []
    for subject, group in data.groupby(
        "subject_id",
        sort=True,
    ):
        counts = group.groupby("task").size().to_dict()
        rows.append(
            {
                "training_scope": "common16_matched",
                "subject_id": subject,
                "n_trials": int(len(group)),
                "n_nr": int(counts.get("NR", 0)),
                "n_tsr": int(counts.get("TSR", 0)),
                "n_unique_sentences": int(
                    group["sentence_key"].nunique()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


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
        "--raw-features",
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
        "--soft-features",
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
        "--sentence-folds-file",
        default=(
            "outputs/soft_aoi_confound_controls_30hz/"
            "dual_holdout_sentence_folds.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/frozen_zuco_external_validation_v1"
        ),
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--skip-all18-compact",
        action="store_true",
        help=(
            "Do not create the secondary 18-subject "
            "max-data Compact model."
        ),
    )
    args = parser.parse_args()

    if args.c_value <= 0:
        raise ValueError("--c-value must be positive.")
    if not np.isfinite(args.decision_threshold):
        raise ValueError(
            "--decision-threshold must be finite."
        )

    input_paths = {
        "common_map": require_file(
            Path(args.common_map)
        ),
        "raw_features": require_file(
            Path(args.raw_features)
        ),
        "raw_feature_sets": require_file(
            Path(args.raw_feature_sets)
        ),
        "soft_features": require_file(
            Path(args.soft_features)
        ),
        "soft_feature_manifest": require_file(
            Path(args.soft_feature_manifest)
        ),
        "sentence_folds_file": require_file(
            Path(args.sentence_folds_file)
        ),
    }

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    common_map, official_columns = (
        common16.load_common_map(
            input_paths["common_map"]
        )
    )
    (
        data,
        prefixed_feature_sets,
        load_metadata,
    ) = common16.load_matched_table(
        common=common_map,
        official_columns=official_columns,
        raw_path=input_paths["raw_features"],
        raw_sets_path=input_paths["raw_feature_sets"],
        soft_path=input_paths["soft_features"],
        soft_manifest_path=(
            input_paths["soft_feature_manifest"]
        ),
        sentence_folds_path=(
            input_paths["sentence_folds_file"]
        ),
    )

    if len(data) != EXPECTED_COMMON16_TRIALS:
        raise ValueError(
            f"Common16 rows={len(data)}, "
            f"expected={EXPECTED_COMMON16_TRIALS}."
        )
    if (
        data["subject_id"].nunique()
        != EXPECTED_COMMON16_SUBJECTS
    ):
        raise ValueError(
            f"Common16 subjects="
            f"{data['subject_id'].nunique()}, "
            f"expected={EXPECTED_COMMON16_SUBJECTS}."
        )

    print("=== FREEZE ZUCO EXTERNAL-VALIDATION MODELS ===")
    print(
        f"Primary matched training: "
        f"{data['subject_id'].nunique()} subjects, "
        f"{len(data)} trials"
    )
    print(f"Fixed C: {args.c_value}")
    print(
        f"Fixed decision threshold: "
        f"{args.decision_threshold}"
    )

    (
        primary_models,
        schema_rows,
        preprocessing_rows,
    ) = fit_primary_models(
        data=data,
        prefixed_feature_sets=(
            prefixed_feature_sets
        ),
        c_value=args.c_value,
        decision_threshold=args.decision_threshold,
        seed=args.seed,
        models_dir=models_dir,
    )

    secondary_models: dict[str, Any] = {}
    if not args.skip_all18_compact:
        compact18_path = (
            models_dir
            / "soft_compact_all18_secondary.joblib"
        )
        (
            compact18_manifest,
            compact18_schema,
            compact18_preprocessing,
        ) = fit_compact_all18(
            soft_features_path=(
                input_paths["soft_features"]
            ),
            soft_manifest_path=(
                input_paths["soft_feature_manifest"]
            ),
            c_value=args.c_value,
            decision_threshold=(
                args.decision_threshold
            ),
            seed=args.seed,
            output_path=compact18_path,
        )
        secondary_models[
            "soft_compact_all18_secondary"
        ] = compact18_manifest
        schema_rows.extend(compact18_schema)
        preprocessing_rows.extend(
            compact18_preprocessing
        )

    feature_schema_path = (
        output_dir / "feature_schema.csv"
    )
    preprocessing_path = (
        output_dir / "preprocessing_statistics.csv"
    )
    subject_audit_path = (
        output_dir / "common16_training_subject_audit.csv"
    )
    training_trials_path = (
        output_dir / "common16_training_trials.csv"
    )
    manifest_path = output_dir / "model_manifest.json"
    checksums_path = output_dir / "checksums_sha256.csv"

    pd.DataFrame(schema_rows).to_csv(
        feature_schema_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(preprocessing_rows).to_csv(
        preprocessing_path,
        index=False,
        encoding="utf-8-sig",
    )
    write_subject_audit(
        data,
        subject_audit_path,
    )
    data[
        [
            "trial_id",
            "subject_id",
            "task",
            "label_nr",
            "sentence_key",
            "sentence_fold",
        ]
    ].to_csv(
        training_trials_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "bundle_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "purpose": (
            "No-retraining webcam external validation"
        ),
        "primary_analysis": {
            "name": (
                "matched_common16_five_condition"
            ),
            "eligible_models": PRIMARY_MODEL_ORDER,
            "rule": (
                "Use only the Common16 artifacts for the "
                "primary five-condition external comparison."
            ),
            "n_training_subjects": int(
                data["subject_id"].nunique()
            ),
            "n_training_trials": int(len(data)),
            "class_mapping": {
                "TSR": 0,
                "NR": 1,
            },
            "external_label_mapping_planned": {
                "information_seeking_or_targeted_reading": 0,
                "normal_reading": 1,
            },
            "c_value": float(args.c_value),
            "decision_threshold": float(
                args.decision_threshold
            ),
            "imputer": (
                "SimpleImputer(strategy='median', "
                "add_indicator=True)"
            ),
            "scaler": "StandardScaler",
            "classifier": "LinearSVC",
            "soft_aoi": {
                "sampling_rate_hz": 30,
                "sigma_word_height_ratio": (
                    2.0 / 3.0
                ),
                "active_threshold": 0.1,
                "compact_definition": (
                    "15 distribution + "
                    "12 membership/quality features"
                ),
            },
        },
        "official_sent_gaze_sacc_external_requirement": {
            "status": (
                "feature extractor still required"
            ),
            "required_columns_in_order": [
                f"official_feature_{index}"
                for index in range(
                    len(official_columns)
                )
            ],
            "n_features": int(
                len(official_columns)
            ),
            "constraint": (
                "Webcam features must reproduce the same "
                "event-detection algorithm, units, aggregation "
                "definitions, and column order. Merely creating "
                "nine arbitrary fixation/saccade statistics is "
                "not an equivalent official-feature condition."
            ),
        },
        "primary_models": primary_models,
        "secondary_models": secondary_models,
        "input_files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "data_metadata": load_metadata,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "reporting_constraints": [
            (
                "The Common16 five-condition bundle is the "
                "primary fair external comparison."
            ),
            (
                "The all18 Compact artifact is secondary and "
                "must be reported separately."
            ),
            (
                "Webcam labels must not be used for feature "
                "selection, imputation, scaling, C selection, "
                "or threshold tuning in the primary analysis."
            ),
            (
                "Coordinate calibration independent of task "
                "labels is allowed and must be documented."
            ),
        ],
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    files_to_hash = [
        *sorted(models_dir.glob("*.joblib")),
        feature_schema_path,
        preprocessing_path,
        subject_audit_path,
        training_trials_path,
        manifest_path,
    ]
    checksum_rows = [
        {
            "relative_path": str(
                path.relative_to(output_dir)
            ),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in files_to_hash
    ]
    pd.DataFrame(checksum_rows).to_csv(
        checksums_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== FROZEN PRIMARY MODELS ===")
    for model_name in PRIMARY_MODEL_ORDER:
        info = primary_models[model_name]
        print(
            f"{model_name}: "
            f"nfeat={info['n_input_features']}, "
            f"trials={info['n_training_trials']}, "
            f"sha256={info['artifact_sha256'][:12]}..."
        )

    if secondary_models:
        info = secondary_models[
            "soft_compact_all18_secondary"
        ]
        print("\n=== SECONDARY MODEL ===")
        print(
            "soft_compact_all18_secondary: "
            f"nfeat={info['n_input_features']}, "
            f"subjects={info['n_training_subjects']}, "
            f"trials={info['n_training_trials']}, "
            f"sha256={info['artifact_sha256'][:12]}..."
        )

    print("\nWritten:")
    for path in [
        models_dir,
        feature_schema_path,
        preprocessing_path,
        subject_audit_path,
        training_trials_path,
        manifest_path,
        checksums_path,
    ]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
