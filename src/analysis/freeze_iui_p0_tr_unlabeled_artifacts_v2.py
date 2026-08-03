from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


QUALITY_FEATURES = [
    "soft_aoi_valid_rate",
    "soft_aoi_active_rate",
    "soft_aoi_word_box_hit_rate",
    "soft_aoi_sentence_envelope_hit_rate",
    "soft_aoi_qmax_mean",
    "soft_aoi_qmax_median",
    "soft_aoi_qmax_q10",
    "soft_aoi_qmax_ge_0_5_rate",
    "soft_aoi_nearest_distance_mean_word_heights",
    "soft_aoi_nearest_distance_median_word_heights",
    "soft_aoi_nearest_distance_q90_word_heights",
    "soft_aoi_nearest_distance_q95_word_heights",
]

PARTICIPANT_ALIASES = [
    "participant_id",
    "participant",
    "subject_id",
    "subject",
    "worker_id",
    "worker",
]

TRIAL_ALIASES = [
    "external_trial_id",
    "trial_id",
    "trial",
    "trial_index",
    "trial_uid",
    "sample_id",
    "item_id",
]

LABEL_ALIASES = [
    "label_nr",
    "label",
    "y",
    "target",
    "class",
    "y_true",
    "true_label",
]

TASK_ALIASES = [
    "task",
    "condition",
    "reading_goal",
    "label_name",
]


def norm(value: Any) -> str:
    return str(value).strip().lower()


def first_existing(columns: Iterable[str], aliases: list[str]) -> str:
    lower = {norm(column): str(column) for column in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_binary_label(series: pd.Series, column_name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = numeric.astype(int)
        unique = set(values.unique().tolist())
        if unique.issubset({0, 1}):
            return values.to_numpy()

    positive = {
        "1",
        "nr",
        "natural reading",
        "natural_reading",
        "ordinary",
        "ordinary reading",
        "ordinary_reading",
    }
    negative = {
        "0",
        "is",
        "information seeking",
        "information_seeking",
        "tsr",
        "task specific reading",
        "task_specific_reading",
    }

    output: list[int] = []
    unknown: set[str] = set()
    for value in series:
        token = norm(value).replace("-", " ")
        if token in positive:
            output.append(1)
        elif token in negative:
            output.append(0)
        else:
            unknown.add(token)

    if unknown:
        raise ValueError(
            f"Unsupported label values in {column_name}: {sorted(unknown)[:20]}"
        )
    return np.asarray(output, dtype=int)


def load_frozen_model(
    model_path: Path,
) -> tuple[dict[str, Any], Any, list[str], float]:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError(
            f"Expected a dict artifact, found {type(artifact).__name__}"
        )

    required = {"pipeline", "feature_columns", "decision_threshold"}
    missing = required - set(artifact)
    if missing:
        raise KeyError(f"Frozen artifact is missing keys: {sorted(missing)}")

    pipeline = artifact["pipeline"]
    feature_columns = list(artifact["feature_columns"])
    threshold = float(artifact["decision_threshold"])

    if len(feature_columns) != 41:
        raise RuntimeError(
            f"Expected Full 41 features, found {len(feature_columns)}"
        )
    if feature_columns[-12:] != QUALITY_FEATURES:
        raise RuntimeError(
            "The frozen feature order does not end with the locked 12 "
            "measurement-associated features."
        )
    if list(pipeline.classes_) != [0, 1]:
        raise RuntimeError(
            f"Unexpected frozen class order: {pipeline.classes_}"
        )

    return artifact, pipeline, feature_columns, threshold


def transform_before_model(
    pipeline: Any,
    frame: pd.DataFrame,
) -> np.ndarray:
    values: Any = frame
    steps = list(pipeline.steps)

    if not steps or steps[-1][0] != "model":
        raise RuntimeError(
            f"Unexpected pipeline steps: {[name for name, _ in steps]}"
        )

    for _, transformer in steps[:-1]:
        values = transformer.transform(values)

    return np.asarray(values, dtype=float)


def exact_score_components(
    pipeline: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    standardized = transform_before_model(
        pipeline,
        frame[feature_columns],
    )
    model = pipeline.named_steps["model"]
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])

    if standardized.shape[1] != len(feature_columns):
        raise RuntimeError(
            "Transformed feature dimension does not match frozen schema."
        )
    if coefficients.size != len(feature_columns):
        raise RuntimeError(
            "Coefficient dimension does not match frozen schema."
        )

    contributions = standardized * coefficients[None, :]
    quality_indices = np.asarray(
        [feature_columns.index(name) for name in QUALITY_FEATURES],
        dtype=int,
    )
    content_indices = np.asarray(
        [
            index
            for index in range(len(feature_columns))
            if index not in set(quality_indices.tolist())
        ],
        dtype=int,
    )

    quality_contribution = contributions[:, quality_indices].sum(axis=1)
    content_contribution = contributions[:, content_indices].sum(axis=1)
    reconstructed = (
        intercept + content_contribution + quality_contribution
    )
    raw_score = np.asarray(
        pipeline.decision_function(frame[feature_columns]),
        dtype=float,
    )

    max_error = float(np.max(np.abs(raw_score - reconstructed)))
    if max_error > 1e-8:
        raise RuntimeError(
            "Exact score decomposition failed: "
            f"max_abs_error={max_error:.3e}"
        )

    return raw_score, content_contribution, quality_contribution


def participant_loo_median(
    scores: np.ndarray,
    participant_ids: np.ndarray,
) -> np.ndarray:
    centers = np.full(scores.shape, np.nan, dtype=float)

    for participant_id in pd.unique(participant_ids):
        indices = np.flatnonzero(participant_ids == participant_id)
        if len(indices) < 2:
            raise ValueError(
                "Leave-one-trial-out participant calibration requires at "
                f"least two trials: participant={participant_id!r}"
            )
        participant_scores = scores[indices]
        for local_index, global_index in enumerate(indices):
            other_scores = np.delete(participant_scores, local_index)
            centers[global_index] = float(np.median(other_scores))

    if not np.isfinite(centers).all():
        raise RuntimeError("Non-finite LOO participant centers were produced.")

    return centers


def load_source_references(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    references = data["source_references"]

    required = {
        "full_final_reference",
        "quality_contribution_reference",
        "quality_removed_final_reference",
    }
    missing = required - set(references)
    if missing:
        raise KeyError(
            f"Factorization manifest is missing: {sorted(missing)}"
        )

    return {
        key: float(references[key])
        for key in required
    }


def resolve_primary_manifest(
    stage2_dir: Path,
    stage3a_dir: Path,
) -> Path:
    """Locate the primary trial manifest without assuming Stage 3A copied it."""
    preferred = [
        stage2_dir / "trial_manifest_primary_candidates.csv",
        stage3a_dir / "trial_manifest_primary_candidates.csv",
        stage2_dir / "trial_manifest_all.csv",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    discovered: list[Path] = []
    for directory in (stage2_dir, stage3a_dir):
        if directory.exists():
            discovered.extend(
                sorted(directory.rglob("*trial*manifest*.csv"))
            )

    if discovered:
        discovered.sort(
            key=lambda path: (
                "primary" not in path.name.lower(),
                "candidate" not in path.name.lower(),
                -path.stat().st_size,
                str(path).lower(),
            )
        )
        return discovered[0]

    raise FileNotFoundError(
        "No trial manifest was found under Stage 2 or Stage 3A.\n"
        f"Stage 2: {stage2_dir}\n"
        f"Stage 3A: {stage3a_dir}"
    )


def read_expected_manifest(path: Path) -> tuple[int, set[str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    trial_column = first_existing(frame.columns, TRIAL_ALIASES)
    if not trial_column:
        raise RuntimeError(
            f"Could not identify trial ID in expected manifest: {path}"
        )

    trial_ids = set(frame[trial_column].astype(str))
    if len(trial_ids) != len(frame):
        raise RuntimeError(
            "Expected primary manifest contains duplicate trial IDs."
        )

    return len(frame), trial_ids


def write_sha_manifest(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        output_dir / "file_hash_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        f"{row['sha256']}  {row['path']}"
        for row in rows
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze unlabeled Turkish Full-41 features and frozen QCN-PC "
            "scores before any predictive metric is calculated."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--stage2-dir",
        default="outputs/webqamgaze_tr_confirmatory_stage2_v1",
    )
    parser.add_argument(
        "--stage3a-dir",
        default="outputs/webqamgaze_tr_confirmatory_stage3a_v1",
    )
    parser.add_argument(
        "--stage3c-dir",
        default="outputs/webqamgaze_tr_confirmatory_stage3c_v1",
    )
    parser.add_argument(
        "--frozen-onestop-model",
        default=(
            "outputs/frozen_onestop_external_validation_v1/"
            "models/soft_full_all.joblib"
        ),
    )
    parser.add_argument(
        "--factor-manifest",
        default=(
            "outputs/linear_feature_group_factorization_v1/"
            "linear_feature_group_factorization_manifest.json"
        ),
    )
    parser.add_argument(
        "--protocol-snapshot",
        default="outputs/iui_p0_protocol_lock_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/iui_p0_confirmatory_TR_v1",
    )
    parser.add_argument(
        "--maximum-feature-failure-rate",
        type=float,
        default=0.20,
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    stage2_dir = (root / args.stage2_dir).resolve()
    stage3a_dir = (root / args.stage3a_dir).resolve()
    stage3c_dir = (root / args.stage3c_dir).resolve()
    model_path = (root / args.frozen_onestop_model).resolve()
    factor_manifest_path = (root / args.factor_manifest).resolve()
    protocol_snapshot = (root / args.protocol_snapshot).resolve()
    output_dir = (root / args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit").mkdir(exist_ok=True)
    (output_dir / "manifests").mkdir(exist_ok=True)
    (output_dir / "features").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "sealed").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)

    primary_manifest_path = resolve_primary_manifest(
        stage2_dir=stage2_dir,
        stage3a_dir=stage3a_dir,
    )

    required_paths = [
        model_path,
        factor_manifest_path,
        protocol_snapshot,
        primary_manifest_path,
        stage3c_dir / "soft_full_features_primary.csv",
        stage3c_dir / "stage3c_manifest.json",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    (
        frozen_artifact,
        pipeline,
        feature_columns,
        frozen_threshold,
    ) = load_frozen_model(model_path)
    source_references = load_source_references(
        factor_manifest_path
    )

    expected_count, expected_trial_ids = read_expected_manifest(
        primary_manifest_path
    )

    source_features_path = (
        stage3c_dir / "soft_full_features_primary.csv"
    )
    source_frame = pd.read_csv(source_features_path)

    participant_column = first_existing(
        source_frame.columns,
        PARTICIPANT_ALIASES,
    )
    trial_column = first_existing(
        source_frame.columns,
        TRIAL_ALIASES,
    )
    label_column = first_existing(
        source_frame.columns,
        LABEL_ALIASES,
    )
    task_column = first_existing(
        source_frame.columns,
        TASK_ALIASES,
    )

    if not participant_column:
        raise RuntimeError("Participant ID column was not found.")
    if not trial_column:
        raise RuntimeError("Trial ID column was not found.")
    if not label_column:
        raise RuntimeError(
            "A label column is required only to verify both classes and "
            "create the separate sealed label file."
        )

    missing_features = [
        name
        for name in feature_columns
        if name not in source_frame.columns
    ]
    if missing_features:
        raise RuntimeError(
            "Turkish Stage 3C output is missing frozen Full features: "
            f"{missing_features}"
        )

    trial_ids = source_frame[trial_column].astype(str)
    if trial_ids.duplicated().any():
        duplicates = trial_ids[trial_ids.duplicated()].head(20).tolist()
        raise RuntimeError(
            f"Duplicate Stage 3C trial IDs: {duplicates}"
        )

    observed_trial_ids = set(trial_ids)
    unexpected_trial_ids = sorted(
        observed_trial_ids - expected_trial_ids
    )
    missing_trial_ids = sorted(
        expected_trial_ids - observed_trial_ids
    )
    failure_rate = (
        len(missing_trial_ids) / expected_count
        if expected_count
        else 1.0
    )

    if unexpected_trial_ids:
        raise RuntimeError(
            "Stage 3C contains trials outside the frozen primary manifest: "
            f"{unexpected_trial_ids[:20]}"
        )
    if failure_rate > args.maximum_feature_failure_rate:
        raise RuntimeError(
            "Full-41 feature extraction failure rate exceeds the locked "
            f"limit: {failure_rate:.3f}"
        )

    label_values = normalize_binary_label(
        source_frame[label_column],
        label_column,
    )
    participant_ids = (
        source_frame[participant_column].astype(str).to_numpy()
    )

    participant_rows: list[dict[str, Any]] = []
    included_participants: list[str] = []

    for participant_id in pd.unique(participant_ids):
        indices = np.flatnonzero(
            participant_ids == participant_id
        )
        participant_labels = label_values[indices]
        class_counts = {
            int(label): int(np.sum(participant_labels == label))
            for label in sorted(np.unique(participant_labels))
        }

        reasons: list[str] = []
        if len(indices) < 2:
            reasons.append("fewer_than_two_trials")
        if set(class_counts) != {0, 1}:
            reasons.append("missing_one_task_class")

        included = not reasons
        if included:
            included_participants.append(participant_id)

        participant_rows.append(
            {
                "participant_id": participant_id,
                "n_trials": int(len(indices)),
                "n_is": int(class_counts.get(0, 0)),
                "n_nr": int(class_counts.get(1, 0)),
                "included": included,
                "exclusion_reason": "|".join(reasons),
            }
        )

    participant_audit = pd.DataFrame(participant_rows)
    participant_audit.to_csv(
        output_dir / "audit" / "inclusion_exclusion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    include_mask = np.isin(
        participant_ids,
        np.asarray(included_participants, dtype=str),
    )
    frame = source_frame.loc[include_mask].reset_index(drop=True)
    labels = label_values[include_mask]
    participants = (
        frame[participant_column].astype(str).to_numpy()
    )
    trials = frame[trial_column].astype(str).to_numpy()

    if pd.unique(participants).size < 40:
        raise RuntimeError(
            "Fewer than 40 eligible participants remain after the "
            "locked participant-level checks."
        )
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError(
            "Both predefined classes are not present after eligibility checks."
        )

    raw_score, content_contribution, quality_contribution = (
        exact_score_components(
            pipeline,
            frame,
            feature_columns,
        )
    )

    full_reference = source_references[
        "full_final_reference"
    ]
    quality_reference = source_references[
        "quality_contribution_reference"
    ]
    qcn_participant_reference = source_references[
        "quality_removed_final_reference"
    ]

    raw_centers = participant_loo_median(
        raw_score,
        participants,
    )
    participant_calibrated_score = (
        raw_score - raw_centers + full_reference
    )

    qcn_score = (
        raw_score
        - quality_contribution
        + quality_reference
    )
    qcn_centers = participant_loo_median(
        qcn_score,
        participants,
    )
    qcn_pc_score = (
        qcn_score
        - qcn_centers
        + qcn_participant_reference
    )

    unlabeled_manifest = pd.DataFrame(
        {
            "participant_id": participants,
            "external_trial_id": trials,
        }
    )
    unlabeled_manifest.to_csv(
        output_dir
        / "manifests"
        / "unlabeled_target_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    unlabeled_features = pd.concat(
        [
            unlabeled_manifest,
            frame[feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    unlabeled_features.to_parquet(
        output_dir
        / "features"
        / "unlabeled_full41_features.parquet",
        index=False,
    )

    predictions = pd.DataFrame(
        {
            "participant_id": participants,
            "external_trial_id": trials,
            "frozen_full41_score": raw_score,
            "participant_median_loo_score": (
                participant_calibrated_score
            ),
            "qcn_only_score": qcn_score,
            "qcn_pc_score": qcn_pc_score,
            "qcn_pc_confidence": np.abs(
                qcn_pc_score - frozen_threshold
            ),
            "frozen_full41_prediction": (
                raw_score >= frozen_threshold
            ).astype(int),
            "participant_median_loo_prediction": (
                participant_calibrated_score >= frozen_threshold
            ).astype(int),
            "qcn_only_prediction": (
                qcn_score >= frozen_threshold
            ).astype(int),
            "qcn_pc_prediction": (
                qcn_pc_score >= frozen_threshold
            ).astype(int),
        }
    )
    predictions.to_csv(
        output_dir
        / "predictions"
        / "unlabeled_frozen_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sealed_labels = pd.DataFrame(
        {
            "participant_id": participants,
            "external_trial_id": trials,
            "label_nr": labels,
        }
    )
    if task_column:
        sealed_labels["task"] = (
            frame[task_column].astype(str).to_numpy()
        )

    sealed_labels.to_csv(
        output_dir / "sealed" / "labels_sealed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    schema_audit = {
        "artifact_version": "iui_p0_tr_feature_schema_audit_v1",
        "language": "TR",
        "expected_primary_trials": expected_count,
        "observed_full41_trials_before_participant_filter": int(
            len(source_frame)
        ),
        "missing_feature_trials": len(missing_trial_ids),
        "unexpected_feature_trials": len(unexpected_trial_ids),
        "feature_extraction_failure_rate": failure_rate,
        "maximum_allowed_failure_rate": (
            args.maximum_feature_failure_rate
        ),
        "eligible_trials": int(len(frame)),
        "eligible_participants": int(
            pd.unique(participants).size
        ),
        "class_counts_used_only_for_eligibility": {
            "IS_0": int(np.sum(labels == 0)),
            "NR_1": int(np.sum(labels == 1)),
        },
        "frozen_feature_count": len(feature_columns),
        "frozen_feature_order": feature_columns,
        "quality_feature_group": QUALITY_FEATURES,
        "frozen_threshold": frozen_threshold,
        "source_references": source_references,
        "model_loaded": True,
        "predictive_scores_generated": True,
        "performance_metrics_calculated": False,
        "labels_joined_to_prediction_table": False,
        "missing_trial_ids_preview": missing_trial_ids[:50],
    }
    (
        output_dir
        / "audit"
        / "feature_schema_audit.json"
    ).write_text(
        json.dumps(
            schema_audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_files = [
        output_dir
        / "audit"
        / "inclusion_exclusion.csv",
        output_dir
        / "audit"
        / "feature_schema_audit.json",
        output_dir
        / "manifests"
        / "unlabeled_target_manifest.csv",
        output_dir
        / "features"
        / "unlabeled_full41_features.parquet",
        output_dir
        / "predictions"
        / "unlabeled_frozen_scores.csv",
        output_dir
        / "sealed"
        / "labels_sealed.csv",
    ]

    input_files = [
        model_path,
        factor_manifest_path,
        primary_manifest_path,
        source_features_path,
        stage3c_dir / "stage3c_manifest.json",
        protocol_snapshot / "CONFIRMATORY_PROTOCOL.md",
        protocol_snapshot / "iui_p0_confirmatory_config.json",
    ]
    input_files = [
        path
        for path in input_files
        if path.exists()
    ]

    hash_rows: list[dict[str, Any]] = []
    for role, paths in [
        ("input", input_files),
        ("output", output_files),
    ]:
        for path in paths:
            hash_rows.append(
                {
                    "role": role,
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "path": str(path),
                }
            )

    write_sha_manifest(output_dir, hash_rows)

    release_manifest = {
        "artifact_version": "iui_p0_confirmatory_TR_unlabeled_freeze_v1",
        "language": "TR",
        "development_language": "EN",
        "source_model": "OneStop Full 41",
        "frozen_model_path": str(model_path),
        "factor_manifest_path": str(
            factor_manifest_path
        ),
        "protocol_snapshot": str(protocol_snapshot),
        "stage2_dir": str(stage2_dir),
        "stage3a_dir": str(stage3a_dir),
        "stage3c_dir": str(stage3c_dir),
        "primary_manifest_path": str(primary_manifest_path),
        "eligible_trials": int(len(frame)),
        "eligible_participants": int(
            pd.unique(participants).size
        ),
        "frozen_threshold": frozen_threshold,
        "source_references": source_references,
        "feature_columns": feature_columns,
        "quality_features": QUALITY_FEATURES,
        "score_methods": [
            "frozen_full41",
            "participant_median_loo",
            "qcn_only",
            "qcn_pc",
        ],
        "performance_calculated": False,
        "labels_joined_to_predictions": False,
        "sealed_label_path": str(
            output_dir / "sealed" / "labels_sealed.csv"
        ),
        "hash_manifest": str(
            output_dir / "file_hash_manifest.csv"
        ),
    }
    (
        output_dir / "release_manifest.json"
    ).write_text(
        json.dumps(
            release_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Hash the release manifest after it has been written.
    with (
        output_dir / "file_hash_manifest.csv"
    ).open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        release_path = output_dir / "release_manifest.json"
        writer.writerow(
            [
                "output",
                sha256(release_path),
                release_path.stat().st_size,
                str(release_path),
            ]
        )

    summary = f"""# IUI P0 Turkish Unlabeled Freeze

- Language: **TR**
- Eligible participants: **{pd.unique(participants).size}**
- Eligible trials: **{len(frame)}**
- Full-41 extraction failure rate: **{failure_rate:.4f}**
- Frozen model applied: **Yes**
- QCN-PC scores generated: **Yes**
- Predictive performance calculated: **No**
- Labels joined to prediction table: **No**

## Frozen artifacts

- `manifests/unlabeled_target_manifest.csv`
- `features/unlabeled_full41_features.parquet`
- `predictions/unlabeled_frozen_scores.csv`
- `sealed/labels_sealed.csv`
- `file_hash_manifest.csv`
- `SHA256SUMS`
- `release_manifest.json`

Do not open or join `sealed/labels_sealed.csv` until the complete output directory has
been copied to independent storage and the verification script reports PASS.
"""
    (
        output_dir
        / "text"
        / "UNLABELED_FREEZE_SUMMARY.md"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print("")
    print("Turkish unlabeled feature/score freeze completed.")
    print(f"Output: {output_dir}")
    print(f"Primary manifest: {primary_manifest_path}")
    print(
        f"Eligible participants: "
        f"{pd.unique(participants).size}"
    )
    print(f"Eligible trials: {len(frame)}")
    print(
        f"Feature extraction failure rate: "
        f"{failure_rate:.4f}"
    )
    print("Performance metrics calculated: False")
    print("Labels joined to predictions: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
