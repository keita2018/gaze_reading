#!/usr/bin/env python
"""Diagnose frozen-model transport failure on WebQAmGaze.

This is a post-hoc diagnostic stage. It does not refit models, change labels,
select thresholds, or alter the primary external-validation results.

For each non-dummy frozen Common16 model, the script:
1. applies the frozen imputer and scaler;
2. decomposes the LinearSVC decision function into per-feature contributions;
3. verifies that contributions + intercept reproduce the saved predictions;
4. measures which transported features dominate the score;
5. measures NR-vs-IS separation for each contribution;
6. audits pooled and within-participant score polarity;
7. merges contribution strength with the Stage 4 transport-shift audit.

Outputs
-------
model_feature_contributions.csv
model_feature_contribution_summary.csv
model_score_polarity_audit.csv
participant_score_offset_audit.csv
transport_contribution_merged.csv
stage5_diagnostic_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


MODEL_ORDER = [
    "rawstat_all",
    "soft_full_all",
    "soft_compact_distribution_membership",
    "official_sent_gaze_sacc",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_model_artifact(frozen_dir: Path, feature_set: str) -> Path:
    expected = frozen_dir / "models" / f"{feature_set}.joblib"
    if expected.is_file():
        return expected
    matches = list(frozen_dir.rglob(f"{feature_set}.joblib"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {feature_set}.joblib under {frozen_dir}"
        )
    raise ValueError(
        f"Multiple artifacts found for {feature_set}: {matches}"
    )


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        return math.nan
    return float(roc_auc_score(y, score))


def load_aligned_table(
    path: Path,
    master_ids: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "external_trial_id" not in frame.columns:
        raise ValueError(f"{path} lacks external_trial_id.")
    frame["external_trial_id"] = frame[
        "external_trial_id"
    ].astype(str)
    if frame["external_trial_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate trial IDs.")
    indexed = frame.set_index("external_trial_id")
    missing = sorted(set(master_ids) - set(indexed.index))
    extra = sorted(set(indexed.index) - set(master_ids))
    if missing or extra:
        raise ValueError(
            f"{path} trial-ID mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return indexed.loc[master_ids].reset_index()


def transformed_feature_names(
    original_names: list[str],
    imputer: Any,
) -> list[tuple[str, str, str | None]]:
    """Return (output_name, kind, source_feature)."""
    names: list[tuple[str, str, str | None]] = [
        (name, "original", name)
        for name in original_names
    ]
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None:
        for index in np.asarray(
            indicator.features_,
            dtype=int,
        ).tolist():
            source = original_names[index]
            names.append(
                (
                    f"missing_indicator__{source}",
                    "missing_indicator",
                    source,
                )
            )
    return names


def decompose_model(
    *,
    feature_set: str,
    artifact: dict[str, Any],
    table: pd.DataFrame,
    trials: pd.DataFrame,
    saved_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = list(artifact["feature_columns"])
    missing = sorted(set(feature_columns) - set(table.columns))
    if missing:
        raise ValueError(
            f"{feature_set} input lacks columns: {missing}"
        )

    x = table[feature_columns].copy()
    for column in feature_columns:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )
    if np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError(f"{feature_set} input contains infinity.")

    pipeline = artifact["pipeline"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    x_imputed = np.asarray(
        imputer.transform(x),
        dtype=float,
    )
    x_scaled = np.asarray(
        scaler.transform(x_imputed),
        dtype=float,
    )
    coefficients = np.asarray(
        model.coef_,
        dtype=float,
    ).reshape(-1)
    intercept = float(
        np.asarray(model.intercept_, dtype=float).reshape(-1)[0]
    )

    names = transformed_feature_names(
        feature_columns,
        imputer,
    )
    if len(names) != x_scaled.shape[1]:
        raise RuntimeError(
            f"{feature_set}: transformed feature-name mismatch: "
            f"{len(names)} names vs {x_scaled.shape[1]} columns."
        )
    if len(coefficients) != x_scaled.shape[1]:
        raise RuntimeError(
            f"{feature_set}: coefficient-length mismatch."
        )

    contributions = x_scaled * coefficients[None, :]
    reconstructed_score = contributions.sum(axis=1) + intercept

    prediction_group = saved_predictions[
        saved_predictions["feature_set"].eq(feature_set)
    ].copy()
    prediction_group["external_trial_id"] = prediction_group[
        "external_trial_id"
    ].astype(str)
    prediction_group = prediction_group.set_index(
        "external_trial_id"
    ).loc[
        trials["external_trial_id"].astype(str)
    ].reset_index()
    saved_score = prediction_group[
        "decision_score_nr"
    ].to_numpy(dtype=float)

    max_error = float(
        np.max(np.abs(reconstructed_score - saved_score))
    )
    if max_error > 1e-8:
        raise RuntimeError(
            f"{feature_set}: contribution reconstruction error "
            f"{max_error:.3g} exceeds tolerance."
        )

    y = trials["label_nr"].to_numpy(dtype=int)
    participant_ids = trials["participant_id"].astype(str).to_numpy()
    trial_ids = trials["external_trial_id"].astype(str).to_numpy()

    long_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for index, (output_name, kind, source_feature) in enumerate(names):
        z = x_scaled[:, index]
        contribution = contributions[:, index]
        coefficient = float(coefficients[index])

        original_raw = (
            pd.to_numeric(
                x[source_feature],
                errors="coerce",
            ).to_numpy(dtype=float)
            if kind == "original"
            else np.full(len(x), np.nan, dtype=float)
        )

        frame = pd.DataFrame(
            {
                "feature_set": feature_set,
                "external_trial_id": trial_ids,
                "participant_id": participant_ids,
                "label_nr": y,
                "transformed_feature_index": index,
                "transformed_feature_name": output_name,
                "feature_kind": kind,
                "source_feature_name": source_feature,
                "coefficient": coefficient,
                "raw_external_value": original_raw,
                "standardized_value": z,
                "decision_contribution": contribution,
            }
        )
        long_rows.append(frame)

        nr = y == 1
        is_task = y == 0
        auc_z = safe_auc(y, z)
        auc_contribution = safe_auc(y, contribution)

        summary_rows.append(
            {
                "feature_set": feature_set,
                "transformed_feature_index": index,
                "transformed_feature_name": output_name,
                "feature_kind": kind,
                "source_feature_name": source_feature,
                "coefficient": coefficient,
                "coefficient_abs": abs(coefficient),
                "external_standardized_mean": float(
                    np.mean(z)
                ),
                "external_standardized_median": float(
                    np.median(z)
                ),
                "external_standardized_std": float(
                    np.std(z, ddof=1)
                ),
                "mean_contribution": float(
                    np.mean(contribution)
                ),
                "median_contribution": float(
                    np.median(contribution)
                ),
                "mean_abs_contribution": float(
                    np.mean(np.abs(contribution))
                ),
                "median_abs_contribution": float(
                    np.median(np.abs(contribution))
                ),
                "nr_mean_contribution": float(
                    np.mean(contribution[nr])
                ),
                "is_mean_contribution": float(
                    np.mean(contribution[is_task])
                ),
                "nr_minus_is_mean_contribution": float(
                    np.mean(contribution[nr])
                    - np.mean(contribution[is_task])
                ),
                "auc_nr_standardized_feature": auc_z,
                "auc_nr_decision_contribution": auc_contribution,
                "external_raw_missing_rate": (
                    float(np.mean(~np.isfinite(original_raw)))
                    if kind == "original"
                    else math.nan
                ),
                "reconstruction_max_abs_error": max_error,
                "model_intercept": intercept,
            }
        )

    participant_rows: list[dict[str, Any]] = []
    scores = reconstructed_score
    for participant_id in sorted(set(participant_ids)):
        mask = participant_ids == participant_id
        participant_y = y[mask]
        participant_scores = scores[mask]
        nr_mask = participant_y == 1
        is_mask = participant_y == 0
        participant_rows.append(
            {
                "feature_set": feature_set,
                "participant_id": participant_id,
                "n_trials": int(np.sum(mask)),
                "n_nr": int(np.sum(nr_mask)),
                "n_is": int(np.sum(is_mask)),
                "mean_decision_score": float(
                    np.mean(participant_scores)
                ),
                "median_decision_score": float(
                    np.median(participant_scores)
                ),
                "nr_mean_decision_score": (
                    float(np.mean(participant_scores[nr_mask]))
                    if np.any(nr_mask)
                    else math.nan
                ),
                "is_mean_decision_score": (
                    float(np.mean(participant_scores[is_mask]))
                    if np.any(is_mask)
                    else math.nan
                ),
                "nr_minus_is_mean_score": (
                    float(
                        np.mean(participant_scores[nr_mask])
                        - np.mean(participant_scores[is_mask])
                    )
                    if np.any(nr_mask) and np.any(is_mask)
                    else math.nan
                ),
                "within_participant_auc": safe_auc(
                    participant_y,
                    participant_scores,
                ),
                "within_participant_reversed_auc": safe_auc(
                    participant_y,
                    -participant_scores,
                ),
            }
        )

    polarity = pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "n_trials": len(y),
                "pooled_auc": safe_auc(y, scores),
                "pooled_reversed_auc": safe_auc(y, -scores),
                "pooled_nr_mean_score": float(
                    np.mean(scores[y == 1])
                ),
                "pooled_is_mean_score": float(
                    np.mean(scores[y == 0])
                ),
                "pooled_nr_minus_is_mean_score": float(
                    np.mean(scores[y == 1])
                    - np.mean(scores[y == 0])
                ),
                "participant_mean_auc": float(
                    pd.DataFrame(participant_rows)[
                        "within_participant_auc"
                    ].mean()
                ),
                "participant_mean_reversed_auc": float(
                    pd.DataFrame(participant_rows)[
                        "within_participant_reversed_auc"
                    ].mean()
                ),
                "participant_score_offset_std": float(
                    pd.DataFrame(participant_rows)[
                        "mean_decision_score"
                    ].std(ddof=1)
                ),
                "reconstruction_max_abs_error": max_error,
            }
        ]
    )

    return (
        pd.concat(long_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(participant_rows),
        polarity,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage2"
        ),
    )
    parser.add_argument(
        "--stage3a-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage3a"
        ),
    )
    parser.add_argument(
        "--stage3c-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage3c"
        ),
    )
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage4"
        ),
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path(
            "outputs/frozen_zuco_external_validation_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage5_diagnostics"
        ),
    )
    args = parser.parse_args()

    stage2_dir = args.stage2_dir.resolve()
    stage3a_dir = args.stage3a_dir.resolve()
    stage3c_dir = args.stage3c_dir.resolve()
    stage4_dir = args.stage4_dir.resolve()
    frozen_dir = args.frozen_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "trials": require_file(
            stage2_dir
            / "trial_manifest_primary_candidates.csv"
        ),
        "rawstat": require_file(
            stage3a_dir / "rawstat_features_primary.csv"
        ),
        "soft_full": require_file(
            stage3c_dir / "soft_full_features_primary.csv"
        ),
        "soft_compact": require_file(
            stage3c_dir / "soft_compact_features_primary.csv"
        ),
        "official": require_file(
            stage4_dir
            / "corrected_official_features_primary.csv"
        ),
        "predictions": require_file(
            stage4_dir / "external_predictions.csv"
        ),
        "transport_audit": require_file(
            stage4_dir / "feature_transport_audit.csv"
        ),
    }

    trials = pd.read_csv(input_paths["trials"])
    trials["external_trial_id"] = trials[
        "external_trial_id"
    ].astype(str)
    trials["participant_id"] = trials[
        "participant_id"
    ].astype(str)
    trials["label_nr"] = pd.to_numeric(
        trials["label_nr"],
        errors="raise",
    ).astype(int)
    trials = trials.sort_values(
        ["participant_id", "trial_index"]
    ).reset_index(drop=True)
    master_ids = trials["external_trial_id"].tolist()

    predictions = pd.read_csv(input_paths["predictions"])
    predictions["external_trial_id"] = predictions[
        "external_trial_id"
    ].astype(str)

    table_paths = {
        "rawstat_all": input_paths["rawstat"],
        "soft_full_all": input_paths["soft_full"],
        "soft_compact_distribution_membership": (
            input_paths["soft_compact"]
        ),
        "official_sent_gaze_sacc": input_paths["official"],
    }
    tables = {
        feature_set: load_aligned_table(path, master_ids)
        for feature_set, path in table_paths.items()
    }

    artifact_paths = {
        feature_set: find_model_artifact(
            frozen_dir,
            feature_set,
        )
        for feature_set in MODEL_ORDER
    }

    contribution_frames = []
    summary_frames = []
    participant_frames = []
    polarity_frames = []

    for feature_set in MODEL_ORDER:
        artifact = joblib.load(artifact_paths[feature_set])
        (
            contribution_long,
            contribution_summary,
            participant_offsets,
            polarity,
        ) = decompose_model(
            feature_set=feature_set,
            artifact=artifact,
            table=tables[feature_set],
            trials=trials,
            saved_predictions=predictions,
        )
        contribution_frames.append(contribution_long)
        summary_frames.append(contribution_summary)
        participant_frames.append(participant_offsets)
        polarity_frames.append(polarity)

    contributions = pd.concat(
        contribution_frames,
        ignore_index=True,
    )
    summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )
    participant_offsets = pd.concat(
        participant_frames,
        ignore_index=True,
    )
    polarity = pd.concat(
        polarity_frames,
        ignore_index=True,
    )

    transport = pd.read_csv(input_paths["transport_audit"])
    merged = summary.merge(
        transport,
        left_on=[
            "feature_set",
            "source_feature_name",
        ],
        right_on=[
            "feature_set",
            "feature_name",
        ],
        how="left",
        validate="many_to_one",
    )
    merged["abs_transport_shift"] = (
        pd.to_numeric(
            merged["external_median_standardized_by_zuco"],
            errors="coerce",
        ).abs()
    )
    merged["shift_weighted_by_coefficient"] = (
        pd.to_numeric(
            merged["external_median_standardized_by_zuco"],
            errors="coerce",
        )
        * merged["coefficient"]
    )

    output_paths = {
        "contributions": (
            output_dir / "model_feature_contributions.csv"
        ),
        "summary": (
            output_dir
            / "model_feature_contribution_summary.csv"
        ),
        "polarity": (
            output_dir / "model_score_polarity_audit.csv"
        ),
        "participant_offsets": (
            output_dir
            / "participant_score_offset_audit.csv"
        ),
        "merged": (
            output_dir / "transport_contribution_merged.csv"
        ),
        "manifest": (
            output_dir / "stage5_diagnostic_manifest.json"
        ),
    }

    contributions.to_csv(
        output_paths["contributions"],
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_paths["summary"],
        index=False,
        encoding="utf-8-sig",
    )
    polarity.to_csv(
        output_paths["polarity"],
        index=False,
        encoding="utf-8-sig",
    )
    participant_offsets.to_csv(
        output_paths["participant_offsets"],
        index=False,
        encoding="utf-8-sig",
    )
    merged.to_csv(
        output_paths["merged"],
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "stage": "webqamgaze_external_validation_stage5_diagnostics",
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "post_hoc_only": True,
        "primary_stage4_results_modified": False,
        "n_trials": len(trials),
        "n_participants": int(
            trials["participant_id"].nunique()
        ),
        "models": MODEL_ORDER,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "artifacts": {
            feature_set: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for feature_set, path in artifact_paths.items()
        },
        "outputs": {
            name: str(path)
            for name, path in output_paths.items()
            if name != "manifest"
        },
    }
    output_paths["manifest"].write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE STAGE 5 TRANSPORT DIAGNOSTICS ===")
    print("Score polarity audit:")
    print(
        polarity.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    for feature_set in MODEL_ORDER:
        subset = summary[
            summary["feature_set"].eq(feature_set)
        ].sort_values(
            "mean_abs_contribution",
            ascending=False,
        )
        print(
            f"=== TOP SCORE CONTRIBUTIONS: {feature_set} ==="
        )
        print(
            subset[
                [
                    "transformed_feature_name",
                    "coefficient",
                    "external_standardized_median",
                    "mean_contribution",
                    "mean_abs_contribution",
                    "nr_minus_is_mean_contribution",
                    "auc_nr_decision_contribution",
                ]
            ]
            .head(15)
            .to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )

    print("Written:")
    for path in output_paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
