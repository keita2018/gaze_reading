#!/usr/bin/env python
"""Evaluate Full-41 Event-Free AOI + LinearSVM under ACL-2025-compatible splits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC



def _find_feature_list(obj):
    """Recursively recover the exact frozen Full-41 feature schema."""
    if isinstance(obj, dict):
        for key in (
            "feature_columns",
            "feature_names",
            "features",
            "input_features",
            "selected_features",
        ):
            if key in obj and isinstance(obj[key], (list, tuple)):
                values = [str(value) for value in obj[key]]
                if (
                    len(values) == 41
                    and all(value.startswith("soft_aoi_") for value in values)
                ):
                    return values
        for value in obj.values():
            found = _find_feature_list(value)
            if found is not None:
                return found

    if hasattr(obj, "feature_names_in_"):
        values = [str(value) for value in obj.feature_names_in_]
        if (
            len(values) == 41
            and all(value.startswith("soft_aoi_") for value in values)
        ):
            return values

    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_feature_list(value)
            if found is not None:
                return found
    return None


def resolve_full41_features(
    frame: pd.DataFrame,
    frozen_model_path: Path,
) -> list[str]:
    """Use the already-frozen OneStop model as the source of truth.

    The quality-view table contains additional soft_aoi_* columns, so a prefix
    match is insufficient. The frozen Full-41 artifact fixes the exact schema
    and order used in the main external-validation pipeline.
    """
    if not frozen_model_path.is_file():
        raise FileNotFoundError(
            "Frozen OneStop Full-41 model was not found: "
            f"{frozen_model_path}\\n"
            "Expected the previously frozen artifact at "
            "outputs/frozen_onestop_external_validation_v1/"
            "models/soft_full_all.joblib"
        )

    artifact = joblib.load(frozen_model_path)
    feature_columns = _find_feature_list(artifact)
    if feature_columns is None:
        raise RuntimeError(
            "Could not recover an exact 41-feature soft_aoi_* schema from "
            f"{frozen_model_path}"
        )

    missing = [
        column for column in feature_columns
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            "The OneStop quality-view table is missing frozen Full-41 "
            f"features: {missing}"
        )

    if len(set(feature_columns)) != 41:
        raise RuntimeError("Frozen Full-41 schema contains duplicate columns.")

    return feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data/processed/onestop/quality_views_asc344_v1/"
            "onestop_quality_views_all_features.parquet"
        ),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "outputs/onestop_acl2025_compatible_splits_v1/"
            "split_assignments.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/onestop_acl2025_compatible_linear_v1"
        ),
    )
    parser.add_argument(
        "--frozen-model",
        type=Path,
        default=Path(
            "outputs/frozen_onestop_external_validation_v1/"
            "models/soft_full_all.joblib"
        ),
    )
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1,10",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, score)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def metrics(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, float | int]:
    pred = (score >= threshold).astype(int)
    return {
        "n_trials": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc_nr": pr_auc(y, score),
        "average_precision_nr": float(average_precision_score(y, score)),
        "nr_prediction_rate": float(pred.mean()),
    }


def build_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=float(c_value),
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=20000,
                ),
            ),
        ]
    )


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features)
    if "view_condition" in features.columns:
        features = features[features["view_condition"].astype(str).eq("clean")].copy()
    if "onestop_trial_id" in features.columns:
        trial_column = "onestop_trial_id"
    elif "trial_id" in features.columns:
        trial_column = "trial_id"
    else:
        raise RuntimeError("Feature table lacks trial ID.")
    features[trial_column] = features[trial_column].astype(str).str.strip()
    features["participant_id"] = features["participant_id"].astype(str).str.strip()
    features["label"] = pd.to_numeric(features["label"], errors="raise").astype(int)

    frozen_model_path = args.frozen_model.resolve()
    feature_columns = resolve_full41_features(
        features,
        frozen_model_path,
    )
    print(
        "Frozen Full-41 schema recovered: "
        f"{len(feature_columns)} features",
        flush=True,
    )
    print(
        f"Frozen model: {frozen_model_path}",
        flush=True,
    )

    splits = pd.read_parquet(args.splits)
    splits["trial_id"] = splits["trial_id"].astype(str).str.strip()

    base = features[
        [trial_column, "participant_id", "label", *feature_columns]
    ].rename(columns={trial_column: "trial_id"})
    if base["trial_id"].duplicated().any():
        raise RuntimeError("Clean feature table has duplicate trial IDs.")

    c_grid = [float(value) for value in args.c_grid.split(",")]
    split_rows = []
    selection_rows = []
    prediction_parts = []

    for fold in range(10):
        assignment = splits[splits["fold"].eq(fold)][
            ["trial_id", "split"]
        ]
        frame = base.merge(
            assignment,
            on="trial_id",
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != len(base):
            raise RuntimeError(
                f"Fold {fold}: feature/split merge lost rows "
                f"({len(frame)} != {len(base)})."
            )
        train = frame[frame["split"].eq("train")]
        validation = frame[frame["split"].eq("validation")]

        best = None
        for c_value in c_grid:
            model = build_model(c_value, args.seed + fold)
            model.fit(
                train[feature_columns].to_numpy(float),
                train["label"].to_numpy(int),
            )
            score = model.decision_function(
                validation[feature_columns].to_numpy(float)
            )
            result = metrics(validation["label"].to_numpy(int), score, 0.0)
            row = {
                "fold": fold,
                "C": c_value,
                **{f"validation_{key}": value for key, value in result.items()},
            }
            selection_rows.append(row)
            key = (
                result["accuracy"],
                result["macro_f1"],
                result["roc_auc"],
                -c_value,
            )
            if best is None or key > best[0]:
                best = (key, c_value)

        assert best is not None
        selected_c = float(best[1])
        model = build_model(selected_c, args.seed + 1000 + fold)
        model.fit(
            train[feature_columns].to_numpy(float),
            train["label"].to_numpy(int),
        )
        joblib.dump(
            {
                "fold": fold,
                "C": selected_c,
                "feature_columns": feature_columns,
                "estimator": model,
            },
            args.output_dir / f"linear_svm_fold_{fold:02d}.joblib",
        )

        regime_frames = {}
        for regime in (
            "test_new_item",
            "test_new_participant",
            "test_new_item_participant",
        ):
            test = frame[frame["split"].eq(regime)].copy()
            score = model.decision_function(
                test[feature_columns].to_numpy(float)
            )
            result = metrics(test["label"].to_numpy(int), score, 0.0)
            split_rows.append(
                {
                    "fold": fold,
                    "regime": regime,
                    "selected_C": selected_c,
                    **result,
                }
            )
            pred = test[["trial_id", "participant_id", "label"]].copy()
            pred["fold"] = fold
            pred["regime"] = regime
            pred["score"] = score
            pred["prediction"] = (score >= 0.0).astype(int)
            prediction_parts.append(pred)
            regime_frames[regime] = pred

        all_test = pd.concat(regime_frames.values(), ignore_index=True)
        all_result = metrics(
            all_test["label"].to_numpy(int),
            all_test["score"].to_numpy(float),
            0.0,
        )
        split_rows.append(
            {
                "fold": fold,
                "regime": "all",
                "selected_C": selected_c,
                **all_result,
            }
        )

    split_metrics = pd.DataFrame(split_rows)
    selections = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)

    summary_rows = []
    for regime, group in split_metrics.groupby("regime", sort=False):
        row = {"regime": regime, "n_splits": int(len(group))}
        for metric_name in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
            "average_precision_nr",
        ):
            mean, low, high = mean_ci(group[metric_name].to_numpy(float))
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    split_metrics.to_csv(
        args.output_dir / "split_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selections.to_csv(
        args.output_dir / "model_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_parquet(
        args.output_dir / "test_predictions.parquet",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "summary_mean_95ci.csv",
        index=False,
        encoding="utf-8-sig",
    )

    published = pd.DataFrame(
        [
            ["New Item", 0.624, 0.715, 0.704, 0.706, 0.646, 0.899],
            ["New Participant", 0.606, 0.610, 0.637, 0.644, 0.625, 0.909],
            ["New Item & Participant", 0.608, 0.615, 0.640, 0.644, 0.620, 0.882],
            ["All", 0.615, 0.659, 0.669, 0.673, 0.635, 0.903],
        ],
        columns=[
            "regime",
            "published_logistic_regression_accuracy",
            "published_beyelstm_no_text_accuracy",
            "published_convnext_accuracy",
            "published_vit_accuracy",
            "published_roberta_eye_w_accuracy",
            "published_roberta_eye_f_accuracy",
        ],
    )
    published.to_csv(
        args.output_dir / "published_acl2025_accuracy_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "artifact_version": "onestop_acl2025_compatible_linear_v1",
        "model": "Full-41 Event-Free AOI + LinearSVM",
        "subset": "ASC344 clean trials",
        "frozen_full41_schema_source": str(frozen_model_path),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "split_status": "protocol reconstruction, not exact author split assignment",
        "selection_metric": "validation accuracy",
        "decision_threshold": 0.0,
        "c_grid": c_grid,
        "reported_primary_for_compatibility": "accuracy",
        "additional_metrics": [
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
            "average_precision_nr",
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== FULL-41 LINEARSVM ACL-2025-COMPATIBLE SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
