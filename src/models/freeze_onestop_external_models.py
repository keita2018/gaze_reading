#!/usr/bin/env python
r"""
Freeze OneStop Soft-AOI models for external zero-shot validation.

The script selects hyperparameters and a decision threshold using only
participant-grouped out-of-fold predictions from OneStop, then fits the final
pipeline on all OneStop trials. No external target data are read.

Primary frozen models
---------------------
- soft_compact_distribution_membership: frozen Compact27 order
- soft_full_all: exact 41-feature Soft-AOI representation

Example
-------
python -u src\models\freeze_onestop_external_models.py ^
  --features data\processed\onestop\soft_aoi_asc344\onestop_soft_aoi_all_features.parquet ^
  --zuco-schema outputs\frozen_zuco_external_validation_v1\feature_schema.csv ^
  --output-dir outputs\frozen_onestop_external_validation_v1 ^
  --cv-splits 5 --c-grid 0.001,0.01,0.1,1,10 ^
  --threshold-grid-size 1001 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

try:
    import extract_zuco_soft_aoi_features as soft
except ImportError as exc:
    raise ImportError(
        "Could not import src/data/extract_zuco_soft_aoi_features.py."
    ) from exc


FULL41 = list(
    dict.fromkeys(
        list(soft.CONTENT_FEATURES)
        + list(soft.MEMBERSHIP_QUALITY_FEATURES)
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--zuco-schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--c-grid",
        default="0.001,0.01,0.1,1,10",
    )
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=1001,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_c_grid(text: str) -> list[float]:
    values = sorted(
        {
            float(token.strip())
            for token in text.split(",")
            if token.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("C grid must contain positive values.")
    return values


def load_compact27(path: Path) -> list[str]:
    schema = pd.read_csv(require_file(path), low_memory=False)
    required = {"feature_set", "feature_name"}
    missing = sorted(required - set(schema.columns))
    if missing:
        raise ValueError(f"Schema lacks columns: {missing}")

    rows = schema.loc[
        schema["feature_set"].astype(str).eq(
            "soft_compact_distribution_membership"
        ),
        "feature_name",
    ].astype(str).tolist()
    features = list(dict.fromkeys(rows))
    if len(features) != 27:
        raise ValueError(
            f"Compact schema resolves to {len(features)} features, expected 27."
        )
    return features


def make_pipeline(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    dual=False,
                    max_iter=100000,
                    random_state=seed,
                ),
            ),
        ]
    )


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc_nr": float(
            average_precision_score(y_true, scores)
        ),
        "predicted_nr_rate": float(np.mean(predictions == 1)),
    }


def threshold_candidates(
    scores: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("No finite decision scores.")
    candidates = np.quantile(
        finite,
        np.linspace(0.0, 1.0, max(3, grid_size)),
    )
    return np.unique(
        np.concatenate(
            [
                candidates,
                np.asarray([0.0]),
                np.asarray(
                    [np.nextafter(finite.min(), -np.inf)]
                ),
                np.asarray(
                    [np.nextafter(finite.max(), np.inf)]
                ),
            ]
        )
    )


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    grid_size: int,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.0
    best_metrics: dict[str, float] | None = None
    best_key = (-np.inf, -np.inf, -np.inf)

    for threshold in threshold_candidates(scores, grid_size):
        predictions = (scores >= threshold).astype(int)
        metrics = compute_metrics(y_true, scores, predictions)
        key = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            -abs(float(threshold)),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def grouped_oof_scores(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    c_value: float,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    scores = np.full(len(y), np.nan, dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(x, y, groups),
        start=1,
    ):
        if set(groups[train_idx]).intersection(groups[valid_idx]):
            raise RuntimeError(f"Participant leakage in fold {fold}.")
        model = make_pipeline(c_value, seed + fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x.iloc[train_idx], y[train_idx])
        scores[valid_idx] = model.decision_function(
            x.iloc[valid_idx]
        )

    if np.isnan(scores).any():
        raise RuntimeError("OOF score vector contains missing values.")
    return scores


def select_model(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    c_grid: list[float],
    n_splits: int,
    threshold_grid_size: int,
    seed: int,
) -> tuple[float, float, pd.DataFrame, np.ndarray]:
    rows = []
    score_by_c: dict[float, np.ndarray] = {}

    for c_value in c_grid:
        scores = grouped_oof_scores(
            x=x,
            y=y,
            groups=groups,
            c_value=c_value,
            n_splits=n_splits,
            seed=seed,
        )
        threshold, metrics = select_threshold(
            y, scores, threshold_grid_size
        )
        rows.append(
            {
                "C": c_value,
                "decision_threshold": threshold,
                **metrics,
            }
        )
        score_by_c[c_value] = scores
        print(
            f"    C={c_value:g}: "
            f"threshold={threshold:.6f}, "
            f"macroF1={metrics['macro_f1']:.4f}, "
            f"BA={metrics['balanced_accuracy']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC(NR)={metrics['pr_auc_nr']:.4f}"
        )

    table = pd.DataFrame(rows).sort_values(
        [
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "C",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    best = table.iloc[0]
    best_c = float(best["C"])
    best_threshold = float(best["decision_threshold"])
    return (
        best_c,
        best_threshold,
        table,
        score_by_c[best_c],
    )


def validate_participant_labels(frame: pd.DataFrame) -> None:
    counts = (
        frame.groupby("participant_id", observed=True)["label"]
        .nunique()
    )
    if counts.gt(1).any():
        bad = counts[counts.gt(1)].index.tolist()
        raise ValueError(
            f"Participants with multiple labels: {bad[:20]}"
        )


def main() -> int:
    args = parse_args()
    if args.cv_splits < 2:
        raise ValueError("--cv-splits must be at least 2.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    compact27 = load_compact27(args.zuco_schema)
    if len(FULL41) != 41:
        raise ValueError(
            f"Exact extractor exposes {len(FULL41)} full features, expected 41."
        )
    feature_sets = {
        "soft_compact_distribution_membership": compact27,
        "soft_full_all": FULL41,
    }

    frame = pd.read_parquet(require_file(args.features))
    required = {
        "onestop_trial_id",
        "participant_id",
        "label",
        "label_name",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature table lacks columns: {missing}")

    frame["participant_id"] = (
        frame["participant_id"].astype(str).str.strip().str.lower()
    )
    frame["label"] = pd.to_numeric(
        frame["label"], errors="raise"
    ).astype(int)
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Expected OneStop labels {0, 1}.")
    if frame["onestop_trial_id"].duplicated().any():
        raise ValueError("Duplicate OneStop trial IDs.")
    validate_participant_labels(frame)

    c_grid = parse_c_grid(args.c_grid)
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["participant_id"].to_numpy(dtype=object)

    print("=== FREEZE ONESTOP EXTERNAL MODELS ===")
    print(f"trials: {len(frame):,}")
    print(f"participants: {frame['participant_id'].nunique()}")
    print(
        "labels: "
        + json.dumps(
            frame["label_name"].value_counts().to_dict()
        )
    )
    print(f"C grid: {c_grid}")

    selection_parts = []
    oof_parts = []
    manifest_models = []

    for offset, (feature_set, columns) in enumerate(
        feature_sets.items(),
        start=1,
    ):
        missing_features = sorted(set(columns) - set(frame.columns))
        if missing_features:
            raise ValueError(
                f"{feature_set} missing features: {missing_features}"
            )

        print(
            f"\n=== {feature_set} ({len(columns)} features) ==="
        )
        x = frame[columns].apply(pd.to_numeric, errors="coerce")
        (
            selected_c,
            selected_threshold,
            selection_table,
            selected_oof_scores,
        ) = select_model(
            x=x,
            y=y,
            groups=groups,
            c_grid=c_grid,
            n_splits=args.cv_splits,
            threshold_grid_size=args.threshold_grid_size,
            seed=args.seed + offset * 100,
        )
        selection_table.insert(0, "feature_set", feature_set)
        selection_parts.append(selection_table)

        selected_predictions = (
            selected_oof_scores >= selected_threshold
        ).astype(int)
        oof_parts.append(
            pd.DataFrame(
                {
                    "onestop_trial_id": frame[
                        "onestop_trial_id"
                    ].astype(str).to_numpy(),
                    "participant_id": frame[
                        "participant_id"
                    ].astype(str).to_numpy(),
                    "label": y,
                    "label_name": frame[
                        "label_name"
                    ].astype(str).to_numpy(),
                    "feature_set": feature_set,
                    "score": selected_oof_scores,
                    "prediction": selected_predictions,
                    "selected_C": selected_c,
                    "decision_threshold": selected_threshold,
                }
            )
        )

        final_pipeline = make_pipeline(
            selected_c,
            args.seed + offset * 1000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            final_pipeline.fit(x, y)

        artifact = {
            "artifact_version": "onestop_external_validation_v1",
            "feature_set": feature_set,
            "training_scope": (
                "OneStop ASC344, 344 participants, 18,573 "
                "regular first-reading trials"
            ),
            "pipeline": final_pipeline,
            "feature_columns": columns,
            "training_feature_columns": columns,
            "label_to_int": {"IS": 0, "NR": 1},
            "int_to_label": {0: "IS", 1: "NR"},
            "decision_threshold": selected_threshold,
            "prediction_rule": (
                "predict NR (1) when decision_function >= "
                "decision_threshold; otherwise IS (0)"
            ),
            "c_value": selected_c,
            "seed": args.seed,
            "selection_protocol": (
                f"{args.cv_splits}-fold StratifiedGroupKFold by participant; "
                "C and threshold selected from OneStop OOF predictions only"
            ),
            "trained_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "primary_external_comparison_eligible": True,
        }
        model_path = model_dir / f"{feature_set}.joblib"
        joblib.dump(artifact, model_path, compress=3)

        best_row = selection_table.iloc[0].to_dict()
        manifest_models.append(
            {
                "feature_set": feature_set,
                "model_path": str(model_path),
                "n_features": len(columns),
                "selected_C": selected_c,
                "decision_threshold": selected_threshold,
                "oof_macro_f1": float(best_row["macro_f1"]),
                "oof_balanced_accuracy": float(
                    best_row["balanced_accuracy"]
                ),
                "oof_roc_auc": float(best_row["roc_auc"]),
                "oof_pr_auc_nr": float(best_row["pr_auc_nr"]),
                "sha256": sha256_file(model_path),
            }
        )

        print(
            f"  selected C={selected_c:g}, "
            f"threshold={selected_threshold:.6f}"
        )
        print(f"  saved: {model_path}")

    selection = pd.concat(selection_parts, ignore_index=True)
    oof = pd.concat(oof_parts, ignore_index=True)

    selection_path = (
        args.output_dir / "onestop_model_selection.csv"
    )
    oof_path = (
        args.output_dir / "onestop_selected_oof_predictions.parquet"
    )
    schema_path = args.output_dir / "feature_schema.csv"
    manifest_path = args.output_dir / "model_manifest.json"
    checksum_path = args.output_dir / "checksums_sha256.csv"

    selection.to_csv(
        selection_path, index=False, encoding="utf-8-sig"
    )
    oof.to_parquet(
        oof_path, index=False, compression="zstd"
    )

    schema_rows = []
    for feature_set, columns in feature_sets.items():
        for index, feature in enumerate(columns):
            schema_rows.append(
                {
                    "feature_set": feature_set,
                    "feature_index": index,
                    "feature_name": feature,
                }
            )
    pd.DataFrame(schema_rows).to_csv(
        schema_path, index=False, encoding="utf-8-sig"
    )

    model_manifest = {
        "artifact_version": "onestop_external_validation_v1",
        "training_dataset": "OneStop ASC344",
        "n_trials": int(len(frame)),
        "n_participants": int(
            frame["participant_id"].nunique()
        ),
        "label_mapping": {"IS": 0, "NR": 1},
        "selection_protocol": (
            "StratifiedGroupKFold grouped by participant; "
            "all model and threshold selection uses OneStop only"
        ),
        "external_target_data_used": False,
        "models": manifest_models,
        "outputs": {
            "selection_table": str(selection_path),
            "oof_predictions": str(oof_path),
            "feature_schema": str(schema_path),
        },
    }
    manifest_path.write_text(
        json.dumps(
            model_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    checksum_rows = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path != checksum_path:
            checksum_rows.append(
                {
                    "relative_path": path.relative_to(
                        args.output_dir
                    ).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(checksum_rows).to_csv(
        checksum_path, index=False, encoding="utf-8-sig"
    )

    print("\n=== FROZEN ONESTOP MODEL SUMMARY ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 220
    ):
        print(
            pd.DataFrame(manifest_models).to_string(index=False)
        )
    print(f"\nSaved: {selection_path}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {schema_path}")
    print(f"Saved: {manifest_path}")
    print(f"Saved: {checksum_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
