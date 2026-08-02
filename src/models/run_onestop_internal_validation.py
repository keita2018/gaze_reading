#!/usr/bin/env python
r"""
Nested participant-independent validation for OneStop Compact Soft-AOI features.

This is the first stage of the OneStop triangulation analysis. It evaluates
whether reading-condition information is reproducible across held-out OneStop
participants and compares the frozen Compact27 representation with simpler
behavior/quality subsets.

Important interpretation:
Each OneStop participant belongs to only one condition (NR or IS). Therefore,
participant-independent performance is a descriptive cohort-generalization
benchmark, not sufficient evidence that the model has isolated a within-person
task effect. ZuCo remains the within-participant identification dataset.

Example
-------
python -u src\models\run_onestop_internal_validation.py ^
  --features data\processed\onestop\soft_aoi_asc344\onestop_soft_aoi_all_features.parquet ^
  --frozen-schema outputs\frozen_zuco_external_validation_v1\feature_schema.csv ^
  --output-dir outputs\onestop_internal_validation_v1 ^
  --outer-splits 5 --inner-splits 5 --bootstrap 2000 --seed 42

Outputs
-------
- onestop_internal_fold_metrics.csv
- onestop_internal_pooled_metrics.csv
- onestop_internal_bootstrap_cis.csv
- onestop_internal_paired_deltas.csv
- onestop_internal_oof_predictions.parquet
- onestop_internal_selected_hyperparameters.csv
- onestop_internal_validation_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data/processed/onestop/soft_aoi_asc344/"
            "onestop_soft_aoi_all_features.parquet"
        ),
    )
    parser.add_argument(
        "--frozen-schema",
        type=Path,
        default=Path(
            "outputs/frozen_zuco_external_validation_v1/feature_schema.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/onestop_internal_validation_v1"),
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument(
        "--c-grid",
        default="0.001,0.01,0.1,1,10",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=401,
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_compact_schema(path: Path) -> list[str]:
    schema = pd.read_csv(require_file(path))
    required = {"feature_set", "feature_name"}
    missing = required - set(schema.columns)
    if missing:
        raise ValueError(
            f"Frozen schema lacks columns: {sorted(missing)}"
        )

    rows = schema.loc[
        schema["feature_set"].astype(str).eq(
            "soft_compact_distribution_membership"
        ),
        "feature_name",
    ].astype(str).tolist()
    features = list(dict.fromkeys(rows))
    if len(features) != 27:
        raise ValueError(
            f"Compact schema resolves to {len(features)} unique features, "
            "expected 27."
        )
    return features


def parse_c_grid(text: str) -> list[float]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0:
            raise ValueError("All C values must be positive.")
        values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError("C grid is empty.")
    return values


def build_feature_sets(
    frame: pd.DataFrame,
    compact: list[str],
) -> dict[str, list[str]]:
    missing = sorted(set(compact) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Feature table lacks frozen Compact columns: {missing}"
        )

    membership_quality = [
        feature
        for feature in soft.MEMBERSHIP_QUALITY_FEATURES
        if feature in compact
    ]
    distribution = [
        feature for feature in compact
        if feature not in set(membership_quality)
    ]

    feature_sets: dict[str, list[str]] = {
        "duration_only": ["soft_aoi_duration_sec"],
        "coverage_only": ["soft_aoi_dominant_word_coverage"],
        "duration_coverage": [
            "soft_aoi_duration_sec",
            "soft_aoi_dominant_word_coverage",
        ],
        "membership_quality": membership_quality,
        "compact_distribution": distribution,
        "compact27": compact,
    }

    for name, columns in feature_sets.items():
        if not columns:
            raise ValueError(f"Feature set {name} is empty.")
        missing_columns = sorted(set(columns) - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"Feature set {name} lacks columns: {missing_columns}"
            )
    return feature_sets


def make_model(c_value: float, seed: int) -> Pipeline:
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


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "roc_auc": safe_auc(y_true, scores),
        "pr_auc_nr": safe_ap(y_true, scores),
    }


def threshold_candidates(
    scores: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("No finite decision scores.")
    quantiles = np.linspace(0.0, 1.0, max(3, grid_size))
    candidates = np.quantile(finite, quantiles)
    candidates = np.unique(
        np.concatenate(
            [
                candidates,
                np.asarray([0.0], dtype=float),
                np.asarray(
                    [np.nextafter(finite.min(), -np.inf)],
                    dtype=float,
                ),
                np.asarray(
                    [np.nextafter(finite.max(), np.inf)],
                    dtype=float,
                ),
            ]
        )
    )
    return candidates


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    grid_size: int,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.0
    best_key = (-np.inf, -np.inf, -np.inf)

    for threshold in threshold_candidates(scores, grid_size):
        prediction = (scores >= threshold).astype(int)
        metrics = compute_metrics(y_true, scores, prediction)
        # Prefer macro F1, then balanced accuracy, then thresholds nearer zero.
        key = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            -abs(float(threshold)),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)

    prediction = (scores >= best_threshold).astype(int)
    return best_threshold, compute_metrics(
        y_true, scores, prediction
    )


def validate_group_labels(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    participant_labels = (
        frame.groupby("participant_id", observed=True)["label"]
        .agg(["nunique", "first", "size"])
        .reset_index()
    )
    bad = participant_labels[
        participant_labels["nunique"].ne(1)
    ]
    if not bad.empty:
        raise ValueError(
            "OneStop participant has more than one label:\n"
            + bad.to_string(index=False)
        )
    return participant_labels


def nested_select(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    c_grid: list[float],
    inner_splits: int,
    threshold_grid_size: int,
    seed: int,
) -> tuple[float, float, pd.DataFrame]:
    splitter = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )

    rows = []
    for c_value in c_grid:
        oof_scores = np.full(len(y_train), np.nan, dtype=float)
        for inner_fold, (fit_idx, val_idx) in enumerate(
            splitter.split(x_train, y_train, groups_train),
            start=1,
        ):
            model = make_model(c_value, seed + inner_fold)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train.iloc[fit_idx], y_train[fit_idx])
            oof_scores[val_idx] = model.decision_function(
                x_train.iloc[val_idx]
            )

        if np.isnan(oof_scores).any():
            raise RuntimeError(
                "Inner OOF prediction contains missing scores."
            )

        threshold, metrics = select_threshold(
            y_train, oof_scores, threshold_grid_size
        )
        rows.append(
            {
                "C": c_value,
                "threshold": threshold,
                **metrics,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        [
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "C",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    best = result.iloc[0]
    return float(best["C"]), float(best["threshold"]), result


def aggregate_participants(
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    def majority(values: pd.Series) -> int:
        return int(values.mean() >= 0.5)

    return (
        prediction_frame.groupby(
            ["feature_set", "participant_id", "fold"],
            observed=True,
        )
        .agg(
            label=("label", "first"),
            score=("score", "median"),
            prediction=("prediction", majority),
            trials=("onestop_trial_id", "size"),
        )
        .reset_index()
    )


def pooled_metric_rows(
    oof: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for feature_set, frame in oof.groupby(
        "feature_set", observed=True
    ):
        trial_metrics = compute_metrics(
            frame["label"].to_numpy(dtype=int),
            frame["score"].to_numpy(dtype=float),
            frame["prediction"].to_numpy(dtype=int),
        )
        rows.append(
            {
                "feature_set": feature_set,
                "level": "trial",
                "n_units": len(frame),
                "n_participants": frame["participant_id"].nunique(),
                **trial_metrics,
            }
        )

    participant = aggregate_participants(oof)
    for feature_set, frame in participant.groupby(
        "feature_set", observed=True
    ):
        metrics = compute_metrics(
            frame["label"].to_numpy(dtype=int),
            frame["score"].to_numpy(dtype=float),
            frame["prediction"].to_numpy(dtype=int),
        )
        rows.append(
            {
                "feature_set": feature_set,
                "level": "participant",
                "n_units": len(frame),
                "n_participants": len(frame),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def stratified_participant_bootstrap(
    oof: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    participant_labels = (
        oof[["participant_id", "label"]]
        .drop_duplicates()
        .set_index("participant_id")["label"]
    )
    participants_by_label = {
        int(label): group.index.to_numpy(dtype=object)
        for label, group in participant_labels.groupby(
            participant_labels
        )
    }

    feature_sets = sorted(oof["feature_set"].unique())
    grouped = {
        feature_set: {
            participant: group.copy()
            for participant, group in frame.groupby(
                "participant_id", observed=True
            )
        }
        for feature_set, frame in oof.groupby(
            "feature_set", observed=True
        )
    }

    metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
    ]
    replicate_rows = []

    for replicate in range(n_bootstrap):
        sampled_ids: list[str] = []
        for label in sorted(participants_by_label):
            population = participants_by_label[label]
            sampled = rng.choice(
                population,
                size=len(population),
                replace=True,
            )
            sampled_ids.extend(sampled.tolist())

        for feature_set in feature_sets:
            pieces = [
                grouped[feature_set][participant]
                for participant in sampled_ids
            ]
            sample = pd.concat(pieces, ignore_index=True)
            metrics = compute_metrics(
                sample["label"].to_numpy(dtype=int),
                sample["score"].to_numpy(dtype=float),
                sample["prediction"].to_numpy(dtype=int),
            )
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "feature_set": feature_set,
                    **metrics,
                }
            )

    replicates = pd.DataFrame(replicate_rows)

    ci_rows = []
    for feature_set, frame in replicates.groupby(
        "feature_set", observed=True
    ):
        for metric in metric_names:
            values = frame[metric].to_numpy(dtype=float)
            ci_rows.append(
                {
                    "feature_set": feature_set,
                    "metric": metric,
                    "bootstrap_mean": float(np.nanmean(values)),
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    cis = pd.DataFrame(ci_rows)

    delta_rows = []
    reference = replicates[
        replicates["feature_set"].eq("compact27")
    ].set_index("replicate")
    for feature_set in feature_sets:
        if feature_set == "compact27":
            continue
        comparison = replicates[
            replicates["feature_set"].eq(feature_set)
        ].set_index("replicate")
        common = reference.index.intersection(comparison.index)
        for metric in metric_names:
            delta = (
                reference.loc[common, metric]
                - comparison.loc[common, metric]
            ).to_numpy(dtype=float)
            delta_rows.append(
                {
                    "reference": "compact27",
                    "comparison": feature_set,
                    "metric": metric,
                    "delta_mean": float(np.nanmean(delta)),
                    "ci_low": float(np.nanquantile(delta, 0.025)),
                    "ci_high": float(np.nanquantile(delta, 0.975)),
                    "probability_delta_gt_0": float(
                        np.nanmean(delta > 0)
                    ),
                    "n_bootstrap": n_bootstrap,
                }
            )
    deltas = pd.DataFrame(delta_rows)
    return cis, deltas


def main() -> int:
    args = parse_args()
    if args.outer_splits < 2 or args.inner_splits < 2:
        raise ValueError("Outer and inner splits must be at least 2.")
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact = load_compact_schema(args.frozen_schema)
    c_grid = parse_c_grid(args.c_grid)

    frame = pd.read_parquet(require_file(args.features))
    required = {
        "onestop_trial_id",
        "participant_id",
        "label",
        "label_name",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Feature table lacks columns: {sorted(missing)}"
        )

    frame["participant_id"] = (
        frame["participant_id"].astype(str).str.strip().str.lower()
    )
    frame["label"] = pd.to_numeric(
        frame["label"], errors="raise"
    ).astype(int)

    if frame["onestop_trial_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate trial IDs.")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Expected binary labels {0, 1}.")

    participant_labels = validate_group_labels(frame)
    label_participant_counts = (
        participant_labels.groupby("first", observed=True)
        .size()
        .to_dict()
    )
    if min(label_participant_counts.values()) < args.outer_splits:
        raise ValueError("Too few participants for outer splits.")

    feature_sets = build_feature_sets(frame, compact)

    print("=== ONESTOP INTERNAL VALIDATION INPUT ===")
    print(f"trials: {len(frame):,}")
    print(f"participants: {frame['participant_id'].nunique()}")
    print(
        "participant labels: "
        + json.dumps(
            {
                "IS_0": int(label_participant_counts.get(0, 0)),
                "NR_1": int(label_participant_counts.get(1, 0)),
            }
        )
    )
    print(
        "feature sets: "
        + json.dumps(
            {name: len(columns) for name, columns in feature_sets.items()}
        )
    )
    print(f"C grid: {c_grid}")

    y = frame["label"].to_numpy(dtype=int)
    groups = frame["participant_id"].to_numpy(dtype=object)

    outer = StratifiedGroupKFold(
        n_splits=args.outer_splits,
        shuffle=True,
        random_state=args.seed,
    )
    splits = list(outer.split(frame, y, groups))

    fold_rows = []
    hyperparameter_rows = []
    prediction_parts = []

    for feature_name, columns in feature_sets.items():
        print(
            f"\n=== FEATURE SET {feature_name} "
            f"({len(columns)} features) ==="
        )
        x = frame[columns].apply(
            pd.to_numeric, errors="coerce"
        )

        for fold, (train_idx, test_idx) in enumerate(
            splits, start=1
        ):
            train_groups = groups[train_idx]
            test_groups = groups[test_idx]
            overlap = set(train_groups).intersection(test_groups)
            if overlap:
                raise RuntimeError(
                    f"Participant leakage in fold {fold}: {overlap}"
                )

            best_c, threshold, inner_table = nested_select(
                x_train=x.iloc[train_idx].reset_index(drop=True),
                y_train=y[train_idx],
                groups_train=train_groups,
                c_grid=c_grid,
                inner_splits=args.inner_splits,
                threshold_grid_size=args.threshold_grid_size,
                seed=args.seed + fold * 100,
            )

            for rank, row in inner_table.iterrows():
                hyperparameter_rows.append(
                    {
                        "feature_set": feature_name,
                        "outer_fold": fold,
                        "rank": rank + 1,
                        **row.to_dict(),
                    }
                )

            model = make_model(
                best_c, args.seed + fold * 1000
            )
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", ConvergenceWarning
                )
                model.fit(x.iloc[train_idx], y[train_idx])
            scores = model.decision_function(x.iloc[test_idx])
            predictions = (scores >= threshold).astype(int)
            metrics = compute_metrics(
                y[test_idx], scores, predictions
            )

            fold_rows.append(
                {
                    "feature_set": feature_name,
                    "fold": fold,
                    "n_train_trials": len(train_idx),
                    "n_test_trials": len(test_idx),
                    "n_train_participants": len(
                        np.unique(train_groups)
                    ),
                    "n_test_participants": len(
                        np.unique(test_groups)
                    ),
                    "selected_C": best_c,
                    "selected_threshold": threshold,
                    **metrics,
                }
            )

            prediction_parts.append(
                pd.DataFrame(
                    {
                        "onestop_trial_id": frame.iloc[
                            test_idx
                        ]["onestop_trial_id"].astype(str).to_numpy(),
                        "participant_id": groups[test_idx],
                        "label": y[test_idx],
                        "label_name": frame.iloc[
                            test_idx
                        ]["label_name"].astype(str).to_numpy(),
                        "feature_set": feature_name,
                        "fold": fold,
                        "score": scores,
                        "prediction": predictions,
                        "selected_C": best_c,
                        "selected_threshold": threshold,
                    }
                )
            )

            print(
                f"[{feature_name}] fold {fold}/{args.outer_splits}: "
                f"C={best_c:g}, threshold={threshold:.4f}, "
                f"macroF1={metrics['macro_f1']:.4f}, "
                f"BA={metrics['balanced_accuracy']:.4f}, "
                f"AUC={metrics['roc_auc']:.4f}"
            )

    fold_metrics = pd.DataFrame(fold_rows)
    hyperparameters = pd.DataFrame(hyperparameter_rows)
    oof = pd.concat(prediction_parts, ignore_index=True)

    expected_prediction_rows = len(frame) * len(feature_sets)
    if len(oof) != expected_prediction_rows:
        raise RuntimeError(
            f"OOF rows={len(oof)}, expected={expected_prediction_rows}"
        )
    duplicate_prediction = oof.duplicated(
        ["feature_set", "onestop_trial_id"]
    )
    if duplicate_prediction.any():
        raise RuntimeError("Duplicate OOF trial predictions.")

    pooled = pooled_metric_rows(oof)
    cis, deltas = stratified_participant_bootstrap(
        oof=oof,
        n_bootstrap=args.bootstrap,
        seed=args.seed + 9999,
    )

    fold_path = (
        args.output_dir / "onestop_internal_fold_metrics.csv"
    )
    pooled_path = (
        args.output_dir / "onestop_internal_pooled_metrics.csv"
    )
    ci_path = (
        args.output_dir / "onestop_internal_bootstrap_cis.csv"
    )
    delta_path = (
        args.output_dir / "onestop_internal_paired_deltas.csv"
    )
    oof_path = (
        args.output_dir
        / "onestop_internal_oof_predictions.parquet"
    )
    hyperparameter_path = (
        args.output_dir
        / "onestop_internal_selected_hyperparameters.csv"
    )
    audit_path = (
        args.output_dir
        / "onestop_internal_validation_audit.json"
    )

    fold_metrics.to_csv(
        fold_path, index=False, encoding="utf-8-sig"
    )
    pooled.to_csv(
        pooled_path, index=False, encoding="utf-8-sig"
    )
    cis.to_csv(ci_path, index=False, encoding="utf-8-sig")
    deltas.to_csv(
        delta_path, index=False, encoding="utf-8-sig"
    )
    oof.to_parquet(oof_path, index=False, compression="zstd")
    hyperparameters.to_csv(
        hyperparameter_path,
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "analysis": "OneStop nested participant-independent validation",
        "interpretation_caveat": (
            "OneStop participants are condition-specific; this benchmark "
            "does not by itself separate task effects from participant-cohort "
            "effects."
        ),
        "n_trials": int(len(frame)),
        "n_participants": int(
            frame["participant_id"].nunique()
        ),
        "participant_label_counts": {
            "IS_0": int(label_participant_counts.get(0, 0)),
            "NR_1": int(label_participant_counts.get(1, 0)),
        },
        "feature_sets": feature_sets,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "c_grid": c_grid,
        "threshold_selection": (
            "inner out-of-fold macro-F1 maximization; "
            "balanced accuracy tie-break"
        ),
        "bootstrap": {
            "replicates": args.bootstrap,
            "unit": "participant",
            "stratified_by_label": True,
        },
        "outputs": {
            "fold_metrics": str(fold_path),
            "pooled_metrics": str(pooled_path),
            "bootstrap_cis": str(ci_path),
            "paired_deltas": str(delta_path),
            "oof_predictions": str(oof_path),
            "hyperparameters": str(hyperparameter_path),
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ONESTOP INTERNAL POOLED METRICS ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 220
    ):
        print(
            pooled.sort_values(
                ["level", "roc_auc"],
                ascending=[True, False],
            ).to_string(index=False)
        )

    print("\n=== COMPACT27 PAIRED BOOTSTRAP DELTAS ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 220
    ):
        print(deltas.to_string(index=False))

    print(f"\nSaved: {fold_path}")
    print(f"Saved: {pooled_path}")
    print(f"Saved: {ci_path}")
    print(f"Saved: {delta_path}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {audit_path}")
    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
