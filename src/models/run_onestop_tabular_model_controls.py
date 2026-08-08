from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score

from model_class_control_common import (
    CANONICAL_CONDITIONS,
    metric_dict,
    optimize_threshold,
    participant_median_loo,
    prepare_source,
    prepare_target,
    read_table,
    resolve_feature_columns,
    resolve_table,
    sha256,
    write_hash_manifest,
    write_json,
)

SEED = 20260806
MODELS = ("linear_svm", "hist_gradient_boosting", "mlp")
PROTOCOLS = ("clean_only", "augmented")


def parameter_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "linear_svm":
        return [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}]
    if model_name == "hist_gradient_boosting":
        return [
            {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 250, "l2_regularization": 1.0},
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 300, "l2_regularization": 1.0},
            {"learning_rate": 0.03, "max_leaf_nodes": 31, "max_iter": 400, "l2_regularization": 3.0},
        ]
    if model_name == "mlp":
        return [
            {"hidden_layer_sizes": [64], "alpha": 1e-4, "learning_rate_init": 1e-3},
            {"hidden_layer_sizes": [128, 64], "alpha": 1e-4, "learning_rate_init": 1e-3},
            {"hidden_layer_sizes": [128, 64], "alpha": 1e-3, "learning_rate_init": 5e-4},
        ]
    raise ValueError(model_name)


def build_estimator(model_name: str, params: dict[str, Any], seed: int) -> Pipeline:
    if model_name == "linear_svm":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                C=float(params["C"]),
                class_weight="balanced",
                random_state=seed,
                max_iter=20000,
            )),
        ])
    if model_name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=float(params["learning_rate"]),
                max_leaf_nodes=int(params["max_leaf_nodes"]),
                max_iter=int(params["max_iter"]),
                l2_regularization=float(params["l2_regularization"]),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed,
            )),
        ])
    if model_name == "mlp":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                alpha=float(params["alpha"]),
                learning_rate_init=float(params["learning_rate_init"]),
                batch_size=256,
                max_iter=250,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=seed,
            )),
        ])
    raise ValueError(model_name)


def scores(estimator: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float).reshape(-1)
    probability = np.asarray(estimator.predict_proba(X), dtype=float)[:, 1]
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def training_rows(
    frame: pd.DataFrame,
    columns: dict[str, str],
    participants: set[str],
    protocol: str,
) -> pd.DataFrame:
    mask = frame[columns["participant"]].astype(str).isin(participants)
    if protocol == "clean_only":
        mask &= frame[columns["condition"]].eq("clean")
    elif protocol == "augmented":
        mask &= frame[columns["condition"]].isin(CANONICAL_CONDITIONS)
    else:
        raise ValueError(protocol)
    return frame.loc[mask]


def condition_rows(
    frame: pd.DataFrame,
    columns: dict[str, str],
    participants: set[str],
    condition: str,
) -> pd.DataFrame:
    return frame.loc[
        frame[columns["participant"]].astype(str).isin(participants)
        & frame[columns["condition"]].eq(condition)
    ]


def select_hyperparameters(
    frame: pd.DataFrame,
    feature_columns: list[str],
    columns: dict[str, str],
    outer_train_participants: np.ndarray,
    model_name: str,
    protocol: str,
    inner_folds: int,
    seed: int,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    clean = frame.loc[
        frame[columns["participant"]].astype(str).isin(set(map(str, outer_train_participants)))
        & frame[columns["condition"]].eq("clean")
    ].copy()

    rows: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(parameter_grid(model_name)):
        clean_y_parts: list[np.ndarray] = []
        clean_score_parts: list[np.ndarray] = []
        degraded_auc_parts: list[float] = []
        splitter = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed,
        )

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(
                clean,
                clean[columns["label"]].to_numpy(int),
                groups=clean[columns["participant"]].astype(str).to_numpy(),
            )
        ):
            train_participants = set(clean.iloc[train_idx][columns["participant"]].astype(str).unique())
            val_participants = set(clean.iloc[val_idx][columns["participant"]].astype(str).unique())
            train = training_rows(frame, columns, train_participants, protocol)
            clean_val = condition_rows(frame, columns, val_participants, "clean")

            estimator = build_estimator(model_name, params, seed + candidate_index * 100 + fold)
            estimator.fit(
                train[feature_columns].to_numpy(float),
                train[columns["label"]].to_numpy(int),
            )
            clean_score = scores(estimator, clean_val[feature_columns].to_numpy(float))
            clean_y_parts.append(clean_val[columns["label"]].to_numpy(int))
            clean_score_parts.append(clean_score)

            for condition in CANONICAL_CONDITIONS[1:]:
                degraded_val = condition_rows(frame, columns, val_participants, condition)
                degraded_score = scores(estimator, degraded_val[feature_columns].to_numpy(float))
                degraded_auc_parts.append(
                    float(roc_auc_score(degraded_val[columns["label"]].to_numpy(int), degraded_score))
                )

        clean_y = np.concatenate(clean_y_parts)
        clean_score = np.concatenate(clean_score_parts)
        threshold, clean_macro_f1 = optimize_threshold(clean_y, clean_score)
        clean_auc = float(roc_auc_score(clean_y, clean_score))
        mean_degraded_auc = float(np.mean(degraded_auc_parts))

        primary = clean_auc if protocol == "clean_only" else mean_degraded_auc
        secondary = clean_macro_f1 if protocol == "clean_only" else clean_auc
        rows.append({
            "candidate_index": candidate_index,
            "model": model_name,
            "protocol": protocol,
            "params_json": json.dumps(params, sort_keys=True),
            "selection_primary": primary,
            "selection_secondary": secondary,
            "clean_oof_roc_auc": clean_auc,
            "clean_oof_macro_f1": clean_macro_f1,
            "mean_degraded_roc_auc": mean_degraded_auc,
            "clean_oof_threshold": threshold,
        })

    summary = pd.DataFrame(rows).sort_values(
        ["selection_primary", "selection_secondary"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    best = summary.iloc[0]
    return json.loads(best["params_json"]), float(best["clean_oof_threshold"]), summary


def bootstrap_transfer(
    frame: pd.DataFrame,
    score_columns: dict[str, str],
    thresholds: dict[str, float],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    participants = frame["participant_id"].astype(str).to_numpy()
    unique = pd.unique(participants)
    index_map = {participant: np.flatnonzero(participants == participant) for participant in unique}
    y = frame["label_nr"].to_numpy(int)
    rows = []

    for replicate in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([index_map[p] for p in sampled])
        pseudo = np.concatenate([
            np.repeat(f"cluster_{i}", len(index_map[p])) for i, p in enumerate(sampled)
        ])
        for method, column in score_columns.items():
            rows.append({
                "replicate": replicate,
                "method": method,
                **metric_dict(y[indices], frame[column].to_numpy(float)[indices], thresholds[method], pseudo),
            })

    replicates = pd.DataFrame(rows)
    delta_rows = []
    comparisons = []
    for model in MODELS:
        for protocol in PROTOCOLS:
            raw = f"{model}_{protocol}_raw"
            calibrated = f"{model}_{protocol}_participant_calibrated"
            if raw in score_columns and calibrated in score_columns:
                comparisons.append((calibrated, raw))

    for protocol in PROTOCOLS:
        reference = f"linear_svm_{protocol}_participant_calibrated"
        if reference in score_columns:
            for model in ("hist_gradient_boosting", "mlp"):
                method = f"{model}_{protocol}_participant_calibrated"
                if method in score_columns:
                    comparisons.append((method, reference))

    for method, reference in comparisons:
        left = replicates[replicates["method"].eq(method)].set_index("replicate")
        right = replicates[replicates["method"].eq(reference)].set_index("replicate")
        common = left.index.intersection(right.index)
        for metric in ("macro_f1", "balanced_accuracy", "roc_auc", "pr_auc_nr", "average_precision_nr"):
            delta = left.loc[common, metric].to_numpy(float) - right.loc[common, metric].to_numpy(float)
            delta_rows.append({
                "method": method,
                "reference": reference,
                "metric": metric,
                "mean_difference": float(np.mean(delta)),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "n_replicates": int(len(delta)),
            })
    return replicates, pd.DataFrame(delta_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-table", default="data/processed/onestop/quality_views_asc344_v1/onestop_quality_views_all_features.parquet")
    parser.add_argument("--frozen-model", default="outputs/frozen_onestop_external_validation_v1/models/soft_full_all.joblib")
    parser.add_argument("--target-features", default="outputs/webqamgaze_external_validation_stage3c/soft_full_features_primary.csv")
    parser.add_argument("--target-labels", default="outputs/webqamgaze_external_validation_stage2/trial_manifest_primary_candidates.csv")
    parser.add_argument("--output-dir", default="outputs/onestop_tabular_model_controls_v1")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    started = time.time()
    root = Path(args.project_root).resolve()
    output = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_path, source_audit = resolve_table(args.source_table, root, require_condition=True)
    target_path, target_audit = resolve_table(args.target_features, root, require_condition=False)
    source_audit.insert(0, "role", "source")
    target_audit.insert(0, "role", "target")
    pd.concat([source_audit, target_audit], ignore_index=True).to_csv(
        output / "input_candidate_audit.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Source candidate: {source_path}")
    print(f"Target candidate: {target_path}")

    frozen_model = (root / args.frozen_model).resolve() if not Path(args.frozen_model).is_absolute() else Path(args.frozen_model).resolve()
    source_raw = read_table(source_path)
    feature_columns = resolve_feature_columns(source_raw, frozen_model)
    source, source_columns = prepare_source(source_raw, feature_columns)

    source_participants = int(
        source[source_columns["participant"]].nunique()
    )
    source_conditions = (
        source[source_columns["condition"]]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    print(f"Source rows: {len(source):,}")
    print(f"Source participants: {source_participants:,}")
    print(f"Source condition column: {source_columns['condition']}")
    print(f"Source condition counts: {source_conditions}")

    if len(source) < 100000 or source_participants < 300:
        raise RuntimeError(
            "The selected source is not the complete OneStop ASC344 "
            "quality-view table. Expected approximately 130,011 rows "
            "and 344 participants. Use "
            "data\\processed\\onestop\\quality_views_asc344_v1\\"
            "onestop_quality_views_all_features.parquet"
        )

    if args.audit_only:
        print("Full OneStop source validation: PASSED")
        return 0

    target_labels_path = (root / args.target_labels).resolve() if not Path(args.target_labels).is_absolute() else Path(args.target_labels).resolve()
    labels = read_table(target_labels_path) if target_labels_path.exists() else None
    target, target_columns = prepare_target(read_table(target_path), labels, feature_columns)

    write_json(output / "input_audit.json", {
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "target_path": str(target_path),
        "target_sha256": sha256(target_path),
        "target_labels_path": str(target_labels_path) if target_labels_path.exists() else None,
        "feature_columns": feature_columns,
        "source_rows": len(source),
        "source_participants": source[source_columns["participant"]].nunique(),
        "source_condition_counts": source[source_columns["condition"]].value_counts().sort_index().to_dict(),
        "target_rows": len(target),
        "target_participants": target[target_columns["participant"]].nunique(),
        "target_class_counts": target[target_columns["label"]].value_counts().sort_index().to_dict(),
    })

    clean = source[source[source_columns["condition"]].eq("clean")].copy()
    outer_splitter = StratifiedGroupKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )

    prediction_frames = []
    fold_metrics = []
    selections = []
    selected_records = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_splitter.split(
            clean,
            clean[source_columns["label"]].to_numpy(int),
            groups=clean[source_columns["participant"]].astype(str).to_numpy(),
        )
    ):
        train_participants = clean.iloc[train_idx][source_columns["participant"]].astype(str).unique()
        test_participants = set(clean.iloc[test_idx][source_columns["participant"]].astype(str).unique())

        for model_name in args.models:
            for protocol in args.protocols:
                print(f"outer={outer_fold + 1}/{args.outer_folds} model={model_name} protocol={protocol}", flush=True)
                best_params, threshold, selection = select_hyperparameters(
                    source,
                    feature_columns,
                    source_columns,
                    train_participants,
                    model_name,
                    protocol,
                    args.inner_folds,
                    args.seed + outer_fold * 1000,
                )
                selection.insert(0, "outer_fold", outer_fold)
                selections.append(selection)
                selected_records.append({
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "protocol": protocol,
                    "params_json": json.dumps(best_params, sort_keys=True),
                    "threshold": threshold,
                })

                train = training_rows(source, source_columns, set(map(str, train_participants)), protocol)
                estimator = build_estimator(model_name, best_params, args.seed + outer_fold * 100 + 7)
                estimator.fit(train[feature_columns].to_numpy(float), train[source_columns["label"]].to_numpy(int))

                for condition in CANONICAL_CONDITIONS:
                    test = condition_rows(source, source_columns, test_participants, condition).copy()
                    score = scores(estimator, test[feature_columns].to_numpy(float))
                    fold_metrics.append({
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "protocol": protocol,
                        "condition": condition,
                        "threshold": threshold,
                        **metric_dict(
                            test[source_columns["label"]].to_numpy(int),
                            score,
                            threshold,
                            test[source_columns["participant"]].astype(str).to_numpy(),
                        ),
                    })
                    frame = pd.DataFrame({
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "protocol": protocol,
                        "condition": condition,
                        "participant_id": test[source_columns["participant"]].astype(str).to_numpy(),
                        "label_nr": test[source_columns["label"]].to_numpy(int),
                        "score": score,
                        "threshold": threshold,
                        "prediction_nr": (score >= threshold).astype(int),
                    })
                    if source_columns["trial"]:
                        frame["trial_id"] = test[source_columns["trial"]].astype(str).to_numpy()
                    prediction_frames.append(frame)

    cv_predictions = pd.concat(prediction_frames, ignore_index=True)
    pd.DataFrame(fold_metrics).to_csv(output / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(selections, ignore_index=True).to_csv(output / "inner_model_selection.csv", index=False, encoding="utf-8-sig")
    selected = pd.DataFrame(selected_records)
    selected.to_csv(output / "selected_hyperparameters.csv", index=False, encoding="utf-8-sig")
    cv_predictions.to_parquet(output / "cv_predictions.parquet", index=False)

    pooled_rows = []
    source_info: dict[tuple[str, str], dict[str, float]] = {}
    for (model_name, protocol, condition), group in cv_predictions.groupby(["model", "protocol", "condition"]):
        y = group["label_nr"].to_numpy(int)
        score = group["score"].to_numpy(float)
        prediction = group["prediction_nr"].to_numpy(int)
        # Fold-specific thresholds already generated prediction_nr. For pooled reporting,
        # macro-F1 and accuracy use those predictions; ranking metrics use pooled scores.
        metrics = metric_dict(y, score, 0.0, group["participant_id"].astype(str).to_numpy())
        metrics["accuracy"] = float(np.mean(prediction == y))
        metrics["macro_f1"] = float(f1_score(y, prediction, average="macro", zero_division=0))
        metrics["balanced_accuracy"] = float(
            0.5 * (
                np.mean(prediction[y == 1] == 1)
                + np.mean(prediction[y == 0] == 0)
            )
        )
        metrics["nr_prediction_rate"] = float(np.mean(prediction))
        pooled_rows.append({"model": model_name, "protocol": protocol, "condition": condition, **metrics})
        if condition == "clean":
            threshold, _ = optimize_threshold(y, score)
            source_info[(model_name, protocol)] = {
                "threshold": threshold,
                "reference": float(np.median(score)),
            }

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output / "cv_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"model": model, "protocol": protocol, "source_oof_threshold": info["threshold"], "source_oof_reference_median": info["reference"]}
        for (model, protocol), info in source_info.items()
    ]).to_csv(output / "source_oof_thresholds.csv", index=False, encoding="utf-8-sig")

    transfer = pd.DataFrame({
        "participant_id": target[target_columns["participant"]].astype(str).to_numpy(),
        "external_trial_id": target[target_columns["trial"]].astype(str).to_numpy(),
        "label_nr": target[target_columns["label"]].to_numpy(int),
    })
    transfer_metric_rows = []
    score_columns = {}
    thresholds = {}

    for model_name in args.models:
        for protocol in args.protocols:
            subset = selected[selected["model"].eq(model_name) & selected["protocol"].eq(protocol)]
            most_common_json = Counter(subset["params_json"]).most_common(1)[0][0]
            params = json.loads(most_common_json)
            train = training_rows(
                source,
                source_columns,
                set(source[source_columns["participant"]].astype(str).unique()),
                protocol,
            )
            estimator = build_estimator(model_name, params, args.seed + 50000)
            estimator.fit(train[feature_columns].to_numpy(float), train[source_columns["label"]].to_numpy(int))
            raw_score = scores(estimator, target[feature_columns].to_numpy(float))
            info = source_info[(model_name, protocol)]
            calibrated_score = participant_median_loo(
                raw_score,
                transfer["participant_id"].to_numpy(str),
                info["reference"],
            )

            for adaptation, values in (("raw", raw_score), ("participant_calibrated", calibrated_score)):
                method = f"{model_name}_{protocol}_{adaptation}"
                column = f"{method}_score"
                transfer[column] = values
                score_columns[method] = column
                thresholds[method] = info["threshold"]
                transfer_metric_rows.append({
                    "method": method,
                    "model": model_name,
                    "protocol": protocol,
                    "adaptation": adaptation,
                    "source_selected_params_json": most_common_json,
                    "source_threshold": info["threshold"],
                    "source_reference": info["reference"],
                    **metric_dict(
                        transfer["label_nr"].to_numpy(int),
                        values,
                        info["threshold"],
                        transfer["participant_id"].to_numpy(str),
                    ),
                })

            model_dir = output / "models"
            model_dir.mkdir(exist_ok=True)
            joblib.dump({
                "model_name": model_name,
                "protocol": protocol,
                "params": params,
                "feature_columns": feature_columns,
                "threshold": info["threshold"],
                "source_reference": info["reference"],
                "estimator": estimator,
            }, model_dir / f"{model_name}_{protocol}.joblib")

    transfer_metrics = pd.DataFrame(transfer_metric_rows)
    transfer_metrics.to_csv(output / "transfer_metrics.csv", index=False, encoding="utf-8-sig")
    transfer.to_parquet(output / "transfer_predictions.parquet", index=False)

    bootstrap_replicates, bootstrap_deltas = bootstrap_transfer(
        transfer,
        score_columns,
        thresholds,
        args.bootstrap,
        args.seed,
    )
    bootstrap_replicates.to_parquet(output / "transfer_bootstrap_replicates.parquet", index=False)
    bootstrap_deltas.to_csv(output / "transfer_bootstrap_deltas.csv", index=False, encoding="utf-8-sig")

    clean_summary = pooled[pooled["condition"].eq("clean")].sort_values(["protocol", "roc_auc"], ascending=[True, False])
    degraded_summary = (
        pooled[~pooled["condition"].eq("clean")]
        .groupby(["model", "protocol"], as_index=False)
        .agg(
            mean_degraded_macro_f1=("macro_f1", "mean"),
            mean_degraded_roc_auc=("roc_auc", "mean"),
            worst_degraded_macro_f1=("macro_f1", "min"),
            worst_degraded_roc_auc=("roc_auc", "min"),
        )
        .sort_values(["protocol", "mean_degraded_roc_auc"], ascending=[True, False])
    )

    summary = f"""# OneStop Tabular Model-Class Controls

## Scope

The Full-41 Event-Free AOI input and participant splits are held fixed while the
classifier family changes among LinearSVM, HistGradientBoosting, and MLP.
This tests whether the paper's findings are specific to a linear classifier.

## OneStop clean-view pooled results

{clean_summary.to_markdown(index=False)}

## OneStop degraded-view summary

{degraded_summary.to_markdown(index=False)}

## Frozen OneStop-to-WebQAmGaze English transfer

{transfer_metrics.sort_values(["protocol", "adaptation", "roc_auc"], ascending=[True, True, False]).to_markdown(index=False)}

## Interpretation boundary

- English WebQAmGaze is a development target, not a new confirmation target.
- Participant median LOO calibration is applied to every model score with the
  same source-only rule.
- QCN-PC is not redefined for nonlinear models because its exact additive
  feature-group decomposition is specific to the frozen linear model.
- A nonlinear improvement on OneStop clean data does not by itself establish
  measurement robustness or cross-dataset transfer.
"""
    (output / "TABULAR_MODEL_CONTROL_SUMMARY.md").write_text(summary, encoding="utf-8")
    write_json(output / "release_manifest.json", {
        "artifact_version": "onestop_tabular_model_controls_v1",
        "elapsed_seconds": time.time() - started,
        "source_path": str(source_path),
        "target_path": str(target_path),
        "models": args.models,
        "protocols": args.protocols,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "feature_count": len(feature_columns),
    })
    write_hash_manifest(output)
    print(f"Completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
