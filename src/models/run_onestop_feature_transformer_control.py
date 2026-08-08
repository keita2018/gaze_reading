from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

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


class FeatureTransformer:
    """
    Same-input tabular Transformer control.

    Each of the fixed Full-41 Event-Free AOI features becomes one token. The
    model does not consume text tokens, fixation sequences, or saccade events,
    and therefore is not a reproduction of RoBERTa-Eye-F.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        learning_rate: float = 7e-4,
        weight_decay: float = 3e-4,
        batch_size: int = 512,
        epochs: int = 40,
        patience: int = 7,
        seed: int = SEED,
    ) -> None:
        self.config = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
        }
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None
        self.device = None
        self.best_epoch_ = None
        self.history_: list[dict[str, float]] = []

    def _build_model(self, n_features: int):
        import torch
        import torch.nn as nn

        config = self.config
        d_model = int(config["d_model"])

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.value_projection = nn.Linear(1, d_model)
                self.feature_embedding = nn.Parameter(
                    torch.zeros(1, n_features, d_model)
                )
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=int(config["n_heads"]),
                    dim_feedforward=d_model * 4,
                    dropout=float(config["dropout"]),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=int(config["n_layers"]),
                )
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, 1)
                nn.init.normal_(self.feature_embedding, std=0.02)
                nn.init.normal_(self.cls_token, std=0.02)

            def forward(self, x):
                tokens = self.value_projection(x.unsqueeze(-1))
                tokens = tokens + self.feature_embedding
                cls = self.cls_token.expand(x.shape[0], -1, -1)
                encoded = self.encoder(torch.cat([cls, tokens], dim=1))
                return self.head(self.norm(encoded[:, 0])).squeeze(-1)

        return Network()

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "FeatureTransformer":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required. Install torch in the project environment."
            ) from exc

        seed = int(self.config["seed"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        X_std = self.scaler.fit_transform(
            self.imputer.fit_transform(X)
        ).astype(np.float32)
        y_int = np.asarray(y, dtype=np.int64)
        groups = np.asarray(groups).astype(str)

        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )
        train_idx, val_idx = next(
            splitter.split(X_std, y_int, groups=groups)
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = self._build_model(X_std.shape[1]).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )

        train_dataset = TensorDataset(
            torch.from_numpy(X_std[train_idx]),
            torch.from_numpy(y_int[train_idx].astype(np.float32)),
        )
        loader = DataLoader(
            train_dataset,
            batch_size=int(self.config["batch_size"]),
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_x = torch.from_numpy(X_std[val_idx]).to(self.device)
        val_y = torch.from_numpy(
            y_int[val_idx].astype(np.float32)
        ).to(self.device)

        best_state = None
        best_loss = math.inf
        stale = 0
        for epoch in range(int(self.config["epochs"])):
            model.train()
            train_losses: list[float] = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_x), val_y).cpu())
            self.history_.append({
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": val_loss,
            })

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                self.best_epoch_ = epoch
                stale = 0
            else:
                stale += 1
                if stale >= int(self.config["patience"]):
                    break

        if best_state is None:
            raise RuntimeError("Transformer training failed before a valid epoch.")
        model.load_state_dict(best_state)
        self.model = model
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X_std = self.scaler.transform(
            self.imputer.transform(X)
        ).astype(np.float32)
        batch_size = int(self.config["batch_size"])
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(X_std), batch_size):
                batch = torch.from_numpy(
                    X_std[start:start + batch_size]
                ).to(self.device)
                outputs.append(
                    self.model(batch).detach().cpu().numpy()
                )
        return np.concatenate(outputs).astype(float)


def training_rows(frame, columns, participants: set[str], protocol: str):
    mask = frame[columns["participant"]].astype(str).isin(participants)
    if protocol == "clean_only":
        mask &= frame[columns["condition"]].eq("clean")
    elif protocol == "augmented":
        mask &= frame[columns["condition"]].isin(CANONICAL_CONDITIONS)
    else:
        raise ValueError(protocol)
    return frame.loc[mask]


def condition_rows(frame, columns, participants: set[str], condition: str):
    return frame.loc[
        frame[columns["participant"]].astype(str).isin(participants)
        & frame[columns["condition"]].eq(condition)
    ]


def bootstrap(frame, raw_column: str, calibrated_column: str, threshold: float, n_bootstrap: int, seed: int):
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
        for method, column in (("raw", raw_column), ("participant_calibrated", calibrated_column)):
            rows.append({
                "replicate": replicate,
                "method": method,
                **metric_dict(
                    y[indices],
                    frame[column].to_numpy(float)[indices],
                    threshold,
                    pseudo,
                ),
            })

    replicates = pd.DataFrame(rows)
    left = replicates[replicates["method"].eq("participant_calibrated")].set_index("replicate")
    right = replicates[replicates["method"].eq("raw")].set_index("replicate")
    deltas = []
    for metric in ("macro_f1", "balanced_accuracy", "roc_auc", "pr_auc_nr", "average_precision_nr"):
        values = left[metric].to_numpy(float) - right[metric].to_numpy(float)
        deltas.append({
            "method": "participant_calibrated",
            "reference": "raw",
            "metric": metric,
            "mean_difference": float(np.mean(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "n_replicates": int(len(values)),
        })
    return replicates, pd.DataFrame(deltas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-table", default="data/processed/onestop/quality_views_asc344_v1/onestop_quality_views_all_features.parquet")
    parser.add_argument("--frozen-model", default="outputs/frozen_onestop_external_validation_v1/models/soft_full_all.joblib")
    parser.add_argument("--target-features", default="outputs/webqamgaze_external_validation_stage3c/soft_full_features_primary.csv")
    parser.add_argument("--target-labels", default="outputs/webqamgaze_external_validation_stage2/trial_manifest_primary_candidates.csv")
    parser.add_argument("--output-dir", default="outputs/onestop_feature_transformer_control_v1")
    parser.add_argument("--protocol", choices=("clean_only", "augmented"), default="clean_only")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)
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

    frozen_model = (root / args.frozen_model).resolve() if not Path(args.frozen_model).is_absolute() else Path(args.frozen_model).resolve()
    source_raw = read_table(source_path)
    feature_columns = resolve_feature_columns(source_raw, frozen_model)
    source, source_columns = prepare_source(source_raw, feature_columns)

    labels_path = (root / args.target_labels).resolve() if not Path(args.target_labels).is_absolute() else Path(args.target_labels).resolve()
    labels = read_table(labels_path) if labels_path.exists() else None
    target, target_columns = prepare_target(read_table(target_path), labels, feature_columns)

    clean = source[source[source_columns["condition"]].eq("clean")].copy()
    splitter = StratifiedGroupKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )

    prediction_frames = []
    fold_metric_rows = []
    history_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        splitter.split(
            clean,
            clean[source_columns["label"]].to_numpy(int),
            groups=clean[source_columns["participant"]].astype(str).to_numpy(),
        )
    ):
        print(f"Transformer outer fold {outer_fold + 1}/{args.outer_folds}", flush=True)
        train_participants = set(clean.iloc[train_idx][source_columns["participant"]].astype(str).unique())
        test_participants = set(clean.iloc[test_idx][source_columns["participant"]].astype(str).unique())
        train = training_rows(source, source_columns, train_participants, args.protocol)

        model = FeatureTransformer(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed + outer_fold * 100,
        )
        model.fit(
            train[feature_columns].to_numpy(float),
            train[source_columns["label"]].to_numpy(int),
            train[source_columns["participant"]].astype(str).to_numpy(),
        )
        for row in model.history_:
            history_rows.append({"outer_fold": outer_fold, **row})

        for condition in CANONICAL_CONDITIONS:
            test = condition_rows(source, source_columns, test_participants, condition).copy()
            score = model.decision_function(test[feature_columns].to_numpy(float))
            fold_metric_rows.append({
                "outer_fold": outer_fold,
                "model": "feature_transformer",
                "protocol": args.protocol,
                "condition": condition,
                "threshold": 0.0,
                "best_epoch": model.best_epoch_,
                **metric_dict(
                    test[source_columns["label"]].to_numpy(int),
                    score,
                    0.0,
                    test[source_columns["participant"]].astype(str).to_numpy(),
                ),
            })
            frame = pd.DataFrame({
                "outer_fold": outer_fold,
                "model": "feature_transformer",
                "protocol": args.protocol,
                "condition": condition,
                "participant_id": test[source_columns["participant"]].astype(str).to_numpy(),
                "label_nr": test[source_columns["label"]].to_numpy(int),
                "score": score,
                "prediction_nr_at_zero": (score >= 0.0).astype(int),
            })
            if source_columns["trial"]:
                frame["trial_id"] = test[source_columns["trial"]].astype(str).to_numpy()
            prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(output / "cv_predictions.parquet", index=False)
    pd.DataFrame(fold_metric_rows).to_csv(output / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")

    pooled_rows = []
    for condition, group in predictions.groupby("condition"):
        y = group["label_nr"].to_numpy(int)
        score = group["score"].to_numpy(float)
        threshold, _ = optimize_threshold(y, score) if condition == "clean" else (0.0, 0.0)
        pooled_rows.append({
            "model": "feature_transformer",
            "protocol": args.protocol,
            "condition": condition,
            "reporting_threshold": threshold,
            **metric_dict(y, score, threshold, group["participant_id"].astype(str).to_numpy()),
        })
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output / "cv_pooled_metrics.csv", index=False, encoding="utf-8-sig")

    clean_oof = predictions[predictions["condition"].eq("clean")]
    source_threshold, _ = optimize_threshold(
        clean_oof["label_nr"].to_numpy(int),
        clean_oof["score"].to_numpy(float),
    )
    source_reference = float(np.median(clean_oof["score"].to_numpy(float)))

    final_model = FeatureTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed + 50000,
    )
    train_all = training_rows(
        source,
        source_columns,
        set(source[source_columns["participant"]].astype(str).unique()),
        args.protocol,
    )
    final_model.fit(
        train_all[feature_columns].to_numpy(float),
        train_all[source_columns["label"]].to_numpy(int),
        train_all[source_columns["participant"]].astype(str).to_numpy(),
    )

    raw_score = final_model.decision_function(target[feature_columns].to_numpy(float))
    participants = target[target_columns["participant"]].astype(str).to_numpy()
    calibrated_score = participant_median_loo(raw_score, participants, source_reference)
    transfer = pd.DataFrame({
        "participant_id": participants,
        "external_trial_id": target[target_columns["trial"]].astype(str).to_numpy(),
        "label_nr": target[target_columns["label"]].to_numpy(int),
        "feature_transformer_raw_score": raw_score,
        "feature_transformer_participant_calibrated_score": calibrated_score,
    })
    transfer.to_parquet(output / "transfer_predictions.parquet", index=False)

    transfer_metrics = pd.DataFrame([
        {
            "method": "feature_transformer_raw",
            "protocol": args.protocol,
            "source_threshold": source_threshold,
            "source_reference": source_reference,
            **metric_dict(
                transfer["label_nr"].to_numpy(int),
                raw_score,
                source_threshold,
                participants,
            ),
        },
        {
            "method": "feature_transformer_participant_calibrated",
            "protocol": args.protocol,
            "source_threshold": source_threshold,
            "source_reference": source_reference,
            **metric_dict(
                transfer["label_nr"].to_numpy(int),
                calibrated_score,
                source_threshold,
                participants,
            ),
        },
    ])
    transfer_metrics.to_csv(output / "transfer_metrics.csv", index=False, encoding="utf-8-sig")

    bootstrap_replicates, bootstrap_deltas = bootstrap(
        transfer,
        "feature_transformer_raw_score",
        "feature_transformer_participant_calibrated_score",
        source_threshold,
        args.bootstrap,
        args.seed,
    )
    bootstrap_replicates.to_parquet(output / "transfer_bootstrap_replicates.parquet", index=False)
    bootstrap_deltas.to_csv(output / "transfer_bootstrap_deltas.csv", index=False, encoding="utf-8-sig")

    summary = f"""# OneStop Feature-Transformer Control

## Scope

This is a same-input model-class control. It uses the exact Full-41 aggregated
Event-Free AOI features as 41 feature tokens. It does not use fixation sequences,
text tokens, RoBERTa embeddings, or linguistic features and is therefore not a
reproduction of RoBERTa-Eye-F.

## OneStop pooled results

{pooled.to_markdown(index=False)}

## Frozen OneStop-to-WebQAmGaze English transfer

{transfer_metrics.to_markdown(index=False)}

## Interpretation boundary

- English WebQAmGaze is a development target.
- The comparison isolates model class while keeping the Event-Free input fixed.
- QCN-PC is not applied because exact additive feature-group decomposition is a
  property of the linear frozen model.
- Higher OneStop clean accuracy does not automatically imply better degradation
  robustness or external webcam transfer.
"""
    (output / "FEATURE_TRANSFORMER_CONTROL_SUMMARY.md").write_text(summary, encoding="utf-8")
    write_json(output / "release_manifest.json", {
        "artifact_version": "onestop_feature_transformer_control_v1",
        "elapsed_seconds": time.time() - started,
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "target_path": str(target_path),
        "target_sha256": sha256(target_path),
        "protocol": args.protocol,
        "source_threshold": source_threshold,
        "source_reference": source_reference,
        "feature_count": len(feature_columns),
        "architecture": final_model.config,
        "outer_folds": args.outer_folds,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    })
    write_hash_manifest(output)
    print(f"Completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
