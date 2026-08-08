#!/usr/bin/env python
"""Train and evaluate a sample-sequence Event-Free Transformer."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from event_free_sequence_common import (  # noqa: E402
    CHANNEL_NAMES,
    calculate_metrics,
    optimize_macro_f1_threshold,
    write_hash_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path(
            "data/processed/onestop/event_free_sequence_cache_v1"
        ),
    )
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path(
            "data/processed/webqamgaze/event_free_sequence_cache_en_v1"
        ),
    )
    parser.add_argument(
        "--protocols",
        default="clean_only,augmented",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/onestop_event_free_transformer_v1"
        ),
    )
    return parser.parse_args()


class ShardStore:
    def __init__(self, cache_root: Path, max_open: int = 4):
        self.cache_root = cache_root
        self.max_open = max_open
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, relative_path: str) -> dict[str, np.ndarray]:
        if relative_path in self._cache:
            value = self._cache.pop(relative_path)
            self._cache[relative_path] = value
            return value
        path = self.cache_root / relative_path
        with np.load(path, allow_pickle=False) as data:
            value = {key: data[key] for key in data.files}
        self._cache[relative_path] = value
        while len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return value


def deterministic_validation_trials(
    clean_rows: pd.DataFrame,
    fraction: float,
    seed: int,
) -> set[str]:
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for _, group in clean_rows.groupby("participant_id", sort=True):
        trial_ids = group["trial_id"].astype(str).to_numpy()
        count = max(1, int(round(len(trial_ids) * fraction)))
        count = min(count, max(len(trial_ids) - 1, 1))
        selected.update(
            rng.choice(trial_ids, size=count, replace=False).tolist()
        )
    return selected


def participant_calibrate(
    score: np.ndarray,
    participant: np.ndarray,
    source_reference: float,
) -> np.ndarray:
    result = np.empty_like(score, dtype=float)
    frame = pd.DataFrame(
        {
            "participant": participant.astype(str),
            "score": score.astype(float),
            "row": np.arange(len(score)),
        }
    )
    for _, group in frame.groupby("participant", sort=False):
        rows = group["row"].to_numpy(int)
        values = group["score"].to_numpy(float)
        if len(values) < 2:
            result[rows] = values
            continue
        for local, row in enumerate(rows):
            other = np.delete(values, local)
            result[row] = (
                values[local]
                - float(np.median(other))
                + source_reference
            )
    return result


def bootstrap_target(
    frame: pd.DataFrame,
    methods: dict[str, str],
    thresholds: dict[str, float],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    participants = frame["participant_id"].astype(str).to_numpy()
    unique = pd.unique(participants)
    lookup = {
        participant: np.flatnonzero(participants == participant)
        for participant in unique
    }
    replicate_rows = []
    for replicate in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([lookup[p] for p in sampled])
        y = frame["label"].to_numpy(int)[indices]
        for method, column in methods.items():
            metrics = calculate_metrics(
                y,
                frame[column].to_numpy(float)[indices],
                thresholds[method],
            )
            replicate_rows.append(
                {"replicate": replicate, "method": method, **metrics}
            )
    replicates = pd.DataFrame(replicate_rows)

    delta_rows = []
    if "participant_calibrated" in methods and "raw" in methods:
        left = (
            replicates[replicates["method"].eq("participant_calibrated")]
            .set_index("replicate")
        )
        right = (
            replicates[replicates["method"].eq("raw")]
            .set_index("replicate")
        )
        for metric in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
            "average_precision_nr",
        ):
            delta = left[metric].to_numpy() - right[metric].to_numpy()
            delta_rows.append(
                {
                    "method": "participant_calibrated",
                    "reference": "raw",
                    "metric": metric,
                    "mean_difference": float(delta.mean()),
                    "ci_low": float(np.quantile(delta, 0.025)),
                    "ci_high": float(np.quantile(delta, 0.975)),
                    "n_bootstrap": int(len(delta)),
                }
            )
    return replicates, pd.DataFrame(delta_rows)


def main() -> int:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install torch in the active .venv."
        ) from exc

    if args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    protocols = [
        value.strip()
        for value in args.protocols.split(",")
        if value.strip()
    ]
    unknown = sorted(set(protocols) - {"clean_only", "augmented"})
    if unknown:
        raise ValueError(f"Unknown protocols: {unknown}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_index = pd.read_parquet(
        args.source_cache / "sequence_index.parquet"
    )
    target_index = pd.read_parquet(
        args.target_cache / "sequence_index.parquet"
    )
    source_store = ShardStore(args.source_cache)
    target_store = ShardStore(args.target_cache)

    class SequenceDataset(Dataset):
        def __init__(
            self,
            frame: pd.DataFrame,
            store: ShardStore,
        ):
            self.frame = frame.reset_index(drop=True)
            self.store = store

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            shard = self.store.get(str(row["shard_path"]))
            local = int(row["local_index"])
            x = torch.from_numpy(
                shard["x"][local].astype(np.float32)
            )
            length = int(shard["lengths"][local])
            padding = torch.arange(x.shape[0]) >= length
            y = torch.tensor(int(row["label"]), dtype=torch.float32)
            return x, padding, y, index

    class SinusoidalPosition(nn.Module):
        def __init__(self, d_model: int, max_length: int):
            super().__init__()
            position = torch.arange(max_length).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_length, d_model)
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            return x + self.pe[:, : x.shape[1]]

    class EventFreeTransformer(nn.Module):
        def __init__(self, max_length: int):
            super().__init__()
            self.input_projection = nn.Linear(
                len(CHANNEL_NAMES),
                args.d_model,
            )
            self.position = SinusoidalPosition(
                args.d_model,
                max_length + 1,
            )
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, args.d_model)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.n_heads,
                dim_feedforward=args.d_model * args.ff_multiplier,
                dropout=args.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=args.n_layers,
            )
            self.norm = nn.LayerNorm(args.d_model)
            self.head = nn.Linear(args.d_model, 1)
            nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, x, padding):
            token = self.input_projection(x)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            token = torch.cat([cls, token], dim=1)
            token = self.position(token)
            cls_padding = torch.zeros(
                (padding.shape[0], 1),
                dtype=torch.bool,
                device=padding.device,
            )
            full_padding = torch.cat([cls_padding, padding], dim=1)
            encoded = self.encoder(
                token,
                src_key_padding_mask=full_padding,
            )
            return self.head(
                self.norm(encoded[:, 0])
            ).squeeze(-1)

    def make_loader(
        frame: pd.DataFrame,
        store: ShardStore,
        shuffle: bool,
        seed: int,
    ):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            SequenceDataset(frame, store),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def predict(
        model,
        frame: pd.DataFrame,
        store: ShardStore,
    ) -> np.ndarray:
        loader = make_loader(frame, store, False, args.seed)
        model.eval()
        scores = np.empty(len(frame), dtype=float)
        with torch.no_grad():
            for x, padding, _, row_index in loader:
                x = x.to(device, non_blocking=True)
                padding = padding.to(device, non_blocking=True)
                logits = model(x, padding).detach().cpu().numpy()
                scores[row_index.numpy()] = logits
        return scores

    def train_model(
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        fixed_epochs: int | None,
        seed: int,
    ):
        max_length = int(
            max(
                train_frame["stored_length"].max(),
                validation_frame["stored_length"].max()
                if len(validation_frame) else 1,
            )
        )
        # Cache arrays all have the manifest max length. Read one sample to
        # recover the actual padded tensor length.
        first = train_frame.iloc[0]
        padded_length = source_store.get(
            str(first["shard_path"])
        )["x"].shape[1]

        model = EventFreeTransformer(padded_length).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=torch.cuda.is_available(),
        )
        train_loader = make_loader(
            train_frame,
            source_store,
            True,
            seed,
        )
        best_state = None
        best_loss = math.inf
        best_epoch = 0
        stale = 0
        history = []
        epoch_limit = fixed_epochs or args.epochs

        for epoch in range(1, epoch_limit + 1):
            model.train()
            losses = []
            for x, padding, y, _ in train_loader:
                x = x.to(device, non_blocking=True)
                padding = padding.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available(),
                ):
                    logits = model(x, padding)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))

            if fixed_epochs is not None:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "validation_loss": np.nan,
                    }
                )
                continue

            validation_score = predict(
                model,
                validation_frame,
                source_store,
            )
            validation_y = validation_frame["label"].to_numpy(int)
            validation_loss = float(
                np.mean(
                    np.logaddexp(0.0, validation_score)
                    - validation_y * validation_score
                )
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

        if fixed_epochs is None:
            if best_state is None:
                raise RuntimeError("No valid early-stopping checkpoint.")
            model.load_state_dict(best_state)
        else:
            best_epoch = fixed_epochs
        return model, best_epoch, pd.DataFrame(history)

    clean = (
        source_index[source_index["condition"].eq("clean")]
        .drop_duplicates(["participant_id", "trial_id"])
        .reset_index(drop=True)
    )
    splitter = StratifiedGroupKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )

    prediction_parts = []
    history_parts = []
    fold_rows = []
    best_epochs: dict[str, list[int]] = {
        protocol: [] for protocol in protocols
    }

    for fold, (train_index, test_index) in enumerate(
        splitter.split(
            clean,
            clean["label"].to_numpy(int),
            groups=clean["participant_id"].astype(str).to_numpy(),
        )
    ):
        train_participants = set(
            clean.iloc[train_index]["participant_id"].astype(str)
        )
        test_participants = set(
            clean.iloc[test_index]["participant_id"].astype(str)
        )
        train_clean = clean[
            clean["participant_id"].astype(str).isin(train_participants)
        ]
        validation_trials = deterministic_validation_trials(
            train_clean,
            args.validation_fraction,
            args.seed + fold,
        )

        for protocol in protocols:
            print(
                f"outer={fold + 1}/{args.outer_folds} "
                f"protocol={protocol}",
                flush=True,
            )
            condition_mask = (
                source_index["condition"].eq("clean")
                if protocol == "clean_only"
                else source_index["condition"].isin(
                    [
                        "clean",
                        "rate10",
                        "noise50",
                        "random40",
                        "burst40",
                        "high_random",
                        "high_burst",
                    ]
                )
            )
            participant_mask = source_index["participant_id"].astype(str).isin(
                train_participants
            )
            train_rows = source_index[
                participant_mask
                & condition_mask
                & ~source_index["trial_id"].astype(str).isin(
                    validation_trials
                )
            ].copy()
            validation_rows = source_index[
                participant_mask
                & source_index["condition"].eq("clean")
                & source_index["trial_id"].astype(str).isin(
                    validation_trials
                )
            ].copy()

            model, best_epoch, history = train_model(
                train_rows,
                validation_rows,
                fixed_epochs=None,
                seed=args.seed + fold * 100,
            )
            best_epochs[protocol].append(best_epoch)
            history.insert(0, "outer_fold", fold)
            history.insert(1, "protocol", protocol)
            history_parts.append(history)

            validation_score = predict(
                model,
                validation_rows,
                source_store,
            )
            threshold, _ = optimize_macro_f1_threshold(
                validation_rows["label"].to_numpy(int),
                validation_score,
            )

            # Keep the early-stopped model for outer testing. The reporting
            # threshold was selected from this same model's source-only
            # validation scores, so its score scale remains consistent.
            for condition in (
                "clean",
                "rate10",
                "noise50",
                "random40",
                "burst40",
                "high_random",
                "high_burst",
            ):
                test_rows = source_index[
                    source_index["participant_id"].astype(str).isin(
                        test_participants
                    )
                    & source_index["condition"].eq(condition)
                ].copy()
                score = predict(model, test_rows, source_store)
                metrics = calculate_metrics(
                    test_rows["label"].to_numpy(int),
                    score,
                    threshold,
                )
                fold_rows.append(
                    {
                        "outer_fold": fold,
                        "protocol": protocol,
                        "condition": condition,
                        "threshold": threshold,
                        "best_epoch": best_epoch,
                        **metrics,
                    }
                )
                part = test_rows[
                    [
                        "participant_id",
                        "trial_id",
                        "label",
                        "condition",
                        "repeat",
                    ]
                ].copy()
                part["outer_fold"] = fold
                part["protocol"] = protocol
                part["score"] = score
                part["threshold"] = threshold
                part["prediction"] = (score >= threshold).astype(int)
                prediction_parts.append(part)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    history = pd.concat(history_parts, ignore_index=True)
    fold_metrics.to_csv(
        args.output_dir / "cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_parquet(
        args.output_dir / "cv_predictions.parquet",
        index=False,
    )
    history.to_csv(
        args.output_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_rows = []
    for (protocol, condition), group in predictions.groupby(
        ["protocol", "condition"],
        sort=True,
    ):
        y = group["label"].to_numpy(int)
        score = group["score"].to_numpy(float)
        prediction = group["prediction"].to_numpy(int)
        metrics = calculate_metrics(y, score, threshold=0.0)
        # Keep the actual fold-specific threshold classifications.
        metrics["accuracy"] = float(np.mean(prediction == y))
        from sklearn.metrics import balanced_accuracy_score, f1_score
        metrics["macro_f1"] = float(
            f1_score(y, prediction, average="macro", zero_division=0)
        )
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(y, prediction)
        )
        metrics["nr_prediction_rate"] = float(prediction.mean())
        pooled_rows.append(
            {
                "protocol": protocol,
                "condition": condition,
                "n_participants": int(
                    group["participant_id"].nunique()
                ),
                **metrics,
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(
        args.output_dir / "cv_pooled_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    transfer_rows = []
    transfer_predictions = []
    for protocol in protocols:
        median_epoch = int(round(np.median(best_epochs[protocol])))
        source_clean = clean.copy()
        validation_trials = deterministic_validation_trials(
            source_clean,
            args.validation_fraction,
            args.seed + 50_000,
        )
        condition_mask = (
            source_index["condition"].eq("clean")
            if protocol == "clean_only"
            else source_index["condition"].isin(
                [
                    "clean",
                    "rate10",
                    "noise50",
                    "random40",
                    "burst40",
                    "high_random",
                    "high_burst",
                ]
            )
        )
        source_train = source_index[
            condition_mask
            & ~source_index["trial_id"].astype(str).isin(
                validation_trials
            )
        ].copy()
        source_validation = source_index[
            source_index["condition"].eq("clean")
            & source_index["trial_id"].astype(str).isin(
                validation_trials
            )
        ].copy()

        model, _, history_final = train_model(
            source_train,
            source_validation,
            fixed_epochs=median_epoch,
            seed=args.seed + 60_000,
        )
        validation_score = predict(
            model,
            source_validation,
            source_store,
        )
        threshold, _ = optimize_macro_f1_threshold(
            source_validation["label"].to_numpy(int),
            validation_score,
        )
        source_reference = float(np.median(validation_score))

        target_rows = target_index[
            target_index["condition"].eq("clean")
        ].copy()
        raw_score = predict(model, target_rows, target_store)
        calibrated_score = participant_calibrate(
            raw_score,
            target_rows["participant_id"].astype(str).to_numpy(),
            source_reference,
        )

        for adaptation, score in (
            ("raw", raw_score),
            ("participant_calibrated", calibrated_score),
        ):
            metrics = calculate_metrics(
                target_rows["label"].to_numpy(int),
                score,
                threshold,
            )
            transfer_rows.append(
                {
                    "protocol": protocol,
                    "adaptation": adaptation,
                    "source_threshold": threshold,
                    "source_reference": source_reference,
                    "selected_epoch": median_epoch,
                    "n_participants": int(
                        target_rows["participant_id"].nunique()
                    ),
                    **metrics,
                }
            )
            part = target_rows[
                ["participant_id", "trial_id", "label"]
            ].copy()
            part["protocol"] = protocol
            part["adaptation"] = adaptation
            part["score"] = score
            part["threshold"] = threshold
            transfer_predictions.append(part)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "channel_names": CHANNEL_NAMES,
                "protocol": protocol,
                "threshold": threshold,
                "source_reference": source_reference,
                "selected_epoch": median_epoch,
                "architecture": {
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "ff_multiplier": args.ff_multiplier,
                    "dropout": args.dropout,
                },
            },
            args.output_dir / f"event_free_transformer_{protocol}.pt",
        )

    transfer_metrics = pd.DataFrame(transfer_rows)
    transfer_prediction_frame = pd.concat(
        transfer_predictions,
        ignore_index=True,
    )
    transfer_metrics.to_csv(
        args.output_dir / "transfer_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transfer_prediction_frame.to_parquet(
        args.output_dir / "transfer_predictions.parquet",
        index=False,
    )

    bootstrap_parts = []
    delta_parts = []
    for protocol in protocols:
        subset = transfer_prediction_frame[
            transfer_prediction_frame["protocol"].eq(protocol)
        ]
        wide = (
            subset.pivot_table(
                index=["participant_id", "trial_id", "label"],
                columns="adaptation",
                values="score",
                aggfunc="first",
            )
            .reset_index()
        )
        threshold = float(
            transfer_metrics[
                transfer_metrics["protocol"].eq(protocol)
            ]["source_threshold"].iloc[0]
        )
        replicates, deltas = bootstrap_target(
            wide,
            methods={
                "raw": "raw",
                "participant_calibrated": "participant_calibrated",
            },
            thresholds={
                "raw": threshold,
                "participant_calibrated": threshold,
            },
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        replicates.insert(0, "protocol", protocol)
        deltas.insert(0, "protocol", protocol)
        bootstrap_parts.append(replicates)
        delta_parts.append(deltas)

    pd.concat(bootstrap_parts, ignore_index=True).to_parquet(
        args.output_dir / "transfer_bootstrap_replicates.parquet",
        index=False,
    )
    pd.concat(delta_parts, ignore_index=True).to_csv(
        args.output_dir / "transfer_bootstrap_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    clean_table = pooled[pooled["condition"].eq("clean")]
    degraded_table = (
        pooled[~pooled["condition"].eq("clean")]
        .groupby("protocol", as_index=False)
        .agg(
            mean_degraded_accuracy=("accuracy", "mean"),
            mean_degraded_macro_f1=("macro_f1", "mean"),
            mean_degraded_roc_auc=("roc_auc", "mean"),
            worst_degraded_macro_f1=("macro_f1", "min"),
            worst_degraded_roc_auc=("roc_auc", "min"),
        )
    )

    summary = f"""# OneStop Event-Free Sample-Sequence Transformer

## Scope

This model consumes event-free 30 Hz gaze/AOI sample tokens. It does not use
fixation IDs, fixation durations, saccade events, text tokens, RoBERTa
embeddings, or linguistic features.

Accuracy is reported for compatibility with prior OneStop work. The primary
metrics remain Macro F1 for thresholded source/robustness evaluation and
ROC-AUC for frozen cross-dataset transfer.

## Device

`{device}`

## OneStop clean-view results

{clean_table.to_markdown(index=False)}

## OneStop degradation summary

{degraded_table.to_markdown(index=False)}

## Frozen OneStop-to-WebQAmGaze English transfer

{transfer_metrics.to_markdown(index=False)}

## Interpretation boundary

- English WebQAmGaze is a development target, not a new confirmation target.
- Higher OneStop accuracy alone is not sufficient evidence of successful transfer.
- The long-sequence policy uniformly samples across the full trial when the
  raw sequence exceeds the cache maximum length.
- QCN-PC is not applied because exact additive feature-group decomposition is
  unavailable for this nonlinear sequence model.
"""
    (args.output_dir / "EVENT_FREE_TRANSFORMER_SUMMARY.md").write_text(
        summary,
        encoding="utf-8",
    )

    manifest = {
        "artifact_version": "onestop_event_free_transformer_v1",
        "source_cache": str(args.source_cache.resolve()),
        "target_cache": str(args.target_cache.resolve()),
        "protocols": protocols,
        "outer_folds": args.outer_folds,
        "metrics_policy": {
            "primary_source_thresholded": "macro_f1",
            "primary_transfer": "roc_auc",
            "reported_for_prior_work_compatibility": "accuracy",
        },
        "architecture": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "ff_multiplier": args.ff_multiplier,
            "dropout": args.dropout,
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "device": str(device),
        "target_status": "English WebQAmGaze development target",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_hash_manifest(args.output_dir)

    print(f"Completed: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
