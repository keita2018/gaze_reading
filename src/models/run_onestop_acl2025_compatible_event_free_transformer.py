#!/usr/bin/env python
"""Evaluate the Event-Free 30 Hz sequence Transformer under ACL-2025-compatible splits."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

try:
    from event_free_sequence_common import CHANNEL_NAMES
except ImportError as exc:
    raise ImportError(
        "event_free_sequence_common.py must be present in src/data."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/processed/onestop/event_free_sequence_cache_v1"
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
            "outputs/onestop_acl2025_compatible_event_free_transformer_v1"
        ),
    )
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
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, score)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def metrics(
    y: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    prediction = (score >= 0.0).astype(int)
    return {
        "n_trials": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(
            f1_score(y, prediction, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc_nr": pr_auc(y, score),
        "average_precision_nr": float(
            average_precision_score(y, score)
        ),
        "nr_prediction_rate": float(prediction.mean()),
    }


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


class ShardStore:
    def __init__(self, root: Path, max_open: int = 4):
        self.root = root
        self.max_open = max_open
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, relative_path: str) -> dict[str, np.ndarray]:
        if relative_path in self.cache:
            value = self.cache.pop(relative_path)
            self.cache[relative_path] = value
            return value
        path = self.root / relative_path
        with np.load(path, allow_pickle=False) as data:
            value = {key: data[key] for key in data.files}
        self.cache[relative_path] = value
        while len(self.cache) > self.max_open:
            self.cache.popitem(last=False)
        return value


def main() -> int:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required.") from exc

    if args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    index = pd.read_parquet(args.cache / "sequence_index.parquet")
    index = index[index["condition"].astype(str).eq("clean")].copy()
    index["trial_id"] = index["trial_id"].astype(str).str.strip()
    index["participant_id"] = index["participant_id"].astype(str).str.strip()
    splits = pd.read_parquet(args.splits)
    splits["trial_id"] = splits["trial_id"].astype(str).str.strip()

    store = ShardStore(args.cache)

    class SequenceDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i: int):
            row = self.frame.iloc[i]
            shard = store.get(str(row["shard_path"]))
            local = int(row["local_index"])
            x = torch.from_numpy(shard["x"][local].astype(np.float32))
            length = int(shard["lengths"][local])
            padding = torch.arange(x.shape[0]) >= length
            y = torch.tensor(int(row["label"]), dtype=torch.float32)
            return x, padding, y, i

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

    sample_row = index.iloc[0]
    padded_length = store.get(str(sample_row["shard_path"]))["x"].shape[1]

    class EventFreeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(len(CHANNEL_NAMES), args.d_model)
            self.position = SinusoidalPosition(
                args.d_model,
                padded_length + 1,
            )
            self.cls = nn.Parameter(torch.zeros(1, 1, args.d_model))
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
            nn.init.normal_(self.cls, std=0.02)

        def forward(self, x, padding):
            token = self.projection(x)
            cls = self.cls.expand(x.shape[0], -1, -1)
            token = self.position(torch.cat([cls, token], dim=1))
            cls_pad = torch.zeros(
                (padding.shape[0], 1),
                dtype=torch.bool,
                device=padding.device,
            )
            full_padding = torch.cat([cls_pad, padding], dim=1)
            encoded = self.encoder(
                token,
                src_key_padding_mask=full_padding,
            )
            return self.head(
                self.norm(encoded[:, 0])
            ).squeeze(-1)

    def loader(frame: pd.DataFrame, shuffle: bool, seed: int):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            SequenceDataset(frame),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def predict(model, frame: pd.DataFrame) -> np.ndarray:
        model.eval()
        result = np.empty(len(frame), dtype=float)
        with torch.no_grad():
            for x, padding, _, row_index in loader(
                frame, False, args.seed
            ):
                x = x.to(device, non_blocking=True)
                padding = padding.to(device, non_blocking=True)
                score = model(x, padding).detach().cpu().numpy()
                result[row_index.numpy()] = score
        return result

    def train_model(
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        fold_seed: int,
    ):
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)
        model = EventFreeTransformer().to(device)
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
        train_loader = loader(train_frame, True, fold_seed)
        best_state = None
        best_accuracy = -1.0
        best_loss = math.inf
        best_epoch = 0
        stale = 0
        history_rows = []

        for epoch in range(1, args.epochs + 1):
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

            val_score = predict(model, validation_frame)
            val_y = validation_frame["label"].to_numpy(int)
            val_accuracy = float(
                accuracy_score(val_y, (val_score >= 0.0).astype(int))
            )
            val_loss = float(
                np.mean(
                    np.logaddexp(0.0, val_score)
                    - val_y * val_score
                )
            )
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                }
            )
            print(
                f"    epoch={epoch:02d} "
                f"train_loss={np.mean(losses):.5f} "
                f"val_loss={val_loss:.5f} "
                f"val_acc={val_accuracy:.4f}",
                flush=True,
            )

            improved = (
                val_accuracy > best_accuracy + 1e-8
                or (
                    abs(val_accuracy - best_accuracy) <= 1e-8
                    and val_loss < best_loss - 1e-5
                )
            )
            if improved:
                best_accuracy = val_accuracy
                best_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

        if best_state is None:
            raise RuntimeError("Training produced no checkpoint.")
        model.load_state_dict(best_state)
        return model, best_epoch, pd.DataFrame(history_rows)

    split_metric_rows = []
    history_parts = []
    prediction_parts = []

    for fold in range(10):
        print(f"\n=== ACL-compatible fold {fold + 1}/10 ===", flush=True)
        assignment = splits[splits["fold"].eq(fold)][["trial_id", "split"]]
        frame = index.merge(
            assignment,
            on="trial_id",
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != len(index):
            raise RuntimeError(
                f"Fold {fold}: cache/split alignment mismatch."
            )
        train = frame[frame["split"].eq("train")].copy()
        validation = frame[frame["split"].eq("validation")].copy()
        print(
            f"train={len(train):,} validation={len(validation):,}",
            flush=True,
        )

        model, best_epoch, history = train_model(
            train,
            validation,
            args.seed + fold * 100,
        )
        history.insert(0, "fold", fold)
        history_parts.append(history)

        torch.save(
            {
                "fold": fold,
                "state_dict": model.state_dict(),
                "channel_names": CHANNEL_NAMES,
                "best_epoch": best_epoch,
                "architecture": {
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "ff_multiplier": args.ff_multiplier,
                    "dropout": args.dropout,
                },
            },
            args.output_dir / f"event_free_transformer_fold_{fold:02d}.pt",
        )

        regime_predictions = []
        for regime in (
            "test_new_item",
            "test_new_participant",
            "test_new_item_participant",
        ):
            test = frame[frame["split"].eq(regime)].copy()
            score = predict(model, test)
            result = metrics(test["label"].to_numpy(int), score)
            split_metric_rows.append(
                {
                    "fold": fold,
                    "regime": regime,
                    "best_epoch": best_epoch,
                    **result,
                }
            )
            pred = test[["trial_id", "participant_id", "label"]].copy()
            pred["fold"] = fold
            pred["regime"] = regime
            pred["score"] = score
            pred["prediction"] = (score >= 0.0).astype(int)
            prediction_parts.append(pred)
            regime_predictions.append(pred)

        all_test = pd.concat(regime_predictions, ignore_index=True)
        all_result = metrics(
            all_test["label"].to_numpy(int),
            all_test["score"].to_numpy(float),
        )
        split_metric_rows.append(
            {
                "fold": fold,
                "regime": "all",
                "best_epoch": best_epoch,
                **all_result,
            }
        )

    split_metrics = pd.DataFrame(split_metric_rows)
    history = pd.concat(history_parts, ignore_index=True)
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
        row["best_epoch_median"] = float(group["best_epoch"].median())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    split_metrics.to_csv(
        args.output_dir / "split_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    history.to_csv(
        args.output_dir / "training_history.csv",
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
        "artifact_version": (
            "onestop_acl2025_compatible_event_free_transformer_v1"
        ),
        "model": "30 Hz Event-Free sample-sequence Transformer",
        "subset": "ASC344 clean trials",
        "split_status": (
            "protocol reconstruction, not exact author split assignment"
        ),
        "selection": (
            "best epoch separately per split using entire validation-set "
            "accuracy, with validation loss as tie-break"
        ),
        "decision_threshold": 0.0,
        "primary_for_prior_work_compatibility": "accuracy",
        "additional_metrics": [
            "macro_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc_nr",
            "average_precision_nr",
        ],
        "device": str(device),
        "architecture": {
            "channels": CHANNEL_NAMES,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "ff_multiplier": args.ff_multiplier,
            "dropout": args.dropout,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_text = (
        "# ACL-2025-Compatible Event-Free Transformer\n\n"
        "Protocol reconstruction on the ASC344 event-free subset. "
        "Accuracy is the primary compatibility metric; Macro F1 and "
        "threshold-free metrics are also reported.\n\n"
        + summary.to_markdown(index=False)
        + "\n"
    )
    (args.output_dir / "ACL2025_COMPAT_TRANSFORMER_SUMMARY.md").write_text(
        summary_text,
        encoding="utf-8",
    )

    print("\n=== EVENT-FREE TRANSFORMER ACL-2025-COMPATIBLE SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
