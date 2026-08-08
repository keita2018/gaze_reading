#!/usr/bin/env python
"""Reconstruct an ACL-2025-compatible OneStop split protocol on the ASC344 subset.

This is a protocol reconstruction, not the authors' exact released split assignment.
It follows the published structure:
- 10 folds.
- split separately within each 10-article / ~120-participant batch.
- article-level item partitioning.
- participant partitioning stratified by reading goal.
- 80% train / 10% validation / 10% test along both item and participant axes.
- train = train-participants x train-articles (about 64%).
- validation = remaining non-test cells (about 17%).
- test:
    New Item             ~= 9%
    New Participant      ~= 9%
    New Item&Participant ~= 1%

The three batch-specific assignments are then unioned into a full split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 20260808
SPLITS = (
    "train",
    "validation",
    "test_new_item",
    "test_new_participant",
    "test_new_item_participant",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/interim/onestop/onestop_regular_trial_manifest.csv"
        ),
    )
    parser.add_argument(
        "--sequence-index",
        type=Path,
        default=Path(
            "data/processed/onestop/event_free_sequence_cache_v1/"
            "sequence_index.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/onestop_acl2025_compatible_splits_v1"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(value)


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def balanced_participant_fold_map(
    participants: pd.DataFrame,
    seed: int,
    batch_value: str,
) -> dict[str, int]:
    """Assign each participant to exactly one of 10 held-out folds.

    The allocation is independently shuffled within the two reading-goal labels,
    then distributed round-robin so each test fold is approximately balanced.
    """
    result: dict[str, int] = {}
    for label, group in participants.groupby("label", sort=True):
        values = sorted(group["participant_id"].astype(str).unique())
        rng = stable_rng(seed, "participant", batch_value, int(label))
        values = list(np.asarray(values, dtype=object)[rng.permutation(len(values))])
        for position, participant in enumerate(values):
            result[str(participant)] = int(position % 10)
    return result


def article_fold_map(
    article_ids: list[str],
    seed: int,
    batch_value: str,
) -> dict[str, int]:
    if len(article_ids) != 10:
        raise RuntimeError(
            f"Batch {batch_value!r}: expected 10 articles, found {len(article_ids)}."
        )
    rng = stable_rng(seed, "article", batch_value)
    shuffled = list(
        np.asarray(sorted(article_ids), dtype=object)[
            rng.permutation(len(article_ids))
        ]
    )
    return {str(article): fold for fold, article in enumerate(shuffled)}


def classify_cell(participant_role: str, article_role: str) -> str:
    if participant_role == "test" and article_role == "test":
        return "test_new_item_participant"
    if participant_role == "test":
        return "test_new_participant"
    if article_role == "test":
        return "test_new_item"
    if participant_role == "train" and article_role == "train":
        return "train"
    return "validation"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    index = pd.read_parquet(args.sequence_index)
    index = index[index["condition"].astype(str).eq("clean")].copy()

    required_manifest = {
        "onestop_trial_id",
        "participant_id",
        "label",
        "article_batch",
        "article_id",
        "paragraph_id",
        "difficulty_level",
    }
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise RuntimeError(f"Manifest lacks required columns: {missing}")
    required_index = {"participant_id", "trial_id", "label"}
    missing_index = sorted(required_index - set(index.columns))
    if missing_index:
        raise RuntimeError(f"Sequence index lacks required columns: {missing_index}")

    for column in (
        "onestop_trial_id",
        "participant_id",
        "article_batch",
        "article_id",
    ):
        manifest[column] = normalize_text(manifest[column])
    index["participant_id"] = normalize_text(index["participant_id"])
    index["trial_id"] = normalize_text(index["trial_id"])

    available_trial_ids = set(index["trial_id"])
    work = manifest[
        manifest["onestop_trial_id"].isin(available_trial_ids)
    ].copy()
    if len(work) != len(index):
        raise RuntimeError(
            "Manifest/cache alignment mismatch: "
            f"manifest aligned rows={len(work)}, clean cache rows={len(index)}"
        )
    work = work.rename(columns={"onestop_trial_id": "trial_id"})
    work["label"] = pd.to_numeric(work["label"], errors="raise").astype(int)

    # article_id is only locally unique within an article_batch in OneStop.
    # The ACL-style item split is constructed independently inside each batch,
    # so leakage audits must use the composite batch+article identity.
    work["article_item_key"] = (
        work["article_batch"].astype(str)
        + "::"
        + work["article_id"].astype(str)
    )

    label_check = (
        work[["participant_id", "label"]]
        .drop_duplicates()
        .groupby("participant_id")["label"]
        .nunique()
    )
    if int(label_check.max()) != 1:
        raise RuntimeError("A participant has multiple reading-goal labels.")

    batch_check = (
        work[["participant_id", "article_batch"]]
        .drop_duplicates()
        .groupby("participant_id")["article_batch"]
        .nunique()
    )
    if int(batch_check.max()) != 1:
        raise RuntimeError("A participant appears in multiple article batches.")

    batch_values = sorted(work["article_batch"].unique())
    if len(batch_values) != 3:
        raise RuntimeError(
            f"Expected 3 OneStop article batches, found {batch_values}"
        )

    participant_fold_by_batch: dict[str, dict[str, int]] = {}
    article_fold_by_batch: dict[str, dict[str, int]] = {}
    partition_definition_rows: list[dict[str, Any]] = []

    for batch in batch_values:
        batch_frame = work[work["article_batch"].eq(batch)]
        participant_table = (
            batch_frame[["participant_id", "label"]]
            .drop_duplicates()
            .sort_values("participant_id")
        )
        participant_map = balanced_participant_fold_map(
            participant_table,
            args.seed,
            batch,
        )
        participant_fold_by_batch[batch] = participant_map

        articles = sorted(batch_frame["article_id"].unique())
        article_map = article_fold_map(articles, args.seed, batch)
        article_fold_by_batch[batch] = article_map

        for participant, fold in sorted(participant_map.items()):
            label = int(
                participant_table.loc[
                    participant_table["participant_id"].eq(participant),
                    "label",
                ].iloc[0]
            )
            partition_definition_rows.append(
                {
                    "entity_type": "participant",
                    "article_batch": batch,
                    "entity_id": participant,
                    "label_if_participant": label,
                    "test_fold": fold,
                    "validation_fold": (fold - 1) % 10,
                }
            )
        for article, fold in sorted(article_map.items()):
            partition_definition_rows.append(
                {
                    "entity_type": "article",
                    "article_batch": batch,
                    "entity_id": article,
                    "label_if_participant": np.nan,
                    "test_fold": fold,
                    "validation_fold": (fold - 1) % 10,
                }
            )

    assignment_parts: list[pd.DataFrame] = []
    for fold in range(10):
        fold_frame = work.copy()
        split_values: list[str] = []
        participant_roles: list[str] = []
        article_roles: list[str] = []

        for row in fold_frame.itertuples(index=False):
            batch = str(row.article_batch)
            participant_test_fold = participant_fold_by_batch[batch][
                str(row.participant_id)
            ]
            article_test_fold = article_fold_by_batch[batch][
                str(row.article_id)
            ]

            if participant_test_fold == fold:
                p_role = "test"
            elif participant_test_fold == (fold + 1) % 10:
                p_role = "validation"
            else:
                p_role = "train"

            if article_test_fold == fold:
                a_role = "test"
            elif article_test_fold == (fold + 1) % 10:
                a_role = "validation"
            else:
                a_role = "train"

            participant_roles.append(p_role)
            article_roles.append(a_role)
            split_values.append(classify_cell(p_role, a_role))

        fold_frame["fold"] = fold
        fold_frame["participant_role"] = participant_roles
        fold_frame["article_role"] = article_roles
        fold_frame["split"] = split_values
        assignment_parts.append(fold_frame)

    assignments = pd.concat(assignment_parts, ignore_index=True)

    # Strong leakage audit.
    audit_rows: list[dict[str, Any]] = []
    for fold in range(10):
        current = assignments[assignments["fold"].eq(fold)]
        train = current[current["split"].eq("train")]
        validation = current[current["split"].eq("validation")]
        new_item = current[current["split"].eq("test_new_item")]
        new_participant = current[
            current["split"].eq("test_new_participant")
        ]
        both = current[
            current["split"].eq("test_new_item_participant")
        ]

        train_articles = set(train["article_item_key"])
        validation_articles = set(validation["article_item_key"])
        train_participants = set(train["participant_id"])
        validation_participants = set(validation["participant_id"])

        checks = {
            # "Item" means an article inside a batch, not the raw article_id
            # token by itself. The same article_id value can recur in another
            # batch without denoting the same held-out item.
            "new_item_articles_unseen": len(
                train_articles & set(new_item["article_item_key"])
            ) == 0,
            "new_item_participants_seen": set(
                new_item["participant_id"]
            ).issubset(
                train_participants | validation_participants
            ),
            "new_participant_people_unseen": len(
                train_participants & set(new_participant["participant_id"])
            ) == 0,
            "new_participant_articles_seen": set(
                new_participant["article_item_key"]
            ).issubset(
                train_articles | validation_articles
            ),
            "both_people_unseen": len(
                train_participants & set(both["participant_id"])
            ) == 0,
            "both_articles_unseen": len(
                train_articles & set(both["article_item_key"])
            ) == 0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Fold {fold} leakage audit failed: {checks}")

        total = len(current)
        for split_name in SPLITS:
            subset = current[current["split"].eq(split_name)]
            audit_rows.append(
                {
                    "fold": fold,
                    "split": split_name,
                    "n_trials": int(len(subset)),
                    "trial_fraction": float(len(subset) / total),
                    "n_participants": int(subset["participant_id"].nunique()),
                    "n_articles": int(subset["article_item_key"].nunique()),
                    "n_paragraphs": int(subset["paragraph_id"].nunique()),
                    "nr_trials": int((subset["label"] == 1).sum()),
                    "is_trials": int((subset["label"] == 0).sum()),
                    **checks,
                }
            )

    audit = pd.DataFrame(audit_rows)
    expected = {
        "train": 0.64,
        "validation": 0.17,
        "test_new_item": 0.09,
        "test_new_participant": 0.09,
        "test_new_item_participant": 0.01,
    }
    summary = (
        audit.groupby("split", as_index=False)
        .agg(
            mean_trial_fraction=("trial_fraction", "mean"),
            min_trial_fraction=("trial_fraction", "min"),
            max_trial_fraction=("trial_fraction", "max"),
            mean_n_trials=("n_trials", "mean"),
        )
    )
    summary["published_target_fraction"] = summary["split"].map(expected)
    summary["mean_minus_target"] = (
        summary["mean_trial_fraction"] - summary["published_target_fraction"]
    )

    # Across ten folds, every available trial should have the published test
    # exposure pattern: 9x New Item, 9x New Participant, 1x Both is not per
    # trial; that statement applies to aggregate trial appearances. We instead
    # audit the actual count distribution over all trial-fold pairs.
    exposure = (
        assignments.groupby(["trial_id", "split"], as_index=False)
        .size()
        .pivot(index="trial_id", columns="split", values="size")
        .fillna(0)
        .reset_index()
    )

    assignments.to_parquet(
        args.output_dir / "split_assignments.parquet",
        index=False,
    )
    pd.DataFrame(partition_definition_rows).to_csv(
        args.output_dir / "partition_definition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        args.output_dir / "split_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "split_fraction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    exposure.to_csv(
        args.output_dir / "trial_exposure_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_out = {
        "artifact_version": "onestop_acl2025_compatible_splits_v1",
        "status": "protocol_reconstruction_not_exact_author_split",
        "seed": args.seed,
        "source_subset": {
            "participants": int(work["participant_id"].nunique()),
            "trials": int(len(work)),
            "article_batches": int(work["article_batch"].nunique()),
            "articles": int(work["article_item_key"].nunique()),
            "nr_trials": int((work["label"] == 1).sum()),
            "is_trials": int((work["label"] == 0).sum()),
        },
        "published_protocol_reconstructed": {
            "n_splits": 10,
            "per_batch_articles": 10,
            "participant_axis": "80% train / 10% validation / 10% test",
            "article_axis": "80% train / 10% validation / 10% test",
            "item_partition_level": "article within article_batch",
            "item_identity_key": "article_batch + article_id",
            "train_expected": "64%",
            "validation_expected": "17%",
            "new_item_expected": "9%",
            "new_participant_expected": "9%",
            "new_item_and_participant_expected": "1%",
            "batch_assignments_unioned": True,
        },
        "limitation": (
            "The published study used all 360 participants. This reconstruction "
            "uses the 344 participants with available ASC-derived event-free "
            "sample caches, so numerical results are protocol-compatible rather "
            "than an exact split-for-split reproduction."
        ),
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== ACL-2025-COMPATIBLE SPLIT AUDIT ===")
    print(json.dumps(manifest_out, ensure_ascii=False, indent=2))
    print("")
    print(summary.to_string(index=False))
    print("")
    print(f"Saved: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
