#!/usr/bin/env python
"""Build a sentence-index map between ZuCo continuous ET trials and official features.

The session-level ET export is ordered chronologically within each block.
The official ZuCo benchmark uses fixed sentence-index ranges for each block:

NR:
    block1 0..49
    block2 50..99
    block3 100..150
    block4 151..200
    block5 201..250
    block6 251..299
    block7 300..348

TSR:
    block1 0..44
    block2 45..116
    block3 117..170
    block4 171..235
    block5 236..289
    block6 290..349
    block7 350..389

Official feature keys have the form:
    SUBJECT_LABEL_IDX_FULLIDX

Examples:
    YAC_NR_0_0
    YAC_TSR_17_366

This script:
1. Assigns each raw ET trial its original task-level sentence index.
2. Loads one official .npy feature dictionary.
3. Maps official feature keys to raw ET trial IDs.
4. Writes the 739-trial full map, common subset, and raw-only subset.
5. Performs strict count and one-to-one validation.

No fixation/saccade information is added to the sample-level Parquet files.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BLOCK_RANGES = {
    "NR": {
        1: (0, 49),
        2: (50, 99),
        3: (100, 150),
        4: (151, 200),
        5: (201, 250),
        6: (251, 299),
        7: (300, 348),
    },
    "TSR": {
        1: (0, 44),
        2: (45, 116),
        3: (117, 170),
        4: (171, 235),
        5: (236, 289),
        6: (290, 349),
        7: (350, 389),
    },
}

EXPECTED_TASK_COUNTS = {
    "NR": 349,
    "TSR": 390,
}

FEATURE_KEY_RE = re.compile(
    r"^(?P<subject>[^_]+)_(?P<label>.+)_(?P<sentence_index>\d+)_"
    r"(?P<full_index>\d+)$"
)


def expected_block_count(task: str, block: int) -> int:
    start, end = BLOCK_RANGES[task][block]
    return end - start + 1


def add_original_sentence_indices(trials: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject_id",
        "task",
        "block_id",
        "trial_index_in_block",
        "trial_id",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Missing trial columns: {missing}")

    result = trials.copy()
    result["task"] = result["task"].astype(str).str.upper()
    result["block_id"] = result["block_id"].astype(int)
    result["trial_index_in_block"] = (
        result["trial_index_in_block"].astype(int)
    )

    sentence_indices = []
    expected_counts = []

    for row in result.itertuples(index=False):
        task = str(row.task).upper()
        block = int(row.block_id)
        trial_index = int(row.trial_index_in_block)

        if task not in BLOCK_RANGES:
            raise ValueError(f"Unexpected task: {task}")
        if block not in BLOCK_RANGES[task]:
            raise ValueError(f"Unexpected block: task={task}, block={block}")

        start, end = BLOCK_RANGES[task][block]
        count = expected_block_count(task, block)

        if trial_index < 0 or trial_index >= count:
            raise ValueError(
                f"Trial index outside official block range: "
                f"task={task}, block={block}, trial_index={trial_index}, "
                f"expected 0..{count - 1}"
            )

        sentence_index = start + trial_index
        if sentence_index > end:
            raise AssertionError("Computed sentence index exceeds block end.")

        sentence_indices.append(sentence_index)
        expected_counts.append(count)

    result["task_sentence_index"] = sentence_indices
    result["official_block_expected_trials"] = expected_counts

    # Strict block-count validation.
    actual_block_counts = (
        result.groupby(["task", "block_id"])
        .size()
        .to_dict()
    )
    for task, blocks in BLOCK_RANGES.items():
        for block in blocks:
            actual = int(actual_block_counts.get((task, block), 0))
            expected = expected_block_count(task, block)
            if actual != expected:
                raise ValueError(
                    f"Block count mismatch for {task}{block}: "
                    f"actual={actual}, expected={expected}"
                )

    # Strict task-count validation.
    actual_task_counts = result.groupby("task").size().to_dict()
    for task, expected in EXPECTED_TASK_COUNTS.items():
        actual = int(actual_task_counts.get(task, 0))
        if actual != expected:
            raise ValueError(
                f"Task count mismatch for {task}: "
                f"actual={actual}, expected={expected}"
            )

    key_columns = ["subject_id", "task", "task_sentence_index"]
    if result.duplicated(key_columns).any():
        duplicates = result.loc[
            result.duplicated(key_columns, keep=False),
            key_columns + ["trial_id"],
        ]
        raise ValueError(
            "Duplicate raw-trial sentence keys:\n"
            + duplicates.head(30).to_string(index=False)
        )

    return result


def infer_task_from_label(label: str) -> str:
    upper = label.upper()
    if upper == "NR" or upper.startswith("NR_"):
        return "NR"
    if upper == "TSR" or upper.startswith("TSR_"):
        return "TSR"
    raise ValueError(f"Cannot infer NR/TSR from feature label: {label!r}")


def load_feature_index(
    feature_path: Path,
    expected_subject: str,
) -> pd.DataFrame:
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)

    loaded = np.load(feature_path, allow_pickle=True)
    feature_dict = loaded.item()

    if not isinstance(feature_dict, dict):
        raise TypeError(
            f"Expected a dict in {feature_path}, got {type(feature_dict)}"
        )

    rows: list[dict[str, Any]] = []
    feature_lengths = set()

    for key, value in feature_dict.items():
        match = FEATURE_KEY_RE.match(str(key))
        if match is None:
            raise ValueError(f"Unexpected official feature key: {key!r}")

        subject = match.group("subject").upper()
        label = match.group("label")
        task = infer_task_from_label(label)
        sentence_index = int(match.group("sentence_index"))
        full_index = int(match.group("full_index"))

        if subject != expected_subject:
            raise ValueError(
                f"Feature file contains subject {subject}, "
                f"expected {expected_subject}"
            )

        values = list(value)
        if len(values) < 2:
            raise ValueError(
                f"Feature value must contain feature(s) and label: {key}"
            )

        numeric_features = values[:-1]
        stored_label = str(values[-1])
        feature_lengths.add(len(numeric_features))

        row: dict[str, Any] = {
            "subject_id": subject,
            "task": task,
            "task_sentence_index": sentence_index,
            "official_feature_key": str(key),
            "official_feature_label_in_key": label,
            "official_feature_stored_label": stored_label,
            "official_feature_full_index": full_index,
            "official_feature_dimension": len(numeric_features),
        }
        for index, feature_value in enumerate(numeric_features):
            try:
                row[f"official_feature_{index}"] = float(feature_value)
            except (TypeError, ValueError):
                row[f"official_feature_{index}"] = math.nan

        rows.append(row)

    features = pd.DataFrame(rows)

    if features.empty:
        raise ValueError(f"No feature rows found in {feature_path}")

    key_columns = ["subject_id", "task", "task_sentence_index"]
    if features.duplicated(key_columns).any():
        duplicates = features.loc[
            features.duplicated(key_columns, keep=False),
            key_columns + ["official_feature_key"],
        ]
        raise ValueError(
            "Duplicate official feature sentence keys:\n"
            + duplicates.head(30).to_string(index=False)
        )

    if len(feature_lengths) != 1:
        raise ValueError(
            f"Inconsistent official feature dimensions: {feature_lengths}"
        )

    return features.sort_values(
        ["task", "task_sentence_index"]
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials",
        default="data/interim/zuco2/YAC/YAC_all_trials.csv",
    )
    parser.add_argument(
        "--official-feature",
        default=(
            "external/zuco-benchmark/src/features/"
            "YAC_sent_gaze_sacc.npy"
        ),
    )
    parser.add_argument("--subject", default="YAC")
    parser.add_argument(
        "--output-dir",
        default="data/processed/zuco2/YAC",
    )
    parser.add_argument(
        "--expected-common-count",
        type=int,
        default=521,
    )
    args = parser.parse_args()

    subject = args.subject.upper()
    trials_path = Path(args.trials)
    feature_path = Path(args.official_feature)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(trials_path)
    trial_map = add_original_sentence_indices(trials)

    if set(trial_map["subject_id"].astype(str).str.upper()) != {subject}:
        raise ValueError(
            f"Trial file does not contain only subject {subject}."
        )
    trial_map["subject_id"] = subject

    features = load_feature_index(feature_path, subject)

    merged = trial_map.merge(
        features,
        how="left",
        on=["subject_id", "task", "task_sentence_index"],
        validate="one_to_one",
        indicator=True,
    )
    merged["has_official_feature"] = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])

    common = merged[merged["has_official_feature"]].copy()
    raw_only = merged[~merged["has_official_feature"]].copy()

    # Ensure every official feature found a raw ET trial.
    feature_keys = set(
        zip(
            features["subject_id"],
            features["task"],
            features["task_sentence_index"],
        )
    )
    raw_keys = set(
        zip(
            trial_map["subject_id"],
            trial_map["task"],
            trial_map["task_sentence_index"],
        )
    )
    unmatched_feature_keys = sorted(feature_keys - raw_keys)
    if unmatched_feature_keys:
        raise ValueError(
            "Official features without raw ET trial mapping: "
            f"{unmatched_feature_keys[:30]}"
        )

    if len(common) != args.expected_common_count:
        raise ValueError(
            f"Common-subset count mismatch: actual={len(common)}, "
            f"expected={args.expected_common_count}"
        )

    if len(common) != len(features):
        raise ValueError(
            f"Not all official features were mapped: "
            f"features={len(features)}, common={len(common)}"
        )

    # Compact task-level summary.
    summary_rows = []
    for task in ("NR", "TSR"):
        task_all = merged[merged["task"] == task]
        task_common = task_all[task_all["has_official_feature"]]
        task_raw_only = task_all[~task_all["has_official_feature"]]

        summary_rows.append(
            {
                "subject_id": subject,
                "task": task,
                "all_raw_trials": len(task_all),
                "common_official_trials": len(task_common),
                "raw_only_trials": len(task_raw_only),
                "common_rate": (
                    len(task_common) / len(task_all)
                    if len(task_all) else math.nan
                ),
                "median_valid_basic_rate_all": (
                    task_all["valid_basic_rate"].median()
                    if "valid_basic_rate" in task_all else math.nan
                ),
                "median_valid_basic_rate_common": (
                    task_common["valid_basic_rate"].median()
                    if "valid_basic_rate" in task_common else math.nan
                ),
                "median_valid_basic_rate_raw_only": (
                    task_raw_only["valid_basic_rate"].median()
                    if "valid_basic_rate" in task_raw_only else math.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    prefix = f"{subject}_sent_gaze_sacc"
    all_map_path = output_dir / f"{prefix}_trial_map_all.csv"
    common_path = output_dir / f"{prefix}_common_trials.csv"
    raw_only_path = output_dir / f"{prefix}_raw_only_trials.csv"
    feature_index_path = output_dir / f"{prefix}_official_feature_index.csv"
    summary_path = output_dir / f"{prefix}_mapping_summary.csv"

    sort_columns = ["task", "task_sentence_index"]
    merged.sort_values(sort_columns).to_csv(
        all_map_path,
        index=False,
        encoding="utf-8-sig",
    )
    common.sort_values(sort_columns).to_csv(
        common_path,
        index=False,
        encoding="utf-8-sig",
    )
    raw_only.sort_values(sort_columns).to_csv(
        raw_only_path,
        index=False,
        encoding="utf-8-sig",
    )
    features.to_csv(
        feature_index_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=== RAW/OFFICIAL TRIAL MAPPING SUMMARY ===")
    print(summary.to_string(index=False))
    print()
    print(f"all raw trials: {len(merged)}")
    print(f"official feature rows: {len(features)}")
    print(f"common trials: {len(common)}")
    print(f"raw-only trials: {len(raw_only)}")
    print(f"unique raw trial IDs: {merged['trial_id'].nunique()}")
    print(
        "unique common sentence keys: "
        f"{common[['task', 'task_sentence_index']].drop_duplicates().shape[0]}"
    )
    print("\nWritten:")
    print(f"  {all_map_path}")
    print(f"  {common_path}")
    print(f"  {raw_only_path}")
    print(f"  {feature_index_path}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
