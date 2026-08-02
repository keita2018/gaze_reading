#!/usr/bin/env python
"""Audit the WebQAmGaze participant/trial counting unit.

This diagnostic resolves the apparent inconsistency between statements such as
"131 participants", "1,349 trials", and "10 trials per participant".

It does not train, tune, modify, or overwrite any model result.  The audit reads
the Stage 2 primary-candidate trial manifest and reports trial counts at several
possible participant units:

- participant_id
- participant_id + set_name
- source_file
- participant_id + set_name + source_file

The output identifies reused participant IDs, incomplete records, duplicate
trial positions, and the exact unit for which the ten-trial design holds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_TRIALS_PER_RECORD = 10
EXPECTED_NR_PER_RECORD = 5
EXPECTED_IS_PER_RECORD = 5


def describe_counts(
    frame: pd.DataFrame,
    group_columns: list[str],
    unit_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            n_trials=("external_trial_id", "size"),
            n_unique_trials=("external_trial_id", "nunique"),
            n_unique_trial_indices=("trial_index", "nunique"),
            min_trial_index=("trial_index", "min"),
            max_trial_index=("trial_index", "max"),
            nr_trials=("task", lambda value: int((value == "NR").sum())),
            is_trials=("task", lambda value: int((value == "IS").sum())),
            n_set_names=("set_name", "nunique"),
            n_source_files=("source_file", "nunique"),
        )
        .reset_index()
    )
    grouped["complete_10_trial_design"] = (
        grouped["n_trials"].eq(EXPECTED_TRIALS_PER_RECORD)
        & grouped["n_unique_trials"].eq(EXPECTED_TRIALS_PER_RECORD)
        & grouped["n_unique_trial_indices"].eq(EXPECTED_TRIALS_PER_RECORD)
        & grouped["nr_trials"].eq(EXPECTED_NR_PER_RECORD)
        & grouped["is_trials"].eq(EXPECTED_IS_PER_RECORD)
    )
    grouped["trial_count_deviation_from_10"] = (
        grouped["n_trials"] - EXPECTED_TRIALS_PER_RECORD
    )

    distribution = (
        grouped["n_trials"]
        .value_counts()
        .sort_index()
        .rename_axis("n_trials")
        .reset_index(name="n_units")
    )
    summary = {
        "unit_name": unit_name,
        "group_columns": group_columns,
        "n_units": int(len(grouped)),
        "n_complete_10_trial_units": int(
            grouped["complete_10_trial_design"].sum()
        ),
        "n_incomplete_or_nonstandard_units": int(
            (~grouped["complete_10_trial_design"]).sum()
        ),
        "minimum_trials": int(grouped["n_trials"].min()),
        "median_trials": float(grouped["n_trials"].median()),
        "maximum_trials": int(grouped["n_trials"].max()),
        "trial_count_distribution": {
            str(int(row.n_trials)): int(row.n_units)
            for row in distribution.itertuples(index=False)
        },
    }
    return grouped, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial-manifest",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage2/"
            "trial_manifest_primary_candidates.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_participant_trial_count_audit"
        ),
    )
    args = parser.parse_args()

    path = args.trial_manifest.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    trials = pd.read_csv(path)
    required = {
        "external_trial_id",
        "participant_id",
        "set_name",
        "trial_index",
        "task",
        "label_nr",
        "source_file",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(
            f"Primary trial manifest lacks columns: {missing}"
        )

    for column in (
        "external_trial_id",
        "participant_id",
        "set_name",
        "source_file",
    ):
        trials[column] = trials[column].astype(str)
    trials["task"] = trials["task"].astype(str).str.upper()
    trials["trial_index"] = pd.to_numeric(
        trials["trial_index"], errors="raise"
    ).astype(int)
    trials["label_nr"] = pd.to_numeric(
        trials["label_nr"], errors="raise"
    ).astype(int)

    if trials["external_trial_id"].duplicated().any():
        duplicate_ids = trials[
            trials["external_trial_id"].duplicated(keep=False)
        ].sort_values("external_trial_id")
    else:
        duplicate_ids = trials.iloc[0:0].copy()

    expected_task = np.where(
        trials["trial_index"].to_numpy(dtype=int) < 5,
        "NR",
        "IS",
    )
    position_mismatch = trials[
        trials["task"].to_numpy() != expected_task
    ].copy()
    label_mismatch = trials[
        (
            trials["task"].map({"NR": 1, "IS": 0})
            != trials["label_nr"]
        )
    ].copy()

    units = {
        "participant_id": ["participant_id"],
        "participant_set": ["participant_id", "set_name"],
        "source_file": ["source_file"],
        "participant_set_source": [
            "participant_id",
            "set_name",
            "source_file",
        ],
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    unit_tables: dict[str, pd.DataFrame] = {}
    for unit_name, columns in units.items():
        table, summary = describe_counts(
            trials,
            columns,
            unit_name,
        )
        unit_tables[unit_name] = table
        summaries[unit_name] = summary
        table.to_csv(
            output_dir / f"counts_by_{unit_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    reused_participant_ids = (
        trials.groupby("participant_id")
        .agg(
            n_set_names=("set_name", "nunique"),
            n_source_files=("source_file", "nunique"),
            n_trials=("external_trial_id", "size"),
        )
        .reset_index()
    )
    reused_participant_ids = reused_participant_ids[
        (reused_participant_ids["n_set_names"] > 1)
        | (reused_participant_ids["n_source_files"] > 1)
    ].sort_values(
        ["n_trials", "participant_id"],
        ascending=[False, True],
    )

    duplicate_trial_positions = (
        trials.groupby(
            ["participant_id", "set_name", "source_file", "trial_index"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_rows")
    )
    duplicate_trial_positions = duplicate_trial_positions[
        duplicate_trial_positions["n_rows"] > 1
    ].sort_values(
        ["n_rows", "participant_id"],
        ascending=[False, True],
    )

    duplicate_ids.to_csv(
        output_dir / "duplicate_external_trial_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )
    position_mismatch.to_csv(
        output_dir / "trial_position_task_mismatches.csv",
        index=False,
        encoding="utf-8-sig",
    )
    label_mismatch.to_csv(
        output_dir / "task_label_mismatches.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reused_participant_ids.to_csv(
        output_dir / "reused_participant_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )
    duplicate_trial_positions.to_csv(
        output_dir / "duplicate_trial_positions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    complete_units = [
        name
        for name, summary in summaries.items()
        if summary["n_incomplete_or_nonstandard_units"] == 0
    ]
    ten_trial_best_unit = None
    for preferred in (
        "participant_set_source",
        "source_file",
        "participant_set",
        "participant_id",
    ):
        if preferred in complete_units:
            ten_trial_best_unit = preferred
            break

    overall = {
        "trial_manifest": str(path),
        "n_rows": int(len(trials)),
        "n_unique_external_trial_ids": int(
            trials["external_trial_id"].nunique()
        ),
        "n_unique_participant_ids": int(
            trials["participant_id"].nunique()
        ),
        "n_unique_participant_set_pairs": int(
            trials[
                ["participant_id", "set_name"]
            ].drop_duplicates().shape[0]
        ),
        "n_unique_source_files": int(
            trials["source_file"].nunique()
        ),
        "n_unique_participant_set_source_records": int(
            trials[
                ["participant_id", "set_name", "source_file"]
            ].drop_duplicates().shape[0]
        ),
        "task_counts": {
            str(task): int(count)
            for task, count in trials["task"].value_counts().items()
        },
        "n_duplicate_external_trial_id_rows": int(
            len(duplicate_ids)
        ),
        "n_position_task_mismatches": int(
            len(position_mismatch)
        ),
        "n_task_label_mismatches": int(
            len(label_mismatch)
        ),
        "n_reused_participant_ids": int(
            len(reused_participant_ids)
        ),
        "n_duplicate_trial_position_groups": int(
            len(duplicate_trial_positions)
        ),
        "unit_summaries": summaries,
        "unit_with_complete_ten_trial_design": ten_trial_best_unit,
    }

    (output_dir / "participant_trial_count_audit.json").write_text(
        json.dumps(
            overall,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE PARTICIPANT/TRIAL COUNT AUDIT ===")
    print(f"Manifest rows: {len(trials)}")
    print(
        "Unique external trial IDs: "
        f"{trials['external_trial_id'].nunique()}"
    )
    print(
        "Unique participant_id values: "
        f"{trials['participant_id'].nunique()}"
    )
    print(
        "Unique participant_id + set_name records: "
        f"{overall['n_unique_participant_set_pairs']}"
    )
    print(
        "Unique source files: "
        f"{trials['source_file'].nunique()}"
    )
    print(
        "Unique participant/set/source records: "
        f"{overall['n_unique_participant_set_source_records']}"
    )
    print(
        "Task counts: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(
                overall["task_counts"].items()
            )
        )
    )
    print(
        "Participant IDs reused across set/source records: "
        f"{len(reused_participant_ids)}"
    )

    print("\n=== TRIAL COUNT DISTRIBUTIONS BY UNIT ===")
    for unit_name in units:
        summary = summaries[unit_name]
        print(
            f"{unit_name}: n_units={summary['n_units']}, "
            f"complete_10={summary['n_complete_10_trial_units']}, "
            f"nonstandard={summary['n_incomplete_or_nonstandard_units']}, "
            f"distribution={summary['trial_count_distribution']}"
        )

    print("\n=== DESIGN INTEGRITY ===")
    print(
        "Duplicate external trial ID rows: "
        f"{len(duplicate_ids)}"
    )
    print(
        "Duplicate trial-position groups: "
        f"{len(duplicate_trial_positions)}"
    )
    print(
        "Position/task mismatches: "
        f"{len(position_mismatch)}"
    )
    print(
        "Task/label mismatches: "
        f"{len(label_mismatch)}"
    )
    print(
        "Unit with complete ten-trial design: "
        f"{ten_trial_best_unit}"
    )
    print("\nWritten:")
    for path_written in sorted(output_dir.iterdir()):
        print(f"  {path_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
