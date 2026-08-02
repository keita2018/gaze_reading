#!/usr/bin/env python
"""
Build the canonical OneStop first-reading trial manifest and word-AOI table.

Run from the gaze_reading_goal repository root:

python src\data\build_onestop_regular_manifest.py ^
  --input-dir data\raw\onestop ^
  --output-dir data\interim\onestop ^
  --audit-dir outputs\onestop_manifest_audit

The script:
1. Reads the two Paragraph Interest Area reports in chunks.
2. Keeps non-practice, non-repeated paragraph-reading trials.
3. Assigns labels compatible with the current study:
       ordinary reading (NR) = 1
       information seeking (IS) = 0
4. Writes a trial-level manifest.
5. Writes a compact word-rectangle table for later raw-gaze Soft-AOI extraction.
6. Audits trial uniqueness, class counts, participant overlap, stimulus counts,
   per-participant trial counts, and IA geometry validity.

No external labels or model predictions are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REGIMES = {
    "ordinary": {"label": 1, "label_name": "NR"},
    "information_seeking": {"label": 0, "label_name": "IS"},
}

ALIASES: dict[str, list[str]] = {
    "participant_id": [
        "participant_id",
        "PARTICIPANT_ID",
        "RECORDING_SESSION_LABEL",
        "RECORDING_SESSION",
    ],
    "list_number": ["list_number", "LIST_NUMBER"],
    "question_preview": ["question_preview", "QUESTION_PREVIEW"],
    "trial_index": ["trial_index", "TRIAL_INDEX"],
    "practice_trial": ["practice_trial", "PRACTICE_TRIAL"],
    "article_batch": ["article_batch", "ARTICLE_BATCH"],
    "article_id": ["article_id", "ARTICLE_ID"],
    "article_index": ["article_index", "ARTICLE_INDEX"],
    "paragraph_id": ["paragraph_id", "PARAGRAPH_ID"],
    "difficulty_level": ["difficulty_level", "DIFFICULTY_LEVEL"],
    "repeated_reading_trial": [
        "repeated_reading_trial",
        "REPEATED_READING_TRIAL",
    ],
    "onestopqa_question_id": [
        "onestopqa_question_id",
        "ONESTOPQA_QUESTION_ID",
    ],
    "article_title": ["article_title", "ARTICLE_TITLE"],
    "ia_id": ["IA_ID", "ia_id"],
    "ia_label": ["IA_LABEL", "ia_label"],
    "ia_left": ["IA_LEFT", "ia_left"],
    "ia_right": ["IA_RIGHT", "ia_right"],
    "ia_top": ["IA_TOP", "ia_top"],
    "ia_bottom": ["IA_BOTTOM", "ia_bottom"],
    "ia_center_x": ["IA_CENTER_X", "ia_center_x"],
    "ia_center_y": ["IA_CENTER_Y", "ia_center_y"],
    "word_length": ["word_length", "WORD_LENGTH"],
    "word_length_no_punctuation": [
        "word_length_no_punctuation",
        "WORD_LENGTH_NO_PUNCTUATION",
    ],
}

REQUIRED = [
    "participant_id",
    "trial_index",
    "practice_trial",
    "article_batch",
    "article_id",
    "paragraph_id",
    "difficulty_level",
    "repeated_reading_trial",
    "ia_id",
    "ia_label",
    "ia_left",
    "ia_right",
    "ia_top",
    "ia_bottom",
]

TRIAL_METADATA = [
    "participant_id",
    "list_number",
    "question_preview",
    "trial_index",
    "article_batch",
    "article_id",
    "article_index",
    "paragraph_id",
    "difficulty_level",
    "onestopqa_question_id",
    "article_title",
]

AOI_COLUMNS = [
    "onestop_trial_id",
    "regime",
    "label",
    "label_name",
    "participant_id",
    "trial_index",
    "stimulus_id",
    "task_context_id",
    "article_batch",
    "article_id",
    "paragraph_id",
    "difficulty_level",
    "onestopqa_question_id",
    "ia_id",
    "ia_label",
    "ia_left",
    "ia_right",
    "ia_top",
    "ia_bottom",
    "ia_center_x",
    "ia_center_y",
    "word_length",
    "word_length_no_punctuation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/onestop"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/onestop"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/onestop_manifest_audit"),
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def find_ia_file(root: Path, regime: str) -> Path:
    regime_pattern = re.compile(
        "information[_ -]?seeking" if regime == "information_seeking" else "ordinary",
        re.IGNORECASE,
    )
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        name = path.name.lower()
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            continue
        if "ia" not in name or "paragraph" not in name:
            continue
        if regime_pattern.search(str(path)):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one Paragraph IA report for {regime}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def detect_text_format(path: Path) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(64_000)
            try:
                delimiter = csv.Sniffer().sniff(
                    sample, delimiters="\t,;|"
                ).delimiter
            except csv.Error:
                delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
            return encoding, delimiter
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode {path}")


def first_existing(columns: Iterable[str], aliases: list[str]) -> str | None:
    available = set(columns)
    return next((alias for alias in aliases if alias in available), None)


def resolve_columns(columns: list[str]) -> dict[str, str | None]:
    return {
        canonical: first_existing(columns, aliases)
        for canonical, aliases in ALIASES.items()
    }


def parse_bool_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "t": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
            "f": False,
        }
    )
    return mapped.astype("boolean")


def canonicalize(
    chunk: pd.DataFrame,
    resolved: dict[str, str | None],
) -> pd.DataFrame:
    out = pd.DataFrame(index=chunk.index)
    for canonical, actual in resolved.items():
        if actual is not None and actual in chunk.columns:
            out[canonical] = chunk[actual]
        else:
            out[canonical] = pd.NA
    return out


def clean_string(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, ".": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def normalized_id_component(series: pd.Series) -> pd.Series:
    text = clean_string(series)
    numeric = pd.to_numeric(text, errors="coerce")
    integral_mask = numeric.notna() & np.isclose(numeric % 1, 0)
    result = text.copy()
    result.loc[integral_mask] = numeric.loc[integral_mask].astype("Int64").astype("string")
    return result.fillna("NA")


def build_ids(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    participant = clean_string(frame["participant_id"]).fillna("NA")
    trial = normalized_id_component(frame["trial_index"])
    batch = normalized_id_component(frame["article_batch"])
    article = normalized_id_component(frame["article_id"])
    paragraph = normalized_id_component(frame["paragraph_id"])
    difficulty = clean_string(frame["difficulty_level"]).fillna("NA")
    question = normalized_id_component(frame["onestopqa_question_id"])

    frame["participant_id"] = participant
    frame["trial_index"] = trial
    frame["article_batch"] = batch
    frame["article_id"] = article
    frame["paragraph_id"] = paragraph
    frame["difficulty_level"] = difficulty
    frame["onestopqa_question_id"] = question

    frame["regime"] = regime
    frame["label"] = REGIMES[regime]["label"]
    frame["label_name"] = REGIMES[regime]["label_name"]

    frame["onestop_trial_id"] = (
        regime + "::" + participant + "::trial_" + trial
    )
    frame["stimulus_id"] = (
        "batch_"
        + batch
        + "::article_"
        + article
        + "::paragraph_"
        + paragraph
        + "::"
        + difficulty
    )
    frame["task_context_id"] = (
        frame["stimulus_id"] + "::question_" + question
    )
    return frame


def numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for name in (
        "ia_id",
        "ia_left",
        "ia_right",
        "ia_top",
        "ia_bottom",
        "ia_center_x",
        "ia_center_y",
        "word_length",
        "word_length_no_punctuation",
    ):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "ERROR: pyarrow is required. Install with: pip install pyarrow",
            file=sys.stderr,
        )
        return 2

    manifest_parts: list[pd.DataFrame] = []
    parquet_path = args.output_dir / "onestop_regular_word_aoi.parquet"
    writer = None

    audit: dict[str, Any] = {
        "files": {},
        "labels": {"ordinary": 1, "information_seeking": 0},
    }

    total_regular_ia_rows = 0

    try:
        for regime in REGIMES:
            path = find_ia_file(args.input_dir, regime)
            encoding, delimiter = detect_text_format(path)
            header = pd.read_csv(
                path,
                sep=delimiter,
                nrows=0,
                encoding=encoding,
                low_memory=False,
            )
            source_columns = [str(c) for c in header.columns]
            resolved = resolve_columns(source_columns)

            missing_required = [
                name for name in REQUIRED if resolved.get(name) is None
            ]
            if missing_required:
                raise RuntimeError(
                    f"{path} is missing required columns: {missing_required}\n"
                    f"Resolved columns: {resolved}"
                )

            usecols = sorted(
                {actual for actual in resolved.values() if actual is not None}
            )

            file_rows = 0
            regular_rows = 0
            file_trial_ids: set[str] = set()
            geometry_invalid = 0
            geometry_missing = 0

            print(f"Processing {regime}: {path}")
            for chunk_index, chunk in enumerate(
                pd.read_csv(
                    path,
                    sep=delimiter,
                    encoding=encoding,
                    usecols=usecols,
                    dtype="string",
                    chunksize=args.chunksize,
                    low_memory=False,
                    na_values=["."],
                    keep_default_na=True,
                ),
                start=1,
            ):
                file_rows += len(chunk)
                frame = canonicalize(chunk, resolved)

                practice = parse_bool_series(frame["practice_trial"])
                repeated = parse_bool_series(frame["repeated_reading_trial"])
                regular_mask = practice.ne(True) & repeated.ne(True)
                frame = frame.loc[regular_mask].copy()
                regular_rows += len(frame)

                if frame.empty:
                    continue

                frame = build_ids(frame, regime)
                frame = numeric_columns(frame)
                frame["ia_label"] = clean_string(frame["ia_label"])

                geometry_missing += int(
                    frame[["ia_left", "ia_right", "ia_top", "ia_bottom"]]
                    .isna()
                    .any(axis=1)
                    .sum()
                )
                valid_geom = (
                    frame["ia_left"].notna()
                    & frame["ia_right"].notna()
                    & frame["ia_top"].notna()
                    & frame["ia_bottom"].notna()
                )
                geometry_invalid += int(
                    (
                        valid_geom
                        & (
                            (frame["ia_right"] <= frame["ia_left"])
                            | (frame["ia_bottom"] <= frame["ia_top"])
                        )
                    ).sum()
                )

                trial_cols = [
                    "onestop_trial_id",
                    "regime",
                    "label",
                    "label_name",
                    "stimulus_id",
                    "task_context_id",
                    *TRIAL_METADATA,
                ]
                trial_cols = list(dict.fromkeys(trial_cols))
                trial_part = frame[trial_cols].drop_duplicates()
                manifest_parts.append(trial_part)
                file_trial_ids.update(trial_part["onestop_trial_id"].astype(str))

                for column in AOI_COLUMNS:
                    if column not in frame.columns:
                        frame[column] = pd.NA
                aoi = frame[AOI_COLUMNS].copy()

                table = pa.Table.from_pandas(
                    aoi,
                    preserve_index=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        parquet_path,
                        table.schema,
                        compression="zstd",
                    )
                writer.write_table(table)

                total_regular_ia_rows += len(aoi)
                if chunk_index % 10 == 0:
                    print(
                        f"  chunks={chunk_index}, source_rows={file_rows:,}, "
                        f"regular_ia_rows={regular_rows:,}, "
                        f"trials_seen={len(file_trial_ids):,}"
                    )

            audit["files"][regime] = {
                "path": str(path),
                "encoding": encoding,
                "delimiter": repr(delimiter),
                "source_row_count": file_rows,
                "regular_ia_row_count": regular_rows,
                "regular_trial_count": len(file_trial_ids),
                "geometry_missing_rows": geometry_missing,
                "geometry_invalid_rows": geometry_invalid,
                "resolved_columns": resolved,
            }
    finally:
        if writer is not None:
            writer.close()

    if not manifest_parts:
        raise RuntimeError("No regular trial rows were collected.")

    manifest_raw = pd.concat(manifest_parts, ignore_index=True)
    metadata_columns = [
        column
        for column in manifest_raw.columns
        if column != "onestop_trial_id"
    ]

    conflict_rows = []
    for column in metadata_columns:
        counts = (
            manifest_raw.groupby("onestop_trial_id", observed=True)[column]
            .nunique(dropna=True)
        )
        bad = counts[counts > 1]
        for trial_id, count in bad.items():
            conflict_rows.append(
                {
                    "onestop_trial_id": trial_id,
                    "column": column,
                    "n_unique": int(count),
                }
            )

    conflicts = pd.DataFrame(conflict_rows)
    conflict_path = args.audit_dir / "onestop_manifest_metadata_conflicts.csv"
    conflicts.to_csv(conflict_path, index=False, encoding="utf-8-sig")

    manifest = (
        manifest_raw.sort_values(
            ["regime", "participant_id", "trial_index"],
            kind="stable",
        )
        .drop_duplicates("onestop_trial_id", keep="first")
        .reset_index(drop=True)
    )

    manifest_path = args.output_dir / "onestop_regular_trial_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    participant_counts = (
        manifest.groupby(
            ["regime", "label_name", "participant_id"],
            observed=True,
        )
        .size()
        .rename("n_trials")
        .reset_index()
    )
    participant_counts_path = (
        args.audit_dir / "onestop_trials_per_participant.csv"
    )
    participant_counts.to_csv(
        participant_counts_path,
        index=False,
        encoding="utf-8-sig",
    )

    regime_participants = {
        regime: set(
            manifest.loc[
                manifest["regime"].eq(regime), "participant_id"
            ].astype(str)
        )
        for regime in REGIMES
    }
    participant_overlap = sorted(
        regime_participants["ordinary"]
        & regime_participants["information_seeking"]
    )

    class_summary = (
        manifest.groupby(["regime", "label", "label_name"], observed=True)
        .agg(
            n_trials=("onestop_trial_id", "nunique"),
            n_participants=("participant_id", "nunique"),
            n_stimuli=("stimulus_id", "nunique"),
            n_task_contexts=("task_context_id", "nunique"),
        )
        .reset_index()
    )
    class_summary_path = args.audit_dir / "onestop_manifest_class_summary.csv"
    class_summary.to_csv(
        class_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    stimulus_label_counts = (
        manifest.groupby("stimulus_id", observed=True)["label_name"]
        .nunique()
    )
    task_context_label_counts = (
        manifest.groupby("task_context_id", observed=True)["label_name"]
        .nunique()
    )

    per_participant = participant_counts["n_trials"]
    audit.update(
        {
            "total_regular_trials": int(manifest["onestop_trial_id"].nunique()),
            "total_regular_ia_rows": int(total_regular_ia_rows),
            "total_participants": int(manifest["participant_id"].nunique()),
            "participant_overlap_between_regimes": participant_overlap,
            "participant_overlap_count": len(participant_overlap),
            "unique_stimuli": int(manifest["stimulus_id"].nunique()),
            "unique_task_contexts": int(manifest["task_context_id"].nunique()),
            "stimuli_seen_in_both_labels": int((stimulus_label_counts == 2).sum()),
            "task_contexts_seen_in_both_labels": int(
                (task_context_label_counts == 2).sum()
            ),
            "metadata_conflict_count": int(len(conflicts)),
            "trials_per_participant": {
                "min": int(per_participant.min()),
                "median": float(per_participant.median()),
                "max": int(per_participant.max()),
                "distribution": {
                    str(k): int(v)
                    for k, v in Counter(per_participant.astype(int)).items()
                },
            },
            "outputs": {
                "manifest": str(manifest_path),
                "word_aoi": str(parquet_path),
                "class_summary": str(class_summary_path),
                "participant_counts": str(participant_counts_path),
                "metadata_conflicts": str(conflict_path),
            },
        }
    )

    audit_path = args.audit_dir / "onestop_manifest_audit.json"
    audit_path.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            default=safe_json_value,
        ),
        encoding="utf-8",
    )

    print("\n=== ONESTOP REGULAR MANIFEST SUMMARY ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(class_summary.to_string(index=False))
    print(
        f"\nTotal trials: {audit['total_regular_trials']:,}\n"
        f"Participants: {audit['total_participants']:,}\n"
        f"Participant overlap between regimes: "
        f"{audit['participant_overlap_count']}\n"
        f"Unique stimulus versions: {audit['unique_stimuli']:,}\n"
        f"Stimuli observed in both labels: "
        f"{audit['stimuli_seen_in_both_labels']:,}\n"
        f"Unique task contexts: {audit['unique_task_contexts']:,}\n"
        f"Metadata conflicts: {audit['metadata_conflict_count']:,}\n"
        f"Trials/participant: min={audit['trials_per_participant']['min']}, "
        f"median={audit['trials_per_participant']['median']}, "
        f"max={audit['trials_per_participant']['max']}\n"
        f"Regular IA rows: {audit['total_regular_ia_rows']:,}"
    )
    print(f"\nSaved: {manifest_path}")
    print(f"Saved: {parquet_path}")
    print(f"Saved: {audit_path}")

    expected = {
        "ordinary": 9718,
        "information_seeking": 9720,
    }
    actual = {
        row["regime"]: int(row["n_trials"])
        for row in class_summary.to_dict("records")
    }
    if actual != expected:
        print(
            f"WARNING: expected class trial counts {expected}, got {actual}",
            file=sys.stderr,
        )
        return 4
    if len(conflicts) > 0:
        print(
            f"WARNING: found {len(conflicts)} metadata conflicts; "
            f"inspect {conflict_path}",
            file=sys.stderr,
        )
        return 5

    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
