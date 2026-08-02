#!/usr/bin/env python
"""
Stage 1 audit for OneStop paragraph reports.

Usage (from the gaze_reading_goal repository root):
    python src/data/audit_onestop_reports.py --input-dir data/raw/onestop --output-dir outputs/onestop_audit

The script recursively finds the ordinary-reading and information-seeking
paragraph-level Interest Area / Fixation reports, inspects their schemas,
and creates JSON/CSV audit artifacts without loading entire large files at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REGIME_PATTERNS = {
    "ordinary": re.compile(r"ordinary", re.IGNORECASE),
    "information_seeking": re.compile(r"information[_ -]?seeking", re.IGNORECASE),
}

REPORT_PATTERNS = {
    "interest_area": re.compile(r"(^|[/\\])ia[_ -]?paragraph|ia[_ -]?paragraph", re.IGNORECASE),
    "fixation": re.compile(r"fixations?[_ -]?paragraph", re.IGNORECASE),
}

COLUMN_ALIASES = {
    "participant_id": ["participant_id", "RECORDING_SESSION_LABEL", "RECORDING_SESSION"],
    "list_number": ["list_number", "LIST_NUMBER"],
    "question_preview": ["question_preview", "QUESTION_PREVIEW"],
    "trial_index": ["trial_index", "TRIAL_INDEX"],
    "practice_trial": ["practice_trial", "PRACTICE_TRIAL"],
    "article_batch": ["article_batch", "ARTICLE_BATCH"],
    "article_id": ["article_id", "ARTICLE_ID"],
    "paragraph_id": ["paragraph_id", "PARAGRAPH_ID"],
    "difficulty_level": ["difficulty_level", "DIFFICULTY_LEVEL"],
    "repeated_reading_trial": ["repeated_reading_trial", "REPEATED_READING_TRIAL"],
    "onestopqa_question_id": ["onestopqa_question_id", "ONESTOPQA_QUESTION_ID"],
    "ia_id": ["IA_ID", "ia_id", "IA_LABEL", "ia_label"],
    "ia_left": ["IA_LEFT", "ia_left"],
    "ia_right": ["IA_RIGHT", "ia_right"],
    "ia_top": ["IA_TOP", "ia_top"],
    "ia_bottom": ["IA_BOTTOM", "ia_bottom"],
    "ia_center_x": ["IA_CENTER_X", "ia_center_x"],
    "ia_center_y": ["IA_CENTER_Y", "ia_center_y"],
    "fix_x": ["CURRENT_FIX_X", "current_fix_x", "FIX_X", "fix_x"],
    "fix_y": ["CURRENT_FIX_Y", "current_fix_y", "FIX_Y", "fix_y"],
    "fix_duration": [
        "CURRENT_FIX_DURATION",
        "current_fix_duration",
        "FIX_DURATION",
        "fix_duration",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/onestop"),
        help="Root folder containing extracted OneStop report CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/onestop_audit"),
        help="Folder for audit artifacts.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per pandas chunk.",
    )
    return parser.parse_args()


def classify_path(path: Path) -> tuple[str | None, str | None]:
    text = str(path)
    regime = next(
        (name for name, pattern in REGIME_PATTERNS.items() if pattern.search(text)),
        None,
    )
    report = next(
        (name for name, pattern in REPORT_PATTERNS.items() if pattern.search(text)),
        None,
    )
    return regime, report


def find_report_files(root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # ZIP files created on macOS can contain AppleDouble metadata such as
        # __MACOSX/._fixations_....csv. These are not CSV data files.
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            continue
        regime, report = classify_path(path)
        if regime is None or report is None:
            continue
        candidates.append(
            {
                "path": path,
                "regime": regime,
                "report_type": report,
                "size_bytes": path.stat().st_size,
            }
        )
    return sorted(candidates, key=lambda x: str(x["path"]).lower())


def detect_text_format(path: Path) -> tuple[str, str]:
    """Return (encoding, delimiter) for a report file."""
    sample = None
    selected_encoding = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(64_000)
            selected_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if sample is None or selected_encoding is None:
        raise UnicodeError(f"Could not decode text file: {path}")

    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
    except csv.Error:
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","

    return selected_encoding, delimiter


def normalized_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def first_existing(columns: Iterable[str], aliases: list[str]) -> str | None:
    column_set = set(columns)
    return next((alias for alias in aliases if alias in column_set), None)


def resolve_columns(columns: list[str]) -> dict[str, str | None]:
    return {
        canonical: first_existing(columns, aliases)
        for canonical, aliases in COLUMN_ALIASES.items()
    }


def safe_unique_update(counter: Counter[str], series: pd.Series) -> None:
    values = series.dropna().astype(str).str.strip()
    counter.update(values[values != ""].tolist())


def make_trial_key_frame(chunk: pd.DataFrame, resolved: dict[str, str | None]) -> pd.Series | None:
    key_names = [
        "participant_id",
        "trial_index",
        "article_batch",
        "article_id",
        "paragraph_id",
        "difficulty_level",
        "onestopqa_question_id",
    ]
    actual = [resolved[name] for name in key_names if resolved.get(name)]
    if not actual:
        return None
    return chunk[actual].astype("string").fillna("<NA>").agg("|".join, axis=1)


def audit_file(meta: dict[str, Any], chunksize: int) -> dict[str, Any]:
    path: Path = meta["path"]
    encoding, delimiter = detect_text_format(path)

    header = pd.read_csv(
        path,
        sep=delimiter,
        nrows=0,
        low_memory=False,
        encoding=encoding,
    )
    columns = [str(c) for c in header.columns]
    resolved = resolve_columns(columns)

    row_count = 0
    unique_values: dict[str, Counter[str]] = defaultdict(Counter)
    unique_trial_keys: set[str] = set()
    regular_first_trial_keys: set[str] = set()
    missing_counts = Counter()
    coordinate_nonmissing = Counter()

    usecols = sorted({c for c in resolved.values() if c is not None})
    if not usecols:
        usecols = None

    for chunk in pd.read_csv(
        path,
        sep=delimiter,
        chunksize=chunksize,
        low_memory=False,
        encoding=encoding,
        usecols=usecols,
    ):
        row_count += len(chunk)

        for canonical, actual in resolved.items():
            if actual is None or actual not in chunk.columns:
                continue
            missing_counts[canonical] += int(chunk[actual].isna().sum())
            if canonical in {
                "participant_id",
                "list_number",
                "question_preview",
                "article_batch",
                "article_id",
                "paragraph_id",
                "difficulty_level",
                "practice_trial",
                "repeated_reading_trial",
            }:
                safe_unique_update(unique_values[canonical], chunk[actual])

        key_series = make_trial_key_frame(chunk, resolved)
        if key_series is not None:
            unique_trial_keys.update(key_series.dropna().astype(str).tolist())

            mask = pd.Series(True, index=chunk.index)
            practice_col = resolved.get("practice_trial")
            repeated_col = resolved.get("repeated_reading_trial")
            if practice_col and practice_col in chunk:
                practice = chunk[practice_col].map(normalized_bool)
                mask &= practice.ne(True)
            if repeated_col and repeated_col in chunk:
                repeated = chunk[repeated_col].map(normalized_bool)
                mask &= repeated.ne(True)
            regular_keys = key_series[mask]
            regular_first_trial_keys.update(regular_keys.dropna().astype(str).tolist())

        for canonical in (
            "ia_left",
            "ia_right",
            "ia_top",
            "ia_bottom",
            "ia_center_x",
            "ia_center_y",
            "fix_x",
            "fix_y",
            "fix_duration",
        ):
            actual = resolved.get(canonical)
            if actual and actual in chunk:
                coordinate_nonmissing[canonical] += int(chunk[actual].notna().sum())

    result = {
        **{k: v for k, v in meta.items() if k != "path"},
        "path": str(path),
        "encoding": encoding,
        "delimiter": repr(delimiter),
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "resolved_columns": resolved,
        "unique_trial_count": len(unique_trial_keys),
        "regular_first_reading_trial_count": len(regular_first_trial_keys),
        "unique_value_counts": {
            name: len(counter) for name, counter in unique_values.items()
        },
        "top_values": {
            name: counter.most_common(20) for name, counter in unique_values.items()
        },
        "missing_counts": dict(missing_counts),
        "coordinate_nonmissing_counts": dict(coordinate_nonmissing),
    }
    return result


def write_summary_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for result in results:
        u = result["unique_value_counts"]
        rows.append(
            {
                "regime": result["regime"],
                "report_type": result["report_type"],
                "path": result["path"],
                "size_bytes": result["size_bytes"],
                "row_count": result["row_count"],
                "column_count": result["column_count"],
                "unique_participants": u.get("participant_id"),
                "unique_trials": result["unique_trial_count"],
                "regular_first_reading_trials": result[
                    "regular_first_reading_trial_count"
                ],
                "unique_articles": u.get("article_id"),
                "unique_paragraphs": u.get("paragraph_id"),
                "difficulty_levels": u.get("difficulty_level"),
                "has_ia_rectangle": all(
                    result["resolved_columns"].get(name)
                    for name in ("ia_left", "ia_right", "ia_top", "ia_bottom")
                ),
                "has_fix_xy": all(
                    result["resolved_columns"].get(name)
                    for name in ("fix_x", "fix_y")
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        print(f"ERROR: input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    files = find_report_files(args.input_dir)
    if not files:
        print(
            "ERROR: no ordinary/information-seeking Paragraph IA or fixation reports "
            f"were found under {args.input_dir}",
            file=sys.stderr,
        )
        print("Existing files:", file=sys.stderr)
        for path in sorted(args.input_dir.rglob("*"))[:100]:
            if path.is_file():
                print(f"  {path}", file=sys.stderr)
        return 3

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, meta in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Auditing {meta['path']}")
        results.append(audit_file(meta, args.chunksize))

    json_path = args.output_dir / "onestop_report_audit.json"
    csv_path = args.output_dir / "onestop_report_audit_summary.csv"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(results, csv_path)

    print("\n=== ONESTOP REPORT AUDIT SUMMARY ===")
    summary = pd.read_csv(csv_path)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(summary.to_string(index=False))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
