#!/usr/bin/env python
"""
Audit OneStop ASC trial blocks and paragraph-reading intervals.

Run from the gaze_reading_goal repository root:

python src\data\audit_onestop_trial_messages.py ^
  --input-dir data\raw\onestop\asc_samples ^
  --manifest data\interim\onestop\onestop_regular_trial_manifest.csv ^
  --output-dir outputs\onestop_trial_messages

The script reads ASC ZIP archives directly and:
- splits the recording into TRIALID blocks;
- parses !V TRIAL_VAR metadata;
- records START/END and paragraph-related action messages;
- identifies candidate paragraph onset/offset timestamps;
- compares parsed trial_index values with the canonical manifest;
- reports which metadata variables can distinguish practice, repeated, and
  regular first-reading trials.

This is an audit only. It does not yet write the full raw-gaze dataset.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


RAW_STEM_RE = re.compile(r"^(?P<stem>[lL]\d+_\d+)\.asc(?:\.zip)?$", re.IGNORECASE)
MSG_RE = re.compile(r"^MSG\s+(\d+)\s*(.*)$", re.IGNORECASE)
TRIALID_RE = re.compile(r"^TRIALID\s+(.+?)\s*$", re.IGNORECASE)
TRIAL_VAR_RE = re.compile(
    r"^!V\s+TRIAL_VAR\s+(?P<name>\S+)(?:\s+(?P<value>.*?))?\s*$",
    re.IGNORECASE,
)
TRIAL_RESULT_RE = re.compile(r"^TRIAL_RESULT(?:\s+(.*?))?\s*$", re.IGNORECASE)
GAZE_COORDS_RE = re.compile(
    r"^(?:!V\s+)?GAZE_COORDS\s+"
    r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

PARAGRAPH_ONSET_NAMES = {
    "NULL_ACTION_PARAG",
}
PARAGRAPH_OFFSET_NAMES = {
    "UPDATE_PARAG_RT",
}
IMPORTANT_ACTION_PREFIXES = (
    "NULL_ACTION_",
    "UPDATE_",
    "PREPARE_",
    "DISPLAY_",
)

MANIFEST_COLUMNS = [
    "onestop_trial_id",
    "participant_id",
    "regime",
    "label",
    "label_name",
    "trial_index",
    "question_preview",
    "article_batch",
    "article_id",
    "paragraph_id",
    "difficulty_level",
    "onestopqa_question_id",
    "stimulus_id",
    "task_context_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/onestop/asc_samples"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/onestop/onestop_regular_trial_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/onestop_trial_messages"),
    )
    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def parse_bool(value: Any) -> bool | None:
    text = normalize_scalar(value).lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def archive_stem(path: Path) -> str:
    match = RAW_STEM_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported ASC filename: {path}")
    return match.group("stem").lower()


def find_inputs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.lower().endswith(".asc.zip")
            or path.name.lower().endswith(".asc")
        )
    )


def choose_zip_member(zf: zipfile.ZipFile) -> str:
    members = [
        name
        for name in zf.namelist()
        if not name.endswith("/")
        and "__MACOSX" not in name
        and not Path(name).name.startswith("._")
        and name.lower().endswith(".asc")
    ]
    if len(members) != 1:
        raise RuntimeError(
            f"Expected exactly one ASC member, found {len(members)}: {members}"
        )
    return members[0]


def decode_stream(binary: io.BufferedReader) -> Iterator[str]:
    wrapper = io.TextIOWrapper(
        binary,
        encoding="latin-1",
        errors="replace",
        newline="",
    )
    try:
        for line in wrapper:
            yield line.rstrip("\r\n")
    finally:
        wrapper.detach()


def iter_lines(path: Path) -> tuple[str, Iterator[str]]:
    if path.name.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path, "r")
        member = choose_zip_member(zf)
        raw = zf.open(member, "r")

        def generator() -> Iterator[str]:
            try:
                yield from decode_stream(raw)
            finally:
                raw.close()
                zf.close()

        return member, generator()

    raw = path.open("rb")

    def generator() -> Iterator[str]:
        try:
            yield from decode_stream(raw)
        finally:
            raw.close()

    return path.name, generator()


def parse_msg(line: str) -> tuple[int, str] | None:
    match = MSG_RE.match(line.strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2).strip()


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype="string", low_memory=False)
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"Manifest missing required columns: {sorted(missing)}"
        )
    manifest = manifest[MANIFEST_COLUMNS].copy()
    manifest["participant_id"] = (
        manifest["participant_id"].astype("string").str.strip().str.lower()
    )
    manifest["trial_index_norm"] = manifest["trial_index"].map(normalize_scalar)
    return manifest


def empty_trial(stem: str, order: int, trialid: str, timestamp: int) -> dict[str, Any]:
    return {
        "stem": stem,
        "block_order": order,
        "trialid_raw": trialid,
        "trialid_timestamp": timestamp,
        "trial_result": None,
        "trial_result_timestamp": None,
        "start_timestamp": None,
        "end_timestamp": None,
        "paragraph_onset_timestamp": None,
        "paragraph_offset_timestamp": None,
        "paragraph_sample_count": 0,
        "paragraph_first_sample_timestamp": None,
        "paragraph_last_sample_timestamp": None,
        "gaze_coords": None,
        "trial_vars": {},
        "actions": [],
        "message_count": 1,
        "sample_count_total_block": 0,
    }


def action_name(text: str) -> str:
    return text.split(None, 1)[0].upper() if text else ""


def finalize_trial(
    trial: dict[str, Any] | None,
    out: list[dict[str, Any]],
) -> None:
    if trial is None:
        return
    out.append(trial)


def audit_file(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    stem = archive_stem(path)
    _, lines = iter_lines(path)

    trials: list[dict[str, Any]] = []
    trial_var_names: Counter[str] = Counter()
    current: dict[str, Any] | None = None
    block_order = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Raw sample lines begin with a numeric timestamp.
        first_token = stripped.split(None, 1)[0]
        if first_token.isdigit():
            if current is not None:
                timestamp = int(first_token)
                current["sample_count_total_block"] += 1
                onset = current["paragraph_onset_timestamp"]
                offset = current["paragraph_offset_timestamp"]
                if onset is not None and offset is None and timestamp >= onset:
                    current["paragraph_sample_count"] += 1
                    if current["paragraph_first_sample_timestamp"] is None:
                        current["paragraph_first_sample_timestamp"] = timestamp
                    current["paragraph_last_sample_timestamp"] = timestamp
            continue

        record_type = first_token.upper()

        if record_type == "MSG":
            parsed = parse_msg(stripped)
            if parsed is None:
                continue
            timestamp, text = parsed

            trialid_match = TRIALID_RE.match(text)
            if trialid_match:
                finalize_trial(current, trials)
                block_order += 1
                current = empty_trial(
                    stem=stem,
                    order=block_order,
                    trialid=trialid_match.group(1).strip(),
                    timestamp=timestamp,
                )
                continue

            if current is None:
                continue

            current["message_count"] += 1

            var_match = TRIAL_VAR_RE.match(text)
            if var_match:
                name = var_match.group("name").strip()
                value = (var_match.group("value") or "").strip()
                current["trial_vars"][name] = value
                trial_var_names[name] += 1
                continue

            result_match = TRIAL_RESULT_RE.match(text)
            if result_match:
                current["trial_result"] = (result_match.group(1) or "").strip()
                current["trial_result_timestamp"] = timestamp
                continue

            gaze_match = GAZE_COORDS_RE.match(text)
            if gaze_match:
                current["gaze_coords"] = tuple(
                    float(value) for value in gaze_match.groups()
                )
                continue

            name = action_name(text)
            if name in PARAGRAPH_ONSET_NAMES:
                if current["paragraph_onset_timestamp"] is None:
                    current["paragraph_onset_timestamp"] = timestamp
                current["actions"].append((timestamp, name, text))
                continue

            if name in PARAGRAPH_OFFSET_NAMES:
                if current["paragraph_offset_timestamp"] is None:
                    current["paragraph_offset_timestamp"] = timestamp
                current["actions"].append((timestamp, name, text))
                continue

            if name.startswith(IMPORTANT_ACTION_PREFIXES):
                current["actions"].append((timestamp, name, text))
                continue

        elif record_type == "START":
            if current is not None:
                tokens = stripped.split()
                if len(tokens) >= 2 and tokens[1].isdigit():
                    current["start_timestamp"] = int(tokens[1])
        elif record_type == "END":
            if current is not None:
                tokens = stripped.split()
                if len(tokens) >= 2 and tokens[1].isdigit():
                    current["end_timestamp"] = int(tokens[1])

    finalize_trial(current, trials)
    return trials, trial_var_names


def choose_trial_index(vars_: dict[str, str], trialid_raw: str) -> str:
    candidates = [
        "trial_index",
        "TRIAL_INDEX",
        "trial",
        "trial_id",
        "TRIAL_ID",
    ]
    for candidate in candidates:
        if candidate in vars_:
            normalized = normalize_scalar(vars_[candidate])
            if normalized:
                return normalized
    return normalize_scalar(trialid_raw)


def choose_var(vars_: dict[str, str], names: list[str]) -> str:
    lowered = {key.lower(): value for key, value in vars_.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return normalize_scalar(value)
    return ""


def trial_to_row(
    trial: dict[str, Any],
    manifest_for_stem: pd.DataFrame,
) -> dict[str, Any]:
    vars_ = trial["trial_vars"]
    parsed_trial_index = choose_trial_index(vars_, trial["trialid_raw"])

    manifest_matches = manifest_for_stem[
        manifest_for_stem["trial_index_norm"].eq(parsed_trial_index)
    ]
    manifest_match_count = len(manifest_matches)

    practice_from_asc = choose_var(
        vars_,
        ["practice_trial", "practice", "is_practice"],
    )
    repeated_from_asc = choose_var(
        vars_,
        ["repeated_reading_trial", "repeated_reading", "is_repeated"],
    )

    row: dict[str, Any] = {
        "stem": trial["stem"],
        "block_order": trial["block_order"],
        "trialid_raw": trial["trialid_raw"],
        "parsed_trial_index": parsed_trial_index,
        "trialid_timestamp": trial["trialid_timestamp"],
        "start_timestamp": trial["start_timestamp"],
        "end_timestamp": trial["end_timestamp"],
        "trial_result": trial["trial_result"],
        "trial_result_timestamp": trial["trial_result_timestamp"],
        "paragraph_onset_timestamp": trial["paragraph_onset_timestamp"],
        "paragraph_offset_timestamp": trial["paragraph_offset_timestamp"],
        "paragraph_duration_ms": (
            trial["paragraph_offset_timestamp"]
            - trial["paragraph_onset_timestamp"]
            if trial["paragraph_onset_timestamp"] is not None
            and trial["paragraph_offset_timestamp"] is not None
            else None
        ),
        "paragraph_sample_count": trial["paragraph_sample_count"],
        "paragraph_first_sample_timestamp": trial[
            "paragraph_first_sample_timestamp"
        ],
        "paragraph_last_sample_timestamp": trial[
            "paragraph_last_sample_timestamp"
        ],
        "sample_count_total_block": trial["sample_count_total_block"],
        "message_count": trial["message_count"],
        "gaze_coords": (
            json.dumps(trial["gaze_coords"])
            if trial["gaze_coords"] is not None
            else None
        ),
        "has_paragraph_onset": trial["paragraph_onset_timestamp"] is not None,
        "has_paragraph_offset": trial["paragraph_offset_timestamp"] is not None,
        "practice_from_asc": practice_from_asc,
        "repeated_from_asc": repeated_from_asc,
        "practice_bool_from_asc": parse_bool(practice_from_asc),
        "repeated_bool_from_asc": parse_bool(repeated_from_asc),
        "manifest_match_count": manifest_match_count,
        "asc_trial_vars_json": json.dumps(vars_, ensure_ascii=False),
        "action_sequence": " > ".join(
            action[1] for action in trial["actions"]
        ),
        "action_messages_json": json.dumps(
            [
                {"timestamp": ts, "name": name, "message": text}
                for ts, name, text in trial["actions"]
            ],
            ensure_ascii=False,
        ),
    }

    # Keep common trial variables explicitly visible.
    row["article_batch_from_asc"] = choose_var(
        vars_, ["article_batch"]
    )
    row["article_id_from_asc"] = choose_var(
        vars_, ["article_id", "article"]
    )
    row["paragraph_id_from_asc"] = choose_var(
        vars_, ["paragraph_id", "paragraph"]
    )
    row["difficulty_from_asc"] = choose_var(
        vars_, ["difficulty_level", "difficulty"]
    )
    row["question_preview_from_asc"] = choose_var(
        vars_, ["question_preview"]
    )
    row["participant_id_from_asc"] = choose_var(
        vars_, ["participant_id"]
    )

    if manifest_match_count == 1:
        match = manifest_matches.iloc[0]
        for column in MANIFEST_COLUMNS:
            row[f"manifest_{column}"] = match[column]
    else:
        for column in MANIFEST_COLUMNS:
            row[f"manifest_{column}"] = None

    return row


def compare_value(a: Any, b: Any) -> bool | None:
    a_norm = normalize_scalar(a)
    b_norm = normalize_scalar(b)
    if a_norm == "" or b_norm == "":
        return None
    return a_norm.lower() == b_norm.lower()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = find_inputs(args.input_dir)
    if not files:
        print(
            f"ERROR: no ASC files found under {args.input_dir}",
            file=sys.stderr,
        )
        return 2

    manifest = load_manifest(args.manifest)

    all_trials: list[dict[str, Any]] = []
    var_name_rows: list[dict[str, Any]] = []

    for index, path in enumerate(files, start=1):
        stem = archive_stem(path)
        print(f"[{index}/{len(files)}] Parsing trial messages: {path}")
        trials, var_names = audit_file(path)
        manifest_for_stem = manifest[
            manifest["participant_id"].eq(stem)
        ].copy()
        for trial in trials:
            all_trials.append(trial_to_row(trial, manifest_for_stem))
        for name, count in var_names.most_common():
            var_name_rows.append(
                {
                    "stem": stem,
                    "trial_var_name": name,
                    "count": count,
                }
            )

    trials_df = pd.DataFrame(all_trials)
    vars_df = pd.DataFrame(var_name_rows)

    if trials_df.empty:
        print("ERROR: no TRIALID blocks parsed.", file=sys.stderr)
        return 3

    # Agreement fields are intentionally tri-state: True / False / missing.
    agreement_specs = {
        "participant_id": (
            "participant_id_from_asc",
            "manifest_participant_id",
        ),
        "article_batch": (
            "article_batch_from_asc",
            "manifest_article_batch",
        ),
        "article_id": (
            "article_id_from_asc",
            "manifest_article_id",
        ),
        "paragraph_id": (
            "paragraph_id_from_asc",
            "manifest_paragraph_id",
        ),
        "difficulty_level": (
            "difficulty_from_asc",
            "manifest_difficulty_level",
        ),
        "question_preview": (
            "question_preview_from_asc",
            "manifest_question_preview",
        ),
    }
    for name, (asc_col, manifest_col) in agreement_specs.items():
        if asc_col not in trials_df.columns or manifest_col not in trials_df.columns:
            trials_df[f"agree_{name}"] = pd.NA
            continue
        trials_df[f"agree_{name}"] = [
            compare_value(a, b)
            for a, b in zip(trials_df[asc_col], trials_df[manifest_col])
        ]

    trials_path = args.output_dir / "onestop_asc_trial_blocks.csv"
    vars_path = args.output_dir / "onestop_asc_trial_var_names.csv"
    trials_df.to_csv(trials_path, index=False, encoding="utf-8-sig")
    vars_df.to_csv(vars_path, index=False, encoding="utf-8-sig")

    summaries: list[dict[str, Any]] = []
    for stem, frame in trials_df.groupby("stem", observed=True):
        paragraph = frame[
            frame["has_paragraph_onset"] & frame["has_paragraph_offset"]
        ]
        regular_matches = frame[frame["manifest_match_count"].eq(1)]
        unmatched_paragraph = paragraph[
            paragraph["manifest_match_count"].ne(1)
        ]
        matched_paragraph = paragraph[
            paragraph["manifest_match_count"].eq(1)
        ]
        duplicate_manifest_matches = frame[
            frame["manifest_match_count"].gt(1)
        ]

        summaries.append(
            {
                "stem": stem,
                "n_trial_blocks": int(len(frame)),
                "n_paragraph_blocks": int(len(paragraph)),
                "n_manifest_matched_blocks": int(len(regular_matches)),
                "n_manifest_matched_paragraph_blocks": int(
                    len(matched_paragraph)
                ),
                "n_unmatched_paragraph_blocks": int(
                    len(unmatched_paragraph)
                ),
                "n_duplicate_manifest_matches": int(
                    len(duplicate_manifest_matches)
                ),
                "n_paragraph_blocks_missing_offset": int(
                    (
                        frame["has_paragraph_onset"]
                        & ~frame["has_paragraph_offset"]
                    ).sum()
                ),
                "paragraph_duration_ms_median": (
                    float(paragraph["paragraph_duration_ms"].median())
                    if not paragraph.empty
                    else None
                ),
                "paragraph_sample_count_median": (
                    float(paragraph["paragraph_sample_count"].median())
                    if not paragraph.empty
                    else None
                ),
                "gaze_coords_values": sorted(
                    frame["gaze_coords"].dropna().unique().tolist()
                ),
                "practice_values_from_asc": sorted(
                    frame["practice_from_asc"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
                "repeated_values_from_asc": sorted(
                    frame["repeated_from_asc"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            }
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = args.output_dir / "onestop_trial_message_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Compact mismatch report.
    mismatch_columns = [
        "stem",
        "block_order",
        "trialid_raw",
        "parsed_trial_index",
        "has_paragraph_onset",
        "has_paragraph_offset",
        "manifest_match_count",
        "practice_from_asc",
        "repeated_from_asc",
        "article_batch_from_asc",
        "article_id_from_asc",
        "paragraph_id_from_asc",
        "difficulty_from_asc",
        "question_preview_from_asc",
        "action_sequence",
    ]
    mismatches = trials_df[
        (
            trials_df["has_paragraph_onset"]
            & trials_df["manifest_match_count"].ne(1)
        )
        | trials_df["manifest_match_count"].gt(1)
    ][mismatch_columns].copy()
    mismatch_path = args.output_dir / "onestop_trial_manifest_mismatches.csv"
    mismatches.to_csv(mismatch_path, index=False, encoding="utf-8-sig")

    audit = {
        "summary": summaries,
        "total_trial_blocks": int(len(trials_df)),
        "total_paragraph_blocks": int(
            (
                trials_df["has_paragraph_onset"]
                & trials_df["has_paragraph_offset"]
            ).sum()
        ),
        "total_manifest_matched_paragraph_blocks": int(
            (
                trials_df["has_paragraph_onset"]
                & trials_df["has_paragraph_offset"]
                & trials_df["manifest_match_count"].eq(1)
            ).sum()
        ),
        "trial_var_names": (
            vars_df.groupby("trial_var_name", observed=True)["count"]
            .sum()
            .sort_values(ascending=False)
            .to_dict()
            if not vars_df.empty
            else {}
        ),
        "outputs": {
            "trial_blocks": str(trials_path),
            "trial_var_names": str(vars_path),
            "summary": str(summary_path),
            "mismatches": str(mismatch_path),
        },
    }
    audit_path = args.output_dir / "onestop_trial_message_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ONESTOP TRIAL MESSAGE SUMMARY ===")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(summary_df.to_string(index=False))

    print("\n=== TOP TRIAL VARIABLES ===")
    if vars_df.empty:
        print("No !V TRIAL_VAR fields parsed.")
    else:
        top_vars = (
            vars_df.groupby("trial_var_name", observed=True)["count"]
            .sum()
            .sort_values(ascending=False)
            .head(80)
            .reset_index()
        )
        print(top_vars.to_string(index=False))

    print("\n=== PARAGRAPH BLOCKS NOT MATCHED TO REGULAR MANIFEST ===")
    if mismatches.empty:
        print("None")
    else:
        print(mismatches.head(50).to_string(index=False))

    print(f"\nSaved: {trials_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {audit_path}")

    expected_regular = int(
        manifest[
            manifest["participant_id"].isin(trials_df["stem"].unique())
        ]["onestop_trial_id"].nunique()
    )
    matched_regular = int(
        trials_df[
            trials_df["has_paragraph_onset"]
            & trials_df["has_paragraph_offset"]
            & trials_df["manifest_match_count"].eq(1)
        ]["manifest_onestop_trial_id"].nunique()
    )

    if matched_regular != expected_regular:
        print(
            f"WARNING: matched regular paragraph trials {matched_regular} "
            f"!= expected manifest trials {expected_regular}.",
            file=sys.stderr,
        )
        return 4

    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
