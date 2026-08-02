#!/usr/bin/env python
r"""
Diagnose a missing OneStop regular trial by searching one participant's ASC
blocks using both trial identifiers and stimulus/task metadata.

Example:
python src\data\diagnose_onestop_missing_trial.py ^
  --asc data\raw\onestop\asc_all\l35_264.asc.zip ^
  --manifest data\interim\onestop\onestop_regular_trial_manifest.csv ^
  --participant l35_264 ^
  --trial-index 53 ^
  --output-dir outputs\onestop_missing_trial_l35_264_t53
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


MSG_RE = re.compile(r"^MSG\s+(\d+)\s*(.*)$", re.IGNORECASE)
TRIALID_RE = re.compile(r"^TRIALID\s+(.+?)\s*$", re.IGNORECASE)
TRIAL_VAR_RE = re.compile(
    r"^!V\s+TRIAL_VAR\s+(?P<name>\S+)(?:\s+(?P<value>.*?))?\s*$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asc", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--trial-index", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def norm_bool(value: Any) -> str:
    text = norm(value).lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return "true"
    if text in {"0", "false", "no", "n", "f"}:
        return "false"
    return text


def choose_var(vars_: dict[str, str], names: list[str]) -> str:
    lowered = {key.lower(): value for key, value in vars_.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return norm(value)
    return ""


def choose_zip_member(archive: zipfile.ZipFile) -> str:
    members = [
        name
        for name in archive.namelist()
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


def iter_lines(path: Path) -> Iterator[str]:
    if path.name.lower().endswith(".zip"):
        archive = zipfile.ZipFile(path, "r")
        member = choose_zip_member(archive)
        raw = archive.open(member, "r")
        wrapper = io.TextIOWrapper(
            raw, encoding="latin-1", errors="replace", newline=""
        )
        try:
            for line in wrapper:
                yield line.rstrip("\r\n")
        finally:
            wrapper.close()
            archive.close()
        return

    with path.open("r", encoding="latin-1", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def new_block(order: int, trialid_raw: str, timestamp: int) -> dict[str, Any]:
    return {
        "block_order": order,
        "trialid_raw": trialid_raw,
        "trialid_timestamp": timestamp,
        "vars": {},
        "messages": [],
        "paragraph_onsets": [],
        "paragraph_offsets": [],
        "sample_count_total": 0,
        "paragraph_sample_count": 0,
    }


def parse_blocks(path: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    order = 0

    for raw_line in iter_lines(path):
        line = raw_line.strip()
        if not line:
            continue

        first = line.split(None, 1)[0]
        if first.isdigit():
            if current is not None:
                timestamp = int(first)
                current["sample_count_total"] += 1
                if current["paragraph_onsets"]:
                    onset = current["paragraph_onsets"][0]
                    offset = (
                        current["paragraph_offsets"][0]
                        if current["paragraph_offsets"]
                        else None
                    )
                    if timestamp >= onset and (offset is None or timestamp < offset):
                        current["paragraph_sample_count"] += 1
            continue

        if first.upper() != "MSG":
            continue

        msg_match = MSG_RE.match(line)
        if not msg_match:
            continue
        timestamp = int(msg_match.group(1))
        text = msg_match.group(2).strip()

        trialid_match = TRIALID_RE.match(text)
        if trialid_match:
            if current is not None:
                blocks.append(current)
            order += 1
            current = new_block(
                order=order,
                trialid_raw=trialid_match.group(1).strip(),
                timestamp=timestamp,
            )
            continue

        if current is None:
            continue

        var_match = TRIAL_VAR_RE.match(text)
        if var_match:
            name = var_match.group("name").strip()
            value = (var_match.group("value") or "").strip()
            current["vars"][name] = value
            continue

        current["messages"].append(
            {"timestamp": timestamp, "message": text}
        )
        action = text.split(None, 1)[0].upper() if text else ""
        if action == "NULL_ACTION_PARAG":
            current["paragraph_onsets"].append(timestamp)
        elif action == "UPDATE_PARAG_RT":
            current["paragraph_offsets"].append(timestamp)

    if current is not None:
        blocks.append(current)
    return blocks


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    participant = args.participant.strip().lower()
    trial_index = norm(args.trial_index)

    manifest = pd.read_csv(args.manifest, dtype="string", low_memory=False)
    manifest["participant_id"] = (
        manifest["participant_id"].astype("string").str.strip().str.lower()
    )
    manifest["trial_index_norm"] = manifest["trial_index"].map(norm)

    target_rows = manifest[
        manifest["participant_id"].eq(participant)
        & manifest["trial_index_norm"].eq(trial_index)
    ]
    if len(target_rows) != 1:
        raise RuntimeError(
            f"Expected one manifest row, found {len(target_rows)}."
        )
    target = target_rows.iloc[0]

    target_values = {
        "article_batch": norm(target.get("article_batch")),
        "article_id": norm(target.get("article_id")),
        "paragraph_id": norm(target.get("paragraph_id")),
        "difficulty_level": norm(target.get("difficulty_level")),
        "question_id": norm(target.get("onestopqa_question_id")),
        "question_preview": norm_bool(target.get("question_preview")),
    }

    blocks = parse_blocks(args.asc)
    rows: list[dict[str, Any]] = []

    for block in blocks:
        vars_ = block["vars"]
        values = {
            "trial": choose_var(vars_, ["trial"]),
            "trial_index_internal": choose_var(vars_, ["Trial_Index_"]),
            "trial_recycled": choose_var(vars_, ["Trial_Recycled_"]),
            "practice": choose_var(vars_, ["practice"]),
            "reread": choose_var(vars_, ["reread"]),
            "article_batch": choose_var(
                vars_, ["batch", "article_batch"]
            ),
            "article_id": choose_var(vars_, ["article_id", "article"]),
            "paragraph_id": choose_var(
                vars_, ["paragraph_id", "paragraph"]
            ),
            "difficulty_level": choose_var(
                vars_, ["level", "difficulty_level", "difficulty"]
            ),
            "question_id": choose_var(
                vars_, ["q_ind", "onestopqa_question_id", "question_id"]
            ),
            "question_preview": norm_bool(
                choose_var(vars_, ["has_preview", "question_preview"])
            ),
            "article_title": choose_var(vars_, ["article_title"]),
            "condition_symb": choose_var(vars_, ["condition_symb"]),
        }

        comparison_fields = [
            "article_batch",
            "article_id",
            "paragraph_id",
            "difficulty_level",
            "question_id",
            "question_preview",
        ]
        comparisons: dict[str, bool | None] = {}
        score = 0
        comparable = 0
        for field in comparison_fields:
            observed = norm_bool(values[field]) if field == "question_preview" else norm(values[field])
            expected = target_values[field]
            if observed == "" or expected == "":
                comparisons[field] = None
            else:
                same = observed.lower() == expected.lower()
                comparisons[field] = same
                comparable += 1
                score += int(same)

        onset = block["paragraph_onsets"][0] if block["paragraph_onsets"] else None
        offset = block["paragraph_offsets"][0] if block["paragraph_offsets"] else None

        rows.append(
            {
                "block_order": block["block_order"],
                "trialid_raw": block["trialid_raw"],
                **values,
                "paragraph_onset_timestamp": onset,
                "paragraph_offset_timestamp": offset,
                "paragraph_duration_ms": (
                    offset - onset
                    if onset is not None and offset is not None
                    else None
                ),
                "sample_count_total": block["sample_count_total"],
                "paragraph_sample_count": block["paragraph_sample_count"],
                "metadata_match_score": score,
                "metadata_fields_comparable": comparable,
                "exact_target_trial_number": values["trial"] == trial_index,
                "match_article_batch": comparisons["article_batch"],
                "match_article_id": comparisons["article_id"],
                "match_paragraph_id": comparisons["paragraph_id"],
                "match_difficulty_level": comparisons["difficulty_level"],
                "match_question_id": comparisons["question_id"],
                "match_question_preview": comparisons["question_preview"],
                "vars_json": json.dumps(vars_, ensure_ascii=False),
                "messages_json": json.dumps(
                    block["messages"], ensure_ascii=False
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame["has_complete_paragraph"] = (
        frame["paragraph_onset_timestamp"].notna()
        & frame["paragraph_offset_timestamp"].notna()
    )
    frame["rank_key"] = (
        frame["metadata_match_score"] * 100
        + frame["has_complete_paragraph"].astype(int) * 10
        + frame["exact_target_trial_number"].astype(int)
    )
    frame = frame.sort_values(
        ["rank_key", "block_order"],
        ascending=[False, True],
    )

    candidates_path = args.output_dir / "candidate_blocks.csv"
    target_path = args.output_dir / "target_manifest_row.json"
    excerpt_path = args.output_dir / "top_candidate_messages.txt"

    frame.drop(columns=["rank_key"]).to_csv(
        candidates_path, index=False, encoding="utf-8-sig"
    )
    target_path.write_text(
        json.dumps(
            {
                key: (None if pd.isna(value) else str(value))
                for key, value in target.to_dict().items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    excerpt_lines = [
        "=== TARGET MANIFEST ===",
        json.dumps(target_values, ensure_ascii=False, indent=2),
        "",
    ]
    for _, row in frame.head(10).iterrows():
        excerpt_lines.extend(
            [
                (
                    f"=== BLOCK {row['block_order']} | "
                    f"trial={row['trial']} | TRIALID={row['trialid_raw']} | "
                    f"score={row['metadata_match_score']}/"
                    f"{row['metadata_fields_comparable']} | "
                    f"complete_paragraph={row['has_complete_paragraph']} ==="
                ),
                f"article_batch={row['article_batch']}",
                f"article_id={row['article_id']}",
                f"paragraph_id={row['paragraph_id']}",
                f"difficulty_level={row['difficulty_level']}",
                f"question_id={row['question_id']}",
                f"question_preview={row['question_preview']}",
                f"practice={row['practice']}",
                f"reread={row['reread']}",
                f"recycled={row['trial_recycled']}",
                f"onset={row['paragraph_onset_timestamp']}",
                f"offset={row['paragraph_offset_timestamp']}",
                "",
            ]
        )
        messages = json.loads(row["messages_json"])
        for message in messages:
            excerpt_lines.append(
                f"MSG {message['timestamp']} {message['message']}"
            )
        excerpt_lines.append("")

    excerpt_path.write_text(
        "\n".join(excerpt_lines),
        encoding="utf-8",
    )

    display_columns = [
        "block_order",
        "trialid_raw",
        "trial",
        "trial_index_internal",
        "trial_recycled",
        "practice",
        "reread",
        "article_batch",
        "article_id",
        "paragraph_id",
        "difficulty_level",
        "question_id",
        "question_preview",
        "metadata_match_score",
        "metadata_fields_comparable",
        "exact_target_trial_number",
        "has_complete_paragraph",
        "paragraph_onset_timestamp",
        "paragraph_offset_timestamp",
        "paragraph_sample_count",
    ]

    print("\n=== TARGET MANIFEST METADATA ===")
    print(json.dumps(target_values, ensure_ascii=False, indent=2))

    print("\n=== TOP ASC BLOCK CANDIDATES ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 260
    ):
        print(frame[display_columns].head(20).to_string(index=False))

    print(f"\nSaved: {candidates_path}")
    print(f"Saved: {excerpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
