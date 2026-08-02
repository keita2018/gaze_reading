#!/usr/bin/env python
"""
Audit the OneStop OSF raw-file inventory against the canonical trial manifest.

Run from the gaze_reading_goal repository root:

python src\data\audit_onestop_raw_inventory.py ^
  --osf-list logs\onestop_raw_files.txt ^
  --manifest data\interim\onestop\onestop_regular_trial_manifest.csv ^
  --output-dir outputs\onestop_raw_inventory

The script does not download files. It:
- parses ASC/EDF ZIP paths from the saved OSF listing;
- deduplicates repeated OSF list entries;
- maps raw filenames l<list_number>_<participant_id> to manifest participants;
- reports ASC/EDF coverage by reading regime;
- identifies ASC-only, EDF-only, both, and missing participants;
- writes exact unique download manifests;
- selects one ASC sample from each regime for parser development.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_RE = re.compile(
    r"(?P<path>osfstorage[\\/](?P<kind>asc|edf)[\\/]"
    r"(?P<stem>[lL]\d+_\d+)\.(?P=kind)\.zip)",
    re.IGNORECASE,
)

STEM_RE = re.compile(r"^[lL](?P<list_number>\d+)_(?P<participant_id>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--osf-list",
        type=Path,
        default=Path("logs/onestop_raw_files.txt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/interim/onestop/onestop_regular_trial_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/onestop_raw_inventory"),
    )
    return parser.parse_args()


def normalize_integer(value: Any) -> str:
    if pd.isna(value):
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


def canonical_osf_path(path: str) -> str:
    return path.replace("\\", "/")


def parse_inventory(text: str) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for match in RAW_RE.finditer(text):
        path = canonical_osf_path(match.group("path"))
        kind = match.group("kind").lower()
        stem = match.group("stem").lower()
        stem_match = STEM_RE.match(stem)
        if stem_match is None:
            continue
        records.append(
            {
                "osf_path": path,
                "kind": kind,
                "stem": stem,
                "list_number": stem_match.group("list_number"),
                "participant_id": stem_match.group("participant_id"),
            }
        )
    return pd.DataFrame(records)


def load_participants(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype="string", low_memory=False)
    required = {"participant_id", "list_number", "regime", "label", "label_name"}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    participants = manifest[
        ["participant_id", "list_number", "regime", "label", "label_name"]
    ].copy()
    participants["participant_id"] = participants["participant_id"].map(
        normalize_integer
    )
    participants["list_number"] = participants["list_number"].map(
        normalize_integer
    )
    participants = participants.drop_duplicates().reset_index(drop=True)

    conflicts = (
        participants.groupby("participant_id", observed=True)
        .agg(
            n_list_numbers=("list_number", "nunique"),
            n_regimes=("regime", "nunique"),
        )
        .reset_index()
    )
    bad = conflicts[
        (conflicts["n_list_numbers"] != 1) | (conflicts["n_regimes"] != 1)
    ]
    if not bad.empty:
        raise RuntimeError(
            "Participant IDs do not map uniquely to list_number/regime:\n"
            + bad.to_string(index=False)
        )

    # In the OneStop reports, participant_id is typically already the raw-file
    # stem (for example, "l31_388"). Older/alternative exports may contain only
    # the numeric subject component ("388"), so support both representations.
    participant_text = participants["participant_id"].astype(str).str.strip()
    already_stem = participant_text.str.fullmatch(
        r"[lL]\d+_\d+", na=False
    )

    constructed = (
        "l"
        + participants["list_number"].astype(str)
        + "_"
        + participant_text
    ).str.lower()

    participants["stem"] = np.where(
        already_stem,
        participant_text.str.lower(),
        constructed,
    )

    # Audit that the list number embedded in an existing stem agrees with the
    # report's list_number column.
    embedded_list = participants["stem"].str.extract(
        r"^[lL](\d+)_\d+$", expand=False
    )
    mismatch = participants[
        embedded_list.notna()
        & embedded_list.ne(participants["list_number"].astype(str))
    ]
    if not mismatch.empty:
        raise RuntimeError(
            "participant_id stem and list_number disagree:\n"
            + mismatch[
                ["participant_id", "list_number", "stem", "regime"]
            ].to_string(index=False)
        )

    return participants


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.osf_list.exists():
        print(f"ERROR: OSF list not found: {args.osf_list}", file=sys.stderr)
        return 2
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 3

    text = args.osf_list.read_text(encoding="utf-8", errors="replace")
    inventory = parse_inventory(text)
    if inventory.empty:
        print(
            f"ERROR: no ASC/EDF ZIP entries found in {args.osf_list}",
            file=sys.stderr,
        )
        return 4

    raw_counts = (
        inventory.groupby(["kind", "osf_path"], observed=True)
        .size()
        .rename("list_occurrences")
        .reset_index()
    )
    duplicate_entries = raw_counts[raw_counts["list_occurrences"] > 1].copy()

    unique_inventory = (
        inventory.sort_values(["kind", "stem", "osf_path"])
        .drop_duplicates(["kind", "stem"], keep="first")
        .reset_index(drop=True)
    )

    participants = load_participants(args.manifest)

    availability = participants.copy()
    for kind in ("asc", "edf"):
        subset = unique_inventory.loc[
            unique_inventory["kind"].eq(kind),
            ["stem", "osf_path"],
        ].rename(columns={"osf_path": f"{kind}_osf_path"})
        availability = availability.merge(
            subset,
            on="stem",
            how="left",
            validate="one_to_one",
        )
        availability[f"has_{kind}"] = availability[
            f"{kind}_osf_path"
        ].notna()

    availability["raw_status"] = np.select(
        [
            availability["has_asc"] & availability["has_edf"],
            availability["has_asc"] & ~availability["has_edf"],
            ~availability["has_asc"] & availability["has_edf"],
        ],
        ["both", "asc_only", "edf_only"],
        default="neither",
    )

    manifest_stems = set(availability["stem"])
    unexpected = unique_inventory[
        ~unique_inventory["stem"].isin(manifest_stems)
    ].copy()

    coverage = (
        availability.groupby(
            ["regime", "label", "label_name", "raw_status"],
            observed=True,
        )
        .size()
        .rename("n_participants")
        .reset_index()
    )

    regime_summary = (
        availability.groupby(
            ["regime", "label", "label_name"],
            observed=True,
        )
        .agg(
            total_participants=("participant_id", "nunique"),
            asc_available=("has_asc", "sum"),
            edf_available=("has_edf", "sum"),
            both_available=("raw_status", lambda s: int((s == "both").sum())),
            asc_only=("raw_status", lambda s: int((s == "asc_only").sum())),
            edf_only=("raw_status", lambda s: int((s == "edf_only").sum())),
            neither=("raw_status", lambda s: int((s == "neither").sum())),
        )
        .reset_index()
    )

    overall = {
        "manifest_participants": int(availability["participant_id"].nunique()),
        "osf_list_entries": int(len(inventory)),
        "osf_unique_asc_stems": int(
            unique_inventory["kind"].eq("asc").sum()
        ),
        "osf_unique_edf_stems": int(
            unique_inventory["kind"].eq("edf").sum()
        ),
        "participants_with_both": int(
            availability["raw_status"].eq("both").sum()
        ),
        "participants_with_asc_only": int(
            availability["raw_status"].eq("asc_only").sum()
        ),
        "participants_with_edf_only": int(
            availability["raw_status"].eq("edf_only").sum()
        ),
        "participants_with_neither": int(
            availability["raw_status"].eq("neither").sum()
        ),
        "duplicate_osf_list_paths": int(len(duplicate_entries)),
        "unexpected_unique_raw_stems": int(
            unexpected["stem"].nunique() if not unexpected.empty else 0
        ),
    }

    availability_path = (
        args.output_dir / "onestop_raw_participant_availability.csv"
    )
    coverage_path = args.output_dir / "onestop_raw_coverage_by_regime.csv"
    duplicates_path = args.output_dir / "onestop_raw_duplicate_list_entries.csv"
    unexpected_path = args.output_dir / "onestop_raw_unexpected_entries.csv"

    availability.to_csv(
        availability_path, index=False, encoding="utf-8-sig"
    )
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    duplicate_entries.to_csv(
        duplicates_path, index=False, encoding="utf-8-sig"
    )
    unexpected.to_csv(unexpected_path, index=False, encoding="utf-8-sig")

    asc_paths = sorted(
        availability.loc[
            availability["has_asc"], "asc_osf_path"
        ].dropna().unique()
    )
    edf_only_paths = sorted(
        availability.loc[
            availability["raw_status"].eq("edf_only"), "edf_osf_path"
        ].dropna().unique()
    )

    asc_manifest_path = (
        args.output_dir / "onestop_unique_asc_download_manifest.txt"
    )
    edf_only_manifest_path = (
        args.output_dir / "onestop_edf_only_conversion_manifest.txt"
    )
    asc_manifest_path.write_text("\n".join(asc_paths) + "\n", encoding="utf-8")
    edf_only_manifest_path.write_text(
        "\n".join(edf_only_paths) + ("\n" if edf_only_paths else ""),
        encoding="utf-8",
    )

    sample_rows = []
    for regime in ("ordinary", "information_seeking"):
        candidates = availability[
            availability["regime"].eq(regime)
            & availability["has_asc"]
        ].sort_values(["list_number", "participant_id"])
        if not candidates.empty:
            row = candidates.iloc[0]
            sample_rows.append(
                {
                    "regime": regime,
                    "participant_id": row["participant_id"],
                    "list_number": row["list_number"],
                    "stem": row["stem"],
                    "asc_osf_path": row["asc_osf_path"],
                }
            )
    samples = pd.DataFrame(sample_rows)
    samples_path = args.output_dir / "onestop_asc_parser_samples.csv"
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")

    audit = {
        "overall": overall,
        "regime_summary": regime_summary.to_dict("records"),
        "duplicate_entries": duplicate_entries.to_dict("records"),
        "sample_asc_files": samples.to_dict("records"),
        "outputs": {
            "availability": str(availability_path),
            "coverage": str(coverage_path),
            "duplicates": str(duplicates_path),
            "unexpected": str(unexpected_path),
            "asc_download_manifest": str(asc_manifest_path),
            "edf_only_conversion_manifest": str(edf_only_manifest_path),
            "parser_samples": str(samples_path),
        },
    }
    audit_path = args.output_dir / "onestop_raw_inventory_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ONESTOP RAW INVENTORY SUMMARY ===")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n=== COVERAGE BY REGIME ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(regime_summary.to_string(index=False))

    if not duplicate_entries.empty:
        print("\n=== DUPLICATED OSF LIST PATHS ===")
        print(duplicate_entries.to_string(index=False))

    print("\n=== ASC PARSER SAMPLE FILES ===")
    print(samples.to_string(index=False))

    print(f"\nSaved: {availability_path}")
    print(f"Saved: {asc_manifest_path}")
    print(f"Saved: {edf_only_manifest_path}")
    print(f"Saved: {audit_path}")

    if overall["participants_with_neither"] > 0:
        print(
            "WARNING: some manifest participants have neither ASC nor EDF.",
            file=sys.stderr,
        )
        return 5
    if overall["unexpected_unique_raw_stems"] > 0:
        print(
            "WARNING: raw inventory contains stems not found in the manifest.",
            file=sys.stderr,
        )
        return 6

    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
