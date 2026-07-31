#!/usr/bin/env python
"""Create a size-aware inventory of ZuCo files for one subject on OSF.

The script does not download data. It recursively scans an OSF project and
writes a CSV containing every remote path that includes the subject ID.
Files likely to contain eye-tracking data are flagged using their path/name.

Example:
    python src/data/inventory_zuco_subject_files.py \
        --project-id 2urht \
        --subject YAC \
        --output outputs/tables/YAC_osf_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path, PurePosixPath
from typing import Any

from osfclient.api import OSF


EYE_KEYWORDS = (
    "_et.",
    "_et_",
    "eyetrack",
    "eye-track",
    "eye_tracking",
    "eye tracking",
    ".edf",
    ".asc",
)

RESULT_KEYWORDS = (
    "resultsy",
    "results",
)

EEG_KEYWORDS = (
    "_eeg",
    "brainvision",
    ".vhdr",
    ".vmrk",
    ".eeg",
)


def normalize_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def parse_size(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="2urht")
    parser.add_argument("--subject", default="YAC")
    parser.add_argument(
        "--output",
        default="outputs/tables/YAC_osf_inventory.csv",
    )
    args = parser.parse_args()

    subject = args.subject.upper()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    project = OSF().project(args.project_id)
    storage = project.storage("osfstorage")

    rows: list[dict[str, Any]] = []

    print(
        f"Scanning OSF project {args.project_id} for paths containing "
        f"{subject!r} ..."
    )

    for remote_file in storage.files:
        remote_path = normalize_path(remote_file.path)
        if subject.lower() not in remote_path.lower():
            continue

        name = getattr(remote_file, "name", None)
        if not name:
            name = PurePosixPath(remote_path).name

        lower = remote_path.lower()
        size_bytes = parse_size(getattr(remote_file, "size", None))
        if size_bytes is None:
            size_bytes = parse_size(getattr(remote_file, "_json", {}).get("attributes", {}).get("size"))

        likely_eye = any(keyword in lower for keyword in EYE_KEYWORDS)
        likely_eeg = any(keyword in lower for keyword in EEG_KEYWORDS)
        likely_results = any(keyword in lower for keyword in RESULT_KEYWORDS)

        rows.append(
            {
                "subject_id": subject,
                "project_id": args.project_id,
                "name": name,
                "remote_path": remote_path,
                "size_bytes": size_bytes if size_bytes is not None else "",
                "size_mb": (
                    round(size_bytes / (1024 * 1024), 3)
                    if size_bytes is not None
                    else ""
                ),
                "likely_eye_tracking": int(likely_eye),
                "likely_eeg": int(likely_eeg),
                "likely_results_file": int(likely_results),
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row["likely_eye_tracking"]),
            int(row["likely_eeg"]),
            row["remote_path"].lower(),
        )
    )

    with output.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else [
            "subject_id",
            "project_id",
            "name",
            "remote_path",
            "size_bytes",
            "size_mb",
            "likely_eye_tracking",
            "likely_eeg",
            "likely_results_file",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} subject-related files.")
    print(f"Written: {output}")

    eye_rows = [row for row in rows if row["likely_eye_tracking"]]
    print("\n=== LIKELY EYE-TRACKING FILES ===")
    if not eye_rows:
        print("No paths matched the current eye-tracking filename rules.")
    else:
        for row in eye_rows:
            print(
                f"{row['size_mb']} MB\t{row['remote_path']}"
            )

    print("\n=== LARGE NON-EEG CANDIDATES ===")
    candidates = [
        row for row in rows
        if not row["likely_eeg"]
        and not row["likely_results_file"]
        and isinstance(row["size_mb"], (int, float))
    ]
    for row in candidates[:30]:
        print(f"{row['size_mb']} MB\t{row['remote_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
