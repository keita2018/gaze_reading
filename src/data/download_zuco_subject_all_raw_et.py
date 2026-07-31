#!/usr/bin/env python
"""Download all 14 raw eye-tracking block files for one ZuCo 2.0 subject.

Downloads:
- task1 - NR/Raw data/<SUBJECT>/<SUBJECT>_NR1_ET.mat ... NR7
- task2 - TSR/Raw data/<SUBJECT>/<SUBJECT>_TSR1_ET.mat ... TSR7

Example:
    python src/data/download_zuco_subject_all_raw_et.py \
        --subject YAC \
        --output-dir data/raw/zuco2/YAC/session_et/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

from osfclient.api import OSF


def normalize(path: object) -> str:
    return str(path).replace("\\", "/").lstrip("/")


def build_targets(subject: str, output_dir: Path) -> Dict[str, Path]:
    targets: Dict[str, Path] = {}
    for task, folder in (("NR", "task1 - NR"), ("TSR", "task2 - TSR")):
        for block in range(1, 8):
            filename = f"{subject}_{task}{block}_ET.mat"
            remote = f"{folder}/Raw data/{subject}/{filename}"
            targets[normalize(remote)] = output_dir / filename
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="2urht")
    parser.add_argument("--subject", default="YAC")
    parser.add_argument(
        "--output-dir",
        default="data/raw/zuco2/YAC/session_et/raw",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subject = args.subject.upper()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = build_targets(subject, output_dir)

    project = OSF().project(args.project_id)
    storage = project.storage("osfstorage")

    found = {}
    print(
        f"Scanning OSF project {args.project_id} for "
        f"{len(targets)} raw ET files ..."
    )
    for remote_file in storage.files:
        remote_path = normalize(remote_file.path)
        if remote_path in targets:
            found[remote_path] = remote_file
            if len(found) == len(targets):
                break

    missing = sorted(set(targets) - set(found))
    if missing:
        print("Missing remote files:", file=sys.stderr)
        for remote in missing:
            print(f"  /{remote}", file=sys.stderr)
        return 2

    downloaded = 0
    skipped = 0
    total_bytes = 0

    for remote, destination in sorted(targets.items()):
        if destination.exists() and not args.overwrite:
            print(f"Already exists, skipping: {destination}")
            skipped += 1
            total_bytes += destination.stat().st_size
            continue

        print(f"Downloading /{remote}")
        print(f"        -> {destination}")
        with destination.open("wb") as fp:
            found[remote].write_to(fp)

        size = destination.stat().st_size
        total_bytes += size
        downloaded += 1
        print(f"Downloaded {size / (1024 * 1024):.3f} MB")

    print("\n=== DOWNLOAD SUMMARY ===")
    print(f"subject: {subject}")
    print(f"downloaded: {downloaded}")
    print(f"skipped existing: {skipped}")
    print(f"available files: {len(targets)}")
    print(f"total size: {total_bytes / (1024 * 1024):.3f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
