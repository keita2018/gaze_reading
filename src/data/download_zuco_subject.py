#!/usr/bin/env python
"""Download one ZuCo 2.0 subject's sentence-level MATLAB files from OSF.

This script:
1. Recursively lists the public OSF project.
2. Saves all remote paths containing the subject ID.
3. Downloads only results<SUBJECT>_NR.mat and results<SUBJECT>_TSR.mat.
4. Prints ET-related candidate paths without downloading large block files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, List

from osfclient.api import OSF


def normalize_remote_path(value: object) -> str:
    return str(value).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="2urht")
    parser.add_argument("--subject", default="YAC")
    parser.add_argument(
        "--output-dir",
        default="data/raw/zuco2/YAC",
        help="Local output directory",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subject = args.subject.upper()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_names = {
        f"results{subject}_NR.mat",
        f"results{subject}_TSR.mat",
    }

    print(f"Connecting to OSF project {args.project_id} ...")
    project = OSF().project(args.project_id)
    storage = project.storage("osfstorage")

    subject_paths: List[str] = []
    et_candidates: List[str] = []
    matches: Dict[str, list] = {name: [] for name in target_names}

    print("Recursively scanning OSF files. This can take a little while ...")
    for remote_file in storage.files:
        remote_path = normalize_remote_path(remote_file.path)
        remote_name = getattr(remote_file, "name", None)
        if not remote_name:
            remote_name = PurePosixPath(remote_path).name

        if subject.lower() in remote_path.lower():
            subject_paths.append(remote_path)
            lower = remote_path.lower()
            if (
                "_et.mat" in lower
                or "eyetrack" in lower
                or "eye-track" in lower
                or "eye_tracking" in lower
            ):
                et_candidates.append(remote_path)

        if remote_name in matches:
            matches[remote_name].append(remote_file)

    paths_file = output_dir / f"{subject}_remote_paths.txt"
    paths_file.write_text(
        "\n".join(sorted(subject_paths)) + ("\n" if subject_paths else ""),
        encoding="utf-8",
    )
    print(f"Saved {len(subject_paths)} subject-related paths to: {paths_file}")

    et_file = output_dir / f"{subject}_ET_candidate_paths.txt"
    et_file.write_text(
        "\n".join(sorted(et_candidates)) + ("\n" if et_candidates else ""),
        encoding="utf-8",
    )
    print(f"Saved {len(et_candidates)} ET-related candidate paths to: {et_file}")

    missing = []
    ambiguous = []

    for target_name, found in matches.items():
        if len(found) == 0:
            missing.append(target_name)
            continue
        if len(found) > 1:
            ambiguous.append((target_name, [normalize_remote_path(f.path) for f in found]))
            continue

        destination = output_dir / target_name
        if destination.exists() and not args.overwrite:
            print(f"Already exists, skipping: {destination}")
            continue

        remote_file = found[0]
        print(f"Downloading {normalize_remote_path(remote_file.path)}")
        print(f"        -> {destination}")
        with destination.open("wb") as fp:
            remote_file.write_to(fp)

        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"Downloaded: {size_mb:.1f} MB")

    if ambiguous:
        print("\nERROR: Multiple exact-name matches were found:", file=sys.stderr)
        for name, paths in ambiguous:
            print(f"  {name}", file=sys.stderr)
            for path in paths:
                print(f"    {path}", file=sys.stderr)
        return 2

    if missing:
        print("\nExact benchmark result files were not found:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        print(
            f"Inspect {paths_file} and {et_file}; the OSF naming may differ.",
            file=sys.stderr,
        )
        return 3

    print("\nDownload completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
