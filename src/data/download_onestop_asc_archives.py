#!/usr/bin/env python
r"""
Download all unique OneStop ASC archives listed by the raw-inventory audit.

The script is designed for resumable Windows use with osfclient. It removes the
leading ``osfstorage/`` component because ``osf fetch`` uses osfstorage as the
default storage, verifies every resulting ZIP, and writes an audit report.

Example:
python src\data\download_onestop_asc_archives.py ^
  --manifest outputs\onestop_raw_inventory\onestop_unique_asc_download_manifest.txt ^
  --output-dir data\raw\onestop\asc_all ^
  --seed-dir data\raw\onestop\asc_samples ^
  --audit-dir outputs\onestop_asc_download ^
  --workers 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/onestop_raw_inventory/"
            "onestop_unique_asc_download_manifest.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/onestop/asc_all"),
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("data/raw/onestop/asc_samples"),
        help="Existing sample ZIPs copied into the full directory when valid.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/onestop_asc_download"),
    )
    parser.add_argument("--project", default="6f2km")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=3.0,
    )
    return parser.parse_args()


def normalize_remote(raw: str) -> str:
    path = raw.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    lower = path.lower()
    if lower.startswith("osfstorage/"):
        path = path[len("osfstorage/") :]
    return path.lstrip("/")


def read_manifest(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Download manifest not found: {path}")

    remotes: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        remote = normalize_remote(stripped)
        if not remote.lower().endswith(".asc.zip"):
            raise RuntimeError(
                f"Manifest entry is not an ASC ZIP path: {stripped}"
            )
        key = remote.lower()
        if key in seen:
            continue
        seen.add(key)
        remotes.append(remote)
    return remotes


def validate_asc_zip(path: Path) -> tuple[bool, str, int | None]:
    if not path.exists():
        return False, "missing", None
    if path.stat().st_size <= 0:
        return False, "empty", 0
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and "__MACOSX" not in name
                and not Path(name).name.startswith("._")
                and name.lower().endswith(".asc")
            ]
            if len(members) != 1:
                return (
                    False,
                    f"expected_one_asc_member_found_{len(members)}",
                    path.stat().st_size,
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                return (
                    False,
                    f"corrupt_member:{bad_member}",
                    path.stat().st_size,
                )
    except (zipfile.BadZipFile, OSError) as exc:
        return False, f"invalid_zip:{type(exc).__name__}", path.stat().st_size
    return True, "valid", path.stat().st_size


def copy_seed_if_available(
    filename: str,
    seed_dir: Path,
    destination: Path,
) -> tuple[bool, str]:
    seed = seed_dir / filename
    valid, reason, _ = validate_asc_zip(seed)
    if not valid:
        return False, reason
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".seed-part")
    shutil.copy2(seed, temporary)
    os.replace(temporary, destination)
    valid_dest, reason_dest, _ = validate_asc_zip(destination)
    return valid_dest, reason_dest


def run_osf_fetch(
    project: str,
    remote: str,
    destination: Path,
) -> tuple[int, str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="onestop_osf_",
        dir=str(destination.parent),
    ) as temporary_dir:
        temporary_path = Path(temporary_dir) / destination.name
        command = [
            "osf",
            "-p",
            project,
            "fetch",
            remote,
            str(temporary_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        valid, reason, _ = validate_asc_zip(temporary_path)
        if completed.returncode == 0 and valid:
            os.replace(temporary_path, destination)
            return completed.returncode, completed.stdout, completed.stderr
        diagnostics = (
            f"osf_returncode={completed.returncode}; "
            f"download_validation={reason}"
        )
        stderr = "\n".join(
            part
            for part in [completed.stderr.strip(), diagnostics]
            if part
        )
        return completed.returncode, completed.stdout, stderr


def download_one(
    remote: str,
    output_dir: Path,
    seed_dir: Path,
    project: str,
    retries: int,
    retry_wait_seconds: float,
) -> dict[str, Any]:
    filename = posixpath.basename(remote)
    destination = output_dir / filename
    started = time.time()

    valid_existing, existing_reason, existing_size = validate_asc_zip(
        destination
    )
    if valid_existing:
        return {
            "remote": remote,
            "filename": filename,
            "status": "already_valid",
            "attempts": 0,
            "size_bytes": existing_size,
            "elapsed_seconds": round(time.time() - started, 3),
            "message": existing_reason,
        }

    if destination.exists():
        destination.unlink()

    copied, seed_reason = copy_seed_if_available(
        filename=filename,
        seed_dir=seed_dir,
        destination=destination,
    )
    if copied:
        return {
            "remote": remote,
            "filename": filename,
            "status": "copied_from_seed",
            "attempts": 0,
            "size_bytes": destination.stat().st_size,
            "elapsed_seconds": round(time.time() - started, 3),
            "message": "valid",
        }

    last_message = f"seed_unavailable:{seed_reason}"
    for attempt in range(1, retries + 1):
        if destination.exists():
            destination.unlink()
        returncode, stdout, stderr = run_osf_fetch(
            project=project,
            remote=remote,
            destination=destination,
        )
        valid, reason, size = validate_asc_zip(destination)
        if returncode == 0 and valid:
            return {
                "remote": remote,
                "filename": filename,
                "status": "downloaded",
                "attempts": attempt,
                "size_bytes": size,
                "elapsed_seconds": round(time.time() - started, 3),
                "message": "valid",
            }

        last_message = " | ".join(
            item
            for item in [
                f"returncode={returncode}",
                reason,
                stdout.strip(),
                stderr.strip(),
            ]
            if item
        )
        if destination.exists():
            destination.unlink()
        if attempt < retries:
            time.sleep(retry_wait_seconds * attempt)

    return {
        "remote": remote,
        "filename": filename,
        "status": "failed",
        "attempts": retries,
        "size_bytes": None,
        "elapsed_seconds": round(time.time() - started, 3),
        "message": last_message,
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("ERROR: --workers must be at least 1.", file=sys.stderr)
        return 2
    if args.retries < 1:
        print("ERROR: --retries must be at least 1.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    remotes = read_manifest(args.manifest)
    if not remotes:
        print("ERROR: download manifest is empty.", file=sys.stderr)
        return 3

    print(
        f"OneStop ASC download: files={len(remotes)}, "
        f"workers={args.workers}, output={args.output_dir}"
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_remote = {
            executor.submit(
                download_one,
                remote,
                args.output_dir,
                args.seed_dir,
                args.project,
                args.retries,
                args.retry_wait_seconds,
            ): remote
            for remote in remotes
        }
        completed_count = 0
        for future in as_completed(future_to_remote):
            completed_count += 1
            remote = future_to_remote[future]
            try:
                result = future.result()
            except Exception as exc:  # defensive worker isolation
                result = {
                    "remote": remote,
                    "filename": posixpath.basename(remote),
                    "status": "failed",
                    "attempts": None,
                    "size_bytes": None,
                    "elapsed_seconds": None,
                    "message": f"worker_exception:{type(exc).__name__}:{exc}",
                }
            results.append(result)
            print(
                f"[{completed_count}/{len(remotes)}] "
                f"{result['status']}: {result['filename']}"
            )

    results.sort(key=lambda row: row["filename"].lower())
    csv_path = args.audit_dir / "onestop_asc_download_results.csv"
    json_path = args.audit_dir / "onestop_asc_download_audit.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    status_counts: dict[str, int] = {}
    for row in results:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    valid_files = []
    invalid_files = []
    total_bytes = 0
    for remote in remotes:
        path = args.output_dir / posixpath.basename(remote)
        valid, reason, size = validate_asc_zip(path)
        if valid:
            valid_files.append(path.name)
            total_bytes += int(size or 0)
        else:
            invalid_files.append(
                {"filename": path.name, "reason": reason}
            )

    audit = {
        "manifest_entries": len(remotes),
        "unique_expected_archives": len(remotes),
        "status_counts": status_counts,
        "valid_archives": len(valid_files),
        "invalid_or_missing_archives": len(invalid_files),
        "total_size_bytes": total_bytes,
        "total_size_gib": total_bytes / (1024**3),
        "invalid_files": invalid_files,
        "output_directory": str(args.output_dir),
        "results_csv": str(csv_path),
    }
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ONESTOP ASC DOWNLOAD SUMMARY ===")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    if invalid_files:
        print(
            "\nDOWNLOAD STATUS: FAIL — some archives are missing or invalid.",
            file=sys.stderr,
        )
        return 4

    print("\nDOWNLOAD STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
