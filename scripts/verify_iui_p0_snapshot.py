from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an IUI P0 confirmatory snapshot hash manifest.")
    parser.add_argument(
        "--snapshot",
        default="outputs/iui_p0_protocol_lock_v1",
        help="Snapshot directory created by freeze_iui_p0_confirmatory_snapshot.ps1",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot)
    manifest_path = snapshot / "file_hash_manifest.csv"
    release_path = snapshot / "release_manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 2

    failures: list[str] = []
    checked = 0

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sha256", "bytes", "path"}
        if not required.issubset(reader.fieldnames or set()):
            print("ERROR: invalid hash manifest columns", file=sys.stderr)
            return 2

        for row in reader:
            raw_path = row["path"].strip()
            path = Path(raw_path[2:] if raw_path.startswith((".\\", "./")) else raw_path)
            if not path.exists():
                failures.append(f"MISSING: {path}")
                continue

            expected = row["sha256"].lower()
            actual = sha256(path)
            checked += 1
            if actual != expected:
                failures.append(f"HASH MISMATCH: {path}")

            if path.stat().st_size != int(row["bytes"]):
                failures.append(f"SIZE MISMATCH: {path}")

    if release_path.exists():
        try:
            json.loads(release_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            failures.append(f"INVALID JSON: {release_path}: {exc}")
    else:
        failures.append(f"MISSING: {release_path}")

    print(f"Checked files: {checked}")
    if failures:
        print("Snapshot verification FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Snapshot verification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
