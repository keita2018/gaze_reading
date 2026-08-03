from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the independently stored Turkish confirmatory snapshot "
            "is byte-for-byte identical to the project copy."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--original",
        default="outputs/iui_p0_confirmatory_TR_v1",
    )
    parser.add_argument(
        "--backup",
        required=True,
        help="Independent snapshot directory, e.g. D:\\silve\\hozonn\\iui_p0_confirmatory_TR_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/iui_p0_tr_backup_verification_v1",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    original = (project_root / args.original).resolve()
    backup = Path(args.backup).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original.exists():
        raise FileNotFoundError(f"Original snapshot not found: {original}")
    if not backup.exists():
        raise FileNotFoundError(f"Backup snapshot not found: {backup}")

    original_files = file_inventory(original)
    backup_files = file_inventory(backup)

    failures: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for relative_path in sorted(original_files):
        source_path = original_files[relative_path]
        backup_path = backup_files.get(relative_path)

        if backup_path is None:
            failures.append(f"MISSING IN BACKUP: {relative_path}")
            continue

        source_size = source_path.stat().st_size
        backup_size = backup_path.stat().st_size
        source_hash = sha256(source_path)
        backup_hash = sha256(backup_path)

        status = "PASS"
        if source_size != backup_size:
            status = "FAIL"
            failures.append(f"SIZE MISMATCH: {relative_path}")
        if source_hash != backup_hash:
            status = "FAIL"
            failures.append(f"HASH MISMATCH: {relative_path}")

        rows.append(
            {
                "relative_path": relative_path,
                "original_bytes": source_size,
                "backup_bytes": backup_size,
                "original_sha256": source_hash,
                "backup_sha256": backup_hash,
                "status": status,
            }
        )

    extra_files = sorted(set(backup_files) - set(original_files))
    for relative_path in extra_files:
        warnings.append(f"EXTRA FILE IN BACKUP: {relative_path}")

    required = [
        "file_hash_manifest.csv",
        "SHA256SUMS",
        "release_manifest.json",
        "audit/feature_schema_audit.json",
        "features/unlabeled_full41_features.parquet",
        "manifests/unlabeled_target_manifest.csv",
        "predictions/unlabeled_frozen_scores.csv",
        "sealed/labels_sealed.csv",
    ]
    for relative_path in required:
        if relative_path not in backup_files:
            failures.append(f"MISSING REQUIRED BACKUP FILE: {relative_path}")

    import pandas as pd

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output_dir / "backup_file_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "artifact_version": "iui_p0_tr_backup_verification_v1",
        "original": str(original),
        "backup": str(backup),
        "original_file_count": len(original_files),
        "backup_file_count": len(backup_files),
        "checked_common_files": len(rows),
        "failures": failures,
        "warnings": warnings,
        "status": "PASS" if not failures else "FAIL",
    }
    (output_dir / "backup_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = f"""# Turkish Confirmatory Backup Verification

- Status: **{report['status']}**
- Original: `{original}`
- Backup: `{backup}`
- Original files: **{len(original_files)}**
- Backup files: **{len(backup_files)}**
- Byte/hash comparisons: **{len(rows)}**

## Failures

```json
{json.dumps(failures, ensure_ascii=False, indent=2)}
```

## Warnings

```json
{json.dumps(warnings, ensure_ascii=False, indent=2)}
```
"""
    (output_dir / "BACKUP_VERIFICATION_SUMMARY.md").write_text(
        summary,
        encoding="utf-8",
    )

    if failures:
        print("Independent backup verification FAILED.")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Independent backup verification PASSED.")
    print(f"Original files: {len(original_files)}")
    print(f"Backup files: {len(backup_files)}")
    print(f"Report: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
