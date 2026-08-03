from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


FORBIDDEN_PREDICTION_COLUMNS = {
    "label",
    "label_nr",
    "task",
    "condition",
    "reading_goal",
    "target",
    "y",
    "y_true",
    "true_label",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    return list(pd.read_csv(path, nrows=0).columns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Turkish unlabeled confirmatory artifacts before "
            "opening the sealed labels."
        )
    )
    parser.add_argument(
        "--snapshot",
        default="outputs/iui_p0_confirmatory_TR_v1",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    manifest_path = snapshot / "file_hash_manifest.csv"
    release_path = snapshot / "release_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not release_path.exists():
        raise FileNotFoundError(release_path)

    manifest = pd.read_csv(manifest_path)
    required_columns = {"role", "sha256", "bytes", "path"}
    if not required_columns.issubset(manifest.columns):
        raise RuntimeError(
            "file_hash_manifest.csv has an invalid schema."
        )

    failures: list[str] = []
    checked = 0

    for row in manifest.to_dict("records"):
        path = Path(str(row["path"]))
        if not path.exists():
            failures.append(f"MISSING: {path}")
            continue

        actual_hash = sha256(path)
        expected_hash = str(row["sha256"]).lower()
        if actual_hash != expected_hash:
            failures.append(f"HASH MISMATCH: {path}")

        actual_size = path.stat().st_size
        expected_size = int(row["bytes"])
        if actual_size != expected_size:
            failures.append(f"SIZE MISMATCH: {path}")

        checked += 1

    prediction_path = (
        snapshot
        / "predictions"
        / "unlabeled_frozen_scores.csv"
    )
    feature_path = (
        snapshot
        / "features"
        / "unlabeled_full41_features.parquet"
    )
    manifest_unlabeled_path = (
        snapshot
        / "manifests"
        / "unlabeled_target_manifest.csv"
    )
    sealed_path = snapshot / "sealed" / "labels_sealed.csv"

    for path in [
        prediction_path,
        feature_path,
        manifest_unlabeled_path,
        sealed_path,
    ]:
        if not path.exists():
            failures.append(f"MISSING REQUIRED ARTIFACT: {path}")

    if prediction_path.exists():
        columns = {
            column.strip().lower()
            for column in read_columns(prediction_path)
        }
        leaked = sorted(
            columns & FORBIDDEN_PREDICTION_COLUMNS
        )
        if leaked:
            failures.append(
                "LABEL/TASK LEAK IN PREDICTION TABLE: "
                + ", ".join(leaked)
            )

    if feature_path.exists():
        columns = {
            column.strip().lower()
            for column in read_columns(feature_path)
        }
        leaked = sorted(
            columns & FORBIDDEN_PREDICTION_COLUMNS
        )
        if leaked:
            failures.append(
                "LABEL/TASK LEAK IN FEATURE TABLE: "
                + ", ".join(leaked)
            )

    if manifest_unlabeled_path.exists():
        columns = {
            column.strip().lower()
            for column in read_columns(
                manifest_unlabeled_path
            )
        }
        leaked = sorted(
            columns & FORBIDDEN_PREDICTION_COLUMNS
        )
        if leaked:
            failures.append(
                "LABEL/TASK LEAK IN UNLABELED MANIFEST: "
                + ", ".join(leaked)
            )

    release = json.loads(
        release_path.read_text(encoding="utf-8-sig")
    )
    if release.get("performance_calculated") is not False:
        failures.append(
            "release_manifest does not state "
            "performance_calculated=false"
        )
    if release.get("labels_joined_to_predictions") is not False:
        failures.append(
            "release_manifest does not state "
            "labels_joined_to_predictions=false"
        )

    print(f"Checked hashed files: {checked}")

    if failures:
        print("Turkish unlabeled freeze verification FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Turkish unlabeled freeze verification PASSED.")
    print(
        "Copy the entire snapshot directory to independent "
        "storage before opening sealed labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
