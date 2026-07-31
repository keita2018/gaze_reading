#!/usr/bin/env python
"""WebQAmGaze external validation Stage 3A.

Goals
-----
1. Build the frozen-schema-compatible 18-dimensional RawStat table for the
   predeclared WebQAmGaze primary external-validation trials.
2. Audit the repository resources required for Soft-AOI and the official
   nine-dimensional fixation/saccade condition.
3. Confirm feature-schema order and missing-feature handling before loading any
   frozen classifier.

No model is fitted or applied. WebQAmGaze labels are retained only as external
ground truth and are not used for feature design, scaling, imputation, threshold
selection, or calibration.

Cross-device coordinate convention
----------------------------------
ZuCo 2.0 used an 800 x 600 stimulus display. WebQAmGaze coordinates are first
normalized by each participant's reported screen resolution in Stage 2. This
script maps x_norm and y_norm to the fixed ZuCo reference pixel system before
computing pixel dispersion and speed:

    x_zuco_px = x_norm * 800
    y_zuco_px = y_norm * 600

The reference dimensions are command-line arguments and are recorded in the
manifest.

Unavailable pupil information
-----------------------------
WebQAmGaze does not provide pupil area. RawStat features that require pupil data
or joint gaze+pupil validity are written as NaN rather than zero. The frozen
ZuCo pipeline's already-fitted median imputer must handle these values later.
No webcam-derived imputation statistics are estimated here.

RawStat features expected from the frozen bundle
------------------------------------------------
- valid_xy_rate
- valid_pupil_rate                 -> NaN
- valid_basic_rate                 -> NaN
- longest_invalid_run_ratio        -> NaN
- x_std
- x_iqr
- x_range90
- y_std
- y_iqr
- y_range90
- xy_correlation
- speed_mean_px_s
- speed_median_px_s
- speed_p90_px_s
- speed_p95_px_s
- pupil_std_rel                    -> NaN
- pupil_iqr_rel                    -> NaN
- pupil_range90_rel                -> NaN

Outputs
-------
rawstat_features_primary.csv
rawstat_feature_audit.csv
frozen_feature_schema_primary.csv
repository_aoi_resource_inventory.csv
candidate_word_boundary_file_schemas.csv
fixation_data_per_part_file_schemas.csv
official_notebook_word_boundary_code.txt
feature_condition_readiness.csv
stage3a_manifest.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


EXPECTED_RAWSTAT = [
    "valid_xy_rate",
    "valid_pupil_rate",
    "valid_basic_rate",
    "longest_invalid_run_ratio",
    "x_std",
    "x_iqr",
    "x_range90",
    "y_std",
    "y_iqr",
    "y_range90",
    "xy_correlation",
    "speed_mean_px_s",
    "speed_median_px_s",
    "speed_p90_px_s",
    "speed_p95_px_s",
    "pupil_std_rel",
    "pupil_iqr_rel",
    "pupil_range90_rel",
]

PUPIL_UNAVAILABLE_FEATURES = {
    "valid_pupil_rate",
    "valid_basic_rate",
    "longest_invalid_run_ratio",
    "pupil_std_rel",
    "pupil_iqr_rel",
    "pupil_range90_rel",
}

OFFICIAL_FEATURE_NAMES = [
    "omission_rate",
    "fixation_number",
    "reading_speed",
    "mean_sacc_dur",
    "max_sacc_dur",
    "mean_sacc_velocity",
    "max_sacc_velocity",
    "mean_sacc_amplitude",
    "max_sacc_amplitude",
]

RESOURCE_PATTERN = re.compile(
    r"word|bound|aoi|roi|fixation_data_per_part|paragraph|stimulus|text",
    re.IGNORECASE,
)
NOTEBOOK_CODE_PATTERN = re.compile(
    r"word|bound|fixation_data|read_json|read_csv|pickle|json|roi|target",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(values: Iterable[Any]) -> np.ndarray:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def safe_std(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=0)) if len(values) else math.nan


def safe_quantile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else math.nan


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    if np.std(x, ddof=0) <= 0 or np.std(y, ddof=0) <= 0:
        return math.nan
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else math.nan


def choose_primary_schema(schema: pd.DataFrame) -> pd.DataFrame:
    required = {"feature_set", "feature_index", "feature_name"}
    missing = sorted(required - set(schema.columns))
    if missing:
        raise ValueError(f"feature_schema.csv lacks columns: {missing}")

    selected = schema[schema["feature_set"].eq("rawstat_all")].copy()
    if selected.empty:
        raise ValueError("feature_schema.csv has no rawstat_all rows.")

    if "training_scope" in selected.columns:
        common = selected[
            selected["training_scope"].astype(str).str.contains(
                "common16|matched", case=False, regex=True
            )
        ].copy()
        if not common.empty:
            selected = common

    selected["feature_index"] = pd.to_numeric(
        selected["feature_index"], errors="raise"
    ).astype(int)
    selected = selected.sort_values("feature_index").reset_index(drop=True)

    if selected["feature_index"].duplicated().any():
        raise ValueError("rawstat_all schema contains duplicate feature indices.")
    if selected["feature_name"].duplicated().any():
        raise ValueError("rawstat_all schema contains duplicate feature names.")
    if len(selected) != 18:
        raise ValueError(
            f"Frozen rawstat_all schema has {len(selected)} features, expected 18."
        )
    return selected


def segment_speeds(group: pd.DataFrame) -> np.ndarray:
    """Compute speeds only within the same interpolation segment."""
    speeds: list[np.ndarray] = []
    for _, segment in group.groupby("interpolation_segment_id", sort=False):
        segment = segment.sort_values("t_ms")
        if len(segment) < 2:
            continue
        t = segment["t_ms"].to_numpy(dtype=float)
        x = segment["x_zuco_px"].to_numpy(dtype=float)
        y = segment["y_zuco_px"].to_numpy(dtype=float)
        dt = np.diff(t) / 1000.0
        displacement = np.hypot(np.diff(x), np.diff(y))
        valid = (
            np.isfinite(dt)
            & (dt > 0)
            & np.isfinite(displacement)
        )
        if valid.any():
            speeds.append(displacement[valid] / dt[valid])
    return np.concatenate(speeds) if speeds else np.asarray([], dtype=float)


def rawstat_row(
    trial: pd.Series,
    samples: pd.DataFrame,
    schema_features: Sequence[str],
    reference_width: float,
    reference_height: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = samples.sort_values(
        ["interpolation_segment_id", "t_ms", "sample_index_30hz"]
    ).copy()

    x_norm = pd.to_numeric(samples["x_norm"], errors="coerce").to_numpy(dtype=float)
    y_norm = pd.to_numeric(samples["y_norm"], errors="coerce").to_numpy(dtype=float)
    x = x_norm * reference_width
    y = y_norm * reference_height
    samples["x_zuco_px"] = x
    samples["y_zuco_px"] = y

    x_q05 = safe_quantile(x, 0.05)
    x_q25 = safe_quantile(x, 0.25)
    x_q75 = safe_quantile(x, 0.75)
    x_q95 = safe_quantile(x, 0.95)
    y_q05 = safe_quantile(y, 0.05)
    y_q25 = safe_quantile(y, 0.25)
    y_q75 = safe_quantile(y, 0.75)
    y_q95 = safe_quantile(y, 0.95)
    speeds = segment_speeds(samples)

    available: dict[str, float] = {
        "valid_xy_rate": float(trial["valid_xy_rate"]),
        "valid_pupil_rate": math.nan,
        "valid_basic_rate": math.nan,
        "longest_invalid_run_ratio": math.nan,
        "x_std": safe_std(x),
        "x_iqr": (
            x_q75 - x_q25
            if math.isfinite(x_q75) and math.isfinite(x_q25)
            else math.nan
        ),
        "x_range90": (
            x_q95 - x_q05
            if math.isfinite(x_q95) and math.isfinite(x_q05)
            else math.nan
        ),
        "y_std": safe_std(y),
        "y_iqr": (
            y_q75 - y_q25
            if math.isfinite(y_q75) and math.isfinite(y_q25)
            else math.nan
        ),
        "y_range90": (
            y_q95 - y_q05
            if math.isfinite(y_q95) and math.isfinite(y_q05)
            else math.nan
        ),
        "xy_correlation": safe_correlation(x, y),
        "speed_mean_px_s": (
            float(np.mean(speeds)) if len(speeds) else math.nan
        ),
        "speed_median_px_s": safe_quantile(speeds, 0.50),
        "speed_p90_px_s": safe_quantile(speeds, 0.90),
        "speed_p95_px_s": safe_quantile(speeds, 0.95),
        "pupil_std_rel": math.nan,
        "pupil_iqr_rel": math.nan,
        "pupil_range90_rel": math.nan,
    }

    row: dict[str, Any] = {
        "external_trial_id": str(trial["external_trial_id"]),
        "trial_id": str(trial["external_trial_id"]),
        "subject_id": str(trial["participant_id"]),
        "participant_id": str(trial["participant_id"]),
        "set_name": str(trial["set_name"]),
        "trial_index": int(trial["trial_index"]),
        "trial_name": str(trial["trial_name"]),
        "task": str(trial["task"]),
        "label_nr": int(trial["label_nr"]),
        "n_input_30hz_samples": int(len(samples)),
        "n_interpolation_segments": int(
            samples["interpolation_segment_id"].nunique()
        ),
        "coordinate_reference_width_px": float(reference_width),
        "coordinate_reference_height_px": float(reference_height),
    }
    audit_rows: list[dict[str, Any]] = []
    for index, feature in enumerate(schema_features):
        if feature not in available:
            raise ValueError(
                f"No Stage 3A calculation is defined for frozen RawStat feature: "
                f"{feature}"
            )
        value = available[feature]
        row[feature] = value
        audit_rows.append(
            {
                "external_trial_id": row["external_trial_id"],
                "feature_index": index,
                "feature_name": feature,
                "value": value,
                "is_missing": not math.isfinite(float(value)),
                "missing_reason": (
                    "pupil_not_available_in_webqamgaze"
                    if feature in PUPIL_UNAVAILABLE_FEATURES
                    else (
                        "insufficient_valid_samples"
                        if not math.isfinite(float(value))
                        else ""
                    )
                ),
            }
        )
    return row, audit_rows


def file_schema(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    base: dict[str, Any] = {
        "relative_path": relative,
        "filename": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "parse_status": "",
        "object_type": "",
        "n_rows_or_items": "",
        "columns_or_keys": "",
        "example": "",
        "error": "",
    }
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path, nrows=20)
            base.update(
                {
                    "parse_status": "ok",
                    "object_type": "csv",
                    "n_rows_or_items": sum(
                        1
                        for _ in path.open(
                            "r", encoding="utf-8-sig", errors="replace"
                        )
                    )
                    - 1,
                    "columns_or_keys": "|".join(map(str, frame.columns)),
                    "example": frame.head(2).to_json(
                        orient="records", force_ascii=False
                    )[:1200],
                }
            )
        elif suffix == ".json":
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                obj = json.loads(path.read_text(encoding="utf-8-sig"))
            keys = list(obj.keys()) if isinstance(obj, dict) else []
            base.update(
                {
                    "parse_status": "ok",
                    "object_type": type(obj).__name__,
                    "n_rows_or_items": len(obj)
                    if isinstance(obj, (dict, list))
                    else "",
                    "columns_or_keys": "|".join(map(str, keys[:150])),
                    "example": repr(obj)[:1200],
                }
            )
        elif suffix in {".pkl", ".pickle"}:
            obj = pd.read_pickle(path)
            if isinstance(obj, pd.DataFrame):
                columns = list(obj.columns)
                length = len(obj)
                example = obj.head(2).to_json(
                    orient="records", force_ascii=False
                )
            elif isinstance(obj, pd.Series):
                columns = list(obj.index)
                length = len(obj)
                example = obj.head(10).to_json(force_ascii=False)
            else:
                columns = []
                length = len(obj) if hasattr(obj, "__len__") else ""
                example = repr(obj)
            base.update(
                {
                    "parse_status": "ok",
                    "object_type": type(obj).__name__,
                    "n_rows_or_items": length,
                    "columns_or_keys": "|".join(map(str, columns[:150])),
                    "example": example[:1200],
                }
            )
        elif suffix == ".npy":
            obj = np.load(path, allow_pickle=True)
            base.update(
                {
                    "parse_status": "ok",
                    "object_type": f"numpy:{obj.dtype}",
                    "n_rows_or_items": int(obj.size),
                    "columns_or_keys": f"shape={obj.shape}",
                    "example": repr(obj.flat[0])[:1200] if obj.size else "",
                }
            )
        else:
            base.update(
                {
                    "parse_status": "not_parsed",
                    "object_type": "binary_or_other",
                }
            )
    except Exception as exc:  # noqa: BLE001
        base.update(
            {
                "parse_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return base


def notebook_word_boundary_code(root: Path) -> str:
    notebook_path = (
        root / "scripts" / "Loading Eye-Tracking Data + Word Boundaries.ipynb"
    )
    if not notebook_path.is_file():
        return "Notebook not found.\n"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    blocks: list[str] = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", [])
        if isinstance(source, str):
            lines = source.splitlines()
        else:
            lines = [str(line).rstrip("\r\n") for line in source]
        if any(NOTEBOOK_CODE_PATTERN.search(line) for line in lines):
            blocks.append(f"### cell {cell_index}")
            blocks.extend(lines)
            blocks.append("")
    return "\n".join(blocks)


def count_trial_coverage(
    trial_manifest: pd.DataFrame,
    inventory: pd.DataFrame,
) -> dict[str, Any]:
    trial_names = set(trial_manifest["trial_name"].astype(str))
    path_text = "\n".join(inventory["relative_path"].astype(str))
    matched = sum(name in path_text for name in trial_names)
    return {
        "n_primary_trial_names": len(trial_names),
        "n_trial_names_appearing_in_resource_paths": matched,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="WebQAmGaze repository root.",
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage2"),
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path("outputs/frozen_zuco_external_validation_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage3a"),
    )
    parser.add_argument("--zuco-reference-width-px", type=float, default=800.0)
    parser.add_argument("--zuco-reference-height-px", type=float, default=600.0)
    args = parser.parse_args()

    if args.zuco_reference_width_px <= 0 or args.zuco_reference_height_px <= 0:
        raise ValueError("ZuCo reference dimensions must be positive.")

    root = args.root.resolve()
    stage2_dir = args.stage2_dir.resolve()
    frozen_dir = args.frozen_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_paths = {
        "trial_manifest": stage2_dir / "trial_manifest_primary_candidates.csv",
        "samples_30hz": stage2_dir / "samples_30hz.parquet",
        "feature_schema": frozen_dir / "feature_schema.csv",
        "model_manifest": frozen_dir / "model_manifest.json",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    trials = pd.read_csv(required_paths["trial_manifest"])
    samples = pd.read_parquet(required_paths["samples_30hz"])
    schema_all = pd.read_csv(required_paths["feature_schema"])
    schema = choose_primary_schema(schema_all)
    schema_features = schema["feature_name"].astype(str).tolist()

    if schema_features != EXPECTED_RAWSTAT:
        raise ValueError(
            "Frozen rawstat_all feature order differs from the expected "
            f"rate-invariant 18-feature definition.\n"
            f"Frozen: {schema_features}\nExpected: {EXPECTED_RAWSTAT}"
        )

    if trials["external_trial_id"].duplicated().any():
        raise ValueError("Primary trial manifest contains duplicate trial IDs.")
    if samples.empty:
        raise ValueError("samples_30hz.parquet is empty.")

    sample_ids = set(samples["external_trial_id"].astype(str))
    trial_ids = set(trials["external_trial_id"].astype(str))
    missing_samples = sorted(trial_ids - sample_ids)
    if missing_samples:
        raise ValueError(
            f"{len(missing_samples)} primary trials have no 30 Hz samples: "
            f"{missing_samples[:20]}"
        )

    grouped_samples = {
        str(trial_id): group.copy()
        for trial_id, group in samples[
            samples["external_trial_id"].astype(str).isin(trial_ids)
        ].groupby("external_trial_id", sort=False)
    }

    feature_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for trial in trials.itertuples(index=False):
        trial_series = pd.Series(trial._asdict())
        trial_id = str(trial_series["external_trial_id"])
        row, row_audit = rawstat_row(
            trial=trial_series,
            samples=grouped_samples[trial_id],
            schema_features=schema_features,
            reference_width=float(args.zuco_reference_width_px),
            reference_height=float(args.zuco_reference_height_px),
        )
        feature_rows.append(row)
        audit_rows.extend(row_audit)

    features = pd.DataFrame(feature_rows)
    audit = pd.DataFrame(audit_rows)
    if len(features) != len(trials):
        raise RuntimeError("RawStat feature row count mismatch.")
    if features["external_trial_id"].duplicated().any():
        raise RuntimeError("RawStat output contains duplicate trial IDs.")
    if np.isinf(features[schema_features].to_numpy(dtype=float)).any():
        raise RuntimeError("RawStat output contains infinite feature values.")

    # Resource discovery for Soft-AOI and official nine-feature extraction.
    resource_paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and RESOURCE_PATTERN.search(path.as_posix())
    ]
    resource_rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "filename": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "path_matches_fixation_data_per_part": (
                "fixation_data_per_part" in path.as_posix()
            ),
            "path_matches_word_boundary": bool(
                re.search(r"word|bound", path.as_posix(), re.I)
            ),
        }
        for path in sorted(resource_paths)
    ]
    inventory = pd.DataFrame(resource_rows)

    parse_candidates = [
        path
        for path in resource_paths
        if path.suffix.lower() in {
            ".csv", ".json", ".pkl", ".pickle", ".npy"
        }
    ]
    word_like = [
        path for path in parse_candidates
        if re.search(r"word|bound|aoi|roi", path.as_posix(), re.I)
    ]
    fixation_like = [
        path for path in parse_candidates
        if "fixation_data_per_part" in path.as_posix()
    ]

    # Parse a capped but broad sample by schema/filename while retaining full inventory.
    def representative_subset(paths: list[Path], cap: int = 300) -> list[Path]:
        if len(paths) <= cap:
            return sorted(paths)
        selected: list[Path] = []
        by_suffix: dict[str, list[Path]] = {}
        for path in paths:
            by_suffix.setdefault(path.suffix.lower(), []).append(path)
        for values in by_suffix.values():
            selected.extend(sorted(values)[: min(60, len(values))])
        return sorted(dict.fromkeys(selected))[:cap]

    word_schema_rows = [
        file_schema(path, root)
        for path in representative_subset(word_like, cap=300)
    ]
    fixation_schema_rows = [
        file_schema(path, root)
        for path in representative_subset(fixation_like, cap=300)
    ]

    notebook_text = notebook_word_boundary_code(root)

    paths = {
        "rawstat_features": output_dir / "rawstat_features_primary.csv",
        "rawstat_audit": output_dir / "rawstat_feature_audit.csv",
        "frozen_schema": output_dir / "frozen_feature_schema_primary.csv",
        "resource_inventory": output_dir
        / "repository_aoi_resource_inventory.csv",
        "word_schemas": output_dir
        / "candidate_word_boundary_file_schemas.csv",
        "fixation_schemas": output_dir
        / "fixation_data_per_part_file_schemas.csv",
        "notebook_code": output_dir
        / "official_notebook_word_boundary_code.txt",
        "readiness": output_dir / "feature_condition_readiness.csv",
        "manifest": output_dir / "stage3a_manifest.json",
    }

    features.to_csv(
        paths["rawstat_features"], index=False, encoding="utf-8-sig"
    )
    audit.to_csv(paths["rawstat_audit"], index=False, encoding="utf-8-sig")
    schema.to_csv(paths["frozen_schema"], index=False, encoding="utf-8-sig")
    inventory.to_csv(
        paths["resource_inventory"], index=False, encoding="utf-8-sig"
    )
    write_csv(paths["word_schemas"], word_schema_rows)
    write_csv(paths["fixation_schemas"], fixation_schema_rows)
    paths["notebook_code"].write_text(notebook_text, encoding="utf-8")

    finite_counts = {
        feature: int(np.isfinite(features[feature].to_numpy(dtype=float)).sum())
        for feature in schema_features
    }
    missing_counts = {
        feature: int(features[feature].isna().sum())
        for feature in schema_features
    }

    word_parse_ok = sum(
        row.get("parse_status") == "ok" for row in word_schema_rows
    )
    fixation_parse_ok = sum(
        row.get("parse_status") == "ok" for row in fixation_schema_rows
    )
    readiness_rows = [
        {
            "feature_set": "dummy_reference",
            "status": "ready",
            "n_required_features": 0,
            "n_available_features": 0,
            "blocking_issue": "",
            "next_action": "apply frozen dummy artifact",
        },
        {
            "feature_set": "rawstat_all",
            "status": "ready_with_predeclared_missing_pupil",
            "n_required_features": 18,
            "n_available_features": 12,
            "blocking_issue": "",
            "next_action": (
                "apply frozen Common16 RawStat pipeline; preserve NaN pupil "
                "features for frozen ZuCo imputer"
            ),
        },
        {
            "feature_set": "soft_full_all",
            "status": (
                "word_boundary_schema_found"
                if word_parse_ok > 0
                else "blocked_word_boundary_schema_not_found"
            ),
            "n_required_features": 41,
            "n_available_features": 0,
            "blocking_issue": (
                "" if word_parse_ok > 0
                else "WebQAmGaze word rectangles not yet resolved"
            ),
            "next_action": "construct per-trial word AOI template",
        },
        {
            "feature_set": "soft_compact_distribution_membership",
            "status": (
                "word_boundary_schema_found"
                if word_parse_ok > 0
                else "blocked_word_boundary_schema_not_found"
            ),
            "n_required_features": 27,
            "n_available_features": 0,
            "blocking_issue": (
                "" if word_parse_ok > 0
                else "WebQAmGaze word rectangles not yet resolved"
            ),
            "next_action": "reuse exact ZuCo Soft-AOI extractor on WebQAmGaze AOIs",
        },
        {
            "feature_set": "official_sent_gaze_sacc",
            "status": (
                "fixations_ready_word_count_pending"
                if fixation_parse_ok > 0
                else "blocked_fixation_schema_not_found"
            ),
            "n_required_features": 9,
            "n_available_features": 6 if fixation_parse_ok > 0 else 0,
            "blocking_issue": (
                "word count/word fixation assignment needed for omission_rate, "
                "fixation_number, reading_speed"
                if fixation_parse_ok > 0
                else "official fixation files not resolved"
            ),
            "next_action": (
                "derive the six saccade aggregates from consecutive filtered "
                "fixations and three word-normalized fixation features from AOIs"
            ),
        },
    ]
    write_csv(paths["readiness"], readiness_rows)

    coverage = count_trial_coverage(trials, inventory)
    manifest = {
        "stage": "webqamgaze_external_validation_stage3a",
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "webqamgaze_root": str(root),
        "webqamgaze_git_commit": git_commit(root),
        "stage2_dir": str(stage2_dir),
        "frozen_dir": str(frozen_dir),
        "frozen_feature_schema_sha256": sha256_file(
            required_paths["feature_schema"]
        ),
        "frozen_model_manifest_sha256": sha256_file(
            required_paths["model_manifest"]
        ),
        "coordinate_mapping": {
            "source": "Stage 2 screen-normalized x_norm/y_norm",
            "target_reference_width_px": float(
                args.zuco_reference_width_px
            ),
            "target_reference_height_px": float(
                args.zuco_reference_height_px
            ),
            "formula": (
                "x_zuco_px=x_norm*reference_width; "
                "y_zuco_px=y_norm*reference_height"
            ),
        },
        "rawstat": {
            "feature_order": schema_features,
            "n_trials": len(features),
            "n_participants": int(features["participant_id"].nunique()),
            "task_counts": features["task"].value_counts().sort_index().to_dict(),
            "finite_counts": finite_counts,
            "missing_counts": missing_counts,
            "pupil_unavailable_features": sorted(
                PUPIL_UNAVAILABLE_FEATURES
            ),
            "missing_value_rule": (
                "write NaN; never estimate imputation statistics from "
                "WebQAmGaze"
            ),
        },
        "resource_audit": {
            "n_resource_files": len(resource_rows),
            "n_candidate_word_boundary_files": len(word_like),
            "n_parsed_word_boundary_schema_samples": len(word_schema_rows),
            "n_word_schema_parse_ok": word_parse_ok,
            "n_fixation_data_per_part_files": len(fixation_like),
            "n_parsed_fixation_schema_samples": len(fixation_schema_rows),
            "n_fixation_schema_parse_ok": fixation_parse_ok,
            **coverage,
        },
        "official_feature_definition": OFFICIAL_FEATURE_NAMES,
        "constraints": [
            "No classifier was fitted or applied.",
            "No WebQAmGaze label was used to select a feature formula.",
            "No WebQAmGaze value was used to estimate imputation or scaling.",
            "RawStat feature order is taken from the frozen Common16 schema.",
            "Pupil-dependent values are NaN and must be handled only by the frozen ZuCo preprocessing pipeline.",
            "Soft-AOI and official features are not fabricated when word AOIs are unresolved.",
        ],
        "outputs": {
            key: str(value) for key, value in paths.items()
            if key != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE EXTERNAL VALIDATION STAGE 3A ===")
    print(f"WebQAmGaze commit: {manifest['webqamgaze_git_commit']}")
    print(
        "Frozen RawStat schema: "
        f"{len(schema_features)} features, exact expected order"
    )
    print(
        "Coordinate reference: "
        f"{args.zuco_reference_width_px:g} x "
        f"{args.zuco_reference_height_px:g}"
    )
    print(
        "RawStat trials: "
        f"{len(features)} across "
        f"{features['participant_id'].nunique()} participants"
    )
    print("RawStat task counts:")
    for task, count in sorted(
        features["task"].value_counts().to_dict().items()
    ):
        print(f"  {task}: {count}")
    print("RawStat missing counts:")
    for feature in schema_features:
        count = missing_counts[feature]
        if count:
            print(f"  {feature}: {count}/{len(features)}")
    print(f"AOI/resource files found: {len(resource_rows)}")
    print(f"Candidate word-boundary files: {len(word_like)}")
    print(f"Parsed word-boundary schema samples: {word_parse_ok}")
    print(
        "fixation_data_per_part files: "
        f"{len(fixation_like)}"
    )
    print(f"Parsed fixation schema samples: {fixation_parse_ok}")
    print(
        "Primary trial names appearing in resource paths: "
        f"{coverage['n_trial_names_appearing_in_resource_paths']}/"
        f"{coverage['n_primary_trial_names']}"
    )
    print("Written:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
