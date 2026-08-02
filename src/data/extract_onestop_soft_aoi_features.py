#!/usr/bin/env python
r"""
Extract exact ZuCo-compatible Soft-AOI features from OneStop 30 Hz raw gaze.

This adapter deliberately reuses the existing implementation in
``src/data/extract_zuco_soft_aoi_features.py`` rather than reimplementing the
41 Soft-AOI formulas. It converts OneStop's trial metadata, 30 Hz sample table,
and paragraph IA report geometry into the schema expected by that extractor.

Recommended first run (two audited participants):
python -u src\data\extract_onestop_soft_aoi_features.py ^
  --samples-dir data\interim\onestop\raw_samples_30hz_asc344 ^
  --manifest data\interim\onestop\onestop_regular_trial_manifest.csv ^
  --word-aoi data\interim\onestop\onestop_regular_word_aoi.parquet ^
  --raw-trial-summary outputs\onestop_raw_extraction_asc344\onestop_raw_extraction_trial_summary.csv ^
  --frozen-schema outputs\frozen_zuco_external_validation_v1\feature_schema.csv ^
  --participants l1_155,l1_161 ^
  --output-dir data\processed\onestop\soft_aoi_sample ^
  --audit-dir outputs\onestop_soft_aoi_sample

After the sample audit passes, use ``--participants all`` for the 344-participant
ASC cohort. Participant-level Parquet caches make the extraction resumable.

Important design choices
------------------------
- NR=1 and IS=0 are taken directly from the canonical OneStop manifest.
- Word AOIs are taken from the participant/trial-specific Paragraph IA report.
- Reading order is reconstructed by line clustering followed by left-to-right
  word order, matching the transport logic used for the external webcam data.
- Samples are already on a regular 30 Hz grid. The imported extractor is called
  with its BASE_RATE_HZ to prevent a second sampling operation, then the stored
  rate metadata is corrected to 30 Hz.
- OneStop has no usable pupil requirement for this feature family; ``valid_basic``
  is exactly the existing 30 Hz sample validity flag plus finite x/y.
- The Compact feature order is read from the frozen ZuCo schema; no OneStop
  labels or performance results are used to choose features.

Outputs
-------
<output-dir>/participant_features/<participant>_soft_aoi_30hz.parquet
<output-dir>/onestop_soft_aoi_all_features.parquet
<output-dir>/onestop_soft_aoi_full_features.csv
<output-dir>/onestop_soft_aoi_compact27_features.csv
<audit-dir>/onestop_soft_aoi_trial_audit.csv
<audit-dir>/onestop_soft_aoi_participant_summary.csv
<audit-dir>/onestop_soft_aoi_feature_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import extract_zuco_soft_aoi_features as soft
except ImportError as exc:
    raise ImportError(
        "Could not import src/data/extract_zuco_soft_aoi_features.py. "
        "Place this script in src/data beside the existing exact extractor."
    ) from exc


FULL_FEATURES = list(
    dict.fromkeys(
        list(soft.CONTENT_FEATURES)
        + list(soft.MEMBERSHIP_QUALITY_FEATURES)
    )
)

CORE_METADATA = [
    "onestop_trial_id",
    "participant_id",
    "subject_id",
    "regime",
    "label",
    "label_name",
    "trial_index",
    "stimulus_id",
    "task_context_id",
    "article_batch",
    "article_id",
    "article_index",
    "paragraph_id",
    "difficulty_level",
    "onestopqa_question_id",
    "question_preview",
    "article_title",
]

REQUIRED_MANIFEST = {
    "onestop_trial_id",
    "participant_id",
    "regime",
    "label",
    "label_name",
    "trial_index",
    "stimulus_id",
    "task_context_id",
    "article_batch",
    "article_id",
    "article_index",
    "paragraph_id",
    "difficulty_level",
    "onestopqa_question_id",
    "question_preview",
    "article_title",
}

REQUIRED_SAMPLE = {
    "onestop_trial_id",
    "participant_id",
    "regime",
    "label",
    "label_name",
    "trial_index",
    "sample_index_30hz",
    "target_timestamp_ms",
    "x_px",
    "y_px",
    "valid",
}

REQUIRED_AOI = {
    "onestop_trial_id",
    "participant_id",
    "ia_id",
    "ia_label",
    "ia_left",
    "ia_right",
    "ia_top",
    "ia_bottom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/interim/onestop/raw_samples_30hz_asc344"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/interim/onestop/onestop_regular_trial_manifest.csv"
        ),
    )
    parser.add_argument(
        "--word-aoi",
        type=Path,
        default=Path(
            "data/interim/onestop/onestop_regular_word_aoi.parquet"
        ),
    )
    parser.add_argument(
        "--raw-trial-summary",
        type=Path,
        default=Path(
            "outputs/onestop_raw_extraction_asc344/"
            "onestop_raw_extraction_trial_summary.csv"
        ),
    )
    parser.add_argument(
        "--frozen-schema",
        type=Path,
        default=Path(
            "outputs/frozen_zuco_external_validation_v1/feature_schema.csv"
        ),
    )
    parser.add_argument(
        "--participants",
        default="l1_155,l1_161",
        help="Comma-separated participant IDs, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/onestop/soft_aoi_sample"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/onestop_soft_aoi_sample"),
    )
    parser.add_argument(
        "--sigma-word-height-ratio",
        type=float,
        default=2.0 / 3.0,
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument("--target-rate-hz", type=int, default=30)
    parser.add_argument(
        "--severe-alignment-threshold",
        type=float,
        default=0.10,
        help=(
            "Fail only when participant-median active rate, envelope hit, or "
            "qmax mean falls below this severe coordinate-mismatch threshold."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute participant caches that already exist.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def normalize_id(value: Any) -> str:
    return str(value).strip().lower()


def normalize_integer(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(
        {"true", "1", "1.0", "yes", "y", "t"}
    ).fillna(False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_participants(
    argument: str,
    manifest: pd.DataFrame,
    samples_dir: Path,
) -> list[str]:
    available_files = {
        path.name[: -len("_samples_30hz.parquet")].lower(): path
        for path in samples_dir.glob("*_samples_30hz.parquet")
    }
    if argument.strip().lower() == "all":
        participants = sorted(available_files)
    else:
        participants = [
            normalize_id(item)
            for item in argument.split(",")
            if item.strip()
        ]
    if not participants:
        raise ValueError("No participants were selected.")
    if len(participants) != len(set(participants)):
        raise ValueError("Participant selection contains duplicates.")

    manifest_participants = set(manifest["participant_id"])
    missing_manifest = sorted(set(participants) - manifest_participants)
    missing_samples = sorted(set(participants) - set(available_files))
    if missing_manifest:
        raise ValueError(
            f"Participants absent from manifest: {missing_manifest}"
        )
    if missing_samples:
        raise FileNotFoundError(
            f"Participant sample files are missing: {missing_samples}"
        )
    return participants


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path), dtype="string", low_memory=False)
    missing = sorted(REQUIRED_MANIFEST - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest lacks columns: {missing}")
    frame["participant_id"] = (
        frame["participant_id"].astype("string").str.strip().str.lower()
    )
    frame["onestop_trial_id"] = frame["onestop_trial_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if frame["onestop_trial_id"].duplicated().any():
        raise ValueError("Canonical manifest contains duplicate trial IDs.")
    expected_labels = {
        ("ordinary", "NR", 1),
        ("information_seeking", "IS", 0),
    }
    observed_labels = {
        (str(row.regime), str(row.label_name), int(row.label))
        for row in frame[["regime", "label_name", "label"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if observed_labels != expected_labels:
        raise ValueError(
            f"Unexpected OneStop label semantics: {observed_labels}"
        )
    return frame


def load_compact_schema(path: Path) -> list[str]:
    """
    Load the frozen Compact feature order.

    The frozen schema can contain the same 27-column feature definition more
    than once (for example, once per frozen model/artifact).  For transport to
    OneStop we need the ordered *unique feature names*, not the number of schema
    rows.  De-duplication therefore preserves first appearance and then applies
    strict 27-column and exact-extractor checks.
    """
    schema = pd.read_csv(require_file(path))
    required = {"feature_set", "feature_name"}
    missing = sorted(required - set(schema.columns))
    if missing:
        raise ValueError(f"Frozen schema lacks columns: {missing}")

    selected_rows = schema.loc[
        schema["feature_set"].astype(str).eq(
            "soft_compact_distribution_membership"
        ),
        "feature_name",
    ].astype(str)

    raw_features = selected_rows.tolist()
    selected = list(dict.fromkeys(raw_features))
    duplicate_rows = len(raw_features) - len(selected)

    print(
        "Frozen Compact schema rows: "
        f"{len(raw_features)}; ordered unique features: {len(selected)}; "
        f"duplicate rows ignored: {duplicate_rows}"
    )

    if len(selected) != 27:
        counts = selected_rows.value_counts().sort_index().to_dict()
        raise ValueError(
            "Frozen Compact schema does not resolve to 27 ordered unique "
            f"features: raw_rows={len(raw_features)}, "
            f"unique_features={len(selected)}, counts={counts}"
        )

    missing_from_exact = sorted(set(selected) - set(FULL_FEATURES))
    if missing_from_exact:
        raise ValueError(
            "Frozen Compact schema differs from the exact extractor; "
            f"unknown_features={missing_from_exact}"
        )

    if len(FULL_FEATURES) != 41:
        raise ValueError(
            f"Exact Soft-AOI extractor exposes {len(FULL_FEATURES)} features, "
            "expected 41."
        )

    return selected


def sample_path(samples_dir: Path, participant: str) -> Path:
    return require_file(
        samples_dir / f"{participant}_samples_30hz.parquet"
    )


def read_participant_aoi(path: Path, participant: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(
            require_file(path),
            filters=[("participant_id", "==", participant)],
        )
    except Exception:
        # Compatibility fallback for older pandas/pyarrow combinations.
        frame = pd.read_parquet(require_file(path))
        frame = frame.loc[
            frame["participant_id"].astype(str).str.lower().eq(participant)
        ].copy()

    missing = sorted(REQUIRED_AOI - set(frame.columns))
    if missing:
        raise ValueError(f"Word-AOI table lacks columns: {missing}")
    frame["participant_id"] = (
        frame["participant_id"].astype(str).str.strip().str.lower()
    )
    frame["onestop_trial_id"] = frame["onestop_trial_id"].astype(str)
    return frame


def cluster_lines(aoi: pd.DataFrame) -> pd.DataFrame:
    result = aoi.copy()
    result["x_min"] = pd.to_numeric(
        result["ia_left"], errors="coerce"
    )
    result["x_max"] = pd.to_numeric(
        result["ia_right"], errors="coerce"
    )
    result["y_min"] = pd.to_numeric(
        result["ia_top"], errors="coerce"
    )
    result["y_max"] = pd.to_numeric(
        result["ia_bottom"], errors="coerce"
    )
    result["height"] = result["y_max"] - result["y_min"]
    result["width"] = result["x_max"] - result["x_min"]
    result["x_center"] = (result["x_min"] + result["x_max"]) / 2.0
    result["y_center"] = (result["y_min"] + result["y_max"]) / 2.0

    finite_geometry = np.isfinite(
        result[
            ["x_min", "x_max", "y_min", "y_max", "height", "width"]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    positive_geometry = (
        result["width"].gt(0) & result["height"].gt(0)
    )
    result = result.loc[finite_geometry & positive_geometry].copy()
    if result.empty:
        raise ValueError("Trial AOI has no valid word rectangles.")

    # Keep exactly one geometry per IA ID. Multiple identical report rows are
    # harmless; conflicting geometries for one IA ID are not.
    conflict = (
        result.groupby("ia_id", dropna=False)[
            ["x_min", "x_max", "y_min", "y_max"]
        ]
        .nunique(dropna=False)
        .max(axis=1)
        .gt(1)
    )
    if conflict.any():
        bad_ids = conflict[conflict].index.astype(str).tolist()
        raise ValueError(
            f"AOI IDs have conflicting geometry: {bad_ids[:20]}"
        )
    result = (
        result.sort_values(["ia_id", "y_center", "x_center"], kind="stable")
        .drop_duplicates("ia_id", keep="first")
        .reset_index(drop=True)
    )

    median_height = float(result["height"].median())
    if not math.isfinite(median_height) or median_height <= 0:
        raise ValueError("Trial AOI median word height is invalid.")
    tolerance = max(1.0, 0.60 * median_height)

    assignments: dict[int, int] = {}
    line_centers: list[float] = []
    for index in result.sort_values(
        ["y_center", "x_center"], kind="stable"
    ).index:
        y = float(result.at[index, "y_center"])
        if not line_centers:
            line_centers.append(y)
            assignments[int(index)] = 0
            continue
        distances = np.abs(np.asarray(line_centers, dtype=float) - y)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= tolerance:
            assignments[int(index)] = nearest
            line_centers[nearest] = (
                line_centers[nearest] + y
            ) / 2.0
        else:
            line_centers.append(y)
            assignments[int(index)] = len(line_centers) - 1

    line_order = np.argsort(np.asarray(line_centers, dtype=float))
    line_remap = {
        int(old): int(new) for new, old in enumerate(line_order)
    }
    result["line_index"] = [
        line_remap[assignments[int(index)]] for index in result.index
    ]

    reading_indices = result.sort_values(
        ["line_index", "x_center", "ia_id"],
        kind="stable",
    ).index
    word_index_map = {
        int(index): position
        for position, index in enumerate(reading_indices)
    }
    result["word_index"] = [
        word_index_map[int(index)] for index in result.index
    ]
    result = result.sort_values("word_index").reset_index(drop=True)

    n_words = len(result)
    n_lines = int(result["line_index"].max()) + 1
    result["word_progress"] = (
        result["word_index"] / float(n_words - 1)
        if n_words > 1
        else 0.0
    )
    result["line_index_norm"] = (
        result["line_index"] / float(n_lines - 1)
        if n_lines > 1
        else 0.0
    )
    result["line_progress"] = 0.0
    for _, indices in result.groupby("line_index", sort=True).groups.items():
        ordered = result.loc[list(indices)].sort_values("word_index").index
        count = len(ordered)
        progress = (
            np.linspace(0.0, 1.0, count)
            if count > 1
            else np.asarray([0.0])
        )
        result.loc[ordered, "line_progress"] = progress

    result["bounds_source_subject"] = (
        result["participant_id"].iloc[0]
    )
    return result


def make_trial_row(
    manifest_row: pd.Series,
    stimulus_index: dict[str, int],
) -> pd.Series:
    return pd.Series(
        {
            "subject_id": str(manifest_row["participant_id"]),
            "task": str(manifest_row["label_name"]),
            "block_id": normalize_integer(
                manifest_row.get("article_batch"), default=0
            ),
            "trial_index_in_block": normalize_integer(
                manifest_row.get("trial_index"), default=0
            ),
            "trial_id": str(manifest_row["onestop_trial_id"]),
            "task_sentence_index": int(
                stimulus_index[str(manifest_row["stimulus_id"])]
            ),
        }
    )


def adapt_samples(samples: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_SAMPLE - set(samples.columns))
    if missing:
        raise ValueError(f"Sample Parquet lacks columns: {missing}")

    adapted = pd.DataFrame(
        {
            "trial_id": samples["onestop_trial_id"].astype(str),
            "sample_index_in_trial": pd.to_numeric(
                samples["sample_index_30hz"], errors="raise"
            ).astype(np.int64),
            "timestamp_ms_abs": pd.to_numeric(
                samples["target_timestamp_ms"], errors="raise"
            ).astype(float),
            "x_px": pd.to_numeric(
                samples["x_px"], errors="coerce"
            ).astype(float),
            "y_px": pd.to_numeric(
                samples["y_px"], errors="coerce"
            ).astype(float),
            "valid_basic": to_bool_series(samples["valid"]),
        }
    )
    adapted["valid_basic"] = (
        adapted["valid_basic"]
        & np.isfinite(adapted["x_px"])
        & np.isfinite(adapted["y_px"])
    )
    return adapted


def validate_cached(
    path: Path,
    expected_ids: set[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "onestop_trial_id",
        "participant_id",
        *FULL_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cached feature file lacks columns: {missing}")
    ids = set(frame["onestop_trial_id"].astype(str))
    if ids != expected_ids:
        raise ValueError(
            f"Cached trial IDs differ: missing={len(expected_ids-ids)}, "
            f"unexpected={len(ids-expected_ids)}"
        )
    if frame["onestop_trial_id"].duplicated().any():
        raise ValueError("Cached feature file contains duplicate trial IDs.")
    if np.isinf(frame[FULL_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Cached feature file contains infinity.")
    return frame


def extract_participant(
    participant: str,
    samples_dir: Path,
    participant_manifest: pd.DataFrame,
    participant_aoi: pd.DataFrame,
    stimulus_index: dict[str, int],
    sigma_ratio: float,
    active_threshold: float,
    target_rate_hz: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_samples = pd.read_parquet(sample_path(samples_dir, participant))
    source_samples["participant_id"] = (
        source_samples["participant_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if set(source_samples["participant_id"]) != {participant}:
        raise ValueError(
            f"{participant} sample file contains another participant."
        )

    available_ids = set(
        source_samples["onestop_trial_id"].astype(str).unique()
    )
    manifest_lookup = participant_manifest.set_index(
        "onestop_trial_id", drop=False, verify_integrity=True
    )
    missing_manifest = sorted(available_ids - set(manifest_lookup.index))
    if missing_manifest:
        raise ValueError(
            f"Raw sample trials absent from manifest: {missing_manifest[:20]}"
        )

    aoi_groups = {
        str(trial_id): group.copy()
        for trial_id, group in participant_aoi.groupby(
            "onestop_trial_id", sort=False
        )
    }
    missing_aoi = sorted(available_ids - set(aoi_groups))
    if missing_aoi:
        raise ValueError(
            f"Raw sample trials absent from AOI table: {missing_aoi[:20]}"
        )

    adapted = adapt_samples(source_samples)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    grouped_source = {
        str(trial_id): group.copy()
        for trial_id, group in source_samples.groupby(
            "onestop_trial_id", sort=False
        )
    }
    grouped_adapted = {
        str(trial_id): group.copy()
        for trial_id, group in adapted.groupby("trial_id", sort=False)
    }

    for position, trial_id in enumerate(sorted(available_ids), start=1):
        manifest_row = manifest_lookup.loc[trial_id]
        aoi = cluster_lines(aoi_groups[trial_id])
        trial_samples = grouped_adapted[trial_id]
        source_trial = grouped_source[trial_id]

        trial_row = make_trial_row(manifest_row, stimulus_index)
        feature = soft.extract_trial_features(
            samples=trial_samples,
            trial_row=trial_row,
            aoi=aoi,
            # The input is already 30 Hz. BASE_RATE_HZ prevents a second
            # selection pass; the rate metadata is corrected below.
            rate_hz=int(soft.BASE_RATE_HZ),
            sigma_word_height_ratio=sigma_ratio,
            active_threshold=active_threshold,
        )
        feature["soft_aoi_rate_hz"] = int(target_rate_hz)

        for column in CORE_METADATA:
            if column == "subject_id":
                feature[column] = str(participant)
            elif column == "participant_id":
                feature[column] = str(participant)
            elif column in manifest_row.index:
                feature[column] = manifest_row[column]

        feature["onestop_trial_id"] = trial_id
        feature["soft_aoi_input_rate_hz"] = int(target_rate_hz)
        feature["soft_aoi_source"] = "OneStop_ASC344"
        rows.append(feature)

        valid_xy = (
            to_bool_series(source_trial["valid"])
            & np.isfinite(
                pd.to_numeric(
                    source_trial["x_px"], errors="coerce"
                ).to_numpy(dtype=float)
            )
            & np.isfinite(
                pd.to_numeric(
                    source_trial["y_px"], errors="coerce"
                ).to_numpy(dtype=float)
            )
        )
        audit_rows.append(
            {
                "onestop_trial_id": trial_id,
                "participant_id": participant,
                "label_name": manifest_row["label_name"],
                "n_input_samples": int(len(source_trial)),
                "n_valid_xy_samples": int(valid_xy.sum()),
                "n_words": int(len(aoi)),
                "n_lines": int(aoi["line_index"].max()) + 1,
                "aoi_x_min": float(aoi["x_min"].min()),
                "aoi_x_max": float(aoi["x_max"].max()),
                "aoi_y_min": float(aoi["y_min"].min()),
                "aoi_y_max": float(aoi["y_max"].max()),
                "median_word_height_px": float(aoi["height"].median()),
                "soft_aoi_n_samples": int(feature["soft_aoi_n_samples"]),
                "sample_count_preserved": (
                    int(feature["soft_aoi_n_samples"])
                    == int(len(source_trial))
                ),
                "soft_aoi_valid_rate": feature["soft_aoi_valid_rate"],
                "soft_aoi_active_rate": feature["soft_aoi_active_rate"],
                "soft_aoi_word_box_hit_rate": (
                    feature["soft_aoi_word_box_hit_rate"]
                ),
                "soft_aoi_sentence_envelope_hit_rate": (
                    feature["soft_aoi_sentence_envelope_hit_rate"]
                ),
                "soft_aoi_qmax_mean": feature["soft_aoi_qmax_mean"],
                "soft_aoi_sigma_px": feature["soft_aoi_sigma_px"],
            }
        )

        if position % 10 == 0 or position == len(available_ids):
            print(
                f"    {participant}: trials {position}/{len(available_ids)}"
            )

    features = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    if set(features["onestop_trial_id"]) != available_ids:
        raise ValueError(f"{participant}: extracted feature IDs differ.")
    if features["onestop_trial_id"].duplicated().any():
        raise ValueError(f"{participant}: duplicate feature trial IDs.")
    if np.isinf(features[FULL_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError(f"{participant}: feature table contains infinity.")
    if not audit["sample_count_preserved"].all():
        bad = audit.loc[
            ~audit["sample_count_preserved"], "onestop_trial_id"
        ].tolist()
        raise ValueError(
            f"{participant}: input sample count changed for {bad[:20]}"
        )
    return features, audit


def main() -> int:
    args = parse_args()
    if args.sigma_word_height_ratio <= 0:
        raise ValueError("--sigma-word-height-ratio must be positive.")
    if not (0 < args.active_threshold <= 1):
        raise ValueError("--active-threshold must be within (0, 1].")
    if args.target_rate_hz <= 0:
        raise ValueError("--target-rate-hz must be positive.")
    if not (0 <= args.severe_alignment_threshold <= 1):
        raise ValueError(
            "--severe-alignment-threshold must be within [0, 1]."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "participant_features"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    compact_features = load_compact_schema(args.frozen_schema)
    participants = parse_participants(
        args.participants, manifest, args.samples_dir
    )

    selected_manifest = manifest.loc[
        manifest["participant_id"].isin(participants)
    ].copy()
    stimulus_values = sorted(selected_manifest["stimulus_id"].astype(str).unique())
    stimulus_index = {
        value: index for index, value in enumerate(stimulus_values)
    }

    raw_summary_ids: set[str] | None = None
    if args.raw_trial_summary.is_file():
        raw_summary = pd.read_csv(
            args.raw_trial_summary, dtype="string", low_memory=False
        )
        if "onestop_trial_id" not in raw_summary.columns:
            raise ValueError(
                "Raw extraction trial summary lacks onestop_trial_id."
            )
        if "participant_id" in raw_summary.columns:
            raw_summary["participant_id"] = (
                raw_summary["participant_id"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            raw_summary = raw_summary.loc[
                raw_summary["participant_id"].isin(participants)
            ]
        raw_summary_ids = set(
            raw_summary["onestop_trial_id"].astype(str)
        )

    print("=== ONESTOP SOFT-AOI INPUT ===")
    print(f"participants: {len(participants)}")
    print(
        "selected manifest trials: "
        f"{selected_manifest['onestop_trial_id'].nunique():,}"
    )
    print(f"exact full features: {len(FULL_FEATURES)}")
    print(f"frozen Compact features: {len(compact_features)}")
    print(
        f"sigma ratio={args.sigma_word_height_ratio:g}, "
        f"active threshold={args.active_threshold:g}"
    )

    feature_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []

    for index, participant in enumerate(participants, start=1):
        participant_manifest = selected_manifest.loc[
            selected_manifest["participant_id"].eq(participant)
        ].copy()
        participant_aoi = read_participant_aoi(
            args.word_aoi, participant
        )

        source_path = sample_path(args.samples_dir, participant)
        source_ids = set(
            pd.read_parquet(
                source_path,
                columns=["onestop_trial_id"],
            )["onestop_trial_id"].astype(str)
        )
        cache_path = (
            cache_dir / f"{participant}_soft_aoi_30hz.parquet"
        )

        print(
            f"[{index}/{len(participants)}] {participant}: "
            f"raw_trials={len(source_ids)}"
        )
        if cache_path.is_file() and not args.overwrite:
            features = validate_cached(cache_path, source_ids)
            # Rebuild the compact audit from cached features.
            audit = features[
                [
                    "onestop_trial_id",
                    "participant_id",
                    "label_name",
                    "soft_aoi_n_samples",
                    "soft_aoi_n_valid_samples",
                    "soft_aoi_n_words",
                    "soft_aoi_n_lines",
                    "soft_aoi_sigma_px",
                    "soft_aoi_valid_rate",
                    "soft_aoi_active_rate",
                    "soft_aoi_word_box_hit_rate",
                    "soft_aoi_sentence_envelope_hit_rate",
                    "soft_aoi_qmax_mean",
                ]
            ].copy()
            audit = audit.rename(
                columns={
                    "soft_aoi_n_samples": "n_input_samples",
                    "soft_aoi_n_valid_samples": "n_valid_xy_samples",
                    "soft_aoi_n_words": "n_words",
                    "soft_aoi_n_lines": "n_lines",
                }
            )
            audit["sample_count_preserved"] = True
            print("    loaded valid cache")
        else:
            features, audit = extract_participant(
                participant=participant,
                samples_dir=args.samples_dir,
                participant_manifest=participant_manifest,
                participant_aoi=participant_aoi,
                stimulus_index=stimulus_index,
                sigma_ratio=args.sigma_word_height_ratio,
                active_threshold=args.active_threshold,
                target_rate_hz=args.target_rate_hz,
            )
            features.to_parquet(
                cache_path, index=False, compression="zstd"
            )
            print(f"    saved {cache_path}")

        feature_parts.append(features)
        audit_parts.append(audit)

    all_features = pd.concat(feature_parts, ignore_index=True)
    trial_audit = pd.concat(audit_parts, ignore_index=True)

    if all_features["onestop_trial_id"].duplicated().any():
        raise ValueError("Combined feature table contains duplicate trial IDs.")

    extracted_ids = set(all_features["onestop_trial_id"].astype(str))
    sample_ids: set[str] = set()
    for participant in participants:
        ids = pd.read_parquet(
            sample_path(args.samples_dir, participant),
            columns=["onestop_trial_id"],
        )["onestop_trial_id"].astype(str)
        sample_ids.update(ids)
    if extracted_ids != sample_ids:
        raise ValueError(
            "Combined feature IDs differ from selected raw sample IDs."
        )
    if raw_summary_ids is not None and extracted_ids != raw_summary_ids:
        raise ValueError(
            "Feature IDs differ from selected raw extraction summary: "
            f"missing={len(raw_summary_ids-extracted_ids)}, "
            f"unexpected={len(extracted_ids-raw_summary_ids)}"
        )

    metadata_columns = [
        column for column in CORE_METADATA if column in all_features.columns
    ]
    auxiliary_columns = [
        "soft_aoi_rate_hz",
        "soft_aoi_input_rate_hz",
        "soft_aoi_sigma_word_height_ratio",
        "soft_aoi_sigma_px",
        "soft_aoi_active_threshold",
        "soft_aoi_n_words",
        "soft_aoi_n_lines",
        "soft_aoi_n_samples",
        "soft_aoi_n_valid_samples",
        "soft_aoi_duration_sec",
        "bounds_source_subject",
        "soft_aoi_source",
    ]
    auxiliary_columns = [
        column
        for column in auxiliary_columns
        if column in all_features.columns
    ]

    full_table = all_features[
        metadata_columns + auxiliary_columns + FULL_FEATURES
    ].copy()
    compact_table = all_features[
        metadata_columns + auxiliary_columns + compact_features
    ].copy()

    all_path = args.output_dir / "onestop_soft_aoi_all_features.parquet"
    full_path = args.output_dir / "onestop_soft_aoi_full_features.csv"
    compact_path = (
        args.output_dir / "onestop_soft_aoi_compact27_features.csv"
    )
    trial_audit_path = (
        args.audit_dir / "onestop_soft_aoi_trial_audit.csv"
    )
    participant_summary_path = (
        args.audit_dir / "onestop_soft_aoi_participant_summary.csv"
    )
    manifest_path = (
        args.audit_dir / "onestop_soft_aoi_feature_manifest.json"
    )

    all_features.to_parquet(
        all_path, index=False, compression="zstd"
    )
    full_table.to_csv(full_path, index=False, encoding="utf-8-sig")
    compact_table.to_csv(
        compact_path, index=False, encoding="utf-8-sig"
    )
    trial_audit.to_csv(
        trial_audit_path, index=False, encoding="utf-8-sig"
    )

    participant_summary = (
        all_features.groupby(
            ["participant_id", "label_name"], observed=True
        )
        .agg(
            trials=("onestop_trial_id", "size"),
            valid_rate_median=("soft_aoi_valid_rate", "median"),
            active_rate_median=("soft_aoi_active_rate", "median"),
            box_hit_rate_median=(
                "soft_aoi_word_box_hit_rate", "median"
            ),
            envelope_hit_rate_median=(
                "soft_aoi_sentence_envelope_hit_rate", "median"
            ),
            qmax_mean_median=("soft_aoi_qmax_mean", "median"),
            nearest_distance_median_wh=(
                "soft_aoi_nearest_distance_median_word_heights",
                "median",
            ),
            word_coverage_median=(
                "soft_aoi_dominant_word_coverage", "median"
            ),
            duration_sec_median=("soft_aoi_duration_sec", "median"),
        )
        .reset_index()
    )
    participant_summary.to_csv(
        participant_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest_without_samples = sorted(
        set(selected_manifest["onestop_trial_id"].astype(str))
        - extracted_ids
    )
    severe = participant_summary.loc[
        (
            participant_summary["active_rate_median"]
            < args.severe_alignment_threshold
        )
        | (
            participant_summary["envelope_hit_rate_median"]
            < args.severe_alignment_threshold
        )
        | (
            participant_summary["qmax_mean_median"]
            < args.severe_alignment_threshold
        )
    ].copy()

    feature_manifest = {
        "dataset": "OneStop",
        "cohort": (
            "selected participants from ASC344; raw-unavailable trials "
            "remain excluded"
        ),
        "participants": participants,
        "n_participants": len(participants),
        "n_feature_rows": int(len(all_features)),
        "label_semantics": {
            "NR": 1,
            "IS": 0,
        },
        "sampling_rate_hz": args.target_rate_hz,
        "exact_extractor_module": str(
            Path(soft.__file__).resolve()
        ),
        "exact_extractor_sha256": sha256_file(
            Path(soft.__file__).resolve()
        ),
        "sigma_definition": (
            "participant/trial median word height times ratio"
        ),
        "sigma_word_height_ratio": args.sigma_word_height_ratio,
        "active_threshold": args.active_threshold,
        "aoi_order_definition": (
            "vertical line clustering at 0.60 times median word height, "
            "then left-to-right order within line"
        ),
        "full_feature_columns": FULL_FEATURES,
        "compact_feature_set": (
            "soft_compact_distribution_membership"
        ),
        "compact_feature_columns": compact_features,
        "manifest_trials_selected": int(
            selected_manifest["onestop_trial_id"].nunique()
        ),
        "manifest_without_raw_samples_count": len(
            manifest_without_samples
        ),
        "manifest_without_raw_samples": manifest_without_samples,
        "severe_coordinate_alignment_participants": (
            severe["participant_id"].astype(str).tolist()
        ),
        "outputs": {
            "all_features_parquet": str(all_path),
            "full_features_csv": str(full_path),
            "compact27_features_csv": str(compact_path),
            "trial_audit_csv": str(trial_audit_path),
            "participant_summary_csv": str(
                participant_summary_path
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(feature_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ONESTOP SOFT-AOI FEATURE SUMMARY ===")
    print(
        json.dumps(
            {
                "participants": len(participants),
                "feature_rows": len(all_features),
                "full_features": len(FULL_FEATURES),
                "compact_features": len(compact_features),
                "manifest_without_raw_samples": len(
                    manifest_without_samples
                ),
                "severe_coordinate_alignment_participants": len(severe),
                "nan_cells_full_features": int(
                    all_features[FULL_FEATURES].isna().sum().sum()
                ),
                "infinite_cells_full_features": int(
                    np.isinf(
                        all_features[FULL_FEATURES].to_numpy(dtype=float)
                    ).sum()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== LABEL-WISE MEDIANS ===")
    label_summary = (
        all_features.groupby("label_name", observed=True)
        .agg(
            trials=("onestop_trial_id", "size"),
            participants=("participant_id", "nunique"),
            valid_rate=("soft_aoi_valid_rate", "median"),
            active_rate=("soft_aoi_active_rate", "median"),
            box_hit_rate=("soft_aoi_word_box_hit_rate", "median"),
            envelope_hit_rate=(
                "soft_aoi_sentence_envelope_hit_rate", "median"
            ),
            qmax_mean=("soft_aoi_qmax_mean", "median"),
            word_coverage=(
                "soft_aoi_dominant_word_coverage", "median"
            ),
            duration_sec=("soft_aoi_duration_sec", "median"),
        )
        .reset_index()
    )
    with pd.option_context(
        "display.max_columns", None, "display.width", 220
    ):
        print(label_summary.to_string(index=False))

    print("\n=== PARTICIPANT MEDIAN QUALITY ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 240
    ):
        print(participant_summary.to_string(index=False))

    print(f"\nSaved: {all_path}")
    print(f"Saved: {full_path}")
    print(f"Saved: {compact_path}")
    print(f"Saved: {trial_audit_path}")
    print(f"Saved: {manifest_path}")

    if severe.empty:
        print("\nAUDIT STATUS: PASS")
        return 0

    print(
        "\nAUDIT STATUS: FAIL — severe coordinate/AOI alignment problem "
        f"for participants: {severe['participant_id'].tolist()}",
        file=sys.stderr,
    )
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
