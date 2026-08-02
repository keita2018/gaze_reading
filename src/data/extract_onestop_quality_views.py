#!/usr/bin/env python
r"""
Create paired clean/degraded OneStop Soft-AOI feature views for
quality-contrastive learning.

The degradation conditions intentionally match the existing ZuCo robustness
study at the sample level:

clean
    Existing clean 30 Hz Soft-AOI features are reused unchanged.
rate10
    Keep every third sample from the regular 30 Hz sequence.
noise50
    Add deterministic Gaussian coordinate noise with sigma=50 px.
random40
    Remove exactly 40% of currently valid samples at random.
burst40
    Remove exactly 40% of currently valid samples using one-second windows.
high_random
    Apply noise50, then random40.
high_burst
    Apply noise50, then burst40.

Every stochastic realization is deterministic from
(base seed, trial ID, condition, repeat). The same trial ID is therefore a
positive pair across clean and degraded views, while labels are not used to
construct the degradation.

Recommended first audit:
python -u src\data\extract_onestop_quality_views.py ^
  --samples-dir data\interim\onestop\raw_samples_30hz_asc344 ^
  --manifest data\interim\onestop\onestop_regular_trial_manifest.csv ^
  --word-aoi data\interim\onestop\onestop_regular_word_aoi.parquet ^
  --clean-features data\processed\onestop\soft_aoi_asc344\onestop_soft_aoi_all_features.parquet ^
  --participants l1_155,l1_161 ^
  --conditions clean,rate10,noise50,random40,burst40,high_random,high_burst ^
  --repeats 1 ^
  --output-dir data\processed\onestop\quality_views_sample ^
  --audit-dir outputs\onestop_quality_views_sample ^
  --seed 42

Outputs
-------
<output-dir>/participant_quality_views/<participant>_quality_views.parquet
<output-dir>/onestop_quality_views_all_features.parquet
<audit-dir>/quality_view_condition_summary.csv
<audit-dir>/quality_view_pair_audit.csv
<audit-dir>/quality_view_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import extract_zuco_soft_aoi_features as soft
    from extract_onestop_soft_aoi_features import (
        adapt_samples,
        cluster_lines,
        load_manifest,
        make_trial_row,
        read_participant_aoi,
        sample_path,
    )
except ImportError as exc:
    raise ImportError(
        "Place this script in src/data beside "
        "extract_zuco_soft_aoi_features.py and "
        "extract_onestop_soft_aoi_features.py."
    ) from exc


FULL_FEATURES = list(
    dict.fromkeys(
        list(soft.CONTENT_FEATURES)
        + list(soft.MEMBERSHIP_QUALITY_FEATURES)
    )
)

SUPPORTED_CONDITIONS = (
    "clean",
    "rate10",
    "noise50",
    "random40",
    "burst40",
    "high_random",
    "high_burst",
)

QUALITY_COLUMNS = [
    "soft_aoi_valid_rate",
    "soft_aoi_active_rate",
    "soft_aoi_word_box_hit_rate",
    "soft_aoi_sentence_envelope_hit_rate",
    "soft_aoi_qmax_mean",
    "soft_aoi_dominant_word_coverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--word-aoi", type=Path, required=True)
    parser.add_argument("--clean-features", type=Path, required=True)
    parser.add_argument(
        "--participants",
        required=True,
        help="Comma-separated participant IDs, or 'all'.",
    )
    parser.add_argument(
        "--conditions",
        default="clean,rate10,noise50,random40,burst40,"
        "high_random,high_burst",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-sigma-px",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--missing-percent",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--burst-seconds",
        type=float,
        default=1.0,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def stable_seed(
    base_seed: int,
    trial_id: str,
    condition: str,
    repeat: int,
) -> int:
    payload = (
        f"{base_seed}|{trial_id}|{condition}|{repeat}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def parse_conditions(text: str) -> list[str]:
    conditions = [
        item.strip().lower()
        for item in text.split(",")
        if item.strip()
    ]
    conditions = list(dict.fromkeys(conditions))
    unknown = sorted(
        set(conditions) - set(SUPPORTED_CONDITIONS)
    )
    if unknown:
        raise ValueError(
            f"Unsupported conditions={unknown}; "
            f"supported={SUPPORTED_CONDITIONS}"
        )
    if "clean" not in conditions:
        conditions.insert(0, "clean")
    return conditions


def parse_participants(
    text: str,
    manifest: pd.DataFrame,
    samples_dir: Path,
) -> list[str]:
    available = {
        path.name[: -len("_samples_30hz.parquet")].lower()
        for path in samples_dir.glob("*_samples_30hz.parquet")
    }
    if text.strip().lower() == "all":
        participants = sorted(available)
    else:
        participants = [
            item.strip().lower()
            for item in text.split(",")
            if item.strip()
        ]
    participants = list(dict.fromkeys(participants))
    if not participants:
        raise ValueError("No participants selected.")

    manifest_ids = set(manifest["participant_id"])
    missing_manifest = sorted(
        set(participants) - manifest_ids
    )
    missing_samples = sorted(
        set(participants) - available
    )
    if missing_manifest:
        raise ValueError(
            f"Participants absent from manifest: {missing_manifest}"
        )
    if missing_samples:
        raise FileNotFoundError(
            f"Participant sample files missing: {missing_samples}"
        )
    return participants


def repeat_values(
    condition: str,
    repeats: int,
) -> list[int]:
    if condition in {"clean", "rate10"}:
        return [0]
    return list(range(1, repeats + 1))


def valid_mask(samples: pd.DataFrame) -> np.ndarray:
    x = pd.to_numeric(
        samples["x_px"], errors="coerce"
    ).to_numpy(dtype=float)
    y = pd.to_numeric(
        samples["y_px"], errors="coerce"
    ).to_numpy(dtype=float)
    basic = samples["valid_basic"].astype(bool).to_numpy()
    return (
        basic
        & np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )


def synchronize_validity(samples: pd.DataFrame) -> pd.DataFrame:
    result = samples.copy()
    mask = valid_mask(result)
    result["valid_basic"] = mask
    return result


def add_coordinate_noise(
    samples: pd.DataFrame,
    sigma_px: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = samples.copy()
    if sigma_px <= 0:
        return result

    x = result["x_px"].to_numpy(dtype=float)
    y = result["y_px"].to_numpy(dtype=float)
    mask = valid_mask(result)
    count = int(mask.sum())
    if count:
        x[mask] = np.maximum(
            x[mask] + rng.normal(0.0, sigma_px, count),
            1e-6,
        )
        y[mask] = np.maximum(
            y[mask] + rng.normal(0.0, sigma_px, count),
            1e-6,
        )
    result["x_px"] = x
    result["y_px"] = y
    return synchronize_validity(result)


def apply_missing_indices(
    samples: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    result = samples.copy()
    if indices.size:
        x_index = result.columns.get_loc("x_px")
        y_index = result.columns.get_loc("y_px")
        valid_index = result.columns.get_loc("valid_basic")
        result.iloc[indices, x_index] = 0.0
        result.iloc[indices, y_index] = 0.0
        result.iloc[indices, valid_index] = False
    return synchronize_validity(result)


def add_random_missingness(
    samples: pd.DataFrame,
    percentage: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    valid_indices = np.flatnonzero(valid_mask(samples))
    target = int(
        round(valid_indices.size * percentage / 100.0)
    )
    target = min(target, valid_indices.size)
    if target <= 0:
        return samples.copy(), 0

    selected = np.asarray(
        rng.choice(
            valid_indices,
            size=target,
            replace=False,
        ),
        dtype=np.int64,
    )
    return apply_missing_indices(samples, selected), target


def add_burst_missingness(
    samples: pd.DataFrame,
    percentage: float,
    burst_seconds: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    valid_before = valid_mask(samples)
    valid_count = int(valid_before.sum())
    target = int(round(valid_count * percentage / 100.0))
    target = min(target, valid_count)
    if target <= 0:
        return samples.copy(), 0

    n_samples = len(samples)
    burst_samples = max(
        1,
        int(round(burst_seconds * 30.0)),
    )
    selected = np.zeros(n_samples, dtype=bool)
    selected_valid = 0
    attempts = 0
    max_attempts = max(1000, n_samples * 20)

    while (
        selected_valid < target
        and attempts < max_attempts
    ):
        attempts += 1
        start = int(rng.integers(0, n_samples))
        end = min(n_samples, start + burst_samples)
        candidates = np.arange(start, end, dtype=np.int64)
        candidates = candidates[
            valid_before[candidates]
            & ~selected[candidates]
        ]
        remaining = target - selected_valid
        if candidates.size > remaining:
            candidates = candidates[:remaining]
        if candidates.size:
            selected[candidates] = True
            selected_valid += int(candidates.size)

    if selected_valid < target:
        remaining_indices = np.flatnonzero(
            valid_before & ~selected
        )
        needed = target - selected_valid
        selected[remaining_indices[:needed]] = True
        selected_valid += needed

    indices = np.flatnonzero(selected)
    return apply_missing_indices(samples, indices), selected_valid


def rate10_view(samples: pd.DataFrame) -> pd.DataFrame:
    ordered = samples.sort_values(
        "sample_index_in_trial"
    ).reset_index(drop=True)
    keep = (
        pd.to_numeric(
            ordered["sample_index_in_trial"],
            errors="raise",
        ).astype(np.int64)
        % 3
        == 0
    )
    result = ordered.loc[keep].copy().reset_index(drop=True)
    result["sample_index_in_trial"] = np.arange(
        len(result), dtype=np.int64
    )
    return result


def degrade(
    clean_samples: pd.DataFrame,
    condition: str,
    repeat: int,
    seed: int,
    trial_id: str,
    noise_sigma_px: float,
    missing_percent: float,
    burst_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean_samples = synchronize_validity(clean_samples)
    original_valid = int(valid_mask(clean_samples).sum())

    if condition == "rate10":
        degraded = rate10_view(clean_samples)
        return degraded, {
            "noise_sigma_px": 0.0,
            "missingness_mode": "none",
            "additional_missing_percent": 0.0,
            "additional_missing_samples": 0,
            "original_valid_samples": original_valid,
            "realized_additional_missing_percent": 0.0,
            "effective_rate_hz": 10,
        }

    rng = np.random.default_rng(
        stable_seed(seed, trial_id, condition, repeat)
    )
    degraded = clean_samples.copy()
    removed = 0
    missingness_mode = "none"
    applied_sigma = 0.0
    applied_missing = 0.0

    if condition in {"noise50", "high_random", "high_burst"}:
        degraded = add_coordinate_noise(
            degraded,
            noise_sigma_px,
            rng,
        )
        applied_sigma = noise_sigma_px

    if condition in {"random40", "high_random"}:
        degraded, removed = add_random_missingness(
            degraded,
            missing_percent,
            rng,
        )
        missingness_mode = "random"
        applied_missing = missing_percent
    elif condition in {"burst40", "high_burst"}:
        degraded, removed = add_burst_missingness(
            degraded,
            missing_percent,
            burst_seconds,
            rng,
        )
        missingness_mode = "burst"
        applied_missing = missing_percent

    realized = (
        100.0 * removed / original_valid
        if original_valid > 0
        else math.nan
    )
    return degraded, {
        "noise_sigma_px": applied_sigma,
        "missingness_mode": missingness_mode,
        "additional_missing_percent": applied_missing,
        "additional_missing_samples": int(removed),
        "original_valid_samples": original_valid,
        "realized_additional_missing_percent": realized,
        "effective_rate_hz": 30,
    }


def extract_view_features(
    samples: pd.DataFrame,
    manifest_row: pd.Series,
    aoi: pd.DataFrame,
    stimulus_index: dict[str, int],
    sigma_word_height_ratio: float,
    active_threshold: float,
) -> dict[str, Any]:
    trial_row = make_trial_row(manifest_row, stimulus_index)
    row = soft.extract_trial_features(
        samples=samples,
        trial_row=trial_row,
        aoi=aoi,
        # The view has already been formed at either 30 or 10 Hz.
        # BASE_RATE_HZ prevents a second selection pass.
        rate_hz=int(soft.BASE_RATE_HZ),
        sigma_word_height_ratio=sigma_word_height_ratio,
        active_threshold=active_threshold,
    )
    return row


def validate_cache(
    frame: pd.DataFrame,
    expected_trial_ids: set[str],
    expected_conditions: list[str],
    repeats: int,
) -> pd.DataFrame:
    required = {
        "onestop_trial_id",
        "participant_id",
        "view_condition",
        "corruption_repeat",
        *FULL_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cache lacks columns: {missing}")

    expected_keys = set()
    for trial_id in expected_trial_ids:
        for condition in expected_conditions:
            for repeat in repeat_values(condition, repeats):
                expected_keys.add((trial_id, condition, repeat))

    observed_keys = set(
        zip(
            frame["onestop_trial_id"].astype(str),
            frame["view_condition"].astype(str),
            pd.to_numeric(
                frame["corruption_repeat"],
                errors="raise",
            ).astype(int),
        )
    )
    if observed_keys != expected_keys:
        raise ValueError(
            "Cache key mismatch: "
            f"missing={len(expected_keys-observed_keys)}, "
            f"unexpected={len(observed_keys-expected_keys)}"
        )
    if np.isinf(
        frame[FULL_FEATURES].to_numpy(dtype=float)
    ).any():
        raise ValueError("Cache contains infinity.")
    return frame


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if args.noise_sigma_px < 0:
        raise ValueError("--noise-sigma-px must be nonnegative.")
    if not (0 < args.missing_percent < 100):
        raise ValueError(
            "--missing-percent must be within (0,100)."
        )
    if args.burst_seconds <= 0:
        raise ValueError("--burst-seconds must be positive.")

    conditions = parse_conditions(args.conditions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "participant_quality_views"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(require_file(args.manifest))
    participants = parse_participants(
        args.participants,
        manifest,
        args.samples_dir,
    )
    selected_manifest = manifest[
        manifest["participant_id"].isin(participants)
    ].copy()

    clean = pd.read_parquet(require_file(args.clean_features))
    clean["participant_id"] = (
        clean["participant_id"].astype(str).str.strip().str.lower()
    )
    clean["onestop_trial_id"] = clean[
        "onestop_trial_id"
    ].astype(str)
    clean = clean[
        clean["participant_id"].isin(participants)
    ].copy()
    missing_clean = sorted(
        set(selected_manifest["onestop_trial_id"].astype(str))
        - set(clean["onestop_trial_id"])
    )
    if missing_clean:
        # The known raw-unavailable trial is expected to be absent.
        selected_manifest = selected_manifest[
            selected_manifest["onestop_trial_id"].isin(
                set(clean["onestop_trial_id"])
            )
        ].copy()

    stimulus_values = sorted(
        selected_manifest["stimulus_id"].astype(str).unique()
    )
    stimulus_index = {
        value: index
        for index, value in enumerate(stimulus_values)
    }

    print("=== ONESTOP QUALITY-VIEW INPUT ===")
    print(f"participants: {len(participants)}")
    print(
        f"clean trials selected: "
        f"{clean['onestop_trial_id'].nunique():,}"
    )
    print(f"conditions: {conditions}")
    print(f"stochastic repeats: {args.repeats}")
    print(
        f"noise sigma={args.noise_sigma_px:g}px, "
        f"missing={args.missing_percent:g}%, "
        f"burst={args.burst_seconds:g}s"
    )

    participant_parts = []

    for participant_index, participant in enumerate(
        participants, start=1
    ):
        participant_manifest = selected_manifest[
            selected_manifest["participant_id"].eq(participant)
        ].copy()
        participant_clean = clean[
            clean["participant_id"].eq(participant)
        ].copy()
        expected_ids = set(
            participant_clean["onestop_trial_id"].astype(str)
        )
        cache_path = (
            cache_dir / f"{participant}_quality_views.parquet"
        )

        print(
            f"[{participant_index}/{len(participants)}] "
            f"{participant}: trials={len(expected_ids)}"
        )

        if cache_path.is_file() and not args.overwrite:
            cached = pd.read_parquet(cache_path)
            cached = validate_cache(
                cached,
                expected_ids,
                conditions,
                args.repeats,
            )
            participant_parts.append(cached)
            print("    loaded valid cache")
            continue

        participant_aoi = read_participant_aoi(
            args.word_aoi,
            participant,
        )
        aoi_lookup = {
            str(trial_id): cluster_lines(group)
            for trial_id, group in participant_aoi.groupby(
                "onestop_trial_id",
                sort=False,
            )
        }

        source_samples = pd.read_parquet(
            sample_path(args.samples_dir, participant)
        )
        adapted = adapt_samples(source_samples)
        sample_lookup = {
            str(trial_id): group.copy()
            for trial_id, group in adapted.groupby(
                "trial_id",
                sort=False,
            )
        }
        manifest_lookup = participant_manifest.set_index(
            "onestop_trial_id",
            drop=False,
            verify_integrity=True,
        )
        clean_lookup = participant_clean.set_index(
            "onestop_trial_id",
            drop=False,
            verify_integrity=True,
        )

        missing_samples = sorted(
            expected_ids - set(sample_lookup)
        )
        missing_aoi = sorted(
            expected_ids - set(aoi_lookup)
        )
        if missing_samples or missing_aoi:
            raise ValueError(
                f"{participant}: missing samples={missing_samples[:5]}, "
                f"missing AOI={missing_aoi[:5]}"
            )

        rows: list[dict[str, Any]] = []
        for trial_number, trial_id in enumerate(
            sorted(expected_ids), start=1
        ):
            manifest_row = manifest_lookup.loc[trial_id]
            base_metadata = {
                "onestop_trial_id": trial_id,
                "participant_id": participant,
                "regime": manifest_row["regime"],
                "label": int(manifest_row["label"]),
                "label_name": manifest_row["label_name"],
                "trial_index": manifest_row["trial_index"],
                "stimulus_id": manifest_row["stimulus_id"],
                "task_context_id": manifest_row["task_context_id"],
            }

            if "clean" in conditions:
                clean_row = clean_lookup.loc[trial_id].to_dict()
                clean_row.update(base_metadata)
                clean_row.update(
                    {
                        "view_condition": "clean",
                        "corruption_repeat": 0,
                        "noise_sigma_px": 0.0,
                        "missingness_mode": "none",
                        "additional_missing_percent": 0.0,
                        "additional_missing_samples": 0,
                        "original_valid_samples": int(
                            clean_row.get(
                                "soft_aoi_n_valid_samples",
                                0,
                            )
                        ),
                        "realized_additional_missing_percent": 0.0,
                        "effective_rate_hz": 30,
                        "quality_pair_source": "existing_clean_features",
                    }
                )
                rows.append(clean_row)

            for condition in conditions:
                if condition == "clean":
                    continue
                for repeat in repeat_values(
                    condition,
                    args.repeats,
                ):
                    degraded, corruption = degrade(
                        clean_samples=sample_lookup[trial_id],
                        condition=condition,
                        repeat=repeat,
                        seed=args.seed,
                        trial_id=trial_id,
                        noise_sigma_px=args.noise_sigma_px,
                        missing_percent=args.missing_percent,
                        burst_seconds=args.burst_seconds,
                    )
                    feature = extract_view_features(
                        samples=degraded,
                        manifest_row=manifest_row,
                        aoi=aoi_lookup[trial_id],
                        stimulus_index=stimulus_index,
                        sigma_word_height_ratio=(
                            args.sigma_word_height_ratio
                        ),
                        active_threshold=args.active_threshold,
                    )
                    feature.update(base_metadata)
                    feature.update(corruption)
                    feature.update(
                        {
                            "view_condition": condition,
                            "corruption_repeat": repeat,
                            "soft_aoi_rate_hz": corruption[
                                "effective_rate_hz"
                            ],
                            "quality_pair_source": (
                                "deterministic_sample_degradation"
                            ),
                        }
                    )
                    rows.append(feature)

            if trial_number % 10 == 0 or trial_number == len(
                expected_ids
            ):
                print(
                    f"    trials {trial_number}/{len(expected_ids)}"
                )

        participant_frame = pd.DataFrame(rows)
        participant_frame = validate_cache(
            participant_frame,
            expected_ids,
            conditions,
            args.repeats,
        )
        participant_frame.to_parquet(
            cache_path,
            index=False,
            compression="zstd",
        )
        participant_parts.append(participant_frame)
        print(f"    saved {cache_path}")

    combined = pd.concat(
        participant_parts,
        ignore_index=True,
    )
    key_columns = [
        "onestop_trial_id",
        "view_condition",
        "corruption_repeat",
    ]
    if combined.duplicated(key_columns).any():
        raise ValueError("Combined table has duplicate view keys.")

    combined_path = (
        args.output_dir
        / "onestop_quality_views_all_features.parquet"
    )
    condition_summary_path = (
        args.audit_dir / "quality_view_condition_summary.csv"
    )
    pair_audit_path = (
        args.audit_dir / "quality_view_pair_audit.csv"
    )
    manifest_path = (
        args.audit_dir / "quality_view_manifest.json"
    )

    combined.to_parquet(
        combined_path,
        index=False,
        compression="zstd",
    )

    condition_summary = (
        combined.groupby(
            ["view_condition", "corruption_repeat"],
            observed=True,
        )
        .agg(
            rows=("onestop_trial_id", "size"),
            participants=("participant_id", "nunique"),
            valid_rate_median=(
                "soft_aoi_valid_rate", "median"
            ),
            active_rate_median=(
                "soft_aoi_active_rate", "median"
            ),
            box_hit_rate_median=(
                "soft_aoi_word_box_hit_rate", "median"
            ),
            envelope_hit_rate_median=(
                "soft_aoi_sentence_envelope_hit_rate",
                "median",
            ),
            qmax_mean_median=("soft_aoi_qmax_mean", "median"),
            word_coverage_median=(
                "soft_aoi_dominant_word_coverage",
                "median",
            ),
            effective_rate_hz_median=(
                "effective_rate_hz", "median"
            ),
            realized_missing_percent_median=(
                "realized_additional_missing_percent",
                "median",
            ),
        )
        .reset_index()
    )
    condition_summary.to_csv(
        condition_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    expected_views_per_trial = sum(
        len(repeat_values(condition, args.repeats))
        for condition in conditions
    )
    pair_audit = (
        combined.groupby(
            ["onestop_trial_id", "participant_id"],
            observed=True,
        )
        .agg(
            n_views=("view_condition", "size"),
            n_conditions=("view_condition", "nunique"),
            label_nunique=("label", "nunique"),
            clean_views=(
                "view_condition",
                lambda values: int(
                    pd.Series(values).eq("clean").sum()
                ),
            ),
        )
        .reset_index()
    )
    pair_audit["expected_views"] = expected_views_per_trial
    pair_audit["pair_complete"] = (
        pair_audit["n_views"].eq(expected_views_per_trial)
        & pair_audit["n_conditions"].eq(len(conditions))
        & pair_audit["label_nunique"].eq(1)
        & pair_audit["clean_views"].eq(1)
    )
    pair_audit.to_csv(
        pair_audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest_payload = {
        "dataset": "OneStop ASC344",
        "purpose": (
            "paired sample-level quality views for "
            "quality-contrastive learning"
        ),
        "participants": participants,
        "n_participants": len(participants),
        "n_clean_trials": int(
            combined.loc[
                combined["view_condition"].eq("clean"),
                "onestop_trial_id",
            ].nunique()
        ),
        "conditions": conditions,
        "repeats": args.repeats,
        "condition_definitions": {
            "rate10": "keep every third sample from 30 Hz",
            "noise50": (
                f"Gaussian coordinate noise sigma="
                f"{args.noise_sigma_px:g}px"
            ),
            "random40": (
                f"exact random removal of "
                f"{args.missing_percent:g}% currently valid samples"
            ),
            "burst40": (
                f"exact removal of {args.missing_percent:g}% "
                f"currently valid samples using "
                f"{args.burst_seconds:g}s windows"
            ),
            "high_random": "noise50 followed by random40",
            "high_burst": "noise50 followed by burst40",
        },
        "seed": args.seed,
        "positive_pair_key": "onestop_trial_id",
        "label_used_for_degradation": False,
        "full_feature_columns": FULL_FEATURES,
        "output": str(combined_path),
        "audit": {
            "pair_complete_trials": int(
                pair_audit["pair_complete"].sum()
            ),
            "pair_incomplete_trials": int(
                (~pair_audit["pair_complete"]).sum()
            ),
            "nan_feature_cells": int(
                combined[FULL_FEATURES].isna().sum().sum()
            ),
            "infinite_feature_cells": int(
                np.isinf(
                    combined[FULL_FEATURES].to_numpy(dtype=float)
                ).sum()
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== QUALITY-VIEW CONDITION SUMMARY ===")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
    ):
        print(condition_summary.to_string(index=False))

    print("\n=== QUALITY-VIEW PAIR AUDIT ===")
    print(
        json.dumps(
            {
                "rows": len(combined),
                "trials": combined[
                    "onestop_trial_id"
                ].nunique(),
                "expected_views_per_trial": (
                    expected_views_per_trial
                ),
                "complete_pairs": int(
                    pair_audit["pair_complete"].sum()
                ),
                "incomplete_pairs": int(
                    (~pair_audit["pair_complete"]).sum()
                ),
                "nan_feature_cells": int(
                    combined[FULL_FEATURES].isna().sum().sum()
                ),
                "infinite_feature_cells": int(
                    np.isinf(
                        combined[
                            FULL_FEATURES
                        ].to_numpy(dtype=float)
                    ).sum()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"\nSaved: {combined_path}")
    print(f"Saved: {condition_summary_path}")
    print(f"Saved: {pair_audit_path}")
    print(f"Saved: {manifest_path}")

    if not pair_audit["pair_complete"].all():
        print("\nAUDIT STATUS: FAIL — incomplete view pairs")
        return 5

    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
