#!/usr/bin/env python
"""Build fixed-length event-free gaze-sequence caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from event_free_sequence_common import (  # noqa: E402
    CHANNEL_NAMES,
    canonicalize_aoi,
    canonicalize_samples,
    compute_event_free_tokens,
    normalize_id,
    resize_sequence,
    sha256_file,
    write_hash_manifest,
)


ONESTOP_CONDITIONS = (
    "clean",
    "rate10",
    "noise50",
    "random40",
    "burst40",
    "high_random",
    "high_burst",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["onestop", "webqamgaze"],
        required=True,
    )
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
        "--web-stage2-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage2"),
    )
    parser.add_argument(
        "--web-stage3c-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage3c"),
    )
    parser.add_argument("--participants", default="all")
    parser.add_argument(
        "--conditions",
        default=",".join(ONESTOP_CONDITIONS),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--sigma-word-height-ratio",
        type=float,
        default=2.0 / 3.0,
    )
    parser.add_argument("--active-threshold", type=float, default=0.1)
    parser.add_argument("--noise-sigma-px", type=float, default=50.0)
    parser.add_argument("--missing-percent", type=float, default=40.0)
    parser.add_argument("--burst-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_conditions(text: str) -> list[str]:
    values = [
        value.strip().lower()
        for value in text.split(",")
        if value.strip()
    ]
    values = list(dict.fromkeys(values))
    unknown = sorted(set(values) - set(ONESTOP_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if "clean" not in values:
        values.insert(0, "clean")
    return values


def repeat_values(condition: str, repeats: int) -> list[int]:
    if condition in {"clean", "rate10"}:
        return [0]
    return list(range(1, repeats + 1))


def save_shard(
    path: Path,
    sequences: list[np.ndarray],
    lengths: list[int],
    labels: list[int],
    conditions: list[str],
    repeats: list[int],
    trial_ids: list[str],
    participant_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.asarray(sequences, dtype=np.float16),
        lengths=np.asarray(lengths, dtype=np.int16),
        labels=np.asarray(labels, dtype=np.int8),
        conditions=np.asarray(conditions, dtype="U16"),
        repeats=np.asarray(repeats, dtype=np.int8),
        trial_ids=np.asarray(trial_ids, dtype="U160"),
        participant_ids=np.asarray(participant_ids, dtype="U48"),
    )


def build_onestop(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from extract_onestop_soft_aoi_features import (
            adapt_samples,
            cluster_lines,
            load_manifest,
            read_participant_aoi,
            sample_path,
        )
        from extract_onestop_quality_views import degrade
    except ImportError as exc:
        raise ImportError(
            "Place this file in src/data beside the existing OneStop "
            "Soft-AOI and quality-view scripts."
        ) from exc

    manifest = load_manifest(args.manifest)
    available = sorted(
        path.name[: -len("_samples_30hz.parquet")].lower()
        for path in args.samples_dir.glob("*_samples_30hz.parquet")
    )
    if args.participants.strip().lower() == "all":
        participants = available
    else:
        participants = [
            normalize_id(value)
            for value in args.participants.split(",")
            if value.strip()
        ]
    if not participants:
        raise ValueError("No participants selected.")

    conditions = parse_conditions(args.conditions)
    input_audit = {
        "dataset": "OneStop ASC344",
        "samples_dir": str(args.samples_dir.resolve()),
        "sample_file_count": len(available),
        "participants_selected": len(participants),
        "manifest_trials": int(manifest["onestop_trial_id"].nunique()),
        "manifest_participants": int(manifest["participant_id"].nunique()),
        "word_aoi": str(args.word_aoi.resolve()),
        "word_aoi_sha256": sha256_file(args.word_aoi),
        "conditions": conditions,
        "repeats": args.repeats,
        "max_length": args.max_length,
        "channels": CHANNEL_NAMES,
    }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "input_audit.json").write_text(
        json.dumps(input_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Perform a real schema test before returning from audit-only.
    first = participants[0]
    sample_test = pd.read_parquet(sample_path(args.samples_dir, first))
    adapted_test = adapt_samples(sample_test)
    aoi_test = read_participant_aoi(args.word_aoi, first)
    if adapted_test.empty or aoi_test.empty:
        raise RuntimeError("Sample schema audit found an empty input.")
    first_trial = str(adapted_test["trial_id"].astype(str).iloc[0])
    test_samples = adapted_test[
        adapted_test["trial_id"].astype(str).eq(first_trial)
    ]
    test_aoi = cluster_lines(
        aoi_test[aoi_test["onestop_trial_id"].astype(str).eq(first_trial)]
    )
    test_tokens, test_stats = compute_event_free_tokens(
        test_samples,
        test_aoi,
        args.sigma_word_height_ratio,
        args.active_threshold,
    )
    resized, length, mode = resize_sequence(test_tokens, args.max_length)
    schema_audit = {
        "participant_id": first,
        "trial_id": first_trial,
        "token_shape_before_resize": list(test_tokens.shape),
        "token_shape_after_resize": list(resized.shape),
        "stored_length": length,
        "resize_mode": mode,
        **test_stats,
    }
    (args.audit_dir / "schema_smoke_test.json").write_text(
        json.dumps(schema_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.audit_only:
        print(json.dumps(input_audit, ensure_ascii=False, indent=2))
        print(json.dumps(schema_audit, ensure_ascii=False, indent=2))
        print("AUDIT STATUS: PASS")
        return pd.DataFrame(), pd.DataFrame()

    index_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    for participant_index, participant in enumerate(participants, start=1):
        print(
            f"[{participant_index}/{len(participants)}] {participant}",
            flush=True,
        )
        participant_manifest = manifest[
            manifest["participant_id"].eq(participant)
        ].copy()
        participant_aoi = read_participant_aoi(args.word_aoi, participant)
        aoi_lookup = {
            str(trial_id): cluster_lines(group)
            for trial_id, group in participant_aoi.groupby(
                "onestop_trial_id",
                sort=False,
            )
        }
        source = pd.read_parquet(sample_path(args.samples_dir, participant))
        adapted = adapt_samples(source)
        sample_lookup = {
            str(trial_id): group.copy()
            for trial_id, group in adapted.groupby("trial_id", sort=False)
        }
        manifest_lookup = participant_manifest.set_index(
            "onestop_trial_id",
            drop=False,
            verify_integrity=True,
        )
        trial_ids = sorted(
            set(sample_lookup)
            & set(aoi_lookup)
            & set(manifest_lookup.index.astype(str))
        )
        if not trial_ids:
            raise RuntimeError(f"{participant}: no aligned trials.")

        shard_path = shard_dir / f"{participant}.npz"
        if shard_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Shard exists: {shard_path}. Use --overwrite to rebuild."
            )

        sequences: list[np.ndarray] = []
        lengths: list[int] = []
        labels: list[int] = []
        condition_values: list[str] = []
        repeat_list: list[int] = []
        stored_trials: list[str] = []
        stored_participants: list[str] = []

        for trial_number, trial_id in enumerate(trial_ids, start=1):
            manifest_row = manifest_lookup.loc[trial_id]
            clean_samples = sample_lookup[trial_id]
            label = int(manifest_row["label"])

            for condition in conditions:
                for repeat in repeat_values(condition, args.repeats):
                    view, corruption = degrade(
                        clean_samples=clean_samples,
                        condition=condition,
                        repeat=repeat,
                        seed=args.seed,
                        trial_id=trial_id,
                        noise_sigma_px=args.noise_sigma_px,
                        missing_percent=args.missing_percent,
                        burst_seconds=args.burst_seconds,
                    )
                    tokens, token_audit = compute_event_free_tokens(
                        view,
                        aoi_lookup[trial_id],
                        args.sigma_word_height_ratio,
                        args.active_threshold,
                    )
                    fixed, stored_length, resize_mode = resize_sequence(
                        tokens,
                        args.max_length,
                    )
                    local_index = len(sequences)
                    sequences.append(fixed)
                    lengths.append(stored_length)
                    labels.append(label)
                    condition_values.append(condition)
                    repeat_list.append(repeat)
                    stored_trials.append(trial_id)
                    stored_participants.append(participant)

                    index_rows.append(
                        {
                            "dataset": "onestop",
                            "participant_id": participant,
                            "trial_id": trial_id,
                            "label": label,
                            "condition": condition,
                            "repeat": repeat,
                            "shard_path": shard_path.relative_to(
                                args.output_dir
                            ).as_posix(),
                            "local_index": local_index,
                            "stored_length": stored_length,
                            "original_length": int(len(tokens)),
                            "resize_mode": resize_mode,
                        }
                    )
                    audit_rows.append(
                        {
                            "dataset": "onestop",
                            "participant_id": participant,
                            "trial_id": trial_id,
                            "label": label,
                            "condition": condition,
                            "repeat": repeat,
                            "corruption_json": json.dumps(
                                corruption, sort_keys=True
                            ),
                            "stored_length": stored_length,
                            "original_length": int(len(tokens)),
                            "resize_mode": resize_mode,
                            **token_audit,
                        }
                    )

            if trial_number % 10 == 0 or trial_number == len(trial_ids):
                print(
                    f"    trials {trial_number}/{len(trial_ids)}",
                    flush=True,
                )

        save_shard(
            shard_path,
            sequences,
            lengths,
            labels,
            condition_values,
            repeat_list,
            stored_trials,
            stored_participants,
        )

    return pd.DataFrame(index_rows), pd.DataFrame(audit_rows)


def _web_dense_samples(group: pd.DataFrame) -> pd.DataFrame:
    if {
        "t_ms",
        "interpolation_segment_id",
        "sample_index_30hz",
        "valid_xy",
        "x_px",
        "y_px",
    }.issubset(group.columns):
        try:
            from prepare_webqamgaze_external_validation_stage3c import (
                build_dense_30hz_samples,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import WebQAmGaze Stage 3C dense-grid builder."
            ) from exc
        dense, _ = build_dense_30hz_samples(group, rate_hz=30.0)
        return dense
    return canonicalize_samples(group)


def build_webqamgaze(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_path = (
        args.web_stage2_dir / "trial_manifest_primary_candidates.csv"
    )
    sample_path = args.web_stage2_dir / "samples_30hz.parquet"
    aoi_path = args.web_stage3c_dir / "set_specific_word_aoi.csv"
    for path in (trial_path, sample_path, aoi_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    trials = pd.read_csv(trial_path)
    samples = pd.read_parquet(sample_path)
    aoi = pd.read_csv(aoi_path)

    required_trials = {
        "external_trial_id",
        "participant_id",
        "label_nr",
        "set_name",
        "trial_name",
    }
    missing = sorted(required_trials - set(trials.columns))
    if missing:
        raise ValueError(f"Web trial manifest lacks: {missing}")
    if "external_trial_id" not in samples.columns:
        raise ValueError("Web samples lack external_trial_id.")

    # Stage 3C deliberately stores AOIs by shared layout key
    # (set_name, trial_name), not by participant-specific external_trial_id.
    # Multiple external trials can therefore reuse one set-specific word layout.
    required_aoi = {
        "set_name",
        "trial_name",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    }
    missing_aoi = sorted(required_aoi - set(aoi.columns))
    if missing_aoi:
        raise ValueError(
            "Web AOI table lacks shared layout columns: "
            f"{missing_aoi}; available={list(aoi.columns)}"
        )

    trials["external_trial_id"] = (
        trials["external_trial_id"].astype(str).str.strip()
    )
    trials["participant_id"] = (
        trials["participant_id"].astype(str).str.strip()
    )
    trials["set_name"] = trials["set_name"].astype(str).str.strip()
    trials["trial_name"] = trials["trial_name"].astype(str).str.strip()
    trials["label_nr"] = pd.to_numeric(
        trials["label_nr"], errors="raise"
    ).astype(int)

    samples["external_trial_id"] = (
        samples["external_trial_id"].astype(str).str.strip()
    )
    aoi["set_name"] = aoi["set_name"].astype(str).str.strip()
    aoi["trial_name"] = aoi["trial_name"].astype(str).str.strip()

    trials["__layout_key__"] = list(
        zip(
            trials["set_name"].str.casefold(),
            trials["trial_name"].str.casefold(),
        )
    )
    aoi["__layout_key__"] = list(
        zip(
            aoi["set_name"].str.casefold(),
            aoi["trial_name"].str.casefold(),
        )
    )

    trial_layout_keys = set(trials["__layout_key__"])
    aoi_layout_keys = set(aoi["__layout_key__"])
    missing_layout_keys = sorted(trial_layout_keys - aoi_layout_keys)
    if missing_layout_keys:
        preview = missing_layout_keys[:20]
        raise RuntimeError(
            "Set-specific AOI lookup coverage mismatch. "
            f"Missing {len(missing_layout_keys)} of "
            f"{len(trial_layout_keys)} trial layout keys. "
            f"First missing keys={preview}"
        )

    input_audit = {
        "dataset": "WebQAmGaze English development target",
        "trial_path": str(trial_path.resolve()),
        "sample_path": str(sample_path.resolve()),
        "aoi_path": str(aoi_path.resolve()),
        "n_trials": int(trials["external_trial_id"].nunique()),
        "n_participants": int(trials["participant_id"].nunique()),
        "sample_rows": int(len(samples)),
        "aoi_rows": int(len(aoi)),
        "trial_layout_keys": int(len(trial_layout_keys)),
        "aoi_layout_keys": int(len(aoi_layout_keys)),
        "missing_layout_keys": int(len(missing_layout_keys)),
        "aoi_join_rule": "set_name + trial_name",
        "max_length": args.max_length,
        "channels": CHANNEL_NAMES,
        "target_status": "development target; not independent confirmation",
    }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "input_audit.json").write_text(
        json.dumps(input_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    first_row = trials.iloc[0]
    first_trial = str(first_row["external_trial_id"])
    first_layout_key = first_row["__layout_key__"]
    dense_test = _web_dense_samples(
        samples[samples["external_trial_id"].eq(first_trial)]
    )
    # pandas.Series.eq(tuple) treats the tuple as an array-like object and
    # raises "Lengths must be equal". Compare the two scalar key columns
    # explicitly instead.
    first_set_key = str(first_row["set_name"]).casefold()
    first_trial_key = str(first_row["trial_name"]).casefold()
    aoi_test = canonicalize_aoi(
        aoi[
            aoi["set_name"].str.casefold().eq(first_set_key)
            & aoi["trial_name"].str.casefold().eq(first_trial_key)
        ]
    )
    tokens, token_stats = compute_event_free_tokens(
        dense_test,
        aoi_test,
        args.sigma_word_height_ratio,
        args.active_threshold,
    )
    resized, length, mode = resize_sequence(tokens, args.max_length)
    schema_audit = {
        "trial_id": first_trial,
        "set_name": str(first_row["set_name"]),
        "trial_name": str(first_row["trial_name"]),
        "aoi_join_rule": "set_name + trial_name",
        "token_shape_before_resize": list(tokens.shape),
        "token_shape_after_resize": list(resized.shape),
        "stored_length": length,
        "resize_mode": mode,
        **token_stats,
    }
    (args.audit_dir / "schema_smoke_test.json").write_text(
        json.dumps(schema_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.audit_only:
        print(json.dumps(input_audit, ensure_ascii=False, indent=2))
        print(json.dumps(schema_audit, ensure_ascii=False, indent=2))
        print("AUDIT STATUS: PASS")
        return pd.DataFrame(), pd.DataFrame()

    index_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    trial_lookup = trials.set_index(
        "external_trial_id",
        drop=False,
        verify_integrity=True,
    )
    sample_lookup = {
        str(trial_id): group.copy()
        for trial_id, group in samples.groupby(
            "external_trial_id",
            sort=False,
        )
    }
    aoi_lookup = {
        layout_key: canonicalize_aoi(group.drop(columns=["__layout_key__"]))
        for layout_key, group in aoi.groupby(
            "__layout_key__",
            sort=False,
        )
    }

    for participant_index, (participant, participant_trials) in enumerate(
        trials.groupby("participant_id", sort=True),
        start=1,
    ):
        participant = str(participant)
        print(
            f"[{participant_index}/{trials['participant_id'].nunique()}] "
            f"{participant}",
            flush=True,
        )
        shard_path = shard_dir / f"{participant}.npz"
        if shard_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Shard exists: {shard_path}. Use --overwrite to rebuild."
            )

        sequences: list[np.ndarray] = []
        lengths: list[int] = []
        labels: list[int] = []
        conditions: list[str] = []
        repeats: list[int] = []
        trial_ids: list[str] = []
        participant_ids: list[str] = []

        for trial_id in participant_trials["external_trial_id"].astype(str):
            if trial_id not in sample_lookup:
                raise RuntimeError(
                    f"Missing Web sample sequence: {trial_id}"
                )
            trial = trial_lookup.loc[trial_id]
            layout_key = trial["__layout_key__"]
            if layout_key not in aoi_lookup:
                raise RuntimeError(
                    "Missing Web AOI layout for "
                    f"external_trial_id={trial_id}, "
                    f"set_name={trial['set_name']!r}, "
                    f"trial_name={trial['trial_name']!r}"
                )
            dense = _web_dense_samples(sample_lookup[trial_id])
            tokens, token_audit = compute_event_free_tokens(
                dense,
                aoi_lookup[layout_key],
                args.sigma_word_height_ratio,
                args.active_threshold,
            )
            fixed, stored_length, resize_mode = resize_sequence(
                tokens,
                args.max_length,
            )
            local_index = len(sequences)
            label = int(trial["label_nr"])
            sequences.append(fixed)
            lengths.append(stored_length)
            labels.append(label)
            conditions.append("clean")
            repeats.append(0)
            trial_ids.append(trial_id)
            participant_ids.append(participant)

            index_rows.append(
                {
                    "dataset": "webqamgaze",
                    "participant_id": participant,
                    "trial_id": trial_id,
                    "label": label,
                    "set_name": str(trial["set_name"]),
                    "trial_name": str(trial["trial_name"]),
                    "condition": "clean",
                    "repeat": 0,
                    "shard_path": shard_path.relative_to(
                        args.output_dir
                    ).as_posix(),
                    "local_index": local_index,
                    "stored_length": stored_length,
                    "original_length": int(len(tokens)),
                    "resize_mode": resize_mode,
                }
            )
            audit_rows.append(
                {
                    "dataset": "webqamgaze",
                    "participant_id": participant,
                    "trial_id": trial_id,
                    "label": label,
                    "set_name": str(trial["set_name"]),
                    "trial_name": str(trial["trial_name"]),
                    "condition": "clean",
                    "repeat": 0,
                    "stored_length": stored_length,
                    "original_length": int(len(tokens)),
                    "resize_mode": resize_mode,
                    **token_audit,
                }
            )

        save_shard(
            shard_path,
            sequences,
            lengths,
            labels,
            conditions,
            repeats,
            trial_ids,
            participant_ids,
        )

    return pd.DataFrame(index_rows), pd.DataFrame(audit_rows)


def main() -> int:
    args = parse_args()
    if args.max_length > np.iinfo(np.int16).max:
        raise ValueError("max_length is too large for the cache schema.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "onestop":
        index, audit = build_onestop(args)
    else:
        index, audit = build_webqamgaze(args)

    if args.audit_only:
        return 0

    if index.empty:
        raise RuntimeError("No cache rows were created.")
    index_path = args.output_dir / "sequence_index.parquet"
    audit_path = args.audit_dir / "sequence_trial_audit.parquet"
    index.to_parquet(index_path, index=False)
    audit.to_parquet(audit_path, index=False)

    condition_summary = (
        index.groupby(["condition", "repeat"], observed=True)
        .agg(
            rows=("trial_id", "size"),
            trials=("trial_id", "nunique"),
            participants=("participant_id", "nunique"),
            stored_length_median=("stored_length", "median"),
            original_length_median=("original_length", "median"),
            uniformly_subsampled=("resize_mode", lambda x: int((x == "uniform_subsample").sum())),
        )
        .reset_index()
    )
    condition_summary.to_csv(
        args.audit_dir / "sequence_condition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "artifact_version": "event_free_sequence_cache_v1",
        "dataset": args.dataset,
        "n_rows": int(len(index)),
        "n_trials": int(index["trial_id"].nunique()),
        "n_participants": int(index["participant_id"].nunique()),
        "label_counts_clean": (
            index[index["condition"].eq("clean")]["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "conditions": sorted(index["condition"].unique()),
        "max_length": args.max_length,
        "channel_names": CHANNEL_NAMES,
        "n_channels": len(CHANNEL_NAMES),
        "long_sequence_policy": (
            "uniformly subsample across the full trial to max_length"
        ),
        "event_definition_used": False,
        "fixation_or_saccade_columns_used": False,
        "sigma_word_height_ratio": args.sigma_word_height_ratio,
        "active_threshold": args.active_threshold,
        "seed": args.seed,
        "index": str(index_path),
        "trial_audit": str(audit_path),
    }
    (args.output_dir / "sequence_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "channel_names.json").write_text(
        json.dumps(CHANNEL_NAMES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_hash_manifest(args.output_dir)

    print("\n=== EVENT-FREE SEQUENCE CACHE ===")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved: {index_path}")
    print(f"Saved: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
