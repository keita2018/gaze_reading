#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def refs(ds):
    for x in np.asarray(ds[()]).ravel(order="C"):
        if isinstance(x, h5py.Reference) and bool(x):
            yield x


def first_ref(ds):
    return next(refs(ds), None)


def task_from_name(path):
    name = path.name.upper()
    return "TSR" if "_TSR" in name else "NR" if "_NR" in name else "UNKNOWN"


def is_ref_dtype(dtype):
    return h5py.check_dtype(ref=dtype) is not None or dtype == object


def safe(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def med(values):
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return safe(np.median(a)) if a.size else math.nan


def pct(values, q):
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return safe(np.percentile(a, q)) if a.size else math.nan


def get_sentence_word_group(h5, sentence_data, idx):
    container = sentence_data.get("word")
    if not isinstance(container, h5py.Dataset):
        return None
    sr = np.asarray(container[()]).ravel(order="C")
    if idx >= len(sr) or not isinstance(sr[idx], h5py.Reference) or not bool(sr[idx]):
        return None
    obj = h5[sr[idx]]
    return obj if isinstance(obj, h5py.Group) else None


def count_all_fixations(h5, sentence_data, idx):
    container = sentence_data.get("allFixations")
    if not isinstance(container, h5py.Dataset):
        return 0
    sr = np.asarray(container[()]).ravel(order="C")
    if idx >= len(sr) or not isinstance(sr[idx], h5py.Reference) or not bool(sr[idx]):
        return 0
    group = h5[sr[idx]]
    if not isinstance(group, h5py.Group) or "duration" not in group:
        return 0
    return int(np.asarray(group["duration"][()]).size)


def get_segments(h5, sentence_data, idx):
    group = get_sentence_word_group(h5, sentence_data, idx)
    if group is None or "rawET" not in group:
        return 0, 0, []
    raw_words = group["rawET"]
    if not isinstance(raw_words, h5py.Dataset):
        return 0, 0, []

    word_refs = np.asarray(raw_words[()]).ravel(order="C")
    n_words = len(word_refs)
    words_with = 0
    segments = []

    for wr in word_refs:
        if not isinstance(wr, h5py.Reference) or not bool(wr):
            continue
        cell = h5[wr]
        if not isinstance(cell, h5py.Dataset) or not is_ref_dtype(cell.dtype):
            continue
        seg_refs = list(refs(cell))
        if not seg_refs:
            continue
        words_with += 1
        for sr in seg_refs:
            obj = h5[sr]
            if not isinstance(obj, h5py.Dataset):
                continue
            arr = np.asarray(obj[()])
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                continue
            if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                arr = arr.T
            segments.append(arr.astype(float, copy=False))
    return n_words, words_with, segments


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mat_files", nargs="+")
    p.add_argument("--subject", required=True)
    p.add_argument("--output-dir", default="outputs/tables")
    args = p.parse_args()

    subject = args.subject.upper()
    outdir = Path(args.output_dir)
    sentence_rows = []
    segment_rows = []

    for filename in args.mat_files:
        path = Path(filename)
        task = task_from_name(path)
        print(f"Scanning {path} ({task})")

        with h5py.File(path, "r") as h5:
            sd = h5["sentenceData"]
            n_sentences = int(np.asarray(sd["content"][()]).size)

            for idx in range(n_sentences):
                n_words, n_words_with, segments = get_segments(h5, sd, idx)
                n_all_fix = count_all_fixations(h5, sd, idx)

                col_counts = Counter()
                within_dts = []
                intervals = []
                n_samples = 0

                for seg_idx, seg in enumerate(segments):
                    n_samples += seg.shape[0]
                    n_cols = seg.shape[1]
                    col_counts[n_cols] += 1

                    first_ts = last_ts = median_dt = p95_dt = math.nan
                    if n_cols >= 1 and seg.shape[0] >= 1:
                        ts = np.sort(seg[:, 0][np.isfinite(seg[:, 0])])
                        if ts.size:
                            first_ts, last_ts = safe(ts[0]), safe(ts[-1])
                            intervals.append((first_ts, last_ts))
                            if ts.size >= 2:
                                d = np.diff(ts)
                                d = d[np.isfinite(d) & (d >= 0)]
                                within_dts.extend(d.tolist())
                                median_dt, p95_dt = med(d), pct(d, 95)

                    segment_rows.append({
                        "subject_id": subject,
                        "task": task,
                        "sentence_index": idx,
                        "segment_index": seg_idx,
                        "n_samples": int(seg.shape[0]),
                        "n_columns": int(n_cols),
                        "first_timestamp_ms": first_ts,
                        "last_timestamp_ms": last_ts,
                        "median_dt_ms": median_dt,
                        "p95_dt_ms": p95_dt,
                        "x_min": safe(np.nanmin(seg[:, 1])) if n_cols >= 2 else math.nan,
                        "x_max": safe(np.nanmax(seg[:, 1])) if n_cols >= 2 else math.nan,
                        "y_min": safe(np.nanmin(seg[:, 2])) if n_cols >= 3 else math.nan,
                        "y_max": safe(np.nanmax(seg[:, 2])) if n_cols >= 3 else math.nan,
                        "pupil_min": safe(np.nanmin(seg[:, 3])) if n_cols >= 4 else math.nan,
                        "pupil_max": safe(np.nanmax(seg[:, 3])) if n_cols >= 4 else math.nan,
                    })

                intervals.sort()
                gaps = [b[0] - a[1] for a, b in zip(intervals, intervals[1:])]
                first_ts = intervals[0][0] if intervals else math.nan
                last_ts = max(x[1] for x in intervals) if intervals else math.nan
                span = last_ts - first_ts if intervals else math.nan
                dt = med(within_dts)
                sampled_ms = n_samples * dt if n_samples and math.isfinite(dt) else math.nan
                coverage = sampled_ms / span if math.isfinite(span) and span > 0 else math.nan

                sentence_rows.append({
                    "subject_id": subject,
                    "task": task,
                    "sentence_index": idx,
                    "n_words": n_words,
                    "n_words_with_raw_et": n_words_with,
                    "n_raw_et_segments": len(segments),
                    "n_raw_et_samples": n_samples,
                    "n_all_fixations": n_all_fix,
                    "segment_to_all_fixation_ratio": len(segments) / n_all_fix if n_all_fix else math.nan,
                    "n_columns_mode": col_counts.most_common(1)[0][0] if col_counts else 0,
                    "median_within_segment_dt_ms": dt,
                    "p95_within_segment_dt_ms": pct(within_dts, 95),
                    "median_between_segment_gap_ms": med(gaps),
                    "max_between_segment_gap_ms": safe(max(gaps)) if gaps else math.nan,
                    "observed_span_ms": safe(span),
                    "sampled_duration_ms": safe(sampled_ms),
                    "temporal_coverage_ratio": safe(coverage),
                })

                if (idx + 1) % 50 == 0 or idx + 1 == n_sentences:
                    print(f"  {idx + 1}/{n_sentences}")

    file_rows = []
    for task in sorted({r["task"] for r in sentence_rows}):
        sub = [r for r in sentence_rows if r["task"] == task]
        words = sum(r["n_words"] for r in sub)
        words_with = sum(r["n_words_with_raw_et"] for r in sub)
        segs = sum(r["n_raw_et_segments"] for r in sub)
        fixs = sum(r["n_all_fixations"] for r in sub)
        file_rows.append({
            "subject_id": subject,
            "task": task,
            "n_sentences": len(sub),
            "n_words": words,
            "n_words_with_raw_et": words_with,
            "word_raw_et_coverage": words_with / words if words else math.nan,
            "n_raw_et_segments": segs,
            "n_raw_et_samples": sum(r["n_raw_et_samples"] for r in sub),
            "n_all_fixations": fixs,
            "segment_to_all_fixation_ratio": segs / fixs if fixs else math.nan,
            "median_within_segment_dt_ms": med(r["median_within_segment_dt_ms"] for r in sub),
            "median_between_segment_gap_ms": med(r["median_between_segment_gap_ms"] for r in sub),
            "median_temporal_coverage_ratio": med(r["temporal_coverage_ratio"] for r in sub),
            "sentences_with_no_raw_et": sum(r["n_raw_et_segments"] == 0 for r in sub),
            "sentences_with_non_4_column_mode": sum(r["n_columns_mode"] not in (0, 4) for r in sub),
        })

    write_csv(outdir / f"{subject}_raw_et_file_summary.csv", file_rows)
    write_csv(outdir / f"{subject}_raw_et_sentence_summary.csv", sentence_rows)
    write_csv(outdir / f"{subject}_raw_et_segment_summary.csv", segment_rows)

    print("\n=== FILE SUMMARY ===")
    for row in file_rows:
        print(row)
    print("\nWritten to", outdir)


if __name__ == "__main__":
    main()
