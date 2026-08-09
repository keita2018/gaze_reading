#!/usr/bin/env python
"""Audit prerequisites for the native-rate ZuCo I-VT fairness control."""
from __future__ import annotations
import sys
import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))
import numpy as np
import pandas as pd
from extract_zuco_soft_aoi_features import BASE_RATE_HZ
import run_zuco_ivt_event_baseline_fixed as fixed
import run_zuco_ivt_event_baseline_nested_threshold as nested

MANIFEST = Path("data/interim/zuco2/all_subjects/complete_raw_sample_manifest.csv")
OUT = Path("outputs/ivt_native_rate_audit_v1")

def main() -> int:
    if int(BASE_RATE_HZ) != 500:
        raise RuntimeError(f"Expected ZuCo BASE_RATE_HZ=500, found {BASE_RATE_HZ}.")
    required_fixed = ["main","extract_conditions","condition_rate_hz","detect_ivt_fixations","apply_condition","event_formula_features"]
    missing_fixed = [name for name in required_fixed if not hasattr(fixed, name)]
    if missing_fixed:
        raise RuntimeError(f"Current fixed I-VT script lacks required API: {missing_fixed}")
    if not hasattr(nested, "main"):
        raise RuntimeError("Current nested I-VT script lacks main().")
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    manifest = pd.read_csv(MANIFEST)
    required_manifest = {"subject_id","task","block_id","sample_parquet"}
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise RuntimeError(f"Manifest lacks columns: {missing}")
    manifest["subject_id"] = manifest["subject_id"].astype(str).str.upper()
    subjects = sorted(manifest["subject_id"].unique())
    if len(subjects) != 18:
        raise RuntimeError(f"Expected 18 subjects, found {len(subjects)}.")

    rate_rows = []
    for subject in subjects:
        row = manifest[manifest["subject_id"].eq(subject)].sort_values(["task","block_id"]).iloc[0]
        path = Path(str(row["sample_parquet"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        sample = pd.read_parquet(path, columns=["trial_id","timestamp_ms_abs"])
        if sample.empty:
            raise RuntimeError(f"Empty sample parquet: {path}")
        trial_id = str(sample["trial_id"].astype(str).iloc[0])
        trial = sample[sample["trial_id"].astype(str).eq(trial_id)].sort_values("timestamp_ms_abs")
        t = pd.to_numeric(trial["timestamp_ms_abs"], errors="coerce").to_numpy(float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            raise RuntimeError(f"No valid time increments: {path}")
        median_dt = float(np.median(dt))
        inferred_hz = 1000.0 / median_dt
        rate_rows.append({"subject_id":subject,"sample_parquet":str(path),"trial_id":trial_id,"median_dt_ms":median_dt,"inferred_rate_hz":inferred_hz})

    rates = pd.DataFrame(rate_rows)
    if not rates["inferred_rate_hz"].between(450.0,550.0).all():
        bad = rates[~rates["inferred_rate_hz"].between(450.0,550.0)]
        raise RuntimeError("Some inspected raw streams are not approximately 500 Hz:\n" + bad.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    rates.to_csv(OUT/"native_rate_sample_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status":"PASS",
        "base_rate_hz":int(BASE_RATE_HZ),
        "n_subjects":len(subjects),
        "inspected_blocks":len(rates),
        "median_inferred_rate_hz":float(rates["inferred_rate_hz"].median()),
        "min_inferred_rate_hz":float(rates["inferred_rate_hz"].min()),
        "max_inferred_rate_hz":float(rates["inferred_rate_hz"].max()),
        "native_rate_policy":{"clean":500,"noise50":500,"random40":500,"burst40":500,"high_random":500,"high_burst":500,"rate10":10},
        "interpretation":"All non-rate10 I-VT conditions operate on the original approximately 500-Hz stream before event detection; rate10 remains 10 Hz by definition."
    }
    (OUT/"native_rate_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("=== NATIVE-RATE I-VT INPUT AUDIT ===")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
