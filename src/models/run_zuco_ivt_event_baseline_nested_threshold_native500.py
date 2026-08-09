#!/usr/bin/env python
"""Native-rate nested quality-tuned I-VT control for ZuCo."""
from __future__ import annotations
import json, sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))
from extract_zuco_soft_aoi_features import BASE_RATE_HZ
import run_zuco_ivt_event_baseline_fixed as fixed_base

DEFAULT_FIXED_OUT = Path("outputs/ivt_event_baseline_fixed_native500_v1")
DEFAULT_OUT = Path("outputs/ivt_event_baseline_nested_threshold_native500_v1")

def native_condition_rate_hz(condition: str) -> int:
    return 10 if str(condition).lower() == "rate10" else int(BASE_RATE_HZ)

def ensure_arg(flag: str, value: str) -> None:
    if flag not in sys.argv[1:]:
        sys.argv.extend([flag, value])

def annotate_manifest() -> None:
    path = DEFAULT_OUT / "ivt_nested_manifest.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["native_rate_fairness_control"] = True
    data["native_base_rate_hz"] = int(BASE_RATE_HZ)
    data["sampling_policy"] = {"clean":int(BASE_RATE_HZ),"noise50":int(BASE_RATE_HZ),"random40":int(BASE_RATE_HZ),"burst40":int(BASE_RATE_HZ),"high_random":int(BASE_RATE_HZ),"high_burst":int(BASE_RATE_HZ),"rate10":10}
    data["comparison_role"] = "Native-rate, condition-specific threshold-tuned event baseline; outer-test labels are never used for threshold selection."
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")

def main() -> int:
    if int(BASE_RATE_HZ) != 500:
        raise RuntimeError(f"Expected BASE_RATE_HZ=500, found {BASE_RATE_HZ}.")
    fixed_base.condition_rate_hz = native_condition_rate_hz
    import run_zuco_ivt_event_baseline_nested_threshold as nested
    ensure_arg("--output-dir", str(DEFAULT_OUT))
    ensure_arg("--fixed-feature-root", str(DEFAULT_FIXED_OUT/"subject_ivt_features"))
    ensure_arg("--fixed-subject-metrics", str(DEFAULT_FIXED_OUT/"ivt_fixed_subject_metrics.csv"))
    ensure_arg("--conditions", "clean,rate10,noise50,random40,burst40,high_random,high_burst")
    ensure_arg("--velocity-thresholds", "30,60,120,240,480")
    ensure_arg("--standard-threshold", "30")
    ensure_arg("--inner-splits", "4")
    ensure_arg("--repeats", "5")
    ensure_arg("--min-fixation-ms", "100")
    ensure_arg("--c-value", "0.01")
    ensure_arg("--seed", "1")
    result = nested.main()
    annotate_manifest()
    return 0 if result is None else int(result)

if __name__ == "__main__":
    raise SystemExit(main())
