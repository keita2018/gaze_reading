#!/usr/bin/env python
r"""
Audit frozen ZuCo model artifacts before zero-shot transport to OneStop.

The script inventories the frozen directory, summarizes feature schemas, and
loads trusted local joblib/pickle artifacts to report object types, pipeline
steps, class order, feature counts, and any stored threshold/metadata. It does
not modify the frozen artifacts.

Example
-------
python -u src\models\audit_frozen_zuco_models.py ^
  --frozen-dir outputs\frozen_zuco_external_validation_v1 ^
  --output-dir outputs\frozen_zuco_external_validation_v1_audit
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None


SERIALIZED_SUFFIXES = {".joblib", ".pkl", ".pickle"}
TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-text-bytes",
        type=int,
        default=1_000_000,
    )
    return parser.parse_args()


def safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, (str, int, float, bool, list, dict)):
                return converted
        except Exception:
            pass
    return repr(value)


def summarize_estimator(obj: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python_type": f"{type(obj).__module__}.{type(obj).__qualname__}",
    }

    for attribute in (
        "n_features_in_",
        "feature_names_in_",
        "classes_",
        "coef_",
        "intercept_",
    ):
        if hasattr(obj, attribute):
            value = getattr(obj, attribute)
            if attribute in {"coef_", "intercept_"}:
                try:
                    summary[f"{attribute}_shape"] = list(value.shape)
                except Exception:
                    summary[attribute] = safe_scalar(value)
            else:
                summary[attribute] = safe_scalar(value)

    if hasattr(obj, "named_steps"):
        steps = {}
        for name, step in obj.named_steps.items():
            steps[name] = summarize_estimator(step)
        summary["named_steps"] = steps
    elif hasattr(obj, "steps"):
        summary["steps"] = [
            {
                "name": name,
                "estimator": summarize_estimator(step),
            }
            for name, step in obj.steps
        ]

    if isinstance(obj, dict):
        summary["dict_keys"] = [str(key) for key in obj.keys()]
        important = {}
        for key, value in obj.items():
            key_text = str(key).lower()
            if any(
                token in key_text
                for token in (
                    "threshold",
                    "feature",
                    "class",
                    "label",
                    "model",
                    "pipeline",
                    "scaler",
                    "imputer",
                    "subject",
                    "task",
                    "metric",
                    "version",
                )
            ):
                if isinstance(value, dict):
                    important[str(key)] = {
                        "python_type": (
                            f"{type(value).__module__}."
                            f"{type(value).__qualname__}"
                        ),
                        "keys": [str(k) for k in value.keys()],
                    }
                elif hasattr(value, "fit") or hasattr(
                    value, "predict"
                ):
                    important[str(key)] = summarize_estimator(value)
                else:
                    important[str(key)] = safe_scalar(value)
        summary["important_dict_values"] = important

    if isinstance(obj, (list, tuple)):
        summary["length"] = len(obj)
        summary["item_types"] = sorted(
            {
                f"{type(item).__module__}.{type(item).__qualname__}"
                for item in obj
            }
        )

    return summary


def load_serialized(path: Path) -> Any:
    if path.suffix.lower() == ".joblib":
        if joblib is None:
            raise RuntimeError("joblib is not installed.")
        return joblib.load(path)

    with path.open("rb") as handle:
        return pickle.load(handle)


def audit_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, low_memory=False)
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": frame.columns.astype(str).tolist(),
    }

    if {"feature_set", "feature_name"}.issubset(frame.columns):
        grouped = (
            frame.groupby("feature_set", dropna=False)["feature_name"]
            .agg(
                raw_rows="size",
                unique_features="nunique",
            )
            .reset_index()
        )
        summary["feature_sets"] = grouped.to_dict(orient="records")
        summary["ordered_unique_features"] = {
            str(feature_set): list(
                dict.fromkeys(
                    group["feature_name"].astype(str).tolist()
                )
            )
            for feature_set, group in frame.groupby(
                "feature_set", dropna=False, sort=False
            )
        }

    preview = frame.head(10).where(pd.notna(frame), None)
    summary["preview"] = preview.to_dict(orient="records")
    return summary


def audit_text(path: Path, max_bytes: int) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        return {
            "skipped": True,
            "reason": f"larger than {max_bytes} bytes",
        }

    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return {
                    "json_type": "dict",
                    "keys": [str(key) for key in data.keys()],
                    "content": data,
                }
            return {
                "json_type": type(data).__name__,
                "content": data,
            }
        except json.JSONDecodeError as exc:
            return {
                "json_parse_error": str(exc),
                "text_preview": text[:5000],
            }

    return {
        "text_preview": text[:10000],
        "truncated": len(text) > 10000,
    }


def main() -> int:
    args = parse_args()
    frozen_dir = args.frozen_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not frozen_dir.is_dir():
        raise FileNotFoundError(frozen_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [path for path in frozen_dir.rglob("*") if path.is_file()]
    )
    if not files:
        raise RuntimeError(f"No files found in {frozen_dir}")

    inventory_rows = []
    detail: dict[str, Any] = {}

    print("=== FROZEN ZUCO ARTIFACT INVENTORY ===")
    print(f"root: {frozen_dir}")
    print(f"files: {len(files)}")

    for path in files:
        relative = path.relative_to(frozen_dir).as_posix()
        suffix = path.suffix.lower()
        size = path.stat().st_size
        kind = "other"
        status = "not_inspected"
        information: dict[str, Any] = {}

        try:
            if suffix in SERIALIZED_SUFFIXES:
                kind = "serialized_model"
                obj = load_serialized(path)
                information = summarize_estimator(obj)
                status = "loaded"
            elif suffix == ".csv":
                kind = "csv"
                information = audit_csv(path)
                status = "loaded"
            elif suffix in TEXT_SUFFIXES:
                kind = "text"
                information = audit_text(
                    path, args.max_text_bytes
                )
                status = "loaded"
            else:
                kind = "other"
                status = "listed"
        except Exception as exc:
            status = "error"
            information = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        inventory_rows.append(
            {
                "relative_path": relative,
                "suffix": suffix,
                "size_bytes": size,
                "kind": kind,
                "status": status,
            }
        )
        detail[relative] = information

        print(
            f"{relative} | {kind} | {size:,} bytes | {status}"
        )
        if kind == "serialized_model":
            print(
                "  type: "
                + str(information.get("python_type", "unknown"))
            )
            if "dict_keys" in information:
                print(
                    "  dict keys: "
                    + ", ".join(information["dict_keys"])
                )
            if "n_features_in_" in information:
                print(
                    f"  n_features_in_: "
                    f"{information['n_features_in_']}"
                )
            if "classes_" in information:
                print(
                    f"  classes_: {information['classes_']}"
                )
            if "named_steps" in information:
                print(
                    "  pipeline steps: "
                    + ", ".join(
                        information["named_steps"].keys()
                    )
                )
        elif kind == "csv" and "feature_sets" in information:
            print("  feature sets:")
            for row in information["feature_sets"]:
                print(
                    f"    {row['feature_set']}: "
                    f"rows={row['raw_rows']}, "
                    f"unique={row['unique_features']}"
                )

    inventory = pd.DataFrame(inventory_rows)
    inventory_path = output_dir / "frozen_artifact_inventory.csv"
    detail_path = output_dir / "frozen_artifact_details.json"
    report_path = output_dir / "frozen_artifact_report.txt"

    inventory.to_csv(
        inventory_path, index=False, encoding="utf-8-sig"
    )
    detail_path.write_text(
        json.dumps(detail, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "=== FROZEN ZUCO ARTIFACT INVENTORY ===",
        f"root: {frozen_dir}",
        f"files: {len(files)}",
        "",
    ]
    for row in inventory_rows:
        report_lines.append(
            f"{row['relative_path']} | {row['kind']} | "
            f"{row['size_bytes']} bytes | {row['status']}"
        )
        info = detail[row["relative_path"]]
        if row["kind"] == "serialized_model":
            report_lines.append(
                f"  type: {info.get('python_type', 'unknown')}"
            )
            if "dict_keys" in info:
                report_lines.append(
                    "  dict keys: "
                    + ", ".join(info["dict_keys"])
                )
            if "n_features_in_" in info:
                report_lines.append(
                    f"  n_features_in_: {info['n_features_in_']}"
                )
            if "classes_" in info:
                report_lines.append(
                    f"  classes_: {info['classes_']}"
                )
            if "named_steps" in info:
                report_lines.append(
                    "  pipeline steps: "
                    + ", ".join(info["named_steps"].keys())
                )
        elif row["kind"] == "csv" and "feature_sets" in info:
            report_lines.append("  feature sets:")
            for feature_row in info["feature_sets"]:
                report_lines.append(
                    f"    {feature_row['feature_set']}: "
                    f"rows={feature_row['raw_rows']}, "
                    f"unique={feature_row['unique_features']}"
                )

    errors = inventory[inventory["status"].eq("error")]
    report_lines.extend(
        [
            "",
            "=== SUMMARY ===",
            f"serialized models: "
            f"{int(inventory['kind'].eq('serialized_model').sum())}",
            f"csv files: {int(inventory['kind'].eq('csv').sum())}",
            f"text files: {int(inventory['kind'].eq('text').sum())}",
            f"inspection errors: {len(errors)}",
        ]
    )
    if not errors.empty:
        report_lines.append("")
        report_lines.append("=== ERRORS ===")
        for row in errors.itertuples(index=False):
            report_lines.append(
                f"{row.relative_path}: "
                f"{detail[row.relative_path]}"
            )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    print(
        f"serialized models: "
        f"{int(inventory['kind'].eq('serialized_model').sum())}"
    )
    print(f"csv files: {int(inventory['kind'].eq('csv').sum())}")
    print(
        f"text files: {int(inventory['kind'].eq('text').sum())}"
    )
    print(f"inspection errors: {len(errors)}")
    print(f"\nSaved: {inventory_path}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {report_path}")

    if not errors.empty:
        print(
            "\nAUDIT STATUS: PARTIAL — some artifacts could not be "
            "inspected.",
            file=sys.stderr,
        )
        return 3

    print("\nAUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
