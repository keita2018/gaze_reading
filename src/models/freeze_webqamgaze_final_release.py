#!/usr/bin/env python
"""Freeze and audit the final WebQAmGaze external-validation release.

This stage performs no analysis changes. It:
1. verifies that Stage 8 primary values exactly match Stage 4;
2. verifies that Stage 8 secondary values exactly match Stage 7;
3. recomputes the four Holm-adjusted Stage 7 p-values;
4. verifies the label-orientation audit is PASS;
5. copies only final thesis/reporting artifacts into a release directory;
6. writes SHA-256 checksums and a reproducibility README.

No model, label, threshold, prediction, metric, table value, or figure is altered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_STAGE8_TO_STAGE4 = {
    "Dummy": "dummy_reference",
    "RawStat": "rawstat_all",
    "Full Soft-AOI": "soft_full_all",
    "Compact Soft-AOI": "soft_compact_distribution_membership",
    "Transferred official 9": "official_sent_gaze_sacc",
}

PRIMARY_METRICS = {
    "Macro F1": "macro_f1",
    "Balanced Accuracy": "balanced_accuracy",
    "ROC-AUC": "roc_auc",
    "PR-AUC (NR)": "pr_auc_nr",
    "Predicted NR rate": "predicted_nr_rate",
}

SECONDARY_STAGE8_TO_STAGE7 = {
    "Common16 Compact": "compact_common16_primary",
    "All18 Compact": "compact_all18_secondary",
}

SECONDARY_METRICS = PRIMARY_METRICS.copy()


class ReleaseAuditError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_true(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ReleaseAuditError(message)


def assert_close(
    actual: float,
    expected: float,
    message: str,
    atol: float = 1e-12,
) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=atol,
    ):
        raise ReleaseAuditError(
            f"{message}: actual={actual}, expected={expected}"
        )


def holm_adjust(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    adjusted = np.empty(len(array), dtype=float)
    running = 0.0
    n = len(array)

    for rank, index in enumerate(order):
        candidate = (n - rank) * array[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)

    return adjusted.tolist()


def copy_file(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def audit_primary(
    stage4_pooled_path: Path,
    stage8_primary_path: Path,
) -> list[str]:
    stage4 = pd.read_csv(stage4_pooled_path).set_index(
        "feature_set"
    )
    stage8 = pd.read_csv(stage8_primary_path).set_index(
        "Condition"
    )

    messages: list[str] = []
    for display_name, feature_set in (
        PRIMARY_STAGE8_TO_STAGE4.items()
    ):
        assert_true(
            display_name in stage8.index,
            f"Stage 8 primary lacks {display_name}",
        )
        assert_true(
            feature_set in stage4.index,
            f"Stage 4 pooled lacks {feature_set}",
        )
        for stage8_name, stage4_name in (
            PRIMARY_METRICS.items()
        ):
            assert_close(
                stage8.loc[
                    display_name,
                    stage8_name,
                ],
                stage4.loc[
                    feature_set,
                    stage4_name,
                ],
                (
                    f"Primary mismatch: {display_name} "
                    f"{stage8_name}"
                ),
            )
        messages.append(
            f"{display_name}: all primary metrics match Stage 4"
        )
    return messages


def audit_secondary(
    stage7_pooled_path: Path,
    stage8_secondary_path: Path,
) -> list[str]:
    stage7 = pd.read_csv(stage7_pooled_path).set_index(
        "condition"
    )
    stage8 = pd.read_csv(stage8_secondary_path).set_index(
        "Condition"
    )

    messages: list[str] = []
    for display_name, condition in (
        SECONDARY_STAGE8_TO_STAGE7.items()
    ):
        assert_true(
            display_name in stage8.index,
            f"Stage 8 secondary lacks {display_name}",
        )
        assert_true(
            condition in stage7.index,
            f"Stage 7 pooled lacks {condition}",
        )
        for stage8_name, stage7_name in (
            SECONDARY_METRICS.items()
        ):
            assert_close(
                stage8.loc[
                    display_name,
                    stage8_name,
                ],
                stage7.loc[
                    condition,
                    stage7_name,
                ],
                (
                    f"Secondary mismatch: {display_name} "
                    f"{stage8_name}"
                ),
            )
        messages.append(
            f"{display_name}: all secondary metrics match Stage 7"
        )
    return messages


def audit_holm(
    stage7_tests_path: Path,
    stage8_tests_path: Path,
) -> list[dict[str, Any]]:
    stage7 = pd.read_csv(stage7_tests_path)
    stage8 = pd.read_csv(stage8_tests_path)

    expected = holm_adjust(
        stage7["p_value"].astype(float).tolist()
    )
    expected_by_metric = {
        metric: adjusted
        for metric, adjusted in zip(
            stage7["metric"],
            expected,
        )
    }

    stage8_metric_map = {
        "Macro F1": "macro_f1",
        "Balanced Accuracy": "balanced_accuracy",
        "ROC-AUC": "roc_auc",
        "PR-AUC (NR)": "pr_auc_nr",
    }

    rows: list[dict[str, Any]] = []
    for metric_label in stage8["Metric"].astype(str):
        if metric_label not in stage8_metric_map:
            raise ReleaseAuditError(
                f"Unexpected Stage 8 metric label: {metric_label}"
            )
        source_metric = stage8_metric_map[metric_label]
        expected_value = expected_by_metric[source_metric]
        actual_value = float(
            stage8.loc[
                stage8["Metric"].astype(str).eq(metric_label),
                "Holm p across 4 metrics",
            ].iloc[0]
        )
        assert_close(
            actual_value,
            expected_value,
            f"Holm mismatch for {metric_label}",
        )
        rows.append(
            {
                "metric": metric_label,
                "source_metric": source_metric,
                "holm_p_recomputed": expected_value,
                "holm_p_stage8": actual_value,
            }
        )
    return rows


def build_readme(
    release_name: str,
    created_at: str,
) -> str:
    return f"""# {release_name}

## Purpose

This directory is the frozen reporting package for the ZuCo-to-WebQAmGaze external validation.

Created at: {created_at}

## Analysis hierarchy

### Primary analysis

Matched Common16 five-condition comparison:

- Dummy
- RawStat
- Full Soft-AOI
- Compact Soft-AOI
- Transferred event-based official-feature-formula condition

Training scope:

- 16 ZuCo subjects
- 11,324 matched trials

### Secondary analysis

Compact training-scope comparison:

- Common16 Compact: 16 subjects, 11,324 trials
- All18 Compact: 18 subjects, 13,302 trials

The All18 condition must not be inserted into the primary five-condition table.

## Fixed labels

- ZuCo: TSR=0, NR=1
- WebQAmGaze: IS=0, NR=1
- Positive class: NR=1

The completed label audit reports:

`NO LABEL REVERSAL DETECTED`

## Fixed modeling protocol

- 27 Compact Soft-AOI features for Compact conditions
- median imputation estimated on ZuCo
- missing indicators estimated on ZuCo
- StandardScaler estimated on ZuCo
- LinearSVC
- C=0.01
- decision threshold=0.0
- no WebQAmGaze retraining
- no external feature selection
- no external threshold tuning
- no external calibration

## Reporting constraints

- Compact Soft-AOI was relatively stronger than RawStat.
- Compact Soft-AOI did not achieve absolute external generalization.
- All18 produced only small improvements.
- No Stage 7 paired metric remained significant after Holm correction.
- The transferred official 9-feature condition is not identical to EyeLink official events.

## Integrity

Use `SHA256SUMS.txt` or `release_manifest.json` to verify file integrity.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage4"
        ),
    )
    parser.add_argument(
        "--stage5-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage5_diagnostics"
        ),
    )
    parser.add_argument(
        "--stage6-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage6_thesis"
        ),
    )
    parser.add_argument(
        "--stage7-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage7_all18"
        ),
    )
    parser.add_argument(
        "--stage8-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage8_final"
        ),
    )
    parser.add_argument(
        "--label-audit",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_label_audit.json"
        ),
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_final_release_v1"
        ),
    )
    args = parser.parse_args()

    stage4 = args.stage4_dir.resolve()
    stage5 = args.stage5_dir.resolve()
    stage6 = args.stage6_dir.resolve()
    stage7 = args.stage7_dir.resolve()
    stage8 = args.stage8_dir.resolve()
    label_audit_path = args.label_audit.resolve()
    release = args.release_dir.resolve()

    # Do not create the release directory until every audit has passed.
    # A previous failed run may have left an empty directory; remove only
    # that empty partial directory. Never overwrite a non-empty release.
    if release.exists():
        if any(release.iterdir()):
            raise FileExistsError(
                f"Non-empty release directory already exists: {release}"
            )
        release.rmdir()

    source_paths = {
        "stage4_pooled": require_file(
            stage4 / "external_pooled_metrics.csv"
        ),
        "stage4_manifest": require_file(
            stage4 / "stage4_manifest.json"
        ),
        "stage5_polarity": require_file(
            stage5 / "model_score_polarity_audit.csv"
        ),
        "stage5_contributions": require_file(
            stage5
            / "model_feature_contribution_summary.csv"
        ),
        "stage5_manifest": require_file(
            stage5 / "stage5_diagnostic_manifest.json"
        ),
        "stage6_primary": require_file(
            stage6
            / "thesis_external_validation_results.csv"
        ),
        "stage6_text": require_file(
            stage6
            / "thesis_external_validation_results_ja.md"
        ),
        "stage6_metrics_figure": require_file(
            stage6
            / "figure_external_validation_metrics.png"
        ),
        "stage6_orientation_figure": require_file(
            stage6
            / "figure_score_orientation.png"
        ),
        "stage6_transport_figure": require_file(
            stage6
            / "figure_transport_shift_top20.png"
        ),
        "stage6_compact_figure": require_file(
            stage6
            / "figure_compact_feature_contributions.png"
        ),
        "stage6_full_figure": require_file(
            stage6
            / "figure_full_dynamic_feature_contributions.png"
        ),
        "stage6_official_figure": require_file(
            stage6
            / "figure_official_feature_contributions.png"
        ),
        "stage6_manifest": require_file(
            stage6 / "stage6_manifest.json"
        ),
        "stage7_pooled": require_file(
            stage7
            / "secondary_compact_pooled_metrics.csv"
        ),
        "stage7_tests": require_file(
            stage7
            / "secondary_compact_paired_subject_tests.csv"
        ),
        "stage7_bootstrap": require_file(
            stage7
            / "secondary_compact_bootstrap_differences.csv"
        ),
        "stage7_figure": require_file(
            stage7
            / "figure_compact_common16_vs_all18.png"
        ),
        "stage7_text": require_file(
            stage7 / "secondary_compact_results_ja.md"
        ),
        "stage7_manifest": require_file(
            stage7 / "stage7_manifest.json"
        ),
        "stage8_primary": require_file(
            stage8 / "final_primary_results.csv"
        ),
        "stage8_primary_md": require_file(
            stage8 / "final_primary_results.md"
        ),
        "stage8_secondary": require_file(
            stage8 / "final_secondary_all18_results.csv"
        ),
        "stage8_secondary_md": require_file(
            stage8 / "final_secondary_all18_results.md"
        ),
        "stage8_tests": require_file(
            stage8
            / "final_secondary_all18_tests_holm.csv"
        ),
        "stage8_tests_md": require_file(
            stage8
            / "final_secondary_all18_tests_holm.md"
        ),
        "stage8_overview_figure": require_file(
            stage8
            / "figure_final_external_validation_overview.png"
        ),
        "stage8_final_text": require_file(
            stage8
            / "final_external_validation_results_ja.md"
        ),
        "stage8_checklist": require_file(
            stage8
            / "external_validation_reporting_checklist.md"
        ),
        "stage8_manifest": require_file(
            stage8 / "stage8_manifest.json"
        ),
        "label_audit": require_file(
            label_audit_path
        ),
    }

    primary_messages = audit_primary(
        source_paths["stage4_pooled"],
        source_paths["stage8_primary"],
    )
    secondary_messages = audit_secondary(
        source_paths["stage7_pooled"],
        source_paths["stage8_secondary"],
    )
    holm_rows = audit_holm(
        source_paths["stage7_tests"],
        source_paths["stage8_tests"],
    )

    label_audit = json.loads(
        source_paths["label_audit"].read_text(
            encoding="utf-8"
        )
    )
    assert_true(
        label_audit.get("status") == "PASS",
        (
            "Label audit status is not PASS: "
            f"{label_audit.get('status')}"
        ),
    )
    assert_true(
        label_audit.get("final_result")
        == "NO LABEL REVERSAL DETECTED",
        (
            "Label audit final_result is unexpected: "
            f"{label_audit.get('final_result')}"
        ),
    )

    # All integrity checks have now passed; create the frozen release.
    release.mkdir(parents=True)

    release_name = release.name
    created_at = utc_now_iso()

    copy_plan = {
        "tables/final_primary_results.csv": (
            source_paths["stage8_primary"]
        ),
        "tables/final_primary_results.md": (
            source_paths["stage8_primary_md"]
        ),
        "tables/final_secondary_all18_results.csv": (
            source_paths["stage8_secondary"]
        ),
        "tables/final_secondary_all18_results.md": (
            source_paths["stage8_secondary_md"]
        ),
        "tables/final_secondary_all18_tests_holm.csv": (
            source_paths["stage8_tests"]
        ),
        "tables/final_secondary_all18_tests_holm.md": (
            source_paths["stage8_tests_md"]
        ),
        "text/final_external_validation_results_ja.md": (
            source_paths["stage8_final_text"]
        ),
        "text/reporting_checklist.md": (
            source_paths["stage8_checklist"]
        ),
        "text/primary_results_detail_ja.md": (
            source_paths["stage6_text"]
        ),
        "text/secondary_all18_detail_ja.md": (
            source_paths["stage7_text"]
        ),
        "figures/final_external_validation_overview.png": (
            source_paths["stage8_overview_figure"]
        ),
        "figures/primary_external_validation_metrics.png": (
            source_paths["stage6_metrics_figure"]
        ),
        "figures/score_orientation.png": (
            source_paths["stage6_orientation_figure"]
        ),
        "figures/transport_shift_top20.png": (
            source_paths["stage6_transport_figure"]
        ),
        "figures/compact_feature_contributions.png": (
            source_paths["stage6_compact_figure"]
        ),
        "figures/full_dynamic_feature_contributions.png": (
            source_paths["stage6_full_figure"]
        ),
        "figures/official_feature_contributions.png": (
            source_paths["stage6_official_figure"]
        ),
        "figures/compact_common16_vs_all18.png": (
            source_paths["stage7_figure"]
        ),
        "audit/label_orientation_audit.json": (
            source_paths["label_audit"]
        ),
        "audit/model_score_polarity_audit.csv": (
            source_paths["stage5_polarity"]
        ),
        "audit/model_feature_contribution_summary.csv": (
            source_paths["stage5_contributions"]
        ),
        "manifests/stage4_manifest.json": (
            source_paths["stage4_manifest"]
        ),
        "manifests/stage5_manifest.json": (
            source_paths["stage5_manifest"]
        ),
        "manifests/stage6_manifest.json": (
            source_paths["stage6_manifest"]
        ),
        "manifests/stage7_manifest.json": (
            source_paths["stage7_manifest"]
        ),
        "manifests/stage8_manifest.json": (
            source_paths["stage8_manifest"]
        ),
    }

    copied: dict[str, dict[str, Any]] = {}
    for relative_name, source in copy_plan.items():
        destination = release / relative_name
        copied[relative_name] = copy_file(
            source,
            destination,
        )

    readme_path = release / "README.md"
    readme_path.write_text(
        build_readme(
            release_name,
            created_at,
        ),
        encoding="utf-8",
    )

    release_manifest = {
        "release_name": release_name,
        "release_version": "1.1",
        "created_at_utc": created_at,
        "status": "PASS",
        "analysis_changes": False,
        "label_orientation": (
            "NO LABEL REVERSAL DETECTED"
        ),
        "primary_integrity_checks": (
            primary_messages
        ),
        "secondary_integrity_checks": (
            secondary_messages
        ),
        "holm_recomputation": holm_rows,
        "copied_files": copied,
        "reporting_hierarchy": {
            "primary": (
                "Common16 matched five-condition comparison"
            ),
            "secondary": (
                "Common16 Compact vs All18 Compact"
            ),
        },
    }

    manifest_path = release / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(
            release_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    checksum_entries = []
    for path in sorted(
        release.rglob("*")
    ):
        if not path.is_file():
            continue
        if path.name == "SHA256SUMS.txt":
            continue
        checksum_entries.append(
            (
                sha256_file(path),
                path.relative_to(release).as_posix(),
            )
        )

    checksum_path = release / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{digest}  {relative}\n"
            for digest, relative in checksum_entries
        ),
        encoding="ascii",
    )

    print(
        "=== WEBQAMGAZE FINAL RELEASE AUDIT ==="
    )
    print(
        "FINAL RESULT: PASS - RELEASE FROZEN"
    )
    print(
        "Label orientation: "
        "NO LABEL REVERSAL DETECTED"
    )
    print()
    print("Primary integrity:")
    for message in primary_messages:
        print(f"  [PASS] {message}")
    print()
    print("Secondary integrity:")
    for message in secondary_messages:
        print(f"  [PASS] {message}")
    print()
    print("Holm recomputation:")
    print(
        pd.DataFrame(holm_rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Release directory: {release}")
    print(f"Release files: {len(checksum_entries) + 1}")
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {checksum_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseAuditError as error:
        print(
            "FINAL RESULT: FAIL - RELEASE NOT CREATED"
        )
        print(str(error))
        raise SystemExit(1)
