#!/usr/bin/env python
"""End-to-end audit of label orientation for the ZuCo -> WebQAmGaze transfer.

This script does not train, tune, modify, or overwrite any model or result.
It verifies the actual files used in the completed external validation.

Required semantics
------------------
ZuCo:
    TSR = 0
    NR  = 1

WebQAmGaze:
    IS  = 0
    NR  = 1

Thus, NR remains class 1 in both datasets. TSR and IS occupy the common
goal-directed/non-NR side, class 0. They are not claimed to be identical tasks.

Audited links
-------------
1. ZuCo Common16 training task -> label_nr
2. Frozen joblib label metadata and classifier classes_
3. WebQAmGaze condition -> label_nr
4. WebQAmGaze trial position -> condition
5. Stage 3A / Stage 3C metadata propagation
6. Stage 4 decision_function -> predicted class
7. Stage 4 predicted class -> predicted task
8. Saved predictions and metrics vs fresh recomputation
9. Stage 5 AUC vs Stage 4 AUC

A successful run ends with:
    FINAL RESULT: PASS — NO LABEL REVERSAL DETECTED
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


MODEL_ORDER = [
    "dummy_reference",
    "rawstat_all",
    "soft_full_all",
    "soft_compact_distribution_membership",
    "official_sent_gaze_sacc",
]

ZUCO_MAP = {"TSR": 0, "NR": 1}
WEB_MAP = {"IS": 0, "NR": 1}


class AuditFailure(RuntimeError):
    pass


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def assert_true(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AuditFailure(message)


def assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AuditFailure(
            f"{message}: actual={actual!r}, expected={expected!r}"
        )


def assert_allclose(
    actual: np.ndarray | pd.Series,
    expected: np.ndarray | pd.Series,
    message: str,
    atol: float = 1e-10,
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise AuditFailure(
            f"{message}: shape mismatch "
            f"{actual_array.shape} vs {expected_array.shape}"
        )
    if not np.allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=atol,
        equal_nan=True,
    ):
        difference = np.abs(
            actual_array.astype(float)
            - expected_array.astype(float)
        )
        max_difference = float(np.nanmax(difference))
        raise AuditFailure(
            f"{message}: max_abs_difference={max_difference}"
        )


def normalize_mapping(mapping: Any) -> dict[str, int]:
    if not isinstance(mapping, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in mapping.items():
        normalized[str(key).upper()] = int(value)
    return normalized


def find_artifact(frozen_dir: Path, feature_set: str) -> Path:
    exact = frozen_dir / "models" / f"{feature_set}.joblib"
    if exact.is_file():
        return exact
    matches = list(frozen_dir.rglob(f"{feature_set}.joblib"))
    if len(matches) != 1:
        raise AuditFailure(
            f"Expected exactly one artifact for {feature_set}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def aligned_table(path: Path, trial_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path))
    assert_true(
        "external_trial_id" in frame.columns,
        f"{path} lacks external_trial_id",
    )
    frame["external_trial_id"] = frame[
        "external_trial_id"
    ].astype(str)
    assert_true(
        not frame["external_trial_id"].duplicated().any(),
        f"{path} has duplicate trial IDs",
    )
    indexed = frame.set_index("external_trial_id")
    assert_equal(
        set(indexed.index),
        set(trial_ids),
        f"{path} trial-ID set",
    )
    return indexed.loc[trial_ids].reset_index()


def check_task_label_frame(
    frame: pd.DataFrame,
    mapping: dict[str, int],
    frame_name: str,
) -> None:
    required = {"task", "label_nr"}
    missing = required - set(frame.columns)
    assert_true(
        not missing,
        f"{frame_name} lacks columns {sorted(missing)}",
    )
    task = frame["task"].astype(str).str.upper()
    label = pd.to_numeric(
        frame["label_nr"],
        errors="raise",
    ).astype(int)
    expected = task.map(mapping)
    assert_true(
        expected.notna().all(),
        f"{frame_name} has unexpected tasks: "
        f"{sorted(task[expected.isna()].unique())}",
    )
    mismatch = label.ne(expected.astype(int))
    if mismatch.any():
        examples = frame.loc[
            mismatch,
            ["task", "label_nr"],
        ].head(20)
        raise AuditFailure(
            f"{frame_name} task/label mismatch:\n"
            f"{examples.to_string(index=False)}"
        )


def compare_metadata(
    master: pd.DataFrame,
    other: pd.DataFrame,
    name: str,
) -> None:
    other_aligned = other.set_index(
        other["external_trial_id"].astype(str)
    ).loc[
        master["external_trial_id"].astype(str)
    ].reset_index(drop=True)

    assert_equal(
        other_aligned["task"].astype(str).str.upper().tolist(),
        master["task"].astype(str).str.upper().tolist(),
        f"{name} task propagation",
    )
    assert_equal(
        pd.to_numeric(
            other_aligned["label_nr"],
            errors="raise",
        ).astype(int).tolist(),
        master["label_nr"].astype(int).tolist(),
        f"{name} label propagation",
    )


def recompute_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "macro_f1": float(
            f1_score(
                y,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc_nr": float(
            average_precision_score(y, score)
        ),
        "positive_rate_nr": float(np.mean(y == 1)),
        "predicted_nr_rate": float(np.mean(prediction == 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage2"
        ),
    )
    parser.add_argument(
        "--stage3a-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage3a"
        ),
    )
    parser.add_argument(
        "--stage3c-dir",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_stage3c"
        ),
    )
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
        "--frozen-dir",
        type=Path,
        default=Path(
            "outputs/frozen_zuco_external_validation_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/webqamgaze_external_validation_label_audit.json"
        ),
    )
    args = parser.parse_args()

    stage2 = args.stage2_dir.resolve()
    stage3a = args.stage3a_dir.resolve()
    stage3c = args.stage3c_dir.resolve()
    stage4 = args.stage4_dir.resolve()
    stage5 = args.stage5_dir.resolve()
    frozen = args.frozen_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "zuco_training_trials": require_file(
            frozen / "common16_training_trials.csv"
        ),
        "external_trials": require_file(
            stage2 / "trial_manifest_primary_candidates.csv"
        ),
        "rawstat": require_file(
            stage3a / "rawstat_features_primary.csv"
        ),
        "soft_full": require_file(
            stage3c / "soft_full_features_primary.csv"
        ),
        "soft_compact": require_file(
            stage3c / "soft_compact_features_primary.csv"
        ),
        "official": require_file(
            stage4 / "corrected_official_features_primary.csv"
        ),
        "predictions": require_file(
            stage4 / "external_predictions.csv"
        ),
        "pooled_metrics": require_file(
            stage4 / "external_pooled_metrics.csv"
        ),
        "stage4_manifest": require_file(
            stage4 / "stage4_manifest.json"
        ),
        "stage5_polarity": require_file(
            stage5 / "model_score_polarity_audit.csv"
        ),
    }

    report: dict[str, Any] = {
        "audit_version": "1.1",
        "status": "running",
        "required_semantics": {
            "ZuCo": ZUCO_MAP,
            "WebQAmGaze": WEB_MAP,
            "shared_positive_class": "NR=1",
            "shared_negative_side": "ZuCo TSR=0; WebQAmGaze IS=0",
        },
        "checks": [],
    }

    def passed(name: str, detail: Any = None) -> None:
        report["checks"].append(
            {
                "name": name,
                "status": "PASS",
                "detail": detail,
            }
        )
        print(f"[PASS] {name}")
        if detail is not None:
            print(f"       {detail}")

    # 1. ZuCo training labels.
    zuco = pd.read_csv(paths["zuco_training_trials"])
    check_task_label_frame(
        zuco,
        ZUCO_MAP,
        "ZuCo Common16 training trials",
    )
    assert_equal(
        set(zuco["task"].astype(str).str.upper().unique()),
        {"NR", "TSR"},
        "ZuCo task set",
    )
    zuco_counts_raw = (
        zuco.groupby(
            [
                zuco["task"].astype(str).str.upper(),
                pd.to_numeric(
                    zuco["label_nr"],
                    errors="raise",
                ).astype(int),
            ]
        )
        .size()
        .to_dict()
    )
    zuco_counts = {
        f"{task}={int(label)}": int(count)
        for (task, label), count in zuco_counts_raw.items()
    }
    passed(
        "ZuCo training mapping is TSR=0 and NR=1",
        zuco_counts,
    )

    # 2. External trial labels.
    external = pd.read_csv(paths["external_trials"])
    external["external_trial_id"] = external[
        "external_trial_id"
    ].astype(str)
    external["task"] = external[
        "task"
    ].astype(str).str.upper()
    external["label_nr"] = pd.to_numeric(
        external["label_nr"],
        errors="raise",
    ).astype(int)
    check_task_label_frame(
        external,
        WEB_MAP,
        "WebQAmGaze primary trials",
    )
    assert_equal(
        set(external["task"].unique()),
        {"NR", "IS"},
        "WebQAmGaze task set",
    )

    expected_by_position = np.where(
        pd.to_numeric(
            external["trial_index"],
            errors="raise",
        ).to_numpy(dtype=int) < 5,
        "NR",
        "IS",
    )
    assert_equal(
        external["task"].tolist(),
        expected_by_position.tolist(),
        "WebQAmGaze position/condition mapping",
    )
    if "fixed_label_mapping" in external.columns:
        assert_equal(
            set(
                external["fixed_label_mapping"]
                .dropna()
                .astype(str)
                .unique()
            ),
            {"NR=1;IS=0"},
            "WebQAmGaze recorded fixed mapping",
        )
    external_counts_raw = (
        external.groupby(["task", "label_nr"])
        .size()
        .to_dict()
    )
    external_counts = {
        f"{task}={int(label)}": int(count)
        for (task, label), count in external_counts_raw.items()
    }
    passed(
        "WebQAmGaze mapping is IS=0 and NR=1",
        external_counts,
    )
    passed(
        "WebQAmGaze first five trials are NR and last five are IS",
    )

    # 3. Stage 3 propagation.
    master_ids = external["external_trial_id"].tolist()
    rawstat = aligned_table(paths["rawstat"], master_ids)
    soft_full = aligned_table(paths["soft_full"], master_ids)
    soft_compact = aligned_table(
        paths["soft_compact"],
        master_ids,
    )
    official = aligned_table(paths["official"], master_ids)

    for frame, name in (
        (rawstat, "Stage 3A RawStat"),
        (soft_full, "Stage 3C Full Soft-AOI"),
        (soft_compact, "Stage 3C Compact Soft-AOI"),
        (official, "Stage 4 corrected official"),
    ):
        compare_metadata(external, frame, name)
        passed(f"{name} preserves task and label_nr")

    assert_true(
        "zuco_task_equivalent" in soft_full.columns,
        "Stage 3C Full lacks zuco_task_equivalent",
    )
    expected_equivalent = np.where(
        external["task"].eq("NR"),
        "NR",
        "TSR",
    )
    assert_equal(
        soft_full[
            "zuco_task_equivalent"
        ].astype(str).str.upper().tolist(),
        expected_equivalent.tolist(),
        "Stage 3C external-to-ZuCo task equivalence",
    )
    passed(
        "Stage 3C maps WebQAmGaze NR->ZuCo NR and IS->ZuCo TSR",
        {
            "NR": "NR",
            "IS": "TSR",
        },
    )

    # 4. Stage 4 manifest.
    manifest = json.loads(
        paths["stage4_manifest"].read_text(
            encoding="utf-8"
        )
    )
    manifest_mapping = normalize_mapping(
        manifest.get("fixed_protocol", {}).get(
            "label_mapping",
            {},
        )
    )
    assert_equal(
        manifest_mapping,
        WEB_MAP,
        "Stage 4 manifest label mapping",
    )
    passed("Stage 4 manifest records IS=0 and NR=1")

    # 5. Frozen artifact orientation and exact saved predictions.
    predictions = pd.read_csv(paths["predictions"])
    predictions["external_trial_id"] = predictions[
        "external_trial_id"
    ].astype(str)
    pooled = pd.read_csv(paths["pooled_metrics"])
    pooled_lookup = pooled.set_index("feature_set")

    feature_tables = {
        "rawstat_all": rawstat,
        "soft_full_all": soft_full,
        "soft_compact_distribution_membership": (
            soft_compact
        ),
        "official_sent_gaze_sacc": official,
    }

    y = external["label_nr"].to_numpy(dtype=int)

    artifact_details: dict[str, Any] = {}
    for feature_set in MODEL_ORDER:
        artifact_path = find_artifact(
            frozen,
            feature_set,
        )
        artifact = joblib.load(artifact_path)

        assert_equal(
            artifact.get("feature_set"),
            feature_set,
            f"{feature_set} artifact identity",
        )
        assert_equal(
            normalize_mapping(
                artifact.get("label_to_int", {})
            ),
            ZUCO_MAP,
            f"{feature_set} artifact label_to_int",
        )

        prediction_group = predictions[
            predictions["feature_set"].eq(feature_set)
        ].copy()
        prediction_group = prediction_group.set_index(
            "external_trial_id"
        ).loc[master_ids].reset_index()

        compare_metadata(
            external,
            prediction_group,
            f"Stage 4 predictions/{feature_set}",
        )

        saved_prediction = pd.to_numeric(
            prediction_group["predicted_label_nr"],
            errors="raise",
        ).to_numpy(dtype=int)
        saved_score = pd.to_numeric(
            prediction_group["decision_score_nr"],
            errors="raise",
        ).to_numpy(dtype=float)

        expected_task = np.where(
            saved_prediction == 1,
            "NR",
            "IS",
        )
        assert_equal(
            prediction_group[
                "predicted_task"
            ].astype(str).str.upper().tolist(),
            expected_task.tolist(),
            f"{feature_set} predicted-label/task mapping",
        )
        expected_correct = (
            saved_prediction == y
        ).astype(int)
        assert_equal(
            pd.to_numeric(
                prediction_group["correct"],
                errors="raise",
            ).to_numpy(dtype=int).tolist(),
            expected_correct.tolist(),
            f"{feature_set} correct flag",
        )

        if feature_set == "dummy_reference":
            model = artifact["model"]
            classes = np.asarray(
                model.classes_,
                dtype=int,
            ).tolist()
            assert_equal(
                classes,
                [0, 1],
                "Dummy classifier classes_",
            )
            direct_prediction = np.asarray(
                model.predict(
                    np.zeros((len(external), 1))
                ),
                dtype=int,
            )
            assert_equal(
                direct_prediction.tolist(),
                saved_prediction.tolist(),
                "Dummy saved predictions",
            )
            fresh_score = saved_score
            threshold = None
        else:
            pipeline = artifact["pipeline"]
            model = pipeline.named_steps["model"]
            classes = np.asarray(
                model.classes_,
                dtype=int,
            ).tolist()
            assert_equal(
                classes,
                [0, 1],
                f"{feature_set} LinearSVC classes_",
            )

            threshold = float(
                artifact["decision_threshold"]
            )
            assert_equal(
                threshold,
                0.0,
                f"{feature_set} frozen threshold",
            )

            columns = list(
                artifact["feature_columns"]
            )
            table = feature_tables[feature_set]
            x = table[columns].copy()
            for column in columns:
                x[column] = pd.to_numeric(
                    x[column],
                    errors="coerce",
                )

            fresh_score = np.asarray(
                pipeline.decision_function(x),
                dtype=float,
            ).reshape(-1)
            direct_prediction = np.asarray(
                pipeline.predict(x),
                dtype=int,
            ).reshape(-1)
            threshold_prediction = (
                fresh_score >= threshold
            ).astype(int)

            assert_equal(
                direct_prediction.tolist(),
                threshold_prediction.tolist(),
                (
                    f"{feature_set} positive decision score "
                    "maps to class 1"
                ),
            )
            assert_allclose(
                fresh_score,
                saved_score,
                f"{feature_set} saved decision scores",
            )
            assert_equal(
                threshold_prediction.tolist(),
                saved_prediction.tolist(),
                f"{feature_set} saved predicted labels",
            )

        metrics = recompute_metrics(
            y,
            saved_prediction,
            fresh_score,
        )
        saved_metric_row = pooled_lookup.loc[
            feature_set
        ]
        for metric_name, metric_value in metrics.items():
            assert_true(
                math.isclose(
                    float(saved_metric_row[metric_name]),
                    metric_value,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ),
                (
                    f"{feature_set} pooled metric mismatch "
                    f"for {metric_name}: "
                    f"saved={saved_metric_row[metric_name]}, "
                    f"fresh={metric_value}"
                ),
            )

        artifact_details[feature_set] = {
            "classes": classes,
            "threshold": threshold,
            "ordinary_auc_nr_positive": metrics[
                "roc_auc"
            ],
            "reversed_auc": float(
                roc_auc_score(y, -fresh_score)
            ),
            "predicted_nr_rate": metrics[
                "predicted_nr_rate"
            ],
        }
        passed(
            (
                f"{feature_set}: class 1 is NR, "
                "saved scores/predictions/metrics reproduce exactly"
            ),
            artifact_details[feature_set],
        )

    # 6. Stage 5 uses the same orientation as Stage 4.
    polarity = pd.read_csv(paths["stage5_polarity"])
    polarity_lookup = polarity.set_index("feature_set")
    for feature_set in MODEL_ORDER[1:]:
        stage4_auc = float(
            pooled_lookup.loc[feature_set, "roc_auc"]
        )
        stage5_auc = float(
            polarity_lookup.loc[
                feature_set,
                "pooled_auc",
            ]
        )
        reversed_auc = float(
            polarity_lookup.loc[
                feature_set,
                "pooled_reversed_auc",
            ]
        )
        assert_true(
            math.isclose(
                stage4_auc,
                stage5_auc,
                rel_tol=0.0,
                abs_tol=1e-10,
            ),
            (
                f"{feature_set} Stage 4/5 AUC mismatch: "
                f"{stage4_auc} vs {stage5_auc}"
            ),
        )
        assert_true(
            math.isclose(
                stage5_auc + reversed_auc,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-10,
            ),
            (
                f"{feature_set} ordinary/reversed AUC "
                "does not sum to 1"
            ),
        )
    passed(
        "Stage 5 polarity audit uses the same NR=1 orientation as Stage 4"
    )

    # A global inversion would reverse all models. The official condition
    # retaining ordinary AUC > .5 is an additional empirical consistency check.
    official_auc = artifact_details[
        "official_sent_gaze_sacc"
    ]["ordinary_auc_nr_positive"]
    official_reversed = artifact_details[
        "official_sent_gaze_sacc"
    ]["reversed_auc"]
    assert_true(
        official_auc > 0.5
        and official_reversed < 0.5,
        (
            "Official condition does not support expected "
            "ordinary score orientation"
        ),
    )
    passed(
        (
            "Empirical global-inversion check: official ordinary AUC "
            "is above 0.5 and reversed AUC is below 0.5"
        ),
        {
            "ordinary_auc": official_auc,
            "reversed_auc": official_reversed,
        },
    )

    report["artifact_details"] = artifact_details
    report["status"] = "PASS"
    report["final_result"] = (
        "NO LABEL REVERSAL DETECTED"
    )
    report["interpretation"] = {
        "numeric_orientation": (
            "NR is class 1 in both ZuCo and WebQAmGaze."
        ),
        "negative_class_alignment": (
            "ZuCo TSR and WebQAmGaze IS are both class 0."
        ),
        "caveat": (
            "TSR and IS are related goal-directed reading tasks, "
            "not identical experimental tasks."
        ),
        "stage5_reversed_auc": (
            "Reversed AUC above 0.5 for some models reflects "
            "cross-domain relationship reversal, not numeric label reversal."
        ),
    }

    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "FINAL RESULT: PASS - NO LABEL REVERSAL DETECTED"
    )
    print(
        "ZuCo: TSR=0, NR=1 | "
        "WebQAmGaze: IS=0, NR=1"
    )
    print(
        "The concern is excluded numerically. "
        "TSR-vs-IS remains a conceptual/domain-shift caveat, "
        "not a coding inversion."
    )
    print(f"Audit report: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as error:
        print()
        print("FINAL RESULT: FAIL")
        print(str(error))
        raise SystemExit(1)
