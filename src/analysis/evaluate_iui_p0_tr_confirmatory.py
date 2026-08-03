from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


METHOD_COLUMNS = {
    "frozen_full41": "frozen_full41_score",
    "participant_median_loo": "participant_median_loo_score",
    "qcn_only": "qcn_only_score",
    "qcn_pc": "qcn_pc_score",
}

PRIMARY_METHOD = "qcn_pc"
PRIMARY_REFERENCE = "participant_median_loo"
COVERAGES = [1.0, 0.5, 0.2]


@dataclass
class MetricRow:
    method: str
    n_trials: int
    n_participants: int
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc_nr: float
    average_precision_nr: float
    nr_prediction_rate: float
    mean_within_participant_auc: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pr_auc_trapezoid(y_true: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, score)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def mean_within_participant_auc(
    y_true: np.ndarray,
    score: np.ndarray,
    participant_ids: np.ndarray,
) -> float:
    aucs: list[float] = []
    for participant_id in pd.unique(participant_ids):
        mask = participant_ids == participant_id
        if np.unique(y_true[mask]).size == 2:
            aucs.append(float(roc_auc_score(y_true[mask], score[mask])))
    return float(np.mean(aucs)) if aucs else float("nan")


def calculate_metrics(
    method: str,
    y_true: np.ndarray,
    score: np.ndarray,
    participant_ids: np.ndarray,
    threshold: float,
) -> MetricRow:
    prediction = (score >= threshold).astype(int)
    return MetricRow(
        method=method,
        n_trials=int(len(y_true)),
        n_participants=int(pd.unique(participant_ids).size),
        accuracy=float(accuracy_score(y_true, prediction)),
        macro_f1=float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(y_true, prediction)
        ),
        roc_auc=safe_auc(y_true, score),
        pr_auc_nr=pr_auc_trapezoid(y_true, score),
        average_precision_nr=float(
            average_precision_score(y_true, score)
        ),
        nr_prediction_rate=float(np.mean(prediction)),
        mean_within_participant_auc=(
            mean_within_participant_auc(
                y_true,
                score,
                participant_ids,
            )
        ),
    )


def join_frozen_artifacts(
    snapshot: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_path = (
        snapshot
        / "predictions"
        / "unlabeled_frozen_scores.csv"
    )
    label_path = snapshot / "sealed" / "labels_sealed.csv"
    manifest_path = (
        snapshot
        / "manifests"
        / "unlabeled_target_manifest.csv"
    )
    release_path = snapshot / "release_manifest.json"

    for path in [
        prediction_path,
        label_path,
        manifest_path,
        release_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(prediction_path)
    labels = pd.read_csv(label_path)
    manifest = pd.read_csv(manifest_path)
    release = json.loads(
        release_path.read_text(encoding="utf-8-sig")
    )

    keys = ["participant_id", "external_trial_id"]
    for name, frame in [
        ("predictions", predictions),
        ("labels", labels),
        ("manifest", manifest),
    ]:
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise RuntimeError(
                f"{name} is missing key columns: {missing}"
            )
        if frame.duplicated(keys).any():
            raise RuntimeError(
                f"{name} contains duplicate participant/trial keys."
            )

    missing_score_columns = [
        column
        for column in METHOD_COLUMNS.values()
        if column not in predictions.columns
    ]
    if missing_score_columns:
        raise RuntimeError(
            f"Prediction artifact is missing scores: "
            f"{missing_score_columns}"
        )
    if "label_nr" not in labels.columns:
        raise RuntimeError("Sealed label file lacks label_nr.")

    joined = predictions.merge(
        labels,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    joined = manifest.merge(
        joined,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    expected_rows = int(release["eligible_trials"])
    expected_participants = int(release["eligible_participants"])

    if len(joined) != expected_rows:
        raise RuntimeError(
            f"Joined row count mismatch: "
            f"expected={expected_rows}, observed={len(joined)}"
        )
    observed_participants = joined["participant_id"].nunique()
    if observed_participants != expected_participants:
        raise RuntimeError(
            "Joined participant count mismatch: "
            f"expected={expected_participants}, "
            f"observed={observed_participants}"
        )

    labels_unique = set(
        pd.to_numeric(joined["label_nr"]).astype(int).unique()
    )
    if labels_unique != {0, 1}:
        raise RuntimeError(
            f"Expected labels {{0,1}}, found {labels_unique}"
        )

    return joined, release


def participant_cluster_bootstrap(
    y_true: np.ndarray,
    participant_ids: np.ndarray,
    scores: dict[str, np.ndarray],
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unique_participants = pd.unique(participant_ids)
    participant_indices = {
        participant_id: np.flatnonzero(
            participant_ids == participant_id
        )
        for participant_id in unique_participants
    }

    replicate_rows: list[dict[str, Any]] = []
    metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc_nr",
        "average_precision_nr",
    ]

    for replicate in range(n_bootstrap):
        sampled_participants = rng.choice(
            unique_participants,
            size=len(unique_participants),
            replace=True,
        )
        indices = np.concatenate(
            [
                participant_indices[participant_id]
                for participant_id in sampled_participants
            ]
        )
        y_boot = y_true[indices]
        if np.unique(y_boot).size < 2:
            continue

        for method, score in scores.items():
            metric = calculate_metrics(
                method,
                y_boot,
                score[indices],
                np.asarray(
                    [
                        f"bootstrap_cluster_{cluster_index}"
                        for cluster_index, participant_id
                        in enumerate(sampled_participants)
                        for _ in participant_indices[participant_id]
                    ],
                    dtype=str,
                ),
                threshold,
            )
            row = {
                "replicate": replicate,
                **asdict(metric),
            }
            replicate_rows.append(row)

    replicates = pd.DataFrame(replicate_rows)

    comparisons = [
        (PRIMARY_METHOD, PRIMARY_REFERENCE),
        (PRIMARY_METHOD, "frozen_full41"),
        (PRIMARY_METHOD, "qcn_only"),
    ]
    delta_rows: list[dict[str, Any]] = []

    for method, reference in comparisons:
        method_rows = (
            replicates[replicates["method"] == method]
            .set_index("replicate")
        )
        reference_rows = (
            replicates[replicates["method"] == reference]
            .set_index("replicate")
        )
        common = method_rows.index.intersection(
            reference_rows.index
        )

        for metric_name in metric_names:
            delta = (
                method_rows.loc[common, metric_name]
                - reference_rows.loc[common, metric_name]
            ).to_numpy(float)
            delta_rows.append(
                {
                    "method": method,
                    "reference": reference,
                    "metric": metric_name,
                    "mean_difference": float(np.mean(delta)),
                    "median_difference": float(
                        np.median(delta)
                    ),
                    "ci_low": float(
                        np.quantile(delta, 0.025)
                    ),
                    "ci_high": float(
                        np.quantile(delta, 0.975)
                    ),
                    "n_replicates": int(len(delta)),
                }
            )

    return replicates, pd.DataFrame(delta_rows)


def within_participant_permutation(
    y_true: np.ndarray,
    participant_ids: np.ndarray,
    qcn_pc_score: np.ndarray,
    participant_median_score: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    unique_participants = pd.unique(participant_ids)
    participant_indices = [
        np.flatnonzero(participant_ids == participant_id)
        for participant_id in unique_participants
    ]

    observed_qcn_auc = float(
        roc_auc_score(y_true, qcn_pc_score)
    )
    observed_reference_auc = float(
        roc_auc_score(y_true, participant_median_score)
    )
    observed_delta = (
        observed_qcn_auc - observed_reference_auc
    )

    rows: list[dict[str, Any]] = []
    null_qcn_auc = np.empty(
        n_permutations,
        dtype=float,
    )
    null_delta = np.empty(
        n_permutations,
        dtype=float,
    )

    for permutation in range(n_permutations):
        permuted = y_true.copy()
        for indices in participant_indices:
            permuted[indices] = rng.permutation(
                permuted[indices]
            )

        qcn_auc = float(
            roc_auc_score(permuted, qcn_pc_score)
        )
        reference_auc = float(
            roc_auc_score(
                permuted,
                participant_median_score,
            )
        )
        delta = qcn_auc - reference_auc

        null_qcn_auc[permutation] = qcn_auc
        null_delta[permutation] = delta
        rows.append(
            {
                "permutation": permutation,
                "qcn_pc_roc_auc": qcn_auc,
                "participant_median_loo_roc_auc": (
                    reference_auc
                ),
                "auc_delta": delta,
            }
        )

    p_qcn = float(
        (
            1
            + np.sum(
                null_qcn_auc >= observed_qcn_auc
            )
        )
        / (n_permutations + 1)
    )
    p_delta = float(
        (
            1
            + np.sum(null_delta >= observed_delta)
        )
        / (n_permutations + 1)
    )

    summary = {
        "n_permutations": n_permutations,
        "seed": seed,
        "scheme": (
            "shuffle labels independently within each "
            "participant; preserve participant-level "
            "class counts"
        ),
        "observed_qcn_pc_roc_auc": observed_qcn_auc,
        "observed_participant_median_loo_roc_auc": (
            observed_reference_auc
        ),
        "observed_auc_delta": observed_delta,
        "one_sided_p_qcn_pc_association": p_qcn,
        "one_sided_p_auc_delta": p_delta,
        "null_qcn_pc_auc_mean": float(
            np.mean(null_qcn_auc)
        ),
        "null_qcn_pc_auc_q95": float(
            np.quantile(null_qcn_auc, 0.95)
        ),
        "null_auc_delta_mean": float(
            np.mean(null_delta)
        ),
        "null_auc_delta_q95": float(
            np.quantile(null_delta, 0.95)
        ),
    }
    return pd.DataFrame(rows), summary


def deterministic_selection(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    coverage: float,
    mode: str,
) -> pd.DataFrame:
    work = frame.copy()
    work["_confidence"] = np.abs(
        work[score_column].to_numpy(float)
        - threshold
    )

    if mode == "global":
        ordered = work.sort_values(
            [
                "_confidence",
                "participant_id",
                "external_trial_id",
            ],
            ascending=[False, True, True],
            kind="mergesort",
        )
        keep = max(
            1,
            int(math.ceil(len(ordered) * coverage)),
        )
        return ordered.iloc[:keep].copy()

    if mode == "participant_balanced":
        retained: list[pd.DataFrame] = []
        for _, participant_frame in work.groupby(
            "participant_id",
            sort=True,
        ):
            ordered = participant_frame.sort_values(
                ["_confidence", "external_trial_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            keep = max(
                1,
                int(
                    math.ceil(
                        len(ordered) * coverage
                    )
                ),
            )
            retained.append(ordered.iloc[:keep])
        return pd.concat(
            retained,
            ignore_index=True,
        )

    raise ValueError(f"Unknown selection mode: {mode}")


def selective_metrics(
    frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for method, score_column in METHOD_COLUMNS.items():
        for mode in [
            "global",
            "participant_balanced",
        ]:
            for requested_coverage in COVERAGES:
                selected = deterministic_selection(
                    frame,
                    score_column,
                    threshold,
                    requested_coverage,
                    mode,
                )
                y_true = (
                    selected["label_nr"]
                    .astype(int)
                    .to_numpy()
                )
                scores = selected[
                    score_column
                ].to_numpy(float)
                participant_ids = (
                    selected["participant_id"]
                    .astype(str)
                    .to_numpy()
                )
                metric = calculate_metrics(
                    method,
                    y_true,
                    scores,
                    participant_ids,
                    threshold,
                )
                rows.append(
                    {
                        "method": method,
                        "selection_mode": mode,
                        "requested_coverage": (
                            requested_coverage
                        ),
                        "realized_coverage": (
                            len(selected) / len(frame)
                        ),
                        **{
                            key: value
                            for key, value in asdict(
                                metric
                            ).items()
                            if key != "method"
                        },
                    }
                )

    return pd.DataFrame(rows)


def build_confirmation_decision(
    point_metrics: pd.DataFrame,
    bootstrap_deltas: pd.DataFrame,
    permutation_summary: dict[str, Any],
) -> dict[str, Any]:
    metrics = point_metrics.set_index("method")
    qcn = metrics.loc[PRIMARY_METHOD]
    reference = metrics.loc[PRIMARY_REFERENCE]

    primary_auc_delta = float(
        qcn["roc_auc"] - reference["roc_auc"]
    )
    primary_f1_delta = float(
        qcn["macro_f1"] - reference["macro_f1"]
    )

    primary_bootstrap = bootstrap_deltas[
        (bootstrap_deltas["method"] == PRIMARY_METHOD)
        & (
            bootstrap_deltas["reference"]
            == PRIMARY_REFERENCE
        )
        & (bootstrap_deltas["metric"] == "roc_auc")
    ]
    if len(primary_bootstrap) != 1:
        raise RuntimeError(
            "Primary bootstrap AUC comparison was not found."
        )

    auc_ci_low = float(
        primary_bootstrap.iloc[0]["ci_low"]
    )
    auc_ci_high = float(
        primary_bootstrap.iloc[0]["ci_high"]
    )
    permutation_p = float(
        permutation_summary[
            "one_sided_p_qcn_pc_association"
        ]
    )

    strong_conditions = {
        "auc_delta_positive": primary_auc_delta > 0,
        "auc_ci_excludes_zero": auc_ci_low > 0,
        "macro_f1_difference_nonnegative": (
            primary_f1_delta >= 0
        ),
        "qcn_pc_auc_above_half": (
            float(qcn["roc_auc"]) > 0.5
        ),
        "permutation_p_below_0_05": (
            permutation_p < 0.05
        ),
    }

    if all(strong_conditions.values()):
        classification = "STRONG_CONFIRMATION"
    elif (
        primary_auc_delta > 0
        and float(qcn["roc_auc"]) > 0.5
        and permutation_p < 0.05
    ):
        classification = "PARTIAL_CONFIRMATION"
    else:
        classification = "NON_CONFIRMATION"

    return {
        "classification": classification,
        "primary_method": PRIMARY_METHOD,
        "primary_reference": PRIMARY_REFERENCE,
        "primary_estimand": (
            "pooled ROC-AUC difference: "
            "QCN-PC minus participant median LOO"
        ),
        "qcn_pc_roc_auc": float(qcn["roc_auc"]),
        "participant_median_loo_roc_auc": float(
            reference["roc_auc"]
        ),
        "primary_auc_delta": primary_auc_delta,
        "primary_auc_delta_ci_low": auc_ci_low,
        "primary_auc_delta_ci_high": auc_ci_high,
        "qcn_pc_macro_f1": float(qcn["macro_f1"]),
        "participant_median_loo_macro_f1": float(
            reference["macro_f1"]
        ),
        "primary_macro_f1_delta": primary_f1_delta,
        "qcn_pc_permutation_p": permutation_p,
        "strong_confirmation_conditions": (
            strong_conditions
        ),
        "partial_rule": (
            "positive point AUC delta and significant "
            "within-participant association, but one or "
            "more strong-confirmation conditions fail"
        ),
        "non_confirmation_rule": (
            "nonpositive point AUC delta or nonsignificant "
            "within-participant association"
        ),
    }


def write_output_hashes(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        output_dir.rglob("*"),
        key=lambda item: str(item).lower(),
    ):
        if not path.is_file():
            continue
        if path.name == "file_hash_manifest.csv":
            continue
        rows.append(
            {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "relative_path": (
                    path.relative_to(output_dir).as_posix()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "file_hash_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open the independently frozen Turkish labels "
            "and run the prespecified confirmatory evaluation."
        )
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Independent verified snapshot directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "iui_p0_confirmatory_TR_evaluation_v1"
        ),
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--permutations",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260803,
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    snapshot = Path(args.snapshot).resolve()
    output_dir = (
        project_root / args.output_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, frozen_release = join_frozen_artifacts(
        snapshot
    )
    threshold = float(
        frozen_release["frozen_threshold"]
    )

    y_true = (
        frame["label_nr"].astype(int).to_numpy()
    )
    participant_ids = (
        frame["participant_id"]
        .astype(str)
        .to_numpy()
    )
    scores = {
        method: frame[column].to_numpy(float)
        for method, column in METHOD_COLUMNS.items()
    }

    point_rows = [
        calculate_metrics(
            method,
            y_true,
            score,
            participant_ids,
            threshold,
        )
        for method, score in scores.items()
    ]
    point_metrics = pd.DataFrame(
        [asdict(row) for row in point_rows]
    )
    point_metrics.to_csv(
        output_dir / "confirmatory_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap_replicates, bootstrap_deltas = (
        participant_cluster_bootstrap(
            y_true=y_true,
            participant_ids=participant_ids,
            scores=scores,
            threshold=threshold,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
    )
    bootstrap_replicates.to_parquet(
        output_dir
        / "confirmatory_bootstrap_replicates.parquet",
        index=False,
    )
    bootstrap_deltas.to_csv(
        output_dir / "confirmatory_bootstrap_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    permutation_rows, permutation_summary = (
        within_participant_permutation(
            y_true=y_true,
            participant_ids=participant_ids,
            qcn_pc_score=scores["qcn_pc"],
            participant_median_score=scores[
                "participant_median_loo"
            ],
            n_permutations=args.permutations,
            seed=args.seed,
        )
    )
    permutation_rows.to_parquet(
        output_dir
        / "confirmatory_permutation_replicates.parquet",
        index=False,
    )
    (
        output_dir
        / "confirmatory_permutation_summary.json"
    ).write_text(
        json.dumps(
            permutation_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selective = selective_metrics(
        frame,
        threshold,
    )
    selective.to_csv(
        output_dir / "selective_prediction_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joined_output = frame[
        [
            "participant_id",
            "external_trial_id",
            "label_nr",
            *METHOD_COLUMNS.values(),
        ]
    ].copy()
    if "task" in frame.columns:
        joined_output["task"] = frame["task"]
    joined_output.to_parquet(
        output_dir
        / "confirmatory_joined_trial_predictions.parquet",
        index=False,
    )

    decision = build_confirmation_decision(
        point_metrics,
        bootstrap_deltas,
        permutation_summary,
    )
    (
        output_dir / "confirmation_decision.json"
    ).write_text(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    release = {
        "artifact_version": (
            "iui_p0_confirmatory_TR_evaluation_v1"
        ),
        "snapshot": str(snapshot),
        "snapshot_release_sha256": sha256(
            snapshot / "release_manifest.json"
        ),
        "language": "TR",
        "n_trials": int(len(frame)),
        "n_participants": int(
            frame["participant_id"].nunique()
        ),
        "threshold": threshold,
        "methods": METHOD_COLUMNS,
        "primary_method": PRIMARY_METHOD,
        "primary_reference": PRIMARY_REFERENCE,
        "bootstrap_replicates": args.bootstrap,
        "permutation_replicates": args.permutations,
        "seed": args.seed,
        "selective_coverages": COVERAGES,
        "confirmation_classification": (
            decision["classification"]
        ),
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(
            release,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    qcn_row = point_metrics[
        point_metrics["method"] == "qcn_pc"
    ].iloc[0]
    reference_row = point_metrics[
        point_metrics["method"]
        == "participant_median_loo"
    ].iloc[0]

    primary_bootstrap_rows = bootstrap_deltas[
        (
            bootstrap_deltas["method"]
            == PRIMARY_METHOD
        )
        & (
            bootstrap_deltas["reference"]
            == PRIMARY_REFERENCE
        )
    ]

    selective_qcn = selective[
        selective["method"] == "qcn_pc"
    ]

    summary = f"""# Turkish Confirmatory Evaluation

## Decision

**{decision['classification']}**

## Primary comparison

- QCN-PC ROC-AUC: `{qcn_row['roc_auc']:.6f}`
- Participant median LOO ROC-AUC: `{reference_row['roc_auc']:.6f}`
- ROC-AUC difference: `{decision['primary_auc_delta']:.6f}`
- 95% participant-cluster bootstrap CI:
  `[{decision['primary_auc_delta_ci_low']:.6f}, {decision['primary_auc_delta_ci_high']:.6f}]`
- QCN-PC Macro F1: `{qcn_row['macro_f1']:.6f}`
- Participant median LOO Macro F1: `{reference_row['macro_f1']:.6f}`
- Macro F1 difference: `{decision['primary_macro_f1_delta']:.6f}`
- Within-participant permutation p-value:
  `{decision['qcn_pc_permutation_p']:.6f}`

## Pooled metrics

{point_metrics.to_markdown(index=False)}

## Bootstrap differences

{primary_bootstrap_rows.to_markdown(index=False)}

## QCN-PC selective prediction

{selective_qcn.to_markdown(index=False)}

## Interpretation rule

Strong confirmation requires a positive primary ROC-AUC difference, a 95% bootstrap
interval excluding zero, a nonnegative Macro F1 difference, QCN-PC ROC-AUC above
0.5, and a one-sided within-participant permutation p-value below 0.05.
"""
    (output_dir / "RESULTS_SUMMARY.md").write_text(
        summary,
        encoding="utf-8",
    )

    write_output_hashes(output_dir)

    print("")
    print("Turkish confirmatory evaluation completed.")
    print(f"Output: {output_dir}")
    print(
        f"Decision: {decision['classification']}"
    )
    print(
        f"QCN-PC ROC-AUC: "
        f"{qcn_row['roc_auc']:.6f}"
    )
    print(
        f"Participant median LOO ROC-AUC: "
        f"{reference_row['roc_auc']:.6f}"
    )
    print(
        f"AUC difference: "
        f"{decision['primary_auc_delta']:.6f}"
    )
    print(
        "AUC 95% CI: "
        f"[{decision['primary_auc_delta_ci_low']:.6f}, "
        f"{decision['primary_auc_delta_ci_high']:.6f}]"
    )
    print(
        f"Permutation p: "
        f"{decision['qcn_pc_permutation_p']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
