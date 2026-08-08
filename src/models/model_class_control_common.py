from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
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

PARTICIPANT_CANDIDATES = (
    "participant_id", "participant", "subject_id", "subject",
    "reader_id", "reader", "participant_code",
)
TRIAL_CANDIDATES = (
    "onestop_trial_id", "external_trial_id", "trial_id", "trial", "trial_key",
    "item_id", "paragraph_id", "unique_trial_id",
)
LABEL_CANDIDATES = (
    "label_nr", "label", "target", "y", "reading_goal_label",
    "task_label", "regime_label",
)
CONDITION_CANDIDATES = (
    "view_condition", "quality_condition", "condition", "view", "quality_view",
    "degradation", "degradation_condition",
)

CANONICAL_CONDITIONS = (
    "clean", "rate10", "noise50", "random40", "burst40",
    "high_random", "high_burst",
)

ALIASES = {
    "clean": {"clean", "identity", "original", "baseline", "none"},
    "rate10": {"rate10", "10hz", "downsample10", "downsample_10hz", "rate_10"},
    "noise50": {"noise50", "50px", "noise_50", "noise50px", "gaussian50"},
    "random40": {"random40", "random_40", "random_missing40", "missing_random40"},
    "burst40": {"burst40", "burst_40", "burst_missing40", "missing_burst40"},
    "high_random": {"high_random", "noise_random", "noise50_random40", "noise_plus_random", "combined_random", "noise+random"},
    "high_burst": {"high_burst", "noise_burst", "noise50_burst40", "noise_plus_burst", "combined_burst", "noise+burst"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table format: {path}")


def normalize_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def canonical_condition(value: Any) -> str:
    value = normalize_name(value)
    for canonical, aliases in ALIASES.items():
        if value in {normalize_name(x) for x in aliases}:
            return canonical
    return value


def detect_column(columns: Sequence[str], candidates: Sequence[str], role: str) -> str:
    mapping = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    raise RuntimeError(
        f"Could not detect {role} column. Candidates={list(candidates)}; "
        f"available={list(columns)[:120]}"
    )


def maybe_detect_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    mapping = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None


def normalize_binary_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        result = pd.to_numeric(series, errors="raise").astype(int)
        values = set(result.unique())
        if values != {0, 1}:
            raise RuntimeError(f"Expected binary labels {{0,1}}, found {values}")
        return result

    mapping = {
        "1": 1, "0": 0,
        "nr": 1, "natural_reading": 1, "ordinary": 1,
        "ordinary_reading": 1, "normal_reading": 1,
        "is": 0, "information_seeking": 0,
        "tsr": 0, "task_specific_reading": 0,
    }
    normalized = series.astype(str).map(normalize_name)
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        bad = sorted(normalized[mapped.isna()].unique())[:20]
        raise RuntimeError(f"Could not map labels: {bad}")
    return mapped.astype(int)


def _candidate_score(path: Path, require_condition: bool) -> tuple[int, dict[str, Any]]:
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, nrows=1000)
        else:
            frame = read_table(path)
    except Exception as exc:
        return -10000, {"error": repr(exc)}

    columns = [str(column) for column in frame.columns]
    n_soft = sum(column.startswith("soft_aoi_") for column in columns)
    participant = maybe_detect_column(columns, PARTICIPANT_CANDIDATES)
    label = maybe_detect_column(columns, LABEL_CANDIDATES)
    condition = maybe_detect_column(columns, CONDITION_CANDIDATES)
    score = n_soft * 10
    score += 50 if participant else 0
    score += 50 if label else 0
    score += 50 if condition else 0
    if require_condition and not condition:
        score -= 500

    lower_name = path.name.lower()
    normalized_parts = {part.lower() for part in path.parts}

    # The combined OneStop table is the intended 130,011-row source.
    # Per-participant cache files have the same schema and must not win a
    # lexicographic tie during automatic discovery.
    if lower_name == "onestop_quality_views_all_features.parquet":
        score += 2000
    if "participant_quality_views" in normalized_parts:
        score -= 2000

    if any(token in lower_name for token in ("quality", "view", "soft", "feature")):
        score += 20

    n_rows = int(len(frame))
    if n_rows >= 100000:
        score += 200
    elif n_rows < 1000:
        score -= 100

    return score, {
        "columns": columns,
        "soft_aoi_columns": n_soft,
        "participant_column": participant,
        "label_column": label,
        "condition_column": condition,
        "n_rows": n_rows,
    }


def resolve_table(path_or_dir: str, project_root: Path, require_condition: bool) -> tuple[Path, pd.DataFrame]:
    raw = Path(path_or_dir)
    path = raw if raw.is_absolute() else project_root / raw
    path = path.resolve()
    audit_rows: list[dict[str, Any]] = []

    if path.is_file():
        score, audit = _candidate_score(path, require_condition)
        audit_rows.append({"candidate": str(path), "score": score, **audit})
        if score < 0:
            raise RuntimeError(f"Could not read input table: {path}")
        if require_condition and audit.get("condition_column") is None:
            raise RuntimeError(
                f"Source table lacks a condition column: {path}"
            )
        return path, pd.DataFrame(audit_rows)

    if not path.exists():
        raise FileNotFoundError(path)

    # Prefer the documented combined OneStop quality-view table before any
    # recursive search. This prevents selecting a single participant cache.
    preferred = path / "onestop_quality_views_all_features.parquet"
    if require_condition and preferred.is_file():
        score, preferred_audit = _candidate_score(preferred, require_condition)
        audit_rows.append(
            {"candidate": str(preferred), "score": score, **preferred_audit}
        )
        if preferred_audit.get("condition_column") is None:
            raise RuntimeError(
                f"Preferred source lacks a condition column: {preferred}"
            )
        return preferred, pd.DataFrame(audit_rows)

    candidates: list[Path] = []
    for pattern in ("*.parquet", "*.csv", "*.pkl", "*.pickle"):
        candidates.extend(path.rglob(pattern))

    ranked: list[tuple[int, Path]] = []
    for candidate in sorted(set(candidates)):
        score, audit = _candidate_score(candidate, require_condition)
        ranked.append((score, candidate))
        audit_rows.append({"candidate": str(candidate), "score": score, **audit})

    ranked.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    if not ranked or ranked[0][0] < 100:
        raise RuntimeError(f"No suitable table found under {path}")
    return ranked[0][1], pd.DataFrame(audit_rows)


def _find_feature_list(obj: Any) -> list[str] | None:
    if isinstance(obj, dict):
        for key in ("feature_columns", "feature_names", "features", "input_features", "selected_features"):
            if key in obj and isinstance(obj[key], (list, tuple)):
                values = [str(value) for value in obj[key]]
                if len(values) == 41 and all(value.startswith("soft_aoi_") for value in values):
                    return values
        for value in obj.values():
            found = _find_feature_list(value)
            if found is not None:
                return found
    if hasattr(obj, "feature_names_in_"):
        values = [str(value) for value in obj.feature_names_in_]
        if len(values) == 41 and all(value.startswith("soft_aoi_") for value in values):
            return values
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_feature_list(value)
            if found is not None:
                return found
    return None


def resolve_feature_columns(frame: pd.DataFrame, frozen_model: Path) -> list[str]:
    if frozen_model.exists():
        artifact = joblib.load(frozen_model)
        found = _find_feature_list(artifact)
        if found is not None:
            missing = [column for column in found if column not in frame.columns]
            if missing:
                raise RuntimeError(f"Source table is missing frozen features: {missing}")
            return found
    columns = [str(column) for column in frame.columns if str(column).startswith("soft_aoi_")]
    if len(columns) != 41:
        raise RuntimeError(
            f"Expected exactly 41 soft_aoi_ columns, found {len(columns)}. "
            "Provide the frozen OneStop Full-41 joblib artifact."
        )
    return columns


def prepare_source(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    participant = detect_column(frame.columns, PARTICIPANT_CANDIDATES, "source participant")
    trial = maybe_detect_column(frame.columns, TRIAL_CANDIDATES) or ""
    label = detect_column(frame.columns, LABEL_CANDIDATES, "source label")
    condition = detect_column(frame.columns, CONDITION_CANDIDATES, "source condition")

    work = frame.copy()
    work[participant] = work[participant].astype(str)
    work[label] = normalize_binary_label(work[label])
    work[condition] = work[condition].map(canonical_condition)
    for column in feature_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    observed = set(work[condition].unique())
    missing = [name for name in CANONICAL_CONDITIONS if name not in observed]
    if missing:
        raise RuntimeError(f"Missing quality conditions {missing}; observed={sorted(observed)}")
    return work, {"participant": participant, "trial": trial, "label": label, "condition": condition}


def prepare_target(
    features: pd.DataFrame,
    labels: pd.DataFrame | None,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    participant = detect_column(features.columns, PARTICIPANT_CANDIDATES, "target participant")
    trial = detect_column(features.columns, TRIAL_CANDIDATES, "target trial")
    work = features.copy()

    if labels is not None:
        lp = detect_column(labels.columns, PARTICIPANT_CANDIDATES, "label participant")
        lt = detect_column(labels.columns, TRIAL_CANDIDATES, "label trial")
        ll = detect_column(labels.columns, LABEL_CANDIDATES, "target label")
        label_frame = labels.rename(columns={lp: participant, lt: trial, ll: "__label_nr__"})
        work = work.merge(
            label_frame[[participant, trial, "__label_nr__"]],
            on=[participant, trial],
            how="inner",
            validate="one_to_one",
        )
        label = "__label_nr__"
    else:
        label = detect_column(work.columns, LABEL_CANDIDATES, "target label")

    work[participant] = work[participant].astype(str)
    work[label] = normalize_binary_label(work[label])
    for column in feature_columns:
        if column not in work.columns:
            raise RuntimeError(f"Target is missing feature {column}")
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work, {"participant": participant, "trial": trial, "label": label}


def pr_auc_trapezoid(y_true: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, score)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def optimize_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(
        np.concatenate([
            np.quantile(score, np.linspace(0.01, 0.99, 401)),
            np.asarray([0.0, float(np.median(score))]),
        ])
    )
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in candidates:
        prediction = (score >= threshold).astype(int)
        value = f1_score(y_true, prediction, average="macro", zero_division=0)
        if value > best_f1 + 1e-12:
            best_threshold = float(threshold)
            best_f1 = float(value)
        elif abs(value - best_f1) <= 1e-12 and abs(threshold) < abs(best_threshold):
            best_threshold = float(threshold)
    return best_threshold, best_f1


def metric_dict(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    participants: np.ndarray,
) -> dict[str, Any]:
    prediction = (score >= threshold).astype(int)
    return {
        "n_trials": int(len(y_true)),
        "n_participants": int(pd.unique(participants).size),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc_nr": pr_auc_trapezoid(y_true, score),
        "average_precision_nr": float(average_precision_score(y_true, score)),
        "nr_prediction_rate": float(np.mean(prediction)),
    }


def participant_median_loo(score: np.ndarray, participants: np.ndarray, source_reference: float) -> np.ndarray:
    result = np.empty_like(score, dtype=float)
    frame = pd.DataFrame({"participant": participants.astype(str), "score": score, "row": np.arange(len(score))})
    for _, group in frame.groupby("participant", sort=False):
        rows = group["row"].to_numpy(int)
        values = group["score"].to_numpy(float)
        if len(values) < 2:
            result[rows] = values
            continue
        for local_index, row in enumerate(rows):
            center = float(np.median(np.delete(values, local_index)))
            result[row] = values[local_index] - center + source_reference
    return result


def write_hash_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "file_hash_manifest.csv":
            rows.append({
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    pd.DataFrame(rows).to_csv(output_dir / "file_hash_manifest.csv", index=False, encoding="utf-8-sig")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
