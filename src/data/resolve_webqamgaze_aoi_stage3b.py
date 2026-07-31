#!/usr/bin/env python
"""Resolve WebQAmGaze word-AOI and participant-fixation resources (Stage 3B).

Why this stage exists
---------------------
Stage 3A found 41 parsable word-boundary candidates and 575 files under
``fixation_data_per_part``, but the 101 unique primary trial names did not occur
in the resource *paths*. Therefore mapping must use file contents, nested object
keys, and table columns rather than filenames.

This script performs that content-based resolution. It does NOT fit or apply any
classifier and does NOT inspect external-validation performance.

Main responsibilities
---------------------
1. Parse candidate resources in:
   - scripts/target_dataframes
   - scripts/js_trials
   - scripts/MECO_texts
   - scripts/pre_processed_data/fixation_data_per_part
   - scripts/combined_text_features.csv
   - scripts/text_token_stats.csv
2. Discover nested DataFrames / record tables in CSV, JSON, pickle, NPY, and
   JSON-like text.
3. Match table rows to the 101 primary trial names using explicit fields,
   nested dictionary keys, and conservative name canonicalization.
4. Normalize rectangle representations such as:
   - x_min, y_min, x_max, y_max
   - left, top, right, bottom
   - x, y, width, height
   - list/dict bounding-box cells
5. Rank candidate AOI sources per trial without using NR/IS labels.
6. Create a candidate canonical word-AOI table, deriving reading order only when
   explicit word/line ordering is unavailable.
7. Map participant-specific fixation files to the 131 primary participants and
   inventory their actual columns.
8. Extract relevant mapping/loading lines from the official notebook excerpt
   created in Stage 3A.

Important
---------
The output named ``canonical_word_aoi_candidates.csv`` is still an audited
candidate table. It becomes the final Soft-AOI template only after the console
audit confirms complete and unambiguous trial coverage and a defensible
coordinate-system mapping.

No model predictions are produced in this stage.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


TRIAL_COLUMN_ALIASES = {
    "trial", "trialname", "trialid", "trial_name", "trial_id",
    "text", "textname", "textid", "text_name", "text_id",
    "passage", "passageid", "passage_id",
    "paragraph", "paragraphid", "paragraph_id",
    "stimulus", "stimulusid", "stimulus_id",
    "image", "imagename", "imageid", "image_name", "image_id",
    "item", "itemid", "item_id", "name",
}
SET_COLUMN_ALIASES = {
    "set", "setname", "setid", "set_name", "set_id",
    "experimentset", "experiment_set",
}
PARTICIPANT_COLUMN_ALIASES = {
    "participant", "participantid", "participant_id",
    "subject", "subjectid", "subject_id",
    "worker", "workerid", "worker_id",
    "uniqueid", "unique_id",
}
WORD_COLUMN_ALIASES = {
    "word", "token", "wordtext", "word_text", "tokentext", "token_text",
    "surface", "string", "spantext", "span_text",
}
WORD_INDEX_ALIASES = {
    "wordindex", "word_index", "tokenindex", "token_index",
    "wordid", "word_id", "tokenid", "token_id", "index", "idx",
}
LINE_INDEX_ALIASES = {
    "lineindex", "line_index", "lineid", "line_id", "line", "row",
}
FIXATION_X_ALIASES = {
    "x", "fixationx", "fixation_x", "gazex", "gaze_x",
    "xpos", "x_pos", "xposition", "x_position",
}
FIXATION_Y_ALIASES = {
    "y", "fixationy", "fixation_y", "gazey", "gaze_y",
    "ypos", "y_pos", "yposition", "y_position",
}
FIXATION_T_ALIASES = {
    "t", "time", "timestamp", "timestampms", "timestamp_ms",
    "starttime", "start_time", "fixationstart", "fixation_start",
}
FIXATION_DURATION_ALIASES = {
    "duration", "durationms", "duration_ms", "fixationduration",
    "fixation_duration", "dur", "d",
}

XMIN_ALIASES = {
    "xmin", "x_min", "left", "xleft", "x_left", "x0", "xstart", "x_start",
    "bboxleft", "bbox_left", "boxleft", "box_left",
}
YMIN_ALIASES = {
    "ymin", "y_min", "top", "ytop", "y_top", "y0", "ystart", "y_start",
    "bboxtop", "bbox_top", "boxtop", "box_top",
}
XMAX_ALIASES = {
    "xmax", "x_max", "right", "xright", "x_right", "x1", "xend", "x_end",
    "bboxright", "bbox_right", "boxright", "box_right",
}
YMAX_ALIASES = {
    "ymax", "y_max", "bottom", "ybottom", "y_bottom", "y1", "yend", "y_end",
    "bboxbottom", "bbox_bottom", "boxbottom", "box_bottom",
}
X_ALIASES = {"x", "left", "xpos", "x_pos", "xposition", "x_position"}
Y_ALIASES = {"y", "top", "ypos", "y_pos", "yposition", "y_position"}
WIDTH_ALIASES = {
    "width", "w", "boxwidth", "box_width", "bboxwidth", "bbox_width",
}
HEIGHT_ALIASES = {
    "height", "h", "boxheight", "box_height", "bboxheight", "bbox_height",
}
BBOX_ALIASES = {
    "bbox", "boundingbox", "bounding_box", "bounds", "rectangle", "rect", "box",
}
SCREEN_WIDTH_ALIASES = {
    "screenx", "screen_x", "screenwidth", "screen_width",
    "viewportwidth", "viewport_width", "windowwidth", "window_width",
    "resolutionx", "resolution_x",
}
SCREEN_HEIGHT_ALIASES = {
    "screeny", "screen_y", "screenheight", "screen_height",
    "viewportheight", "viewport_height", "windowheight", "window_height",
    "resolutiony", "resolution_y",
}
IMAGE_WIDTH_ALIASES = {
    "imagewidth", "image_width", "imgwidth", "img_width",
    "stimuluswidth", "stimulus_width", "naturalwidth", "natural_width",
}
IMAGE_HEIGHT_ALIASES = {
    "imageheight", "image_height", "imgheight", "img_height",
    "stimulusheight", "stimulus_height", "naturalheight", "natural_height",
}

SUPPORTED_SUFFIXES = {
    ".csv", ".tsv", ".json", ".jsonl", ".pkl", ".pickle", ".npy", ".npz",
    ".txt", ".js",
}

RESOURCE_ROOT_NAMES = {
    "target_dataframes",
    "js_trials",
    "MECO_texts",
    "fixation_data_per_part",
}

TEXT_HINT_PATTERN = re.compile(
    r"word|token|bound|bbox|box|rect|target|trial|text|passage|paragraph|"
    r"fixation|gaze|screen|resolution|image",
    re.IGNORECASE,
)
NOTEBOOK_HINT_PATTERN = re.compile(
    r"target_dataframes|fixation_data_per_part|word|bound|bbox|"
    r"read_csv|read_json|read_pickle|json.loads|trial|set_name",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def compact(value: Any, limit: int = 400) -> str:
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.replace("\r", " ").replace("\n", " ")).strip()
    else:
        try:
            text = repr(value)
        except Exception:  # noqa: BLE001
            text = f"<{type(value).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def safe_json_or_literal(value: str, max_chars: int = 20_000_000) -> Any | None:
    text = value.strip()
    if not text or len(text) > max_chars:
        return None
    if text[0] not in "[{(" or text[-1] not in "]})":
        return None
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list, tuple)):
            return parsed
    except Exception:  # noqa: BLE001
        pass
    return None


def canonical_trial_candidate(value: Any) -> str:
    """Remove wrappers/extensions while preserving the WebQAmGaze trial core."""
    text = str(value).strip()
    if not text:
        return ""
    text = Path(text.replace("\\", "/")).name
    text = re.sub(r"\.(?:png|jpg|jpeg|gif|bmp|webp|html|js|json|csv)$", "", text, flags=re.I)
    text = re.sub(r"^(?:q_before_|q_after_|before_|after_)", "", text, flags=re.I)
    text = re.sub(r"_qa_\d+$", "", text, flags=re.I)
    return text


def build_trial_lookup(trial_names: Sequence[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    collisions: dict[str, set[str]] = defaultdict(set)
    for name in trial_names:
        variants = {
            name,
            canonical_trial_candidate(name),
            normalize_name(name),
            normalize_name(canonical_trial_candidate(name)),
        }
        for variant in variants:
            key = normalize_name(variant)
            if key:
                collisions[key].add(name)
    for key, values in collisions.items():
        if len(values) == 1:
            lookup[key] = next(iter(values))
    return lookup


def resolve_trial_name(value: Any, lookup: dict[str, str]) -> str:
    candidate = canonical_trial_candidate(value)
    return lookup.get(normalize_name(candidate), "")


def column_by_alias(columns: Iterable[Any], aliases: set[str]) -> str:
    mapping = {normalize_name(column): str(column) for column in columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in mapping:
            return mapping[key]
    return ""


def load_file_object(path: Path) -> tuple[Any | None, str, str]:
    suffix = path.suffix.lower()
    errors: list[str] = []
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False), "csv", ""
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False), "tsv", ""
        if suffix == ".json":
            text = path.read_text(encoding="utf-8-sig")
            return json.loads(text), "json", ""
        if suffix == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records, "jsonl", ""
        if suffix in {".pkl", ".pickle"}:
            return pd.read_pickle(path), "pickle", ""
        if suffix == ".npy":
            value = np.load(path, allow_pickle=True)
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            return value, "npy", ""
        if suffix == ".npz":
            value = np.load(path, allow_pickle=True)
            return {key: value[key] for key in value.files}, "npz", ""
        if suffix in {".txt", ".js"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            parsed = safe_json_or_literal(text)
            return (parsed if parsed is not None else text), suffix[1:], ""
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
    return None, suffix[1:] or "unknown", " || ".join(errors)


def dataframe_from_mapping(value: dict[Any, Any]) -> pd.DataFrame | None:
    if not value:
        return None
    try:
        frame = pd.DataFrame(value)
        if len(frame) and len(frame.columns):
            return frame
    except Exception:  # noqa: BLE001
        pass
    return None


def iter_tables(
    value: Any,
    object_path: str = "$",
    depth: int = 0,
    max_depth: int = 8,
    max_nodes: int = 50_000,
) -> Iterator[tuple[str, pd.DataFrame, int]]:
    """Yield plausible tabular objects from nested repository resources."""
    seen_frames: set[tuple[str, int, int, tuple[str, ...]]] = set()
    stack: list[tuple[str, Any, int]] = [(object_path, value, depth)]
    visited = 0

    while stack and visited < max_nodes:
        path, current, level = stack.pop()
        visited += 1

        frame: pd.DataFrame | None = None
        if isinstance(current, pd.DataFrame):
            frame = current.copy()
        elif isinstance(current, pd.Series):
            frame = current.to_frame().T
        elif isinstance(current, list) and current and all(
            isinstance(item, dict) for item in current[: min(100, len(current))]
        ):
            try:
                frame = pd.DataFrame(current)
            except Exception:  # noqa: BLE001
                frame = None
        elif isinstance(current, dict):
            frame = dataframe_from_mapping(current)

        if frame is not None and len(frame.columns):
            frame.columns = [str(column) for column in frame.columns]
            signature = (
                path,
                len(frame),
                len(frame.columns),
                tuple(frame.columns[:50]),
            )
            if signature not in seen_frames:
                seen_frames.add(signature)
                yield path, frame, level

        if level >= max_depth:
            continue

        if isinstance(current, str):
            parsed = safe_json_or_literal(current)
            if parsed is not None:
                stack.append((path + "::<decoded>", parsed, level + 1))
        elif isinstance(current, dict):
            for key, child in reversed(list(current.items())):
                stack.append((f"{path}[{key!r}]", child, level + 1))
        elif isinstance(current, (list, tuple)):
            indexes = list(range(min(len(current), 1000)))
            for index in reversed(indexes):
                child = current[index]
                if isinstance(child, (dict, list, tuple, pd.DataFrame, pd.Series, str)):
                    stack.append((f"{path}[{index}]", child, level + 1))
        elif isinstance(current, np.ndarray):
            if current.dtype == object:
                flat = current.ravel()
                for index in reversed(range(min(len(flat), 1000))):
                    stack.append((f"{path}[{index}]", flat[index], level + 1))


def values_matching_trials(
    frame: pd.DataFrame,
    trial_lookup: dict[str, str],
    max_unique: int = 5000,
) -> list[tuple[str, dict[Any, str]]]:
    matches: list[tuple[str, dict[Any, str]]] = []
    for column in frame.columns:
        series = frame[column]
        if series.dtype.kind not in {"O", "U", "S"} and not pd.api.types.is_string_dtype(series):
            continue
        values = series.dropna().astype(str)
        if values.empty or values.nunique() > max_unique:
            continue
        mapping: dict[Any, str] = {}
        for index, value in values.items():
            resolved = resolve_trial_name(value, trial_lookup)
            if resolved:
                mapping[index] = resolved
        if mapping:
            matches.append((str(column), mapping))
    return matches


def path_trial_match(object_path: str, trial_lookup: dict[str, str]) -> str:
    parts = re.findall(r"['\"]([^'\"]+)['\"]", object_path)
    for part in reversed(parts):
        resolved = resolve_trial_name(part, trial_lookup)
        if resolved:
            return resolved
    return ""


def parse_bbox_cell(value: Any) -> tuple[float, float, float, float] | None:
    current = value
    if isinstance(current, str):
        parsed = safe_json_or_literal(current, max_chars=100_000)
        if parsed is not None:
            current = parsed

    if isinstance(current, dict):
        keys = {normalize_name(key): child for key, child in current.items()}
        xmin = next((finite_number(keys[key]) for key in map(normalize_name, XMIN_ALIASES) if key in keys), math.nan)
        ymin = next((finite_number(keys[key]) for key in map(normalize_name, YMIN_ALIASES) if key in keys), math.nan)
        xmax = next((finite_number(keys[key]) for key in map(normalize_name, XMAX_ALIASES) if key in keys), math.nan)
        ymax = next((finite_number(keys[key]) for key in map(normalize_name, YMAX_ALIASES) if key in keys), math.nan)
        if all(map(math.isfinite, (xmin, ymin, xmax, ymax))):
            return xmin, ymin, xmax, ymax

        x = next((finite_number(keys[key]) for key in map(normalize_name, X_ALIASES) if key in keys), math.nan)
        y = next((finite_number(keys[key]) for key in map(normalize_name, Y_ALIASES) if key in keys), math.nan)
        width = next((finite_number(keys[key]) for key in map(normalize_name, WIDTH_ALIASES) if key in keys), math.nan)
        height = next((finite_number(keys[key]) for key in map(normalize_name, HEIGHT_ALIASES) if key in keys), math.nan)
        if all(map(math.isfinite, (x, y, width, height))):
            return x, y, x + width, y + height

    if isinstance(current, (list, tuple, np.ndarray)) and len(current) >= 4:
        numbers = [finite_number(current[index]) for index in range(4)]
        if all(map(math.isfinite, numbers)):
            a, b, c, d = numbers
            # Prefer [xmin, ymin, xmax, ymax] when the last coordinates exceed
            # the first; otherwise interpret as [x, y, width, height].
            if c > a and d > b:
                return a, b, c, d
            if c > 0 and d > 0:
                return a, b, a + c, b + d
    return None


def rectangle_columns(frame: pd.DataFrame) -> dict[str, str]:
    columns = list(frame.columns)
    result = {
        "x_min": column_by_alias(columns, XMIN_ALIASES),
        "y_min": column_by_alias(columns, YMIN_ALIASES),
        "x_max": column_by_alias(columns, XMAX_ALIASES),
        "y_max": column_by_alias(columns, YMAX_ALIASES),
        "x": column_by_alias(columns, X_ALIASES),
        "y": column_by_alias(columns, Y_ALIASES),
        "width": column_by_alias(columns, WIDTH_ALIASES),
        "height": column_by_alias(columns, HEIGHT_ALIASES),
        "bbox": column_by_alias(columns, BBOX_ALIASES),
    }
    return result


def row_rectangle(row: pd.Series, columns: dict[str, str]) -> tuple[float, float, float, float] | None:
    if all(columns[name] for name in ("x_min", "y_min", "x_max", "y_max")):
        values = tuple(
            finite_number(row[columns[name]])
            for name in ("x_min", "y_min", "x_max", "y_max")
        )
        if all(map(math.isfinite, values)):
            return values  # type: ignore[return-value]

    if all(columns[name] for name in ("x", "y", "width", "height")):
        x = finite_number(row[columns["x"]])
        y = finite_number(row[columns["y"]])
        width = finite_number(row[columns["width"]])
        height = finite_number(row[columns["height"]])
        if all(map(math.isfinite, (x, y, width, height))):
            return x, y, x + width, y + height

    if columns["bbox"]:
        return parse_bbox_cell(row[columns["bbox"]])
    return None


def derive_line_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive line and word order from rectangle geometry when absent."""
    result = frame.copy()
    if result.empty:
        return result

    result["x_center"] = (result["x_min"] + result["x_max"]) / 2.0
    result["y_center"] = (result["y_min"] + result["y_max"]) / 2.0
    result["height"] = result["y_max"] - result["y_min"]
    positive_heights = result.loc[result["height"] > 0, "height"]
    median_height = float(positive_heights.median()) if len(positive_heights) else 1.0
    tolerance = max(1e-9, median_height * 0.60)

    if "line_index_explicit" in result and result["line_index_explicit"].notna().any():
        explicit = pd.to_numeric(result["line_index_explicit"], errors="coerce")
        if explicit.notna().all():
            codes, _ = pd.factorize(explicit, sort=True)
            result["line_index"] = codes
        else:
            result["line_index"] = np.nan
    else:
        result["line_index"] = np.nan

    if result["line_index"].isna().any():
        sorted_indices = result.sort_values(["y_center", "x_center"]).index
        line_centers: list[float] = []
        assignments: dict[Any, int] = {}
        for index in sorted_indices:
            y = float(result.at[index, "y_center"])
            if not line_centers:
                line_centers.append(y)
                assignments[index] = 0
                continue
            distances = [abs(y - center) for center in line_centers]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= tolerance:
                assignments[index] = nearest
                line_centers[nearest] = (line_centers[nearest] + y) / 2.0
            else:
                line_centers.append(y)
                assignments[index] = len(line_centers) - 1

        order = np.argsort(line_centers)
        remap = {int(old): int(new) for new, old in enumerate(order)}
        result["line_index"] = [
            remap[assignments[index]] for index in result.index
        ]

    result["line_index"] = result["line_index"].astype(int)

    # Explicit word index is used only if complete and unique.
    explicit_word_valid = False
    if "word_index_explicit" in result:
        explicit_word = pd.to_numeric(result["word_index_explicit"], errors="coerce")
        explicit_word_valid = (
            explicit_word.notna().all()
            and not explicit_word.duplicated().any()
        )
    if explicit_word_valid:
        order = result.assign(_explicit=explicit_word).sort_values("_explicit").index
    else:
        order = result.sort_values(["line_index", "x_center", "y_center"]).index

    word_index_map = {index: position for position, index in enumerate(order)}
    result["word_index"] = [word_index_map[index] for index in result.index]
    result = result.sort_values("word_index").reset_index(drop=True)

    n_words = len(result)
    result["word_progress"] = (
        result["word_index"] / float(n_words - 1)
        if n_words > 1
        else 0.0
    )

    line_progress = np.zeros(n_words, dtype=float)
    for _, group in result.groupby("line_index", sort=True):
        positions = np.arange(len(group), dtype=float)
        values = positions / float(len(group) - 1) if len(group) > 1 else np.zeros(1)
        line_progress[group.index.to_numpy()] = values
    result["line_progress"] = line_progress

    n_lines = int(result["line_index"].max()) + 1
    result["line_index_norm"] = (
        result["line_index"] / float(n_lines - 1)
        if n_lines > 1
        else 0.0
    )
    result["width"] = result["x_max"] - result["x_min"]
    result["height"] = result["y_max"] - result["y_min"]
    return result


def coordinate_mode(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "unknown"
    xmax = float(pd.to_numeric(frame["x_max"], errors="coerce").max())
    ymax = float(pd.to_numeric(frame["y_max"], errors="coerce").max())
    if not math.isfinite(xmax) or not math.isfinite(ymax):
        return "unknown"
    if xmax <= 1.5 and ymax <= 1.5:
        return "normalized_0_1"
    if xmax <= 100 and ymax <= 100:
        return "possibly_percentage"
    return "pixel_or_css_coordinate"


def table_trial_assignments(
    frame: pd.DataFrame,
    object_path: str,
    trial_lookup: dict[str, str],
) -> tuple[pd.Series, str, int]:
    assignments = pd.Series("", index=frame.index, dtype=object)
    evidence = ""
    explicit_hits = 0

    # Prefer semantically named trial columns.
    semantic_columns = [
        column for column in frame.columns
        if normalize_name(column) in {normalize_name(alias) for alias in TRIAL_COLUMN_ALIASES}
    ]
    candidate_columns = semantic_columns + [
        column for column, _ in values_matching_trials(frame, trial_lookup)
        if column not in semantic_columns
    ]

    for column in candidate_columns:
        mapped = frame[column].map(lambda value: resolve_trial_name(value, trial_lookup))
        new_mask = assignments.eq("") & mapped.ne("")
        if new_mask.any():
            assignments.loc[new_mask] = mapped.loc[new_mask]
            explicit_hits += int(new_mask.sum())
            evidence = evidence or f"column:{column}"

    path_match = path_trial_match(object_path, trial_lookup)
    if path_match:
        assignments.loc[assignments.eq("")] = path_match
        evidence = evidence or "nested_object_key"

    return assignments, evidence, explicit_hits


def normalize_aoi_table(
    *,
    frame: pd.DataFrame,
    source_file: str,
    object_path: str,
    trial_lookup: dict[str, str],
    primary_sets: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rect_columns = rectangle_columns(frame)
    assignments, assignment_evidence, explicit_hits = table_trial_assignments(
        frame, object_path, trial_lookup
    )

    word_column = column_by_alias(frame.columns, WORD_COLUMN_ALIASES)
    word_index_column = column_by_alias(frame.columns, WORD_INDEX_ALIASES)
    line_index_column = column_by_alias(frame.columns, LINE_INDEX_ALIASES)
    set_column = column_by_alias(frame.columns, SET_COLUMN_ALIASES)
    image_width_column = column_by_alias(frame.columns, IMAGE_WIDTH_ALIASES)
    image_height_column = column_by_alias(frame.columns, IMAGE_HEIGHT_ALIASES)
    screen_width_column = column_by_alias(frame.columns, SCREEN_WIDTH_ALIASES)
    screen_height_column = column_by_alias(frame.columns, SCREEN_HEIGHT_ALIASES)

    rows: list[dict[str, Any]] = []
    valid_rectangles = 0
    matched_rows = 0
    for index, row in frame.iterrows():
        trial_name = str(assignments.loc[index])
        if not trial_name:
            continue
        matched_rows += 1
        rectangle = row_rectangle(row, rect_columns)
        if rectangle is None:
            continue
        xmin, ymin, xmax, ymax = rectangle
        if not all(map(math.isfinite, rectangle)):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        valid_rectangles += 1

        set_name = str(row[set_column]) if set_column and pd.notna(row[set_column]) else ""
        rows.append(
            {
                "source_file": source_file,
                "object_path": object_path,
                "source_row_index": index,
                "trial_name": trial_name,
                "set_name_source": set_name,
                "set_name_matches_primary": (
                    set_name in primary_sets if set_name else ""
                ),
                "word_text": (
                    str(row[word_column]) if word_column and pd.notna(row[word_column]) else ""
                ),
                "word_index_explicit": (
                    row[word_index_column] if word_index_column else np.nan
                ),
                "line_index_explicit": (
                    row[line_index_column] if line_index_column else np.nan
                ),
                "x_min": xmin,
                "y_min": ymin,
                "x_max": xmax,
                "y_max": ymax,
                "image_width_source": (
                    finite_number(row[image_width_column])
                    if image_width_column else math.nan
                ),
                "image_height_source": (
                    finite_number(row[image_height_column])
                    if image_height_column else math.nan
                ),
                "screen_width_source": (
                    finite_number(row[screen_width_column])
                    if screen_width_column else math.nan
                ),
                "screen_height_source": (
                    finite_number(row[screen_height_column])
                    if screen_height_column else math.nan
                ),
                "trial_assignment_evidence": assignment_evidence,
            }
        )

    source_bonus = 0
    lower_source = source_file.lower()
    if "target_dataframes" in lower_source:
        source_bonus += 30
    if "js_trials" in lower_source:
        source_bonus += 15
    if word_column:
        source_bonus += 10
    if word_index_column:
        source_bonus += 5
    if line_index_column:
        source_bonus += 5
    if explicit_hits:
        source_bonus += 20
    if valid_rectangles >= 5:
        source_bonus += 20
    elif valid_rectangles >= 2:
        source_bonus += 10

    table_audit = {
        "source_file": source_file,
        "object_path": object_path,
        "n_rows": len(frame),
        "n_columns": len(frame.columns),
        "columns": "|".join(map(str, frame.columns)),
        "trial_assignment_evidence": assignment_evidence,
        "n_explicit_trial_assignments": explicit_hits,
        "n_rows_assigned_to_primary_trials": matched_rows,
        "n_valid_rectangles": valid_rectangles,
        "word_column": word_column,
        "word_index_column": word_index_column,
        "line_index_column": line_index_column,
        "set_column": set_column,
        "rectangle_columns_json": json.dumps(rect_columns, ensure_ascii=False),
        "candidate_score": source_bonus,
        "is_aoi_candidate": bool(valid_rectangles >= 2),
    }
    return rows, table_audit


def select_best_aoi_sources(
    normalized_rows: pd.DataFrame,
    table_audit: pd.DataFrame,
    primary_trial_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if normalized_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    score_lookup = {
        (str(row.source_file), str(row.object_path)): float(row.candidate_score)
        for row in table_audit.itertuples(index=False)
    }

    selection_rows: list[dict[str, Any]] = []
    canonical_frames: list[pd.DataFrame] = []

    for trial_name in primary_trial_names:
        trial_rows = normalized_rows[
            normalized_rows["trial_name"].eq(trial_name)
        ].copy()
        if trial_rows.empty:
            selection_rows.append(
                {
                    "trial_name": trial_name,
                    "selection_status": "no_candidate",
                    "selected_source_file": "",
                    "selected_object_path": "",
                    "selected_n_boxes": 0,
                    "selected_score": math.nan,
                    "n_candidate_sources": 0,
                    "coordinate_mode": "unknown",
                    "ambiguity_reason": "no content-based rectangle match",
                }
            )
            continue

        source_stats: list[dict[str, Any]] = []
        for (source_file, object_path), group in trial_rows.groupby(
            ["source_file", "object_path"], sort=False
        ):
            unique_boxes = group[
                ["x_min", "y_min", "x_max", "y_max", "word_text"]
            ].drop_duplicates()
            base_score = score_lookup.get((str(source_file), str(object_path)), 0.0)
            score = base_score + min(len(unique_boxes), 200) * 0.05
            if unique_boxes["word_text"].astype(str).str.len().gt(0).mean() >= 0.8:
                score += 5
            source_stats.append(
                {
                    "source_file": source_file,
                    "object_path": object_path,
                    "n_boxes": len(unique_boxes),
                    "score": score,
                    "group": unique_boxes,
                }
            )

        source_stats.sort(
            key=lambda item: (item["score"], item["n_boxes"]),
            reverse=True,
        )
        best = source_stats[0]
        second = source_stats[1] if len(source_stats) > 1 else None
        ambiguous = bool(
            second is not None
            and abs(float(best["score"]) - float(second["score"])) < 1e-9
            and int(best["n_boxes"]) == int(second["n_boxes"])
        )

        chosen = trial_rows[
            trial_rows["source_file"].eq(best["source_file"])
            & trial_rows["object_path"].eq(best["object_path"])
        ].copy()
        chosen = chosen.drop_duplicates(
            ["x_min", "y_min", "x_max", "y_max", "word_text"]
        )
        chosen = derive_line_order(chosen)
        chosen["selected_candidate_score"] = float(best["score"])
        chosen["selection_ambiguous"] = ambiguous
        chosen["coordinate_mode"] = coordinate_mode(chosen)
        canonical_frames.append(chosen)

        selection_rows.append(
            {
                "trial_name": trial_name,
                "selection_status": (
                    "ambiguous_top_tie" if ambiguous else "candidate_selected"
                ),
                "selected_source_file": best["source_file"],
                "selected_object_path": best["object_path"],
                "selected_n_boxes": best["n_boxes"],
                "selected_score": best["score"],
                "n_candidate_sources": len(source_stats),
                "coordinate_mode": coordinate_mode(chosen),
                "ambiguity_reason": (
                    "top candidate tied on score and box count"
                    if ambiguous else ""
                ),
                "second_source_file": (
                    second["source_file"] if second is not None else ""
                ),
                "second_object_path": (
                    second["object_path"] if second is not None else ""
                ),
                "second_n_boxes": (
                    second["n_boxes"] if second is not None else ""
                ),
                "second_score": (
                    second["score"] if second is not None else ""
                ),
            }
        )

    canonical = (
        pd.concat(canonical_frames, ignore_index=True)
        if canonical_frames else pd.DataFrame()
    )
    return canonical, pd.DataFrame(selection_rows)


def resource_files(root: Path) -> list[Path]:
    scripts = root / "scripts"
    selected: list[Path] = []
    for path in scripts.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        parts = set(path.parts)
        is_named_root = bool(parts & RESOURCE_ROOT_NAMES)
        is_named_file = path.name in {
            "combined_text_features.csv",
            "text_token_stats.csv",
        }
        if is_named_root or is_named_file:
            selected.append(path)
    return sorted(selected)


def fixation_file_candidates(
    root: Path,
    participants: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixation_root = (
        root / "scripts" / "pre_processed_data" / "fixation_data_per_part"
    )
    file_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    if not fixation_root.is_dir():
        return file_rows, table_rows

    normalized_participants = {
        normalize_name(participant): participant
        for participant in participants
    }

    for path in sorted(path for path in fixation_root.rglob("*") if path.is_file()):
        stem_norm = normalize_name(path.stem)
        filename_matches = [
            participant
            for key, participant in normalized_participants.items()
            if key and key in stem_norm
        ]
        object_value, loader, error = load_file_object(path)
        relative = path.relative_to(root).as_posix()

        content_participants: set[str] = set()
        n_tables = 0
        n_fixation_tables = 0
        if object_value is not None:
            for object_path, frame, depth in iter_tables(object_value):
                n_tables += 1
                participant_column = column_by_alias(
                    frame.columns, PARTICIPANT_COLUMN_ALIASES
                )
                if participant_column:
                    for value in frame[participant_column].dropna().astype(str).unique():
                        key = normalize_name(value)
                        if key in normalized_participants:
                            content_participants.add(normalized_participants[key])

                x_col = column_by_alias(frame.columns, FIXATION_X_ALIASES)
                y_col = column_by_alias(frame.columns, FIXATION_Y_ALIASES)
                t_col = column_by_alias(frame.columns, FIXATION_T_ALIASES)
                d_col = column_by_alias(frame.columns, FIXATION_DURATION_ALIASES)
                trial_matches = values_matching_trials(
                    frame,
                    build_trial_lookup([]),  # no-op; retained for schema symmetry
                )
                is_fixation = bool(x_col and y_col and d_col)
                if is_fixation:
                    n_fixation_tables += 1
                table_rows.append(
                    {
                        "source_file": relative,
                        "object_path": object_path,
                        "depth": depth,
                        "n_rows": len(frame),
                        "n_columns": len(frame.columns),
                        "columns": "|".join(map(str, frame.columns)),
                        "participant_column": participant_column,
                        "x_column": x_col,
                        "y_column": y_col,
                        "time_column": t_col,
                        "duration_column": d_col,
                        "is_fixation_candidate": is_fixation,
                    }
                )

        matched = sorted(set(filename_matches) | content_participants)
        file_rows.append(
            {
                "source_file": relative,
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "loader": loader,
                "parse_error": error,
                "filename_primary_participant_matches": "|".join(
                    sorted(set(filename_matches))
                ),
                "content_primary_participant_matches": "|".join(
                    sorted(content_participants)
                ),
                "resolved_primary_participants": "|".join(matched),
                "n_resolved_primary_participants": len(matched),
                "n_tables": n_tables,
                "n_fixation_tables": n_fixation_tables,
            }
        )
    return file_rows, table_rows


def notebook_hints(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [line for line in lines if NOTEBOOK_HINT_PATTERN.search(line)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage2"),
    )
    parser.add_argument(
        "--stage3a-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage3a"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage3b"),
    )
    parser.add_argument(
        "--max-resource-files",
        type=int,
        default=5000,
    )
    args = parser.parse_args()

    root = args.root.resolve()
    stage2_dir = args.stage2_dir.resolve()
    stage3a_dir = args.stage3a_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_path = stage2_dir / "trial_manifest_primary_candidates.csv"
    participant_path = stage2_dir / "participant_quality_audit.csv"
    notebook_excerpt_path = (
        stage3a_dir / "official_notebook_word_boundary_code.txt"
    )
    for path in (trial_path, participant_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    trials = pd.read_csv(trial_path)
    participants_audit = pd.read_csv(participant_path)
    primary_participants = set(
        trials["participant_id"].astype(str).unique()
    )
    primary_trial_names = sorted(
        trials["trial_name"].astype(str).unique()
    )
    primary_sets = set(trials["set_name"].astype(str).unique())
    trial_lookup = build_trial_lookup(primary_trial_names)

    files = resource_files(root)
    if len(files) > args.max_resource_files:
        raise ValueError(
            f"Found {len(files)} resource files, above --max-resource-files="
            f"{args.max_resource_files}."
        )

    file_audit_rows: list[dict[str, Any]] = []
    table_audit_rows: list[dict[str, Any]] = []
    normalized_aoi_rows: list[dict[str, Any]] = []

    for file_index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        value, loader, error = load_file_object(path)
        file_trial_hits: set[str] = set()
        n_tables = 0
        n_aoi_tables = 0

        # Text-level exact trial-name hints for JS/text resources.
        if isinstance(value, str):
            for trial_name in primary_trial_names:
                if trial_name in value:
                    file_trial_hits.add(trial_name)

        if value is not None:
            for object_path, frame, depth in iter_tables(value):
                n_tables += 1
                rows, audit = normalize_aoi_table(
                    frame=frame,
                    source_file=relative,
                    object_path=object_path,
                    trial_lookup=trial_lookup,
                    primary_sets=primary_sets,
                )
                audit["depth"] = depth
                table_audit_rows.append(audit)
                normalized_aoi_rows.extend(rows)
                if audit["is_aoi_candidate"]:
                    n_aoi_tables += 1
                    file_trial_hits.update(row["trial_name"] for row in rows)

        file_audit_rows.append(
            {
                "source_file": relative,
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "loader": loader,
                "parse_error": error,
                "n_tables": n_tables,
                "n_aoi_candidate_tables": n_aoi_tables,
                "n_primary_trial_names_detected": len(file_trial_hits),
                "primary_trial_names_detected": "|".join(
                    sorted(file_trial_hits)
                ),
            }
        )

        if file_index % 100 == 0:
            print(f"Parsed resource files: {file_index}/{len(files)}")

    file_audit = pd.DataFrame(file_audit_rows)
    table_audit = pd.DataFrame(table_audit_rows)
    normalized_aoi = pd.DataFrame(normalized_aoi_rows)

    canonical, selection = select_best_aoi_sources(
        normalized_rows=normalized_aoi,
        table_audit=table_audit,
        primary_trial_names=primary_trial_names,
    )

    fixation_file_rows, fixation_table_rows = fixation_file_candidates(
        root=root,
        participants=primary_participants,
    )
    fixation_files = pd.DataFrame(fixation_file_rows)
    fixation_tables = pd.DataFrame(fixation_table_rows)

    mapped_fixation_participants: set[str] = set()
    if not fixation_files.empty:
        for value in fixation_files["resolved_primary_participants"].fillna(""):
            mapped_fixation_participants.update(
                item for item in str(value).split("|") if item
            )

    hints = notebook_hints(notebook_excerpt_path)

    paths = {
        "resource_files": output_dir / "resource_file_content_audit.csv",
        "aoi_tables": output_dir / "aoi_table_candidates.csv",
        "aoi_rows": output_dir / "aoi_rows_normalized_all_candidates.csv",
        "aoi_selection": output_dir / "aoi_trial_source_selection.csv",
        "canonical_aoi": output_dir / "canonical_word_aoi_candidates.csv",
        "fixation_files": output_dir / "primary_fixation_file_map.csv",
        "fixation_tables": output_dir / "fixation_table_candidates.csv",
        "notebook_hints": output_dir / "official_notebook_mapping_hints.txt",
        "readiness": output_dir / "stage3b_readiness.csv",
        "manifest": output_dir / "stage3b_manifest.json",
    }

    file_audit.to_csv(
        paths["resource_files"], index=False, encoding="utf-8-sig"
    )
    table_audit.to_csv(
        paths["aoi_tables"], index=False, encoding="utf-8-sig"
    )
    normalized_aoi.to_csv(
        paths["aoi_rows"], index=False, encoding="utf-8-sig"
    )
    selection.to_csv(
        paths["aoi_selection"], index=False, encoding="utf-8-sig"
    )
    canonical.to_csv(
        paths["canonical_aoi"], index=False, encoding="utf-8-sig"
    )
    fixation_files.to_csv(
        paths["fixation_files"], index=False, encoding="utf-8-sig"
    )
    fixation_tables.to_csv(
        paths["fixation_tables"], index=False, encoding="utf-8-sig"
    )
    paths["notebook_hints"].write_text(
        "\n".join(hints) + ("\n" if hints else ""),
        encoding="utf-8",
    )

    if selection.empty:
        covered_trials = 0
        selected_trials = 0
        ambiguous_trials = 0
        no_candidate_trials = len(primary_trial_names)
        coordinate_counts: dict[str, int] = {}
    else:
        covered_trials = int(
            selection["selection_status"].ne("no_candidate").sum()
        )
        selected_trials = int(
            selection["selection_status"].eq("candidate_selected").sum()
        )
        ambiguous_trials = int(
            selection["selection_status"].eq("ambiguous_top_tie").sum()
        )
        no_candidate_trials = int(
            selection["selection_status"].eq("no_candidate").sum()
        )
        coordinate_counts = (
            selection.loc[
                selection["selection_status"].ne("no_candidate"),
                "coordinate_mode",
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    canonical_valid_trials = (
        int(canonical["trial_name"].nunique())
        if not canonical.empty else 0
    )
    canonical_min_boxes = (
        int(canonical.groupby("trial_name").size().min())
        if not canonical.empty else 0
    )

    fixation_candidate_tables = (
        int(fixation_tables["is_fixation_candidate"].fillna(False).sum())
        if not fixation_tables.empty else 0
    )

    aoi_ready = (
        covered_trials == len(primary_trial_names)
        and ambiguous_trials == 0
        and canonical_valid_trials == len(primary_trial_names)
        and canonical_min_boxes >= 2
    )
    fixation_ready = (
        len(mapped_fixation_participants) == len(primary_participants)
        and fixation_candidate_tables > 0
    )

    readiness_rows = [
        {
            "component": "word_aoi_trial_coverage",
            "status": "ready" if covered_trials == len(primary_trial_names) else "incomplete",
            "actual": covered_trials,
            "required": len(primary_trial_names),
            "detail": f"no_candidate={no_candidate_trials}",
        },
        {
            "component": "word_aoi_unambiguous_selection",
            "status": "ready" if ambiguous_trials == 0 else "ambiguous",
            "actual": ambiguous_trials,
            "required": 0,
            "detail": "number of top-score ties",
        },
        {
            "component": "canonical_minimum_boxes",
            "status": "ready" if canonical_min_boxes >= 2 else "incomplete",
            "actual": canonical_min_boxes,
            "required": 2,
            "detail": "minimum boxes among covered trials",
        },
        {
            "component": "primary_participant_fixation_mapping",
            "status": (
                "ready"
                if len(mapped_fixation_participants) == len(primary_participants)
                else "incomplete"
            ),
            "actual": len(mapped_fixation_participants),
            "required": len(primary_participants),
            "detail": f"fixation_candidate_tables={fixation_candidate_tables}",
        },
        {
            "component": "stage3c_soft_aoi_gate",
            "status": "ready" if aoi_ready else "blocked",
            "actual": int(aoi_ready),
            "required": 1,
            "detail": json.dumps(coordinate_counts, ensure_ascii=False),
        },
        {
            "component": "stage3c_official_event_gate",
            "status": "ready" if fixation_ready else "blocked",
            "actual": int(fixation_ready),
            "required": 1,
            "detail": (
                "Official condition remains a transferred event-based formula "
                "unless event detection is proven identical to ZuCo/EyeLink."
            ),
        },
    ]
    write_csv(paths["readiness"], readiness_rows)

    manifest = {
        "stage": "webqamgaze_external_validation_stage3b_resource_resolution",
        "version": "1.0",
        "created_at_utc": utc_now_iso(),
        "root": str(root),
        "primary_participants": len(primary_participants),
        "primary_trials": len(trials),
        "primary_unique_trial_names": len(primary_trial_names),
        "primary_sets": len(primary_sets),
        "resource_files_scanned": len(files),
        "tables_discovered": len(table_audit),
        "aoi_candidate_tables": int(
            table_audit["is_aoi_candidate"].fillna(False).sum()
        ) if not table_audit.empty else 0,
        "normalized_aoi_candidate_rows": len(normalized_aoi),
        "aoi_trial_coverage": covered_trials,
        "aoi_trials_selected_unambiguously": selected_trials,
        "aoi_trials_ambiguous": ambiguous_trials,
        "aoi_trials_without_candidate": no_candidate_trials,
        "canonical_aoi_trial_count": canonical_valid_trials,
        "canonical_minimum_boxes_per_trial": canonical_min_boxes,
        "coordinate_mode_counts": coordinate_counts,
        "fixation_files_scanned": len(fixation_files),
        "fixation_candidate_tables": fixation_candidate_tables,
        "primary_participants_mapped_to_fixation_files": len(
            mapped_fixation_participants
        ),
        "soft_aoi_gate_ready": aoi_ready,
        "official_event_gate_ready": fixation_ready,
        "constraints": [
            "No model was fitted or applied.",
            "No prediction score or external performance metric was inspected.",
            "Candidate ranking uses schema completeness and resource provenance, never NR/IS labels.",
            "Derived word/line order uses rectangle geometry only when explicit order is unavailable.",
            "The candidate AOI table is not final until coverage, ambiguity, and coordinate mapping are audited.",
        ],
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== WEBQAMGAZE EXTERNAL VALIDATION STAGE 3B ===")
    print(f"Primary participants: {len(primary_participants)}")
    print(f"Primary trials: {len(trials)}")
    print(f"Unique primary trial names: {len(primary_trial_names)}")
    print(f"Resource files scanned: {len(files)}")
    print(f"Nested tables discovered: {len(table_audit)}")
    print(
        "AOI candidate tables: "
        f"{manifest['aoi_candidate_tables']}"
    )
    print(f"Normalized AOI candidate rows: {len(normalized_aoi)}")
    print(
        "AOI trial coverage: "
        f"{covered_trials}/{len(primary_trial_names)}"
    )
    print(f"Unambiguous AOI selections: {selected_trials}")
    print(f"Ambiguous AOI selections: {ambiguous_trials}")
    print(f"Trials without AOI candidate: {no_candidate_trials}")
    print(
        "Canonical AOI trial count: "
        f"{canonical_valid_trials}/{len(primary_trial_names)}"
    )
    print(f"Minimum boxes per covered trial: {canonical_min_boxes}")
    print("Coordinate-mode counts:")
    if coordinate_counts:
        for mode, count in sorted(coordinate_counts.items()):
            print(f"  {mode}: {count}")
    else:
        print("  <none>")
    print(f"Fixation files scanned: {len(fixation_files)}")
    print(f"Fixation candidate tables: {fixation_candidate_tables}")
    print(
        "Primary participants mapped to fixation files: "
        f"{len(mapped_fixation_participants)}/{len(primary_participants)}"
    )
    print(f"Soft-AOI Stage 3C gate ready: {aoi_ready}")
    print(f"Official-event Stage 3C gate ready: {fixation_ready}")
    print("Official notebook mapping hints:")
    for line in hints[:30]:
        print(f"  {line}")
    print("Written:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
