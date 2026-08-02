#!/usr/bin/env python
"""Layer 3 preparation: audit layout descriptors across ZuCo and WebQAmGaze.

This script creates a shared, scale-invariant layout representation from:

- the ZuCo word-AOI template; and
- WebQAmGaze set-specific word AOIs produced by Stage 3C.

No WebQAmGaze task label is used for feature construction, parameter fitting,
descriptor selection, or the domain-support audit.

Outputs
-------
source_layout_descriptors.csv
external_layout_descriptors.csv
layout_descriptor_availability.csv
layout_shift_summary.csv
external_layout_support.csv
layer3_layout_audit_manifest.json

The descriptors intentionally emphasize scale-invariant quantities so that
1280x720 browser AOIs and 800x600 ZuCo AOIs can be compared without pretending
that raw pixels have the same physical meaning.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


TEXT_ALIASES = (
    "word_text",
    "target_name",
    "word",
    "token",
    "text",
    "target",
)
WORD_INDEX_ALIASES = (
    "word_index",
    "source_row_index",
    "target_index",
    "token_index",
    "index",
)
LINE_INDEX_ALIASES = (
    "line_index",
    "line_id",
    "line",
    "row_index",
)
XMIN_ALIASES = (
    "x_min",
    "xmin",
    "left",
    "x_left",
)
YMIN_ALIASES = (
    "y_min",
    "ymin",
    "top",
    "y_top",
)
XMAX_ALIASES = (
    "x_max",
    "xmax",
    "right",
    "x_right",
)
YMAX_ALIASES = (
    "y_max",
    "ymax",
    "bottom",
    "y_bottom",
)

DESCRIPTOR_COLUMNS = [
    "layout_log1p_n_words",
    "layout_log1p_n_lines",
    "layout_lines_per_word",
    "layout_word_char_mean",
    "layout_word_char_std",
    "layout_word_char_median",
    "layout_word_char_q90",
    "layout_envelope_aspect_ratio",
    "layout_word_box_density",
    "layout_word_width_rel_mean",
    "layout_word_width_rel_std",
    "layout_word_width_rel_median",
    "layout_word_width_rel_q90",
    "layout_word_height_rel_mean",
    "layout_word_height_rel_std",
    "layout_word_height_rel_median",
    "layout_word_height_cv",
    "layout_line_word_count_mean",
    "layout_line_word_count_std",
    "layout_line_word_count_max_fraction",
    "layout_line_width_rel_mean",
    "layout_line_width_rel_std",
    "layout_line_width_rel_max",
    "layout_line_height_rel_mean",
    "layout_horizontal_gap_wordwidth_mean",
    "layout_horizontal_gap_wordwidth_std",
    "layout_vertical_gap_wordheight_mean",
    "layout_vertical_gap_wordheight_std",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def find_column(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
    required: bool = False,
) -> str:
    normalized = {
        normalize_name(column): str(column)
        for column in frame.columns
    }
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(
            f"Could not find required column among aliases {aliases}; "
            f"available={list(frame.columns)}"
        )
    return ""


def safe_mean(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else math.nan


def safe_std(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=0)) if len(array) else math.nan


def safe_median(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else math.nan


def safe_quantile(
    values: np.ndarray | list[float],
    q: float,
) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return (
        float(np.quantile(array, q))
        if len(array)
        else math.nan
    )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator == 0
    ):
        return math.nan
    return float(numerator / denominator)


def parse_word_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    # Keep the rule conservative: only strip obvious one-item container syntax.
    if (
        len(text) >= 4
        and text[0] in "[('\""
        and text[-1] in "])'\""
    ):
        text = text.strip("[]()'\" ")
    return text


def infer_lines(
    frame: pd.DataFrame,
) -> np.ndarray:
    """Infer line IDs from vertical box positions.

    Boxes are sorted by vertical center. A box joins the nearest existing line
    when its center is within 0.60 median word heights from that line's current
    median center. Otherwise a new line is created.
    """
    y_center = (
        frame["y_min"].to_numpy(dtype=float)
        + frame["y_max"].to_numpy(dtype=float)
    ) / 2.0
    heights = (
        frame["y_max"].to_numpy(dtype=float)
        - frame["y_min"].to_numpy(dtype=float)
    )
    median_height = safe_median(heights)
    tolerance = (
        0.60 * median_height
        if math.isfinite(median_height)
        and median_height > 0
        else 1.0
    )

    order = np.argsort(y_center)
    line_centers: list[list[float]] = []
    assignments = np.full(
        len(frame),
        -1,
        dtype=int,
    )

    for index in order:
        center = float(y_center[index])
        if not line_centers:
            line_centers.append([center])
            assignments[index] = 0
            continue

        medians = np.asarray(
            [
                np.median(values)
                for values in line_centers
            ],
            dtype=float,
        )
        nearest = int(
            np.argmin(
                np.abs(
                    medians - center
                )
            )
        )
        if abs(medians[nearest] - center) <= tolerance:
            assignments[index] = nearest
            line_centers[nearest].append(center)
        else:
            assignments[index] = len(
                line_centers
            )
            line_centers.append([center])

    # Renumber by top-to-bottom median center.
    line_order = np.argsort(
        [
            np.median(values)
            for values in line_centers
        ]
    )
    remap = {
        int(old): int(new)
        for new, old in enumerate(
            line_order
        )
    }
    return np.asarray(
        [
            remap[int(value)]
            for value in assignments
        ],
        dtype=int,
    )


def canonicalize_boxes(
    frame: pd.DataFrame,
    line_column: str,
    word_index_column: str,
    text_column: str,
) -> pd.DataFrame:
    xmin = find_column(
        frame,
        XMIN_ALIASES,
        required=True,
    )
    ymin = find_column(
        frame,
        YMIN_ALIASES,
        required=True,
    )
    xmax = find_column(
        frame,
        XMAX_ALIASES,
        required=True,
    )
    ymax = find_column(
        frame,
        YMAX_ALIASES,
        required=True,
    )

    result = pd.DataFrame(
        {
            "x_min": pd.to_numeric(
                frame[xmin],
                errors="coerce",
            ),
            "y_min": pd.to_numeric(
                frame[ymin],
                errors="coerce",
            ),
            "x_max": pd.to_numeric(
                frame[xmax],
                errors="coerce",
            ),
            "y_max": pd.to_numeric(
                frame[ymax],
                errors="coerce",
            ),
        }
    )
    if text_column:
        result["word_text"] = (
            frame[text_column]
            .map(parse_word_text)
            .astype(str)
        )
    else:
        result["word_text"] = ""

    if word_index_column:
        result["word_index"] = pd.to_numeric(
            frame[word_index_column],
            errors="coerce",
        )
    else:
        result["word_index"] = np.arange(
            len(result),
            dtype=float,
        )

    finite = np.isfinite(
        result[
            [
                "x_min",
                "y_min",
                "x_max",
                "y_max",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    positive = (
        result["x_max"].gt(
            result["x_min"]
        )
        & result["y_max"].gt(
            result["y_min"]
        )
    )
    result = result[
        finite & positive
    ].copy().reset_index(drop=True)
    if result.empty:
        return result

    if line_column:
        raw_line = pd.to_numeric(
            frame.loc[
                finite & positive,
                line_column,
            ],
            errors="coerce",
        ).reset_index(drop=True)
        if raw_line.notna().all():
            codes, _ = pd.factorize(
                raw_line,
                sort=True,
            )
            result["line_index"] = (
                codes.astype(int)
            )
        else:
            result["line_index"] = infer_lines(
                result
            )
    else:
        result["line_index"] = infer_lines(
            result
        )

    # Stable order: line, x, original index.
    result = result.sort_values(
        [
            "line_index",
            "x_min",
            "word_index",
        ]
    ).reset_index(drop=True)
    return result


def layout_descriptor_row(
    boxes: pd.DataFrame,
) -> dict[str, float]:
    if boxes.empty:
        return {
            column: math.nan
            for column in DESCRIPTOR_COLUMNS
        }

    x_min = boxes[
        "x_min"
    ].to_numpy(dtype=float)
    x_max = boxes[
        "x_max"
    ].to_numpy(dtype=float)
    y_min = boxes[
        "y_min"
    ].to_numpy(dtype=float)
    y_max = boxes[
        "y_max"
    ].to_numpy(dtype=float)
    widths = x_max - x_min
    heights = y_max - y_min
    areas = widths * heights

    envelope_width = float(
        np.max(x_max) - np.min(x_min)
    )
    envelope_height = float(
        np.max(y_max) - np.min(y_min)
    )
    envelope_area = (
        envelope_width
        * envelope_height
    )
    n_words = int(len(boxes))
    n_lines = int(
        boxes["line_index"].nunique()
    )

    word_chars = (
        boxes["word_text"]
        .fillna("")
        .astype(str)
        .str.len()
        .to_numpy(dtype=float)
    )
    valid_chars = word_chars[
        word_chars > 0
    ]

    line_word_counts = []
    line_widths_rel = []
    line_heights_rel = []
    horizontal_gap_rel = []
    line_top_bottom = []
    median_word_width = safe_median(
        widths
    )
    median_word_height = safe_median(
        heights
    )

    for _, line in boxes.groupby(
        "line_index",
        sort=True,
    ):
        line = line.sort_values(
            ["x_min", "word_index"]
        )
        count = float(len(line))
        line_word_counts.append(count)

        line_width = float(
            line["x_max"].max()
            - line["x_min"].min()
        )
        line_height = float(
            line["y_max"].max()
            - line["y_min"].min()
        )
        line_widths_rel.append(
            safe_ratio(
                line_width,
                envelope_width,
            )
        )
        line_heights_rel.append(
            safe_ratio(
                line_height,
                envelope_height,
            )
        )
        line_top_bottom.append(
            (
                float(
                    line["y_min"].min()
                ),
                float(
                    line["y_max"].max()
                ),
            )
        )
        if len(line) >= 2:
            gaps = (
                line["x_min"]
                .to_numpy(dtype=float)[1:]
                - line["x_max"]
                .to_numpy(dtype=float)[:-1]
            )
            horizontal_gap_rel.extend(
                [
                    safe_ratio(
                        float(gap),
                        median_word_width,
                    )
                    for gap in gaps
                ]
            )

    vertical_gap_rel = []
    ordered_lines = sorted(
        line_top_bottom,
        key=lambda pair: pair[0],
    )
    for previous, current in zip(
        ordered_lines[:-1],
        ordered_lines[1:],
    ):
        gap = float(
            current[0] - previous[1]
        )
        vertical_gap_rel.append(
            safe_ratio(
                gap,
                median_word_height,
            )
        )

    word_width_rel = (
        widths / envelope_width
        if envelope_width > 0
        else np.full(
            len(widths),
            np.nan,
        )
    )
    word_height_rel = (
        heights / envelope_height
        if envelope_height > 0
        else np.full(
            len(heights),
            np.nan,
        )
    )

    return {
        "layout_log1p_n_words": float(
            np.log1p(n_words)
        ),
        "layout_log1p_n_lines": float(
            np.log1p(n_lines)
        ),
        "layout_lines_per_word": safe_ratio(
            float(n_lines),
            float(n_words),
        ),
        "layout_word_char_mean": safe_mean(
            valid_chars
        ),
        "layout_word_char_std": safe_std(
            valid_chars
        ),
        "layout_word_char_median": safe_median(
            valid_chars
        ),
        "layout_word_char_q90": safe_quantile(
            valid_chars,
            0.90,
        ),
        "layout_envelope_aspect_ratio": safe_ratio(
            envelope_width,
            envelope_height,
        ),
        "layout_word_box_density": safe_ratio(
            float(np.sum(areas)),
            envelope_area,
        ),
        "layout_word_width_rel_mean": safe_mean(
            word_width_rel
        ),
        "layout_word_width_rel_std": safe_std(
            word_width_rel
        ),
        "layout_word_width_rel_median": safe_median(
            word_width_rel
        ),
        "layout_word_width_rel_q90": safe_quantile(
            word_width_rel,
            0.90,
        ),
        "layout_word_height_rel_mean": safe_mean(
            word_height_rel
        ),
        "layout_word_height_rel_std": safe_std(
            word_height_rel
        ),
        "layout_word_height_rel_median": safe_median(
            word_height_rel
        ),
        "layout_word_height_cv": safe_ratio(
            safe_std(heights),
            safe_mean(heights),
        ),
        "layout_line_word_count_mean": safe_mean(
            line_word_counts
        ),
        "layout_line_word_count_std": safe_std(
            line_word_counts
        ),
        "layout_line_word_count_max_fraction": safe_ratio(
            float(
                np.max(
                    line_word_counts
                )
            )
            if line_word_counts
            else math.nan,
            float(n_words),
        ),
        "layout_line_width_rel_mean": safe_mean(
            line_widths_rel
        ),
        "layout_line_width_rel_std": safe_std(
            line_widths_rel
        ),
        "layout_line_width_rel_max": (
            float(
                np.max(
                    line_widths_rel
                )
            )
            if line_widths_rel
            else math.nan
        ),
        "layout_line_height_rel_mean": safe_mean(
            line_heights_rel
        ),
        "layout_horizontal_gap_wordwidth_mean": safe_mean(
            horizontal_gap_rel
        ),
        "layout_horizontal_gap_wordwidth_std": safe_std(
            horizontal_gap_rel
        ),
        "layout_vertical_gap_wordheight_mean": safe_mean(
            vertical_gap_rel
        ),
        "layout_vertical_gap_wordheight_std": safe_std(
            vertical_gap_rel
        ),
    }


def build_source_descriptors(
    path: Path,
) -> pd.DataFrame:
    words = pd.read_csv(path)
    required = {
        "task",
        "task_sentence_index",
    }
    missing = sorted(
        required - set(words.columns)
    )
    if missing:
        raise ValueError(
            f"Source word template lacks columns: {missing}"
        )

    words["task"] = (
        words["task"]
        .astype(str)
        .str.upper()
    )
    words["task_sentence_index"] = (
        pd.to_numeric(
            words[
                "task_sentence_index"
            ],
            errors="raise",
        ).astype(int)
    )

    if "valid_box" in words.columns:
        valid = words[
            "valid_box"
        ]
        if valid.dtype == bool:
            words = words[
                valid
            ].copy()
        else:
            words = words[
                valid.astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "1.0",
                    ]
                )
            ].copy()

    line_column = find_column(
        words,
        LINE_INDEX_ALIASES,
    )
    word_index_column = find_column(
        words,
        WORD_INDEX_ALIASES,
    )
    text_column = find_column(
        words,
        TEXT_ALIASES,
    )

    rows = []
    for (
        task,
        sentence_index,
    ), group in words.groupby(
        [
            "task",
            "task_sentence_index",
        ],
        sort=True,
    ):
        boxes = canonicalize_boxes(
            group,
            line_column=line_column,
            word_index_column=(
                word_index_column
            ),
            text_column=text_column,
        )
        rows.append(
            {
                "domain": "zuco",
                "source_task": str(
                    task
                ),
                "task_sentence_index": int(
                    sentence_index
                ),
                "layout_id": (
                    f"{task}:"
                    f"{int(sentence_index):03d}"
                ),
                "n_valid_boxes": int(
                    len(boxes)
                ),
                "line_source": (
                    "provided"
                    if line_column
                    else "inferred"
                ),
                **layout_descriptor_row(
                    boxes
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result["layout_id"].duplicated().any():
        raise ValueError(
            "Source descriptors contain duplicate layout IDs."
        )
    return result



def _find_alias_column(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
) -> str:
    normalized = {
        normalize_name(column): str(column)
        for column in frame.columns
    }
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    return ""


def _resolve_external_layout_keys(
    aoi: pd.DataFrame,
    compact: pd.DataFrame,
) -> tuple[list[str], list[str], str]:
    """Resolve a stable layout key shared by AOI and trial tables.

    Stage-3C AOIs are often stored once per set/trial layout rather than once
    per participant trial.  In that case `external_trial_id` is intentionally
    absent from the AOI table, so we map layouts to participant trials by a
    shared set/trial key.
    """
    direct_aliases = (
        "external_trial_id",
        "layout_id",
        "aoi_trial_key",
        "set_trial_key",
        "trial_key",
        "stimulus_id",
        "passage_id",
    )
    for alias in direct_aliases:
        aoi_col = _find_alias_column(
            aoi,
            (alias,),
        )
        compact_col = _find_alias_column(
            compact,
            (alias,),
        )
        if aoi_col and compact_col:
            return (
                [aoi_col],
                [compact_col],
                f"direct:{alias}",
            )

    semantic_aliases = {
        "set": (
            "set_name",
            "set_id",
            "set",
            "experiment_set",
            "dataset_set",
        ),
        "trial_index": (
            "trial_index",
            "trial_idx",
            "trial_position",
            "trial_number",
            "trial_no",
            "item_index",
        ),
        "trial_name": (
            "trial_name",
            "item_name",
            "stimulus_name",
            "page_name",
            "passage_name",
            "question_name",
        ),
        "source_file": (
            "source_file",
            "json_file",
            "json_path",
            "source_path",
            "participant_file",
        ),
    }

    resolved: dict[str, tuple[str, str]] = {}
    for semantic, aliases in semantic_aliases.items():
        aoi_col = _find_alias_column(
            aoi,
            aliases,
        )
        compact_col = _find_alias_column(
            compact,
            aliases,
        )
        if aoi_col and compact_col:
            resolved[semantic] = (
                aoi_col,
                compact_col,
            )

    candidate_semantics = (
        ("set", "trial_index"),
        ("set", "trial_name"),
        ("source_file", "trial_index"),
        ("source_file", "trial_name"),
        ("set", "trial_index", "trial_name"),
    )
    for semantics in candidate_semantics:
        if all(
            semantic in resolved
            for semantic in semantics
        ):
            return (
                [
                    resolved[semantic][0]
                    for semantic in semantics
                ],
                [
                    resolved[semantic][1]
                    for semantic in semantics
                ],
                "composite:"
                + "+".join(semantics),
            )

    # A single trial-name key is acceptable only when it is unique in the AOI
    # layout table and every Compact trial maps to an observed AOI key.
    if "trial_name" in resolved:
        aoi_col, compact_col = resolved[
            "trial_name"
        ]
        aoi_values = (
            aoi[aoi_col]
            .astype(str)
            .str.strip()
        )
        compact_values = (
            compact[compact_col]
            .astype(str)
            .str.strip()
        )
        aoi_unique = set(
            aoi_values.dropna().unique()
        )
        compact_unique = set(
            compact_values.dropna().unique()
        )
        if (
            len(aoi_unique) > 1
            and compact_unique.issubset(
                aoi_unique
            )
        ):
            return (
                [aoi_col],
                [compact_col],
                "single:trial_name",
            )

    raise ValueError(
        "Could not resolve a shared layout key between the external AOI "
        "and Compact tables.\n"
        f"AOI columns: {list(aoi.columns)}\n"
        f"Compact columns: {list(compact.columns)}\n"
        "Expected either external_trial_id, a shared trial/layout key, or "
        "a composite such as set_name + trial_index."
    )


def _key_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    normalized_parts = []
    for column in columns:
        values = frame[column]
        numeric = pd.to_numeric(
            values,
            errors="coerce",
        )
        if numeric.notna().all():
            part = numeric.map(
                lambda value: (
                    str(int(value))
                    if float(value).is_integer()
                    else format(
                        float(value),
                        ".12g",
                    )
                )
            )
        else:
            part = (
                values.astype(str)
                .str.strip()
                .str.replace(
                    r"\s+",
                    " ",
                    regex=True,
                )
            )
        normalized_parts.append(
            part
        )

    key = normalized_parts[0]
    for part in normalized_parts[1:]:
        key = key + "||" + part
    return key


def build_external_descriptors(
    aoi_path: Path,
    compact_path: Path,
) -> pd.DataFrame:
    aoi = pd.read_csv(aoi_path)
    compact = pd.read_csv(
        compact_path
    )
    if "external_trial_id" not in compact.columns:
        raise ValueError(
            "External Compact table lacks external_trial_id."
        )

    compact[
        "external_trial_id"
    ] = compact[
        "external_trial_id"
    ].astype(str)
    if compact[
        "external_trial_id"
    ].duplicated().any():
        raise ValueError(
            "External Compact table has duplicate trial IDs."
        )

    (
        aoi_key_columns,
        compact_key_columns,
        key_strategy,
    ) = _resolve_external_layout_keys(
        aoi,
        compact,
    )
    aoi = aoi.copy()
    compact = compact.copy()
    aoi["_resolved_layout_key"] = (
        _key_frame(
            aoi,
            aoi_key_columns,
        )
    )
    compact["_resolved_layout_key"] = (
        _key_frame(
            compact,
            compact_key_columns,
        )
    )

    line_column = find_column(
        aoi,
        LINE_INDEX_ALIASES,
    )
    word_index_column = find_column(
        aoi,
        WORD_INDEX_ALIASES,
    )
    text_column = find_column(
        aoi,
        TEXT_ALIASES,
    )

    rows = []
    for layout_key, group in aoi.groupby(
        "_resolved_layout_key",
        sort=True,
        dropna=False,
    ):
        boxes = canonicalize_boxes(
            group,
            line_column=line_column,
            word_index_column=(
                word_index_column
            ),
            text_column=text_column,
        )
        rows.append(
            {
                "domain": "webqamgaze",
                "_resolved_layout_key": str(
                    layout_key
                ),
                "layout_id": str(
                    layout_key
                ),
                "n_valid_boxes": int(
                    len(boxes)
                ),
                "line_source": (
                    "provided"
                    if line_column
                    else "inferred"
                ),
                "layout_key_strategy": (
                    key_strategy
                ),
                "aoi_key_columns": (
                    "+".join(
                        aoi_key_columns
                    )
                ),
                "compact_key_columns": (
                    "+".join(
                        compact_key_columns
                    )
                ),
                **layout_descriptor_row(
                    boxes
                ),
            }
        )
    layout_descriptors = pd.DataFrame(
        rows
    )
    if layout_descriptors[
        "_resolved_layout_key"
    ].duplicated().any():
        raise ValueError(
            "Resolved AOI layout keys are not unique."
        )

    metadata_columns = [
        column
        for column in (
            "external_trial_id",
            "participant_id",
            "set_name",
            "trial_index",
            "trial_name",
            "source_file",
            "_resolved_layout_key",
        )
        if column in compact.columns
    ]
    metadata = compact[
        metadata_columns
    ].copy()

    result = metadata.merge(
        layout_descriptors,
        on="_resolved_layout_key",
        how="left",
        validate="many_to_one",
    )
    if len(result) != len(compact):
        raise ValueError(
            f"External descriptor rows={len(result)}, "
            f"expected={len(compact)}."
        )

    missing_layout = result[
        DESCRIPTOR_COLUMNS
    ].isna().all(axis=1)
    if missing_layout.any():
        examples = result.loc[
            missing_layout,
            [
                column
                for column in (
                    "external_trial_id",
                    "participant_id",
                    "set_name",
                    "trial_index",
                    "trial_name",
                    "source_file",
                    "_resolved_layout_key",
                )
                if column in result.columns
            ],
        ].head(20)
        raise ValueError(
            "Some Compact trials could not be mapped to an AOI layout.\n"
            f"Key strategy: {key_strategy}\n"
            f"AOI key columns: {aoi_key_columns}\n"
            f"Compact key columns: {compact_key_columns}\n"
            "Examples:\n"
            + examples.to_string(index=False)
        )

    # The same set/trial layout may be reused across many participants.  This is
    # expected; external_trial_id remains the row-level evaluation key.
    return result.drop(
        columns=[
            "_resolved_layout_key"
        ]
    )

def descriptor_availability(
    source: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for column in DESCRIPTOR_COLUMNS:
        rows.append(
            {
                "descriptor": column,
                "source_missing_rate": float(
                    source[column].isna().mean()
                ),
                "external_missing_rate": float(
                    external[
                        column
                    ].isna().mean()
                ),
                "source_unique_finite": int(
                    source[column]
                    .dropna()
                    .nunique()
                ),
                "external_unique_finite": int(
                    external[column]
                    .dropna()
                    .nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def layout_shift_summary(
    source: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for column in DESCRIPTOR_COLUMNS:
        source_values = pd.to_numeric(
            source[column],
            errors="coerce",
        ).dropna()
        external_values = pd.to_numeric(
            external[column],
            errors="coerce",
        ).dropna()

        source_median = float(
            source_values.median()
        )
        source_q25 = float(
            source_values.quantile(
                0.25
            )
        )
        source_q75 = float(
            source_values.quantile(
                0.75
            )
        )
        source_iqr = (
            source_q75 - source_q25
        )
        robust_shift = (
            (
                float(
                    external_values.median()
                )
                - source_median
            )
            / source_iqr
            if source_iqr > 1e-12
            else math.nan
        )
        source_min = float(
            source_values.min()
        )
        source_max = float(
            source_values.max()
        )
        out_of_range_rate = float(
            np.mean(
                (
                    external_values
                    < source_min
                )
                | (
                    external_values
                    > source_max
                )
            )
        )
        rows.append(
            {
                "descriptor": column,
                "source_median": (
                    source_median
                ),
                "external_median": float(
                    external_values.median()
                ),
                "source_iqr": float(
                    source_iqr
                ),
                "robust_median_shift_iqr": float(
                    robust_shift
                ),
                "source_min": source_min,
                "source_max": source_max,
                "external_outside_source_range_rate": (
                    out_of_range_rate
                ),
            }
        )
    result = pd.DataFrame(rows)
    result[
        "absolute_robust_shift"
    ] = result[
        "robust_median_shift_iqr"
    ].abs()
    return result.sort_values(
        [
            "absolute_robust_shift",
            "external_outside_source_range_rate",
        ],
        ascending=False,
    ).reset_index(drop=True)


def external_support(
    source: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    source_values = source[
        DESCRIPTOR_COLUMNS
    ]
    external_values = external[
        DESCRIPTOR_COLUMNS
    ]

    imputer = SimpleImputer(
        strategy="median",
    )
    source_imputed = (
        imputer.fit_transform(
            source_values
        )
    )
    external_imputed = (
        imputer.transform(
            external_values
        )
    )
    scaler = RobustScaler(
        quantile_range=(
            25.0,
            75.0,
        )
    )
    source_scaled = scaler.fit_transform(
        source_imputed
    )
    external_scaled = scaler.transform(
        external_imputed
    )

    source_center = np.median(
        source_scaled,
        axis=0,
    )
    external_distance = np.sqrt(
        np.mean(
            (
                external_scaled
                - source_center[
                    None,
                    :
                ]
            )
            ** 2,
            axis=1,
        )
    )
    source_distance = np.sqrt(
        np.mean(
            (
                source_scaled
                - source_center[
                    None,
                    :
                ]
            )
            ** 2,
            axis=1,
        )
    )
    q95 = float(
        np.quantile(
            source_distance,
            0.95,
        )
    )
    q99 = float(
        np.quantile(
            source_distance,
            0.99,
        )
    )

    metadata = external[
        [
            column
            for column in (
                "external_trial_id",
                "participant_id",
                "set_name",
                "trial_index",
                "trial_name",
            )
            if column in external.columns
        ]
    ].copy()
    metadata[
        "layout_robust_distance_from_source_center"
    ] = external_distance
    metadata[
        "outside_source_layout_q95"
    ] = external_distance > q95
    metadata[
        "outside_source_layout_q99"
    ] = external_distance > q99
    metadata.attrs[
        "source_q95"
    ] = q95
    metadata.attrs[
        "source_q99"
    ] = q99
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-word-template",
        type=Path,
        default=Path(
            "data/processed/zuco2/"
            "soft_aoi_template/"
            "zuco_soft_aoi_word_template.csv"
        ),
    )
    parser.add_argument(
        "--external-aoi",
        type=Path,
        default=Path(
            "outputs/"
            "webqamgaze_external_validation_stage3c/"
            "set_specific_word_aoi.csv"
        ),
    )
    parser.add_argument(
        "--external-compact",
        type=Path,
        default=Path(
            "outputs/"
            "webqamgaze_external_validation_stage3c/"
            "soft_compact_features_primary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/"
            "layer3_layout_audit"
        ),
    )
    args = parser.parse_args()

    for path in (
        args.source_word_template,
        args.external_aoi,
        args.external_compact,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    source = build_source_descriptors(
        args.source_word_template
    )
    external = build_external_descriptors(
        args.external_aoi,
        args.external_compact,
    )
    availability = descriptor_availability(
        source,
        external,
    )
    shift = layout_shift_summary(
        source,
        external,
    )
    support = external_support(
        source,
        external,
    )

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths = {
        "source": (
            output_dir
            / "source_layout_descriptors.csv"
        ),
        "external": (
            output_dir
            / "external_layout_descriptors.csv"
        ),
        "availability": (
            output_dir
            / "layout_descriptor_availability.csv"
        ),
        "shift": (
            output_dir
            / "layout_shift_summary.csv"
        ),
        "support": (
            output_dir
            / "external_layout_support.csv"
        ),
        "manifest": (
            output_dir
            / "layer3_layout_audit_manifest.json"
        ),
    }

    source.to_csv(
        paths["source"],
        index=False,
        encoding="utf-8-sig",
    )
    external.to_csv(
        paths["external"],
        index=False,
        encoding="utf-8-sig",
    )
    availability.to_csv(
        paths["availability"],
        index=False,
        encoding="utf-8-sig",
    )
    shift.to_csv(
        paths["shift"],
        index=False,
        encoding="utf-8-sig",
    )
    support.to_csv(
        paths["support"],
        index=False,
        encoding="utf-8-sig",
    )

    source_q95 = float(
        support.attrs[
            "source_q95"
        ]
    )
    source_q99 = float(
        support.attrs[
            "source_q99"
        ]
    )
    manifest = {
        "stage": (
            "layer3_layout_descriptor_audit"
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "n_source_layouts": int(
            len(source)
        ),
        "n_external_layouts": int(
            len(external)
        ),
        "descriptor_columns": (
            DESCRIPTOR_COLUMNS
        ),
        "external_labels_used": False,
        "coordinate_policy": (
            "scale-invariant relative layout descriptors; "
            "no raw-pixel alignment"
        ),
        "line_policy": {
            "source": str(
                source[
                    "line_source"
                ].mode().iloc[0]
            ),
            "external": str(
                external[
                    "line_source"
                ].mode().iloc[0]
            ),
            "inference_tolerance": (
                "0.60 * median word-box height"
            ),
        },
        "source_support_thresholds": {
            "robust_distance_q95": (
                source_q95
            ),
            "robust_distance_q99": (
                source_q99
            ),
        },
        "external_outside_support": {
            "q95_rate": float(
                support[
                    "outside_source_layout_q95"
                ].mean()
            ),
            "q99_rate": float(
                support[
                    "outside_source_layout_q99"
                ].mean()
            ),
        },
        "inputs": {
            "source_word_template": str(
                args.source_word_template
            ),
            "external_aoi": str(
                args.external_aoi
            ),
            "external_compact": str(
                args.external_compact
            ),
        },
        "outputs": {
            name: str(path)
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== LAYER-3 LAYOUT AUDIT INPUT ==="
    )
    print(
        f"Source layouts: {len(source)}"
    )
    print(
        f"External layouts: {len(external)}"
    )
    print(
        f"Shared descriptors: "
        f"{len(DESCRIPTOR_COLUMNS)}"
    )
    print(
        "External labels used: False"
    )
    print(
        "Source line source: "
        + ", ".join(
            f"{key}={value}"
            for key, value in source[
                "line_source"
            ].value_counts().to_dict().items()
        )
    )
    print(
        "External line source: "
        + ", ".join(
            f"{key}={value}"
            for key, value in external[
                "line_source"
            ].value_counts().to_dict().items()
        )
    )

    print(
        "\n=== LAYER-3 DESCRIPTOR AVAILABILITY ==="
    )
    print(
        availability.to_string(
            index=False
        )
    )

    print(
        "\n=== LAYER-3 TOP LAYOUT SHIFTS ==="
    )
    print(
        shift.head(15)[
            [
                "descriptor",
                "source_median",
                "external_median",
                "robust_median_shift_iqr",
                "external_outside_source_range_rate",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== LAYER-3 EXTERNAL LAYOUT SUPPORT ==="
    )
    print(
        f"Source robust-distance q95: "
        f"{source_q95:.6f}"
    )
    print(
        f"Source robust-distance q99: "
        f"{source_q99:.6f}"
    )
    print(
        "External outside q95 rate: "
        f"{support['outside_source_layout_q95'].mean():.6f}"
    )
    print(
        "External outside q99 rate: "
        f"{support['outside_source_layout_q99'].mean():.6f}"
    )
    print(
        "External robust-distance median: "
        f"{support['layout_robust_distance_from_source_center'].median():.6f}"
    )
    print(
        "External robust-distance q90: "
        f"{support['layout_robust_distance_from_source_center'].quantile(0.90):.6f}"
    )

    print("\nWritten:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
