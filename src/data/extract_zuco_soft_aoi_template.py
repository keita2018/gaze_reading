#!/usr/bin/env python
"""Extract a reusable ZuCo 2.0 word-AOI template for Soft-AOI features.

The ZuCo reading stimuli are fixed across subjects. Therefore, the sentence
text and word bounding boxes can be extracted once from one complete template
subject (YAC by default) and mapped to every subject through the task-level
sentence index.

Inputs
------
data/raw/zuco2/YAC/resultsYAC_NR.mat
data/raw/zuco2/YAC/resultsYAC_TSR.mat

Outputs
-------
data/processed/zuco2/soft_aoi_template/
├─ zuco_soft_aoi_word_template.csv
├─ zuco_soft_aoi_sentence_template.csv
├─ zuco_wordbounds_layout_audit.csv
├─ zuco_soft_aoi_extraction_issues.csv
├─ preview_NR_sentence_000.png
├─ preview_NR_max_lines.png
├─ preview_TSR_sentence_000.png
└─ preview_TSR_max_lines.png

The script:
1. Reads sentenceData/content, sentenceData/word/content, and wordbounds.
2. Automatically infers the four-column wordbounds layout.
3. Converts every word box to x_min, y_min, x_max, y_max.
4. Assigns line indices and normalized word/line progress variables.
5. Validates the expected 349 NR and 390 TSR sentences.
6. Draws box-and-word previews for visual verification.

No fixation, saccade, duration, or word-level gaze event is used.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


EXPECTED_SENTENCES = {"NR": 349, "TSR": 390}


@dataclass(frozen=True)
class BoundsLayout:
    name: str
    x_min_col: int
    y_min_col: int
    x_max_col: int
    y_max_col: int


LAYOUTS = [
    BoundsLayout("x1_y1_x2_y2", 0, 1, 2, 3),
    BoundsLayout("x1_x2_y1_y2", 0, 2, 1, 3),
    BoundsLayout("y1_x1_y2_x2", 1, 0, 3, 2),
    BoundsLayout("y1_y2_x1_x2", 2, 0, 3, 1),
]


def iter_references(dataset: h5py.Dataset) -> Iterable[h5py.Reference]:
    values = np.asarray(dataset[()]).ravel(order="C")
    for value in values:
        if isinstance(value, h5py.Reference) and bool(value):
            yield value


def indexed_reference(
    dataset: h5py.Dataset,
    index: int,
) -> h5py.Reference | None:
    values = np.asarray(dataset[()]).ravel(order="C")
    if index < 0 or index >= len(values):
        return None
    value = values[index]
    if isinstance(value, h5py.Reference) and bool(value):
        return value
    return None


def decode_matlab_char(
    obj: h5py.Group | h5py.Dataset,
) -> str:
    if not isinstance(obj, h5py.Dataset):
        return ""
    array = np.asarray(obj[()]).ravel(order="F")
    if array.size == 0:
        return ""
    if np.issubdtype(array.dtype, np.integer):
        return "".join(
            chr(int(value))
            for value in array
            if int(value) != 0
        )
    return str(array.tolist())


def read_referenced_text(
    h5: h5py.File,
    container: h5py.Dataset,
    index: int,
) -> str:
    reference = indexed_reference(container, index)
    if reference is None:
        return ""
    return decode_matlab_char(h5[reference])


def normalize_bounds_array(
    obj: h5py.Group | h5py.Dataset,
) -> np.ndarray:
    """Return word bounds as an N x 4 numeric array.

    Some ZuCo result files encode an empty MATLAB array as a referenced
    scalar HDF5 dataset. Such a scalar is not a malformed bounding box; it
    means that this subject does not provide wordbounds for that sentence.
    The caller may then fill the sentence from another subject because the
    reading stimuli are shared across subjects.
    """
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"wordbounds reference is not a dataset: {obj.name}"
        )

    raw_value = obj[()]
    array = np.asarray(raw_value)

    # MATLAB v7.3 empty values may appear as scalar datasets, often with a
    # MATLAB_empty attribute. Treat every scalar reference as unavailable
    # wordbounds rather than terminating the whole extraction.
    if array.ndim == 0:
        return np.empty((0, 4), dtype=float)

    if array.size == 0:
        return np.empty((0, 4), dtype=float)

    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"wordbounds is not numeric: {obj.name}, "
            f"dtype={array.dtype}, shape={array.shape}"
        ) from error

    array = np.squeeze(array)

    # Squeezing a 1 x 1 MATLAB empty marker also produces a scalar.
    if array.ndim == 0:
        return np.empty((0, 4), dtype=float)

    if array.ndim == 1:
        if array.size == 0:
            return np.empty((0, 4), dtype=float)
        if array.size % 4 != 0:
            raise ValueError(
                f"Cannot reshape wordbounds {obj.name}: "
                f"shape={array.shape}"
            )
        array = array.reshape(-1, 4)

    if array.ndim != 2:
        raise ValueError(
            f"wordbounds must be 2-D after squeeze: "
            f"{obj.name}, shape={array.shape}"
        )

    if array.shape[1] == 4:
        result = array
    elif array.shape[0] == 4:
        result = array.T
    else:
        raise ValueError(
            f"No four-column axis in wordbounds: "
            f"{obj.name}, shape={array.shape}"
        )

    return np.asarray(result, dtype=float)


def read_sentence_bounds_raw(
    h5: h5py.File,
    sentence_data: h5py.Group,
    sentence_index: int,
) -> np.ndarray:
    container = sentence_data["wordbounds"]
    if not isinstance(container, h5py.Dataset):
        raise TypeError("sentenceData/wordbounds is not a dataset.")

    reference = indexed_reference(container, sentence_index)
    if reference is None:
        return np.empty((0, 4), dtype=float)

    return normalize_bounds_array(h5[reference])


def _decode_text_object(
    h5: h5py.File,
    obj: h5py.Group | h5py.Dataset,
    depth: int = 0,
) -> list[str]:
    """Decode one MATLAB word object from heterogeneous HDF5 layouts.

    ZuCo result files do not store sentenceData/word uniformly. Depending on
    the sentence, a reference may point to:
    - a MATLAB struct represented as an HDF5 Group with a ``content`` field;
    - a cell/reference Dataset containing word objects;
    - a direct MATLAB char Dataset;
    - another scalar reference Dataset.

    The function recursively follows those representations while preserving
    the order of reference arrays.
    """
    if depth > 8:
        raise RecursionError(
            f"Exceeded word-object reference depth at {obj.name}"
        )

    if isinstance(obj, h5py.Group):
        if "content" in obj:
            return _decode_text_object(
                h5,
                obj["content"],
                depth=depth + 1,
            )

        # Some MATLAB structs are wrapped in an additional group. Prefer
        # semantically relevant fields and otherwise avoid traversing gaze,
        # EEG, or other large fields accidentally.
        for key in ("word", "words", "text"):
            if key in obj:
                return _decode_text_object(
                    h5,
                    obj[key],
                    depth=depth + 1,
                )
        return []

    if not isinstance(obj, h5py.Dataset):
        return []

    array = np.asarray(obj[()])

    # A direct MATLAB char array is decoded as a single word/string.
    if np.issubdtype(array.dtype, np.integer):
        value = decode_matlab_char(obj).replace("\x00", "")
        return [value]

    if array.dtype.kind in {"S", "U"}:
        flattened = array.ravel(order="F")
        value = "".join(
            (
                item.decode("utf-8", errors="replace")
                if isinstance(item, bytes)
                else str(item)
            )
            for item in flattened
        ).replace("\x00", "")
        return [value]

    flattened = array.ravel(order="C")
    contains_reference = any(
        isinstance(value, h5py.Reference)
        for value in flattened
    )

    if contains_reference:
        texts: list[str] = []

        for value in flattened:
            if not (
                isinstance(value, h5py.Reference)
                and bool(value)
            ):
                # Keep one placeholder for an empty cell so that word indices
                # remain aligned with wordbounds whenever possible.
                texts.append("")
                continue

            decoded = _decode_text_object(
                h5,
                h5[value],
                depth=depth + 1,
            )
            if not decoded:
                texts.append("")
            else:
                texts.extend(decoded)

        return texts

    # Numeric non-character datasets are not word text.
    return []


def read_sentence_words(
    h5: h5py.File,
    sentence_data: h5py.Group,
    sentence_index: int,
) -> list[str]:
    word_container = sentence_data["word"]
    if not isinstance(word_container, h5py.Dataset):
        raise TypeError("sentenceData/word is not a dataset.")

    sentence_reference = indexed_reference(
        word_container,
        sentence_index,
    )
    if sentence_reference is None:
        return []

    sentence_word_object = h5[sentence_reference]
    words = _decode_text_object(
        h5,
        sentence_word_object,
    )

    # Normalize MATLAB strings without altering the number or order of words.
    return [
        str(word).replace("\x00", "").strip()
        for word in words
    ]



def convert_layout(
    raw_bounds: np.ndarray,
    layout: BoundsLayout,
) -> np.ndarray:
    if raw_bounds.size == 0:
        return np.empty((0, 4), dtype=float)

    return np.column_stack(
        [
            raw_bounds[:, layout.x_min_col],
            raw_bounds[:, layout.y_min_col],
            raw_bounds[:, layout.x_max_col],
            raw_bounds[:, layout.y_max_col],
        ]
    )


def layout_score_for_sentence(
    converted: np.ndarray,
) -> dict[str, float]:
    if converted.size == 0:
        return {
            "n_boxes": 0,
            "positive_fraction": math.nan,
            "finite_fraction": math.nan,
            "median_width": math.nan,
            "median_height": math.nan,
            "plausible_size_fraction": math.nan,
            "progression_fraction": math.nan,
            "score": -math.inf,
        }

    x_min, y_min, x_max, y_max = converted.T
    width = x_max - x_min
    height = y_max - y_min

    finite = np.all(np.isfinite(converted), axis=1)
    positive = finite & (width > 0) & (height > 0)
    finite_fraction = float(np.mean(finite))
    positive_fraction = float(np.mean(positive))

    if np.any(positive):
        positive_width = width[positive]
        positive_height = height[positive]
        median_width = float(np.median(positive_width))
        median_height = float(np.median(positive_height))
        plausible_size = (
            (positive_width >= 2)
            & (positive_width <= 600)
            & (positive_height >= 5)
            & (positive_height <= 200)
        )
        plausible_size_fraction = float(
            np.mean(plausible_size)
        )
    else:
        median_width = math.nan
        median_height = math.nan
        plausible_size_fraction = 0.0

    centers_x = (x_min + x_max) / 2.0
    centers_y = (y_min + y_max) / 2.0

    if len(converted) >= 2 and math.isfinite(median_height):
        dx = np.diff(centers_x)
        dy = np.abs(np.diff(centers_y))
        same_line = dy <= max(3.0, median_height * 0.60)
        if np.any(same_line):
            progression_fraction = float(
                np.mean(dx[same_line] >= -2.0)
            )
        else:
            progression_fraction = 0.5
    else:
        progression_fraction = 0.5

    aspect_bonus = 0.0
    if (
        math.isfinite(median_width)
        and math.isfinite(median_height)
        and median_height > 0
    ):
        ratio = median_width / median_height
        if 0.25 <= ratio <= 20:
            aspect_bonus = 1.0

    score = (
        100.0 * positive_fraction
        + 20.0 * finite_fraction
        + 20.0 * plausible_size_fraction
        + 10.0 * progression_fraction
        + aspect_bonus
    )

    return {
        "n_boxes": int(len(converted)),
        "positive_fraction": positive_fraction,
        "finite_fraction": finite_fraction,
        "median_width": median_width,
        "median_height": median_height,
        "plausible_size_fraction": plausible_size_fraction,
        "progression_fraction": progression_fraction,
        "score": score,
    }


def infer_layout(
    raw_arrays: list[np.ndarray],
) -> tuple[BoundsLayout, pd.DataFrame]:
    audit_rows: list[dict[str, Any]] = []

    for layout in LAYOUTS:
        sentence_scores: list[dict[str, float]] = []

        for raw_bounds in raw_arrays:
            sentence_scores.append(
                layout_score_for_sentence(
                    convert_layout(raw_bounds, layout)
                )
            )

        score_values = np.asarray(
            [row["score"] for row in sentence_scores],
            dtype=float,
        )
        finite_scores = score_values[np.isfinite(score_values)]

        def mean_field(name: str) -> float:
            values = np.asarray(
                [row[name] for row in sentence_scores],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            return (
                float(np.mean(values))
                if values.size
                else math.nan
            )

        audit_rows.append(
            {
                "layout": layout.name,
                "n_sentences_audited": len(sentence_scores),
                "mean_score": (
                    float(np.mean(finite_scores))
                    if finite_scores.size
                    else -math.inf
                ),
                "mean_positive_fraction": mean_field(
                    "positive_fraction"
                ),
                "mean_finite_fraction": mean_field(
                    "finite_fraction"
                ),
                "mean_plausible_size_fraction": mean_field(
                    "plausible_size_fraction"
                ),
                "mean_progression_fraction": mean_field(
                    "progression_fraction"
                ),
                "mean_median_width": mean_field(
                    "median_width"
                ),
                "mean_median_height": mean_field(
                    "median_height"
                ),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(
        "mean_score",
        ascending=False,
    ).reset_index(drop=True)

    selected_name = str(audit.iloc[0]["layout"])
    selected = next(
        layout
        for layout in LAYOUTS
        if layout.name == selected_name
    )
    audit["selected"] = audit["layout"].eq(selected_name)
    return selected, audit


def assign_line_information(
    boxes: np.ndarray,
) -> pd.DataFrame:
    if boxes.size == 0:
        return pd.DataFrame(
            columns=[
                "line_index",
                "line_word_index",
                "line_word_count",
                "line_index_norm",
                "line_progress",
            ]
        )

    x_min, y_min, x_max, y_max = boxes.T
    centers_x = (x_min + x_max) / 2.0
    centers_y = (y_min + y_max) / 2.0
    heights = y_max - y_min
    finite_heights = heights[
        np.isfinite(heights) & (heights > 0)
    ]
    median_height = (
        float(np.median(finite_heights))
        if finite_heights.size
        else 20.0
    )
    threshold = max(3.0, median_height * 0.60)

    sorted_indices = np.argsort(centers_y)
    line_centers: list[float] = []
    line_members: list[list[int]] = []

    for word_index in sorted_indices:
        y_value = float(centers_y[word_index])

        if not line_centers:
            line_centers.append(y_value)
            line_members.append([int(word_index)])
            continue

        distances = np.abs(
            np.asarray(line_centers, dtype=float) - y_value
        )
        nearest = int(np.argmin(distances))

        if distances[nearest] <= threshold:
            line_members[nearest].append(int(word_index))
            line_centers[nearest] = float(
                np.mean(centers_y[line_members[nearest]])
            )
        else:
            line_centers.append(y_value)
            line_members.append([int(word_index)])

    line_order = np.argsort(np.asarray(line_centers))
    ordered_members = [line_members[index] for index in line_order]
    n_lines = len(ordered_members)

    result = pd.DataFrame(
        index=np.arange(len(boxes)),
        columns=[
            "line_index",
            "line_word_index",
            "line_word_count",
            "line_index_norm",
            "line_progress",
        ],
        dtype=float,
    )

    for line_index, members in enumerate(ordered_members):
        members_sorted = sorted(
            members,
            key=lambda index: centers_x[index],
        )
        count = len(members_sorted)

        for position, word_index in enumerate(members_sorted):
            result.loc[word_index, "line_index"] = line_index
            result.loc[word_index, "line_word_index"] = position
            result.loc[word_index, "line_word_count"] = count
            result.loc[word_index, "line_index_norm"] = (
                line_index / (n_lines - 1)
                if n_lines > 1
                else 0.0
            )
            result.loc[word_index, "line_progress"] = (
                position / (count - 1)
                if count > 1
                else 0.0
            )

    return result


def preview_sentence(
    word_rows: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if word_rows.empty:
        return

    figure, axis = plt.subplots(figsize=(14, 6))

    for row in word_rows.itertuples(index=False):
        rectangle = Rectangle(
            (row.x_min, row.y_min),
            row.width,
            row.height,
            fill=False,
            linewidth=1.0,
        )
        axis.add_patch(rectangle)
        axis.text(
            row.x_center,
            row.y_center,
            str(row.word_text),
            ha="center",
            va="center",
            fontsize=7,
        )

    axis.set_xlim(
        float(word_rows["x_min"].min()) - 30,
        float(word_rows["x_max"].max()) + 30,
    )
    axis.set_ylim(
        float(word_rows["y_min"].min()) - 30,
        float(word_rows["y_max"].max()) + 30,
    )
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Screen x (pixels)")
    axis.set_ylabel("Screen y (pixels)")
    axis.set_title(title)
    axis.grid(True, alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def normalize_sentence_for_match(text: str) -> str:
    """Normalize harmless whitespace differences for cross-subject checks."""
    return " ".join(str(text).replace("\x00", " ").split())


def discover_task_result_files(
    raw_root: Path,
    task: str,
    template_subject: str,
) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []

    for subject_dir in sorted(raw_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name.upper()
        exact = subject_dir / f"results{subject}_{task}.mat"
        if exact.is_file():
            candidates.append((subject, exact))
            continue

        matches = sorted(subject_dir.glob(f"results*_{task}.mat"))
        for match in matches:
            candidates.append((subject, match))
            break

    candidates.sort(
        key=lambda item: (
            item[0] != template_subject.upper(),
            item[0],
        )
    )
    return candidates


def validate_sentence_data(
    h5: h5py.File,
    mat_path: Path,
    task: str,
) -> tuple[h5py.Group, int]:
    if "sentenceData" not in h5:
        raise KeyError(f"No sentenceData in {mat_path}")

    sentence_data = h5["sentenceData"]
    if not isinstance(sentence_data, h5py.Group):
        raise TypeError("sentenceData is not a group.")

    for field in ("content", "word", "wordbounds"):
        if field not in sentence_data:
            raise KeyError(
                f"sentenceData/{field} is missing in {mat_path}"
            )

    sentence_count = int(
        np.asarray(sentence_data["content"][()]).size
    )
    expected = EXPECTED_SENTENCES[task]
    if sentence_count != expected:
        raise ValueError(
            f"{task} sentence count={sentence_count}, "
            f"expected={expected}, file={mat_path}"
        )
    return sentence_data, sentence_count


def extract_task(
    raw_root: Path,
    task: str,
    template_subject: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_files = discover_task_result_files(
        raw_root=raw_root,
        task=task,
        template_subject=template_subject,
    )
    if not candidate_files:
        raise FileNotFoundError(
            f"No result MAT files found for task={task} "
            f"under {raw_root}"
        )

    primary_matches = [
        item
        for item in candidate_files
        if item[0] == template_subject.upper()
    ]
    if not primary_matches:
        raise FileNotFoundError(
            raw_root
            / template_subject.upper()
            / f"results{template_subject.upper()}_{task}.mat"
        )

    primary_subject, primary_path = primary_matches[0]
    if not h5py.is_hdf5(primary_path):
        raise ValueError(
            f"Expected MATLAB v7.3/HDF5 file: {primary_path}"
        )

    word_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    sentence_texts: list[str] = []
    sentence_words: list[list[str]] = []
    raw_arrays: list[np.ndarray] = []
    bounds_sources: list[str] = []

    # Text and word order are taken from the requested template subject when
    # available. Missing word lists are recovered together with wordbounds
    # from a matching fallback subject.
    with h5py.File(primary_path, "r") as h5:
        sentence_data, sentence_count = validate_sentence_data(
            h5=h5,
            mat_path=primary_path,
            task=task,
        )

        for sentence_index in range(sentence_count):
            sentence_texts.append(
                read_referenced_text(
                    h5,
                    sentence_data["content"],
                    sentence_index,
                )
            )
            sentence_words.append(
                read_sentence_words(
                    h5,
                    sentence_data,
                    sentence_index,
                )
            )
            raw_bounds = read_sentence_bounds_raw(
                h5,
                sentence_data,
                sentence_index,
            )
            raw_arrays.append(raw_bounds)
            bounds_sources.append(
                primary_subject if raw_bounds.size else ""
            )

    initially_missing = {
        index
        for index, bounds in enumerate(raw_arrays)
        if bounds.size == 0
    }
    print(
        f"{task}: {len(initially_missing)} sentence(s) have empty "
        f"wordbounds in {primary_subject}; searching fallback subjects."
    )

    # Word locations are stimulus properties. Fill a primary subject's empty
    # wordbounds from another subject who saw the same task and sentence.
    remaining = set(initially_missing)
    for fallback_subject, fallback_path in candidate_files:
        if not remaining:
            break
        if fallback_subject == primary_subject:
            continue
        if not h5py.is_hdf5(fallback_path):
            continue

        filled_now = 0
        try:
            with h5py.File(fallback_path, "r") as h5:
                sentence_data, fallback_count = validate_sentence_data(
                    h5=h5,
                    mat_path=fallback_path,
                    task=task,
                )
                if fallback_count != sentence_count:
                    continue

                for sentence_index in sorted(remaining):
                    fallback_text = read_referenced_text(
                        h5,
                        sentence_data["content"],
                        sentence_index,
                    )
                    if normalize_sentence_for_match(
                        fallback_text
                    ) != normalize_sentence_for_match(
                        sentence_texts[sentence_index]
                    ):
                        issues.append(
                            {
                                "template_subject": template_subject,
                                "task": task,
                                "task_sentence_index": sentence_index,
                                "issue": (
                                    "fallback_sentence_text_mismatch"
                                ),
                                "n_words": len(
                                    sentence_words[sentence_index]
                                ),
                                "n_bounds": math.nan,
                                "detail": (
                                    f"fallback_subject="
                                    f"{fallback_subject}"
                                ),
                            }
                        )
                        continue

                    fallback_words = read_sentence_words(
                        h5,
                        sentence_data,
                        sentence_index,
                    )
                    candidate_bounds = read_sentence_bounds_raw(
                        h5,
                        sentence_data,
                        sentence_index,
                    )
                    if candidate_bounds.size == 0:
                        continue

                    primary_word_count = len(
                        sentence_words[sentence_index]
                    )
                    fallback_word_count = len(fallback_words)
                    bounds_count = len(candidate_bounds)

                    # In some ZuCo result files, an unavailable trial is
                    # represented by both an empty word list and empty
                    # wordbounds. In that case, recover the word list and the
                    # matching bounds together from a fallback subject.
                    if primary_word_count == 0:
                        counts_match = (
                            fallback_word_count > 0
                            and bounds_count == fallback_word_count
                        )
                    else:
                        counts_match = (
                            fallback_word_count == primary_word_count
                            and bounds_count == primary_word_count
                        )

                    if not counts_match:
                        issues.append(
                            {
                                "template_subject": template_subject,
                                "task": task,
                                "task_sentence_index": sentence_index,
                                "issue": (
                                    "fallback_word_count_mismatch"
                                ),
                                "n_words": primary_word_count,
                                "n_bounds": bounds_count,
                                "detail": (
                                    f"fallback_subject="
                                    f"{fallback_subject}; "
                                    f"fallback_words="
                                    f"{fallback_word_count}"
                                ),
                            }
                        )
                        continue

                    if primary_word_count == 0:
                        sentence_words[sentence_index] = fallback_words

                    raw_arrays[sentence_index] = candidate_bounds
                    bounds_sources[sentence_index] = fallback_subject
                    filled_now += 1

            if filled_now:
                remaining = {
                    index
                    for index in remaining
                    if raw_arrays[index].size == 0
                }
                print(
                    f"{task}: filled {filled_now} sentence(s) "
                    f"from {fallback_subject}; remaining={len(remaining)}."
                )
        except (OSError, KeyError, TypeError, ValueError) as error:
            issues.append(
                {
                    "template_subject": template_subject,
                    "task": task,
                    "task_sentence_index": -1,
                    "issue": "fallback_file_skipped",
                    "n_words": math.nan,
                    "n_bounds": math.nan,
                    "detail": (
                        f"{fallback_subject}: {fallback_path}: {error}"
                    ),
                }
            )

    for sentence_index in sorted(initially_missing - remaining):
        issues.append(
            {
                "template_subject": template_subject,
                "task": task,
                "task_sentence_index": sentence_index,
                "issue": "primary_empty_wordbounds_filled",
                "n_words": len(sentence_words[sentence_index]),
                "n_bounds": len(raw_arrays[sentence_index]),
                "detail": (
                    f"bounds_source_subject="
                    f"{bounds_sources[sentence_index]}"
                ),
            }
        )

    for sentence_index in sorted(remaining):
        issues.append(
            {
                "template_subject": template_subject,
                "task": task,
                "task_sentence_index": sentence_index,
                "issue": "wordbounds_unavailable_all_subjects",
                "n_words": len(sentence_words[sentence_index]),
                "n_bounds": 0,
                "detail": "",
            }
        )

    nonempty_arrays = [
        bounds for bounds in raw_arrays if bounds.size
    ]
    if not nonempty_arrays:
        raise ValueError(
            f"No usable wordbounds were found for task={task}."
        )

    selected_layout, audit = infer_layout(nonempty_arrays)
    audit.insert(0, "task", task)
    audit.insert(0, "template_subject", template_subject)

    print(
        f"{task}: selected wordbounds layout = "
        f"{selected_layout.name}"
    )

    for sentence_index in range(sentence_count):
        sentence_text = sentence_texts[sentence_index]
        words = sentence_words[sentence_index]
        raw_bounds = raw_arrays[sentence_index]
        source_subject = bounds_sources[sentence_index]
        boxes = convert_layout(
            raw_bounds,
            selected_layout,
        )

        original_word_count = len(words)
        original_box_count = len(boxes)
        used_count = min(
            original_word_count,
            original_box_count,
        )

        if original_word_count != original_box_count:
            issues.append(
                {
                    "template_subject": template_subject,
                    "task": task,
                    "task_sentence_index": sentence_index,
                    "issue": "word_bounds_count_mismatch",
                    "n_words": original_word_count,
                    "n_bounds": original_box_count,
                    "detail": (
                        "Rows were aligned by index and truncated "
                        f"to {used_count}; "
                        f"bounds_source_subject={source_subject or 'none'}."
                    ),
                }
            )

        words_used = words[:used_count]
        boxes = boxes[:used_count]
        raw_bounds = raw_bounds[:used_count]

        if used_count == 0:
            sentence_rows.append(
                {
                    "template_subject": template_subject,
                    "bounds_source_subject": "",
                    "task": task,
                    "task_sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "has_word_aoi": False,
                    "n_words_text": original_word_count,
                    "n_words": 0,
                    "n_lines": 0,
                    "sentence_x_min": math.nan,
                    "sentence_y_min": math.nan,
                    "sentence_x_max": math.nan,
                    "sentence_y_max": math.nan,
                    "sentence_width": math.nan,
                    "sentence_height": math.nan,
                    "median_word_width": math.nan,
                    "median_word_height": math.nan,
                    "bounds_layout": selected_layout.name,
                }
            )
            continue

        x_min, y_min, x_max, y_max = boxes.T
        width = x_max - x_min
        height = y_max - y_min
        valid_box = (
            np.all(np.isfinite(boxes), axis=1)
            & (width > 0)
            & (height > 0)
        )

        if not np.all(valid_box):
            bad_indices = np.flatnonzero(~valid_box)
            issues.append(
                {
                    "template_subject": template_subject,
                    "task": task,
                    "task_sentence_index": sentence_index,
                    "issue": "invalid_word_box",
                    "n_words": original_word_count,
                    "n_bounds": original_box_count,
                    "detail": (
                        "Invalid indices: "
                        + ",".join(
                            str(int(index))
                            for index in bad_indices[:20]
                        )
                    ),
                }
            )

        line_info = assign_line_information(boxes)
        n_lines = int(
            pd.to_numeric(
                line_info["line_index"],
                errors="coerce",
            ).max()
            + 1
        )

        sentence_rows.append(
            {
                "template_subject": template_subject,
                "bounds_source_subject": source_subject,
                "task": task,
                "task_sentence_index": sentence_index,
                "sentence_text": sentence_text,
                "has_word_aoi": True,
                "n_words_text": original_word_count,
                "n_words": used_count,
                "n_lines": n_lines,
                "sentence_x_min": float(np.nanmin(x_min)),
                "sentence_y_min": float(np.nanmin(y_min)),
                "sentence_x_max": float(np.nanmax(x_max)),
                "sentence_y_max": float(np.nanmax(y_max)),
                "sentence_width": float(
                    np.nanmax(x_max) - np.nanmin(x_min)
                ),
                "sentence_height": float(
                    np.nanmax(y_max) - np.nanmin(y_min)
                ),
                "median_word_width": float(
                    np.nanmedian(width)
                ),
                "median_word_height": float(
                    np.nanmedian(height)
                ),
                "bounds_layout": selected_layout.name,
            }
        )

        for word_index in range(used_count):
            line_row = line_info.loc[word_index]
            word_rows.append(
                {
                    "template_subject": template_subject,
                    "bounds_source_subject": source_subject,
                    "task": task,
                    "task_sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "word_index": word_index,
                    "word_text": words_used[word_index],
                    "raw_bound_0": raw_bounds[word_index, 0],
                    "raw_bound_1": raw_bounds[word_index, 1],
                    "raw_bound_2": raw_bounds[word_index, 2],
                    "raw_bound_3": raw_bounds[word_index, 3],
                    "x_min": x_min[word_index],
                    "y_min": y_min[word_index],
                    "x_max": x_max[word_index],
                    "y_max": y_max[word_index],
                    "x_center": (
                        x_min[word_index] + x_max[word_index]
                    )
                    / 2.0,
                    "y_center": (
                        y_min[word_index] + y_max[word_index]
                    )
                    / 2.0,
                    "width": width[word_index],
                    "height": height[word_index],
                    "word_progress": (
                        word_index / (used_count - 1)
                        if used_count > 1
                        else 0.0
                    ),
                    "line_index": int(
                        line_row["line_index"]
                    ),
                    "line_word_index": int(
                        line_row["line_word_index"]
                    ),
                    "line_word_count": int(
                        line_row["line_word_count"]
                    ),
                    "line_index_norm": float(
                        line_row["line_index_norm"]
                    ),
                    "line_progress": float(
                        line_row["line_progress"]
                    ),
                    "bounds_layout": selected_layout.name,
                    "valid_box": bool(
                        valid_box[word_index]
                    ),
                }
            )

    words_frame = pd.DataFrame(word_rows)
    sentences_frame = pd.DataFrame(sentence_rows)
    issues_frame = pd.DataFrame(issues)

    first_available = sentences_frame[
        sentences_frame["has_word_aoi"]
    ].sort_values("task_sentence_index")
    if not first_available.empty:
        first_index = int(
            first_available.iloc[0]["task_sentence_index"]
        )
        preview_sentence(
            words_frame[
                words_frame["task_sentence_index"].eq(
                    first_index
                )
            ],
            title=(
                f"{task} sentence {first_index}: "
                "word AOI template"
            ),
            output_path=(
                output_dir
                / f"preview_{task}_sentence_000.png"
            ),
        )

    available_sentences = sentences_frame[
        sentences_frame["has_word_aoi"]
    ]
    if not available_sentences.empty:
        max_line_sentence = int(
            available_sentences.sort_values(
                ["n_lines", "n_words"],
                ascending=False,
            ).iloc[0]["task_sentence_index"]
        )
        preview_sentence(
            words_frame[
                words_frame["task_sentence_index"].eq(
                    max_line_sentence
                )
            ],
            title=(
                f"{task} sentence {max_line_sentence}: "
                "maximum-line AOI template"
            ),
            output_path=(
                output_dir
                / f"preview_{task}_max_lines.png"
            ),
        )

    return words_frame, sentences_frame, audit, issues_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template-subject",
        default="YAC",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw/zuco2",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/zuco2/"
            "soft_aoi_template"
        ),
    )
    args = parser.parse_args()

    subject = args.template_subject.upper()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_words: list[pd.DataFrame] = []
    all_sentences: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    all_issues: list[pd.DataFrame] = []

    for task in ("NR", "TSR"):
        (
            word_frame,
            sentence_frame,
            audit_frame,
            issue_frame,
        ) = extract_task(
            raw_root=raw_root,
            task=task,
            template_subject=subject,
            output_dir=output_dir,
        )
        all_words.append(word_frame)
        all_sentences.append(sentence_frame)
        all_audits.append(audit_frame)
        if not issue_frame.empty:
            all_issues.append(issue_frame)

    words = pd.concat(all_words, ignore_index=True)
    sentences = pd.concat(
        all_sentences,
        ignore_index=True,
    )
    audits = pd.concat(all_audits, ignore_index=True)
    issues = (
        pd.concat(all_issues, ignore_index=True)
        if all_issues
        else pd.DataFrame(
            columns=[
                "template_subject",
                "task",
                "task_sentence_index",
                "issue",
                "n_words",
                "n_bounds",
                "detail",
            ]
        )
    )

    if len(sentences) != sum(EXPECTED_SENTENCES.values()):
        raise ValueError(
            f"Sentence template rows={len(sentences)}, "
            f"expected={sum(EXPECTED_SENTENCES.values())}."
        )

    sentence_keys = [
        "task",
        "task_sentence_index",
    ]
    if sentences.duplicated(sentence_keys).any():
        raise ValueError(
            "Duplicate sentence template keys were found."
        )

    word_path = (
        output_dir / "zuco_soft_aoi_word_template.csv"
    )
    sentence_path = (
        output_dir / "zuco_soft_aoi_sentence_template.csv"
    )
    audit_path = (
        output_dir / "zuco_wordbounds_layout_audit.csv"
    )
    issues_path = (
        output_dir / "zuco_soft_aoi_extraction_issues.csv"
    )

    words.to_csv(
        word_path,
        index=False,
        encoding="utf-8-sig",
    )
    sentences.to_csv(
        sentence_path,
        index=False,
        encoding="utf-8-sig",
    )
    audits.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )
    issues.to_csv(
        issues_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== SOFT-AOI TEMPLATE SUMMARY ===")
    print(f"template subject: {subject}")
    print(f"sentences: {len(sentences)}")
    print(f"word AOIs: {len(words)}")
    valid_box_count = (
        int(words["valid_box"].sum())
        if not words.empty
        else 0
    )
    print(
        "valid boxes: "
        f"{valid_box_count}/{len(words)}"
    )
    print(
        "sentences with AOI: "
        f"{int(sentences['has_word_aoi'].sum())}/"
        f"{len(sentences)}"
    )
    print(f"issues: {len(issues)}")

    print("\n=== TASK SUMMARY ===")
    print(
        sentences.groupby("task").agg(
            n_sentences=("task_sentence_index", "size"),
            total_words=("n_words", "sum"),
            median_words=("n_words", "median"),
            max_lines=("n_lines", "max"),
            median_word_width=("median_word_width", "median"),
            median_word_height=("median_word_height", "median"),
        ).to_string()
    )

    print("\n=== SELECTED LAYOUTS ===")
    print(
        audits[audits["selected"]][
            [
                "task",
                "layout",
                "mean_score",
                "mean_positive_fraction",
                "mean_plausible_size_fraction",
                "mean_median_width",
                "mean_median_height",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")
    print(f"  {word_path}")
    print(f"  {sentence_path}")
    print(f"  {audit_path}")
    print(f"  {issues_path}")
    print(
        f"  {output_dir / 'preview_NR_sentence_000.png'}"
    )
    print(
        f"  {output_dir / 'preview_TSR_sentence_000.png'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
