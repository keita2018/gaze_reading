#!/usr/bin/env python
"""Generate WebQAmGaze Soft-AOI and transferred official-formula features.

Stage 3C performs no model fitting and no prediction. It finalizes all external
feature vectors before any frozen ZuCo model is applied.

AOI selection
-------------
The official WebQAmGaze notebook specifies the following deterministic mapping:

    Stage 3B content-based trial mapping
        -> restrict source_file to target_dataframe_{set_name}.csv
        -> restrict source_row_index to rows whose is_word flag is true
        -> deduplicate the resulting word rectangles

Stage 3B already established 101/101 content-based trial coverage. This
set-specific source-file and source-row rule resolves duplicate/tied AOI
candidates without re-guessing the repository's text-name encoding and without
looking at NR/IS labels or classification performance. The target dataframes
were generated for a 1280 x 720 browser.

Soft-AOI
--------
The script imports the existing ZuCo extractor
``extract_zuco_soft_aoi_features.py`` and calls its original
``extract_trial_features`` implementation with the frozen parameters:

- 30 Hz
- sigma = (2/3) * median word height
- active threshold = 0.1

WebQAmGaze has no pupil signal. For the extractor's sample-validity input,
``valid_basic`` is therefore defined as valid x/y. Long Stage-2 interpolation
gaps are restored as invalid rows on the 30 Hz grid before feature extraction,
so temporal/order features are not linked across missing segments.

Transferred official-formula condition
--------------------------------------
The benchmark's nine-dimensional sentence vector is constructed in this order:

1. omission_rate
2. fixation_number
3. reading_speed
4. mean_sacc_dur
5. max_sacc_dur
6. mean_sacc_velocity
7. max_sacc_velocity
8. mean_sacc_amplitude
9. max_sacc_amplitude

The three fixation formulas follow the published definitions. Saccades are not
EyeLink-detected events: they are approximate inter-fixation transitions derived
from consecutive WebQAmGaze filtered fixations. Therefore this condition MUST be
reported as a "transferred event-based official-feature-formula condition", not
as an identical EyeLink/ZuCo official condition.

For scale compatibility only, WebQAmGaze's 1280 x 720 coordinates are mapped to
the ZuCo 800 x 600 display, then converted to approximate visual degrees using
the fixed ZuCo setup (400 x 298.9 mm active display at 680 mm viewing distance).
This does not claim to recover each webcam participant's true visual angle.

No external label is used for AOI selection, feature tuning, imputation, scaling,
threshold selection, or calibration.

Outputs
-------
set_specific_word_aoi.csv
set_specific_word_aoi_audit.csv
soft_full_features_primary.csv
soft_compact_features_primary.csv
soft_feature_audit.csv
official_formula_features_descriptive.csv
official_features_primary.csv
official_feature_audit.csv
frozen_schema_stage3c.csv
stage3c_readiness.csv
stage3c_manifest.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import extract_zuco_soft_aoi_features as soft
except ImportError as exc:
    raise ImportError(
        "Could not import src/data/extract_zuco_soft_aoi_features.py. "
        "Place this Stage 3C script in src/data beside the existing ZuCo "
        "Soft-AOI extractor."
    ) from exc


FORMULA_NAMES = [
    "omission_rate",
    "fixation_number",
    "reading_speed",
    "mean_sacc_dur",
    "max_sacc_dur",
    "mean_sacc_velocity",
    "max_sacc_velocity",
    "mean_sacc_amplitude",
    "max_sacc_amplitude",
]

TEXT_ALIASES = {
    "text_name", "textname", "trial_name", "trialname",
    "text_id", "textid", "trial_id", "trialid",
}
IS_WORD_ALIASES = {"is_word", "isword", "word_flag", "wordflag"}
TARGET_NAME_ALIASES = {
    "target_name", "targetname", "word", "token", "target",
}
XMIN_ALIASES = {"x_min", "xmin", "left", "x_left", "xleft"}
YMIN_ALIASES = {"y_min", "ymin", "top", "y_top", "ytop"}
XMAX_ALIASES = {"x_max", "xmax", "right", "x_right", "xright"}
YMAX_ALIASES = {"y_max", "ymax", "bottom", "y_bottom", "ybottom"}
X_ALIASES = {"x", "left", "x_pos", "xpos"}
Y_ALIASES = {"y", "top", "y_pos", "ypos"}
WIDTH_ALIASES = {"width", "w", "box_width", "boxwidth"}
HEIGHT_ALIASES = {"height", "h", "box_height", "boxheight"}
BBOX_ALIASES = {"bbox", "bounding_box", "boundingbox", "bounds", "rect", "box"}

SOURCE_BROWSER_WIDTH = 1280.0
SOURCE_BROWSER_HEIGHT = 720.0
ZUCO_WIDTH_PX = 800.0
ZUCO_HEIGHT_PX = 600.0
ZUCO_ACTIVE_WIDTH_MM = 400.0
ZUCO_ACTIVE_HEIGHT_MM = 298.9
ZUCO_VIEWING_DISTANCE_MM = 680.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def canonical_trial_name(value: Any) -> str:
    """Canonicalize repository trial/text identifiers conservatively.

    The target dataframes can store one text as values such as
    ``a_Text_4_qa_0`` or ``q_before_a_Text_4`` while the participant manifest
    stores ``a_Text_4``. Only known wrappers, file extensions, and question
    occurrence suffixes are removed. The set-specific target dataframe remains
    fixed before this comparison, so this cannot select an AOI from another
    experiment set.
    """
    parsed = safe_literal(value)
    if isinstance(parsed, (list, tuple)) and parsed:
        parsed = parsed[0]
    elif isinstance(parsed, dict):
        for key in (
            "text_name", "trial_name", "text", "trial",
            "name", "stimulus", "image",
        ):
            if key in parsed:
                parsed = parsed[key]
                break

    raw = str(parsed).strip()
    if not raw:
        return ""
    raw = raw.split("?", 1)[0].split("#", 1)[0]
    raw = Path(raw.replace("\\", "/")).name
    raw = re.sub(
        r"\.(?:png|jpe?g|gif|bmp|webp|html?|js|json|csv)$",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(
        r"^(?:q_before_|q_after_|before_|after_)",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(
        r"_(?:qa|question|target)_?\d+$",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    return normalize_name(raw)


def column_by_alias(columns: Iterable[Any], aliases: set[str]) -> str:
    mapping = {normalize_name(column): str(column) for column in columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in mapping:
            return mapping[key]
    return ""


def finite_number(value: Any) -> float:
    """Parse scalar coordinate values robustly.

    Target-dataframe cells may be numeric, numeric strings, one-element
    containers, or CSS-like strings such as ``"123.4px"``.
    """
    parsed = safe_literal(value) if isinstance(value, str) else value

    if isinstance(parsed, np.ndarray):
        if parsed.size != 1:
            return math.nan
        parsed = parsed.reshape(-1)[0]
    elif isinstance(parsed, (list, tuple)):
        if len(parsed) != 1:
            return math.nan
        parsed = parsed[0]
    elif isinstance(parsed, dict):
        if len(parsed) != 1:
            return math.nan
        parsed = next(iter(parsed.values()))

    if isinstance(parsed, str):
        candidate = parsed.strip().replace(",", "")
        css_match = re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:px)?",
            candidate,
            flags=re.IGNORECASE,
        )
        if css_match is None:
            return math.nan
        candidate = re.sub(r"px$", "", candidate, flags=re.IGNORECASE)
        parsed = candidate

    try:
        number = float(parsed)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def to_bool(series: pd.Series) -> pd.Series:
    """Convert bool, numeric 0/1, and common text flags safely.

    WebQAmGaze target CSVs may deserialize ``is_word`` as float values
    (0.0/1.0). The previous implementation converted them to strings and
    recognized "1" but not "1.0", causing every row to become False.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.fillna(0).ne(0)

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )
    direct = normalized.isin(
        {
            "true",
            "1",
            "1.0",
            "yes",
            "y",
            "t",
            "word",
        }
    )

    # Also accept numeric-looking strings such as "1.00".
    numeric = pd.to_numeric(normalized, errors="coerce")
    numeric_true = numeric.fillna(0).ne(0)
    return (direct | numeric_true).fillna(False)


def safe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{(" or text[-1] not in "]})":
        return value
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def parse_target_word(value: Any) -> str:
    parsed = safe_literal(value)
    if isinstance(parsed, (list, tuple)) and parsed:
        return str(parsed[0])
    if isinstance(parsed, dict):
        for key in ("word", "text", "target", "token"):
            if key in parsed:
                return str(parsed[key])
    return "" if value is None else str(value)


def parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    parsed = safe_literal(value)
    if isinstance(parsed, dict):
        mapping = {normalize_name(k): v for k, v in parsed.items()}
        xmin = next(
            (finite_number(mapping[normalize_name(k)]) for k in XMIN_ALIASES
             if normalize_name(k) in mapping),
            math.nan,
        )
        ymin = next(
            (finite_number(mapping[normalize_name(k)]) for k in YMIN_ALIASES
             if normalize_name(k) in mapping),
            math.nan,
        )
        xmax = next(
            (finite_number(mapping[normalize_name(k)]) for k in XMAX_ALIASES
             if normalize_name(k) in mapping),
            math.nan,
        )
        ymax = next(
            (finite_number(mapping[normalize_name(k)]) for k in YMAX_ALIASES
             if normalize_name(k) in mapping),
            math.nan,
        )
        if all(map(math.isfinite, (xmin, ymin, xmax, ymax))):
            return xmin, ymin, xmax, ymax

        x = next(
            (finite_number(mapping[normalize_name(k)]) for k in X_ALIASES
             if normalize_name(k) in mapping),
            math.nan,
        )
        y = next(
            (finite_number(mapping[normalize_name(k)]) for k in Y_ALIASES
             if normalize_name(k) in mapping),
            math.nan,
        )
        width = next(
            (finite_number(mapping[normalize_name(k)]) for k in WIDTH_ALIASES
             if normalize_name(k) in mapping),
            math.nan,
        )
        height = next(
            (finite_number(mapping[normalize_name(k)]) for k in HEIGHT_ALIASES
             if normalize_name(k) in mapping),
            math.nan,
        )
        if all(map(math.isfinite, (x, y, width, height))):
            return x, y, x + width, y + height

    if isinstance(parsed, (list, tuple, np.ndarray)) and len(parsed) >= 4:
        values = [finite_number(parsed[index]) for index in range(4)]
        if all(map(math.isfinite, values)):
            a, b, c, d = values
            if c > a and d > b:
                return a, b, c, d
            if c > 0 and d > 0:
                return a, b, a + c, b + d
    return None


def rectangle_columns(frame: pd.DataFrame) -> dict[str, str]:
    columns = frame.columns
    return {
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


def row_rectangle(
    row: pd.Series,
    columns: dict[str, str],
) -> tuple[float, float, float, float] | None:
    """Read a rectangle using deterministic, row-level column priorities.

    Some WebQAmGaze target CSVs contain both:
      - left/top/right/bottom, and
      - x/y/width/height.

    Parent rows may populate the first representation while word rows populate
    the second. A single dataframe-level alias choice is therefore unsafe.
    Values are selected per row, preferring exact x/y columns over left/top for
    the width/height representation.
    """
    normalized_to_columns: dict[str, list[str]] = {}
    for column in row.index:
        normalized_to_columns.setdefault(
            normalize_name(column),
            [],
        ).append(str(column))

    def first_finite(preferred_names: Sequence[str]) -> float:
        for name in preferred_names:
            key = normalize_name(name)
            for actual_column in normalized_to_columns.get(key, []):
                value = finite_number(row[actual_column])
                if math.isfinite(value):
                    return value
        return math.nan

    # Representation 1: two corners.
    xmin = first_finite(
        ("x_min", "xmin", "left", "x_left", "xleft", "x0")
    )
    ymin = first_finite(
        ("y_min", "ymin", "top", "y_top", "ytop", "y0")
    )
    xmax = first_finite(
        ("x_max", "xmax", "right", "x_right", "xright", "x1")
    )
    ymax = first_finite(
        ("y_max", "ymax", "bottom", "y_bottom", "ybottom", "y1")
    )
    if (
        all(map(math.isfinite, (xmin, ymin, xmax, ymax)))
        and xmax > xmin
        and ymax > ymin
    ):
        return xmin, ymin, xmax, ymax

    # Representation 2: top-left plus size. Exact x/y are deliberately checked
    # before left/top because word rows commonly use x/y/width/height.
    x = first_finite(("x", "x_pos", "xpos", "x_position", "left"))
    y = first_finite(("y", "y_pos", "ypos", "y_position", "top"))
    width = first_finite(
        ("width", "w", "box_width", "boxwidth", "bbox_width", "bboxwidth")
    )
    height = first_finite(
        ("height", "h", "box_height", "boxheight", "bbox_height", "bboxheight")
    )
    if (
        all(map(math.isfinite, (x, y, width, height)))
        and width > 0
        and height > 0
    ):
        return x, y, x + width, y + height

    # Representation 3: a nested bbox cell.
    bbox_column = columns.get("bbox", "")
    if bbox_column and bbox_column in row.index:
        parsed = parse_bbox(row[bbox_column])
        if parsed is not None:
            xmin, ymin, xmax, ymax = parsed
            if xmax > xmin and ymax > ymin:
                return parsed
    return None

def find_target_dataframe(root: Path, set_name: str) -> Path:
    """Resolve a participant set variant to its base AOI dataframe.

    Participant metadata can append condition suffixes, for example:

        mturk_EN_v02_lab_cognition
        mturk_EN_v02_lab_control

    while the shared word-boundary resource is stored as:

        target_dataframe_mturk_EN_v02.csv

    Resolution order:
      1. exact filename;
      2. normalized exact set ID;
      3. longest underscore-delimited base-set prefix.

    This mapping uses repository naming only, never NR/IS labels or results.
    """
    target_root = root / "scripts" / "target_dataframes"
    if not target_root.is_dir():
        raise FileNotFoundError(target_root)

    exact = target_root / f"target_dataframe_{set_name}.csv"
    if exact.is_file():
        return exact

    paths = sorted(target_root.glob("target_dataframe_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No target_dataframe_*.csv files found under {target_root}"
        )

    file_prefix = "target_dataframe_"
    records: list[tuple[Path, str]] = []
    for path in paths:
        stem = path.stem
        if stem.lower().startswith(file_prefix):
            records.append((path, stem[len(file_prefix):]))

    requested_norm = normalize_name(set_name)
    normalized_exact = [
        path
        for path, base_set in records
        if normalize_name(base_set) == requested_norm
    ]
    if len(normalized_exact) == 1:
        return normalized_exact[0]
    if len(normalized_exact) > 1:
        raise ValueError(
            f"Multiple normalized-exact target dataframes match "
            f"set_name={set_name!r}: {normalized_exact}"
        )

    requested_lower = str(set_name).strip().lower()
    prefix_matches = [
        (path, base_set)
        for path, base_set in records
        if requested_lower.startswith(base_set.lower() + "_")
    ]
    if prefix_matches:
        longest_length = max(
            len(base_set) for _, base_set in prefix_matches
        )
        best = [
            (path, base_set)
            for path, base_set in prefix_matches
            if len(base_set) == longest_length
        ]
        if len(best) == 1:
            return best[0][0]
        raise ValueError(
            f"Ambiguous longest-prefix target dataframe resolution for "
            f"set_name={set_name!r}: {best}"
        )

    # Conservative explicit fallback for the repository's EN version IDs.
    version_match = re.match(
        r"^(.*?_EN_v\d+)(?:_|$)",
        str(set_name),
        flags=re.IGNORECASE,
    )
    if version_match:
        base_requested = version_match.group(1)
        version_matches = [
            path
            for path, base_set in records
            if normalize_name(base_set) == normalize_name(base_requested)
        ]
        if len(version_matches) == 1:
            return version_matches[0]
        if len(version_matches) > 1:
            raise ValueError(
                f"Multiple version-base target dataframes match "
                f"set_name={set_name!r}: {version_matches}"
            )

    available = sorted(base_set for _, base_set in records)
    raise FileNotFoundError(
        f"No target dataframe found for set_name={set_name!r}. "
        f"Available base sets={available}"
    )

def read_target_dataframe(path: Path) -> pd.DataFrame:
    attempts = [
        lambda: pd.read_csv(path, index_col=0, low_memory=False),
        lambda: pd.read_csv(path, low_memory=False),
    ]
    errors: list[str] = []
    for loader in attempts:
        try:
            frame = loader()
            if len(frame.columns):
                return frame
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise ValueError(f"Could not read {path}: {' || '.join(errors)}")


def derive_reading_order(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["x_center"] = (result["x_min"] + result["x_max"]) / 2.0
    result["y_center"] = (result["y_min"] + result["y_max"]) / 2.0
    result["width"] = result["x_max"] - result["x_min"]
    result["height"] = result["y_max"] - result["y_min"]

    positive_heights = result.loc[result["height"] > 0, "height"]
    median_height = (
        float(positive_heights.median()) if len(positive_heights) else math.nan
    )
    if not math.isfinite(median_height) or median_height <= 0:
        raise ValueError("AOI table has no positive word heights.")
    line_tolerance = max(1.0, median_height * 0.60)

    sorted_indices = result.sort_values(["y_center", "x_center"]).index
    line_centers: list[float] = []
    assignments: dict[Any, int] = {}

    for index in sorted_indices:
        y = float(result.at[index, "y_center"])
        if not line_centers:
            line_centers.append(y)
            assignments[index] = 0
            continue
        distances = np.abs(np.asarray(line_centers, dtype=float) - y)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= line_tolerance:
            assignments[index] = nearest
            line_centers[nearest] = (
                line_centers[nearest] + y
            ) / 2.0
        else:
            line_centers.append(y)
            assignments[index] = len(line_centers) - 1

    line_order = np.argsort(np.asarray(line_centers, dtype=float))
    line_remap = {
        int(old): int(new) for new, old in enumerate(line_order)
    }
    result["line_index"] = [
        line_remap[assignments[index]] for index in result.index
    ]

    order = result.sort_values(
        ["line_index", "x_center", "y_center"]
    ).index
    word_map = {index: position for position, index in enumerate(order)}
    result["word_index"] = [word_map[index] for index in result.index]
    result = result.sort_values("word_index").reset_index(drop=True)

    n_words = len(result)
    n_lines = int(result["line_index"].max()) + 1
    result["word_progress"] = (
        result["word_index"] / float(n_words - 1)
        if n_words > 1 else 0.0
    )
    result["line_index_norm"] = (
        result["line_index"] / float(n_lines - 1)
        if n_lines > 1 else 0.0
    )

    result["line_progress"] = 0.0
    for _, group in result.groupby("line_index", sort=True):
        if len(group) > 1:
            values = np.arange(len(group), dtype=float) / float(len(group) - 1)
        else:
            values = np.zeros(1, dtype=float)
        result.loc[group.index, "line_progress"] = values
    return result


def extract_set_trial_aoi(
    frame: pd.DataFrame,
    set_name: str,
    trial_name: str,
    source_path: Path,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    text_column = column_by_alias(frame.columns, TEXT_ALIASES)
    is_word_column = column_by_alias(frame.columns, IS_WORD_ALIASES)
    target_column = column_by_alias(frame.columns, TARGET_NAME_ALIASES)
    rectangles = rectangle_columns(frame)

    if not text_column:
        raise ValueError(
            f"{source_path} has no text_name-like column: "
            f"{list(frame.columns)}"
        )
    if not is_word_column:
        raise ValueError(
            f"{source_path} has no is_word-like column: "
            f"{list(frame.columns)}"
        )

    word_mask = to_bool(frame[is_word_column])
    word_rows = frame[word_mask].copy()
    text_values = word_rows[text_column].astype(str).str.strip()
    trial_text = str(trial_name).strip()

    exact_mask = text_values.eq(trial_text)
    if exact_mask.any():
        matched = word_rows[exact_mask].copy()
        match_method = "exact_text_name"
    else:
        trial_key = canonical_trial_name(trial_name)
        canonical_values = word_rows[text_column].map(canonical_trial_name)
        canonical_mask = canonical_values.eq(trial_key) & canonical_values.ne("")
        matched = word_rows[canonical_mask].copy()
        match_method = "canonical_text_name"

    matched_text_values = sorted(
        matched[text_column].dropna().astype(str).unique().tolist()
    )

    rows: list[dict[str, Any]] = []
    malformed = 0
    for source_index, row in matched.iterrows():
        rectangle = row_rectangle(row, rectangles)
        if rectangle is None:
            malformed += 1
            continue
        xmin, ymin, xmax, ymax = rectangle
        if (
            not all(map(math.isfinite, rectangle))
            or xmax <= xmin
            or ymax <= ymin
        ):
            malformed += 1
            continue
        rows.append(
            {
                "set_name": set_name,
                "trial_name": trial_name,
                "source_file": source_path.relative_to(root).as_posix(),
                "source_row_index": source_index,
                "source_text_name": str(row[text_column]),
                "text_name_match_method": match_method,
                "word_text": (
                    parse_target_word(row[target_column])
                    if target_column else ""
                ),
                "x_min": xmin,
                "y_min": ymin,
                "x_max": xmax,
                "y_max": ymax,
            }
        )

    aoi = pd.DataFrame(rows)
    if aoi.empty:
        available_values = sorted(
            word_rows[text_column].dropna().astype(str).unique().tolist()
        )
        trial_key = canonical_trial_name(trial_name)
        nearby = [
            value
            for value in available_values
            if trial_key
            and (
                trial_key in canonical_trial_name(value)
                or canonical_trial_name(value) in trial_key
            )
        ][:30]
        raise ValueError(
            "No valid word AOIs after exact/canonical matching for "
            f"set={set_name}, trial={trial_name}, source={source_path}. "
            f"trial_canonical={trial_key!r}; "
            f"matched_text_values={matched_text_values[:30]}; "
            f"nearby_text_values={nearby}; "
            f"text_column={text_column!r}; is_word_column={is_word_column!r}; "
            f"rectangle_columns={rectangles}"
        )

    # One text can be repeated once per QA target/occurrence. After the
    # canonical match, collapse identical word rectangles deterministically.
    aoi = aoi.drop_duplicates(
        ["x_min", "y_min", "x_max", "y_max", "word_text"]
    ).reset_index(drop=True)
    aoi = derive_reading_order(aoi)

    audit = {
        "set_name": set_name,
        "trial_name": trial_name,
        "source_file": source_path.relative_to(root).as_posix(),
        "text_column": text_column,
        "is_word_column": is_word_column,
        "target_name_column": target_column,
        "text_name_match_method": match_method,
        "n_matched_text_name_values": len(matched_text_values),
        "matched_text_name_examples": "|".join(matched_text_values[:20]),
        "rectangle_columns_json": json.dumps(rectangles, ensure_ascii=False),
        "n_matching_is_word_rows": len(matched),
        "n_malformed_rectangles": malformed,
        "n_unique_word_boxes": len(aoi),
        "n_lines": int(aoi["line_index"].max()) + 1,
        "median_word_height": float(aoi["height"].median()),
        "x_min": float(aoi["x_min"].min()),
        "x_max": float(aoi["x_max"].max()),
        "y_min": float(aoi["y_min"].min()),
        "y_max": float(aoi["y_max"].max()),
        "within_1280x720": bool(
            (aoi["x_min"] >= 0).all()
            and (aoi["y_min"] >= 0).all()
            and (aoi["x_max"] <= SOURCE_BROWSER_WIDTH).all()
            and (aoi["y_max"] <= SOURCE_BROWSER_HEIGHT).all()
        ),
    }
    return aoi, audit



def extract_set_trial_aoi_from_stage3b(
    *,
    stage3b_rows: pd.DataFrame,
    set_name: str,
    trial_name: str,
    target_path: Path,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve one set/trial AOI using Stage 3B's successful content mapping.

    Stage 3B found all 101 unique primary trial names, but many texts appeared in
    more than one target dataframe. Here we:
      1. select only target_dataframe_{set_name}.csv;
      2. select Stage-3B rows mapped to trial_name;
      3. retain only physical source rows where the target dataframe says
         is_word == True;
      4. deduplicate rectangles and derive reading order.

    This avoids assuming that the repository's ``text_name`` column itself is a
    trial identifier. No task label or model output is used.
    """
    required_columns = {
        "source_file",
        "source_row_index",
        "trial_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    }
    missing = sorted(required_columns - set(stage3b_rows.columns))
    if missing:
        raise ValueError(
            "Stage 3B AOI candidate table lacks required columns: "
            f"{missing}"
        )

    expected_relative = target_path.relative_to(root).as_posix()
    expected_basename = target_path.name.lower()

    source_values = (
        stage3b_rows["source_file"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )
    source_match = source_values.eq(expected_relative) | (
        source_values.map(lambda value: Path(value).name.lower())
        .eq(expected_basename)
    )
    trial_match = stage3b_rows["trial_name"].astype(str).eq(trial_name)

    subset = stage3b_rows[source_match & trial_match].copy()
    initial_candidate_rows = len(subset)

    if subset.empty:
        available_sources = sorted(
            stage3b_rows.loc[
                stage3b_rows["trial_name"].astype(str).eq(trial_name),
                "source_file",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        available_trials_in_source = sorted(
            stage3b_rows.loc[source_match, "trial_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Stage 3B found the trial globally but not in the participant's "
            "set-specific target dataframe. "
            f"set={set_name}, trial={trial_name}, "
            f"expected_source={expected_relative!r}, "
            f"sources_for_trial={available_sources[:30]}, "
            f"trials_in_expected_source={available_trials_in_source[:30]}"
        )

    # Stage 3B loaded CSVs with the default RangeIndex. Load the same way so
    # source_row_index is a stable physical row reference.
    target_frame = pd.read_csv(target_path, low_memory=False)
    text_column = column_by_alias(
        target_frame.columns,
        TEXT_ALIASES,
    )
    is_word_column = column_by_alias(
        target_frame.columns,
        IS_WORD_ALIASES,
    )
    target_name_column = column_by_alias(
        target_frame.columns,
        TARGET_NAME_ALIASES,
    )
    target_rectangles = rectangle_columns(target_frame)
    if not text_column:
        raise ValueError(
            f"{target_path} has no text_name-like grouping column: "
            f"{list(target_frame.columns)}"
        )
    if not is_word_column:
        raise ValueError(
            f"{target_path} has no is_word-like column: "
            f"{list(target_frame.columns)}"
        )

    original_subset = subset.copy()
    source_index_numeric = pd.to_numeric(
        original_subset["source_row_index"],
        errors="coerce",
    )
    is_word_mask = to_bool(target_frame[is_word_column])
    valid_word_source_indexes = set(
        target_frame.index[is_word_mask].astype(int).tolist()
    )
    subset = original_subset[
        source_index_numeric.isin(valid_word_source_indexes)
    ].copy()
    aoi_resolution_method = (
        "stage3b_direct_word_rows+set_specific_source_file+is_word_filter"
    )
    parent_text_group_counts: list[dict[str, Any]] = []

    if subset.empty:
        # The Stage-3B rows that carry a trial identifier can be parent targets
        # (paragraph/question rows), not word rows. Their text_name is the group
        # key shared with the corresponding is_word=True rows.
        candidate_indexes = (
            source_index_numeric.dropna().astype(int).drop_duplicates().tolist()
        )
        in_bounds_indexes = [
            index for index in candidate_indexes
            if 0 <= index < len(target_frame)
        ]
        if not in_bounds_indexes:
            raise ValueError(
                "No Stage 3B source indexes are in bounds for the target CSV. "
                f"set={set_name}, trial={trial_name}, "
                f"candidate_source_indexes={candidate_indexes[:30]}, "
                f"target_rows={len(target_frame)}"
            )

        parent_rows = target_frame.loc[in_bounds_indexes].copy()
        parent_text_values = (
            parent_rows[text_column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        text_as_string = target_frame[text_column].astype("string")
        positive_text_values: list[str] = []
        for value in parent_text_values:
            count = int(
                (is_word_mask & text_as_string.eq(str(value))).sum()
            )
            parent_text_group_counts.append(
                {
                    "text_name_preview": str(value)[:300],
                    "n_is_word_rows": count,
                }
            )
            if count > 0:
                positive_text_values.append(str(value))

        if not positive_text_values:
            raise ValueError(
                "The Stage 3B parent rows were found, but their text_name "
                "values do not link to is_word=True rows. "
                f"set={set_name}, trial={trial_name}, "
                f"source={expected_relative!r}, "
                f"candidate_source_indexes={in_bounds_indexes[:30]}, "
                f"text_column={text_column!r}, "
                f"parent_text_group_counts={parent_text_group_counts[:20]!r}, "
                f"parent_rows={parent_rows.head(20).to_dict(orient='records')!r}"
            )

        grouped_word_rows = target_frame[
            is_word_mask & text_as_string.isin(positive_text_values)
        ].copy()
        counts = grouped_word_rows[text_column].astype(str).value_counts()
        max_count = int(counts.max())
        selected_text_values = sorted(
            counts[counts.eq(max_count)].index.astype(str).tolist()
        )
        grouped_word_rows = grouped_word_rows[
            grouped_word_rows[text_column]
            .astype(str)
            .isin(selected_text_values)
        ].copy()

        normalized_rows: list[dict[str, Any]] = []
        for source_index, target_row in grouped_word_rows.iterrows():
            rectangle = row_rectangle(target_row, target_rectangles)
            if rectangle is None:
                continue
            xmin, ymin, xmax, ymax = rectangle
            if (
                not all(map(math.isfinite, rectangle))
                or xmax <= xmin
                or ymax <= ymin
            ):
                continue
            normalized_rows.append(
                {
                    "source_file": expected_relative,
                    "source_row_index": int(source_index),
                    "trial_name": trial_name,
                    "word_text": (
                        parse_target_word(target_row[target_name_column])
                        if target_name_column else ""
                    ),
                    "x_min": xmin,
                    "y_min": ymin,
                    "x_max": xmax,
                    "y_max": ymax,
                }
            )
        subset = pd.DataFrame(normalized_rows)
        aoi_resolution_method = (
            "stage3b_parent_rows+shared_text_name+set_specific_source_file+"
            "is_word_filter"
        )

        if subset.empty:
            diagnostic_columns = [
                column
                for column in (
                    "left", "top", "right", "bottom",
                    "x", "y", "width", "height",
                    text_column, is_word_column,
                )
                if column and column in grouped_word_rows.columns
            ]
            diagnostic_rows = (
                grouped_word_rows[diagnostic_columns]
                .head(12)
                .to_dict(orient="records")
            )
            diagnostic_dtypes = {
                column: str(grouped_word_rows[column].dtype)
                for column in diagnostic_columns
            }
            raise ValueError(
                "Parent text groups contained is_word=True rows, but no valid "
                "word rectangles could be parsed. "
                f"set={set_name}, trial={trial_name}, "
                f"rectangle_columns={target_rectangles!r}, "
                f"diagnostic_dtypes={diagnostic_dtypes!r}, "
                f"diagnostic_rows={diagnostic_rows!r}"
            )

    word_candidate_rows = len(subset)

    for column in ("x_min", "y_min", "x_max", "y_max"):
        subset[column] = pd.to_numeric(
            subset[column],
            errors="coerce",
        )
    valid_rectangle = (
        np.isfinite(subset["x_min"])
        & np.isfinite(subset["y_min"])
        & np.isfinite(subset["x_max"])
        & np.isfinite(subset["y_max"])
        & subset["x_max"].gt(subset["x_min"])
        & subset["y_max"].gt(subset["y_min"])
    )
    malformed = int((~valid_rectangle).sum())
    subset = subset[valid_rectangle].copy()

    if subset.empty:
        raise ValueError(
            "All Stage 3B set-specific word rows had invalid rectangles. "
            f"set={set_name}, trial={trial_name}, "
            f"source={expected_relative!r}"
        )

    if "word_text" not in subset.columns:
        subset["word_text"] = ""
    subset["word_text"] = subset["word_text"].fillna("").astype(str)

    keep_columns = [
        "source_file",
        "source_row_index",
        "trial_name",
        "word_text",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
    for optional in (
        "object_path",
        "trial_assignment_evidence",
        "set_name_source",
    ):
        if optional in subset.columns:
            keep_columns.append(optional)

    aoi = subset[keep_columns].copy()
    aoi.insert(0, "set_name", set_name)
    aoi["source_file"] = expected_relative
    aoi = aoi.drop_duplicates(
        ["x_min", "y_min", "x_max", "y_max", "word_text"]
    ).reset_index(drop=True)
    aoi = derive_reading_order(aoi)

    if len(aoi) < 2:
        raise ValueError(
            "Set-specific Stage 3B mapping produced fewer than two word boxes. "
            f"set={set_name}, trial={trial_name}, n_boxes={len(aoi)}"
        )

    object_paths = (
        sorted(subset["object_path"].dropna().astype(str).unique().tolist())
        if "object_path" in subset.columns
        else []
    )
    evidence_values = (
        sorted(
            subset["trial_assignment_evidence"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if "trial_assignment_evidence" in subset.columns
        else []
    )
    audit = {
        "set_name": set_name,
        "trial_name": trial_name,
        "source_file": expected_relative,
        "resolved_target_dataframe_base_set": (
            target_path.stem.removeprefix("target_dataframe_")
        ),
        "aoi_resolution_method": aoi_resolution_method,
        "text_column": text_column,
        "is_word_column": is_word_column,
        "target_name_column": target_name_column,
        "parent_text_group_counts_json": json.dumps(
            parent_text_group_counts,
            ensure_ascii=False,
        ),
        "stage3b_candidate_rows_before_is_word": initial_candidate_rows,
        "stage3b_candidate_rows_after_is_word": word_candidate_rows,
        "n_malformed_rectangles": malformed,
        "n_unique_word_boxes": len(aoi),
        "n_lines": int(aoi["line_index"].max()) + 1,
        "median_word_height": float(aoi["height"].median()),
        "x_min": float(aoi["x_min"].min()),
        "x_max": float(aoi["x_max"].max()),
        "y_min": float(aoi["y_min"].min()),
        "y_max": float(aoi["y_max"].max()),
        "stage3b_object_paths": "|".join(object_paths[:20]),
        "stage3b_assignment_evidence": "|".join(evidence_values[:20]),
        "within_1280x720": bool(
            (aoi["x_min"] >= 0).all()
            and (aoi["y_min"] >= 0).all()
            and (aoi["x_max"] <= SOURCE_BROWSER_WIDTH).all()
            and (aoi["y_max"] <= SOURCE_BROWSER_HEIGHT).all()
        ),
    }
    return aoi, audit

def schema_for_feature_set(
    schema: pd.DataFrame,
    feature_set: str,
    expected_count: int,
) -> pd.DataFrame:
    required = {"feature_set", "feature_index", "feature_name"}
    missing = sorted(required - set(schema.columns))
    if missing:
        raise ValueError(f"Frozen feature schema lacks columns: {missing}")

    selected = schema[schema["feature_set"].eq(feature_set)].copy()
    if "training_scope" in selected.columns:
        common = selected[
            selected["training_scope"].astype(str).str.contains(
                "common16|matched", case=False, regex=True
            )
        ].copy()
        if not common.empty:
            selected = common
    selected["feature_index"] = pd.to_numeric(
        selected["feature_index"], errors="raise"
    ).astype(int)
    selected = selected.sort_values("feature_index").reset_index(drop=True)
    if len(selected) != expected_count:
        raise ValueError(
            f"{feature_set} frozen schema has {len(selected)} rows, "
            f"expected {expected_count}."
        )
    if selected["feature_name"].duplicated().any():
        raise ValueError(f"{feature_set} has duplicate frozen feature names.")
    return selected


def build_dense_30hz_samples(
    group: pd.DataFrame,
    rate_hz: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    step_ms = 1000.0 / rate_hz
    group = group.sort_values(
        ["t_ms", "interpolation_segment_id", "sample_index_30hz"]
    ).copy()
    group = group[
        np.isfinite(pd.to_numeric(group["t_ms"], errors="coerce"))
    ]
    if group.empty:
        raise ValueError("Trial has no finite 30 Hz timestamps.")

    start = float(group["t_ms"].min())
    end = float(group["t_ms"].max())
    n_grid = int(round((end - start) / step_ms)) + 1
    if n_grid < 2:
        raise ValueError("Trial dense grid contains fewer than two samples.")

    grid = start + np.arange(n_grid, dtype=float) * step_ms
    x = np.full(n_grid, np.nan, dtype=float)
    y = np.full(n_grid, np.nan, dtype=float)
    valid = np.zeros(n_grid, dtype=bool)
    assigned_segment = np.full(n_grid, -1, dtype=int)
    collisions = 0
    off_grid = 0

    for row in group.itertuples(index=False):
        timestamp = float(row.t_ms)
        position = int(round((timestamp - start) / step_ms))
        if position < 0 or position >= n_grid:
            off_grid += 1
            continue
        if abs(grid[position] - timestamp) > max(0.5, step_ms * 0.05):
            off_grid += 1
            continue
        if valid[position]:
            collisions += 1
            continue
        x_value = finite_number(row.x_px)
        y_value = finite_number(row.y_px)
        is_valid = bool(row.valid_xy) and math.isfinite(x_value) and math.isfinite(y_value)
        if is_valid:
            x[position] = x_value
            y[position] = y_value
            valid[position] = True
            assigned_segment[position] = int(row.interpolation_segment_id)

    dense = pd.DataFrame(
        {
            "sample_index_in_trial": np.arange(n_grid, dtype=np.int64),
            "timestamp_ms_abs": grid,
            "x_px": x,
            "y_px": y,
            # WebQAmGaze has no pupil; for Soft-AOI membership, basic validity
            # is explicitly defined as valid x/y.
            "valid_basic": valid,
        }
    )
    valid_positions = np.flatnonzero(valid)
    segment_break_pairs = 0
    if len(valid_positions) >= 2:
        segment_values = assigned_segment[valid_positions]
        adjacent_grid = np.diff(valid_positions) == 1
        segment_change = np.diff(segment_values) != 0
        segment_break_pairs = int(np.sum(adjacent_grid & segment_change))
        if segment_break_pairs:
            raise RuntimeError(
                "Dense grid accidentally made different interpolation segments "
                "adjacent without an invalid row."
            )

    audit = {
        "dense_n_samples": n_grid,
        "dense_n_valid_samples": int(valid.sum()),
        "dense_valid_rate": float(valid.mean()),
        "source_resampled_rows": len(group),
        "source_interpolation_segments": int(
            group["interpolation_segment_id"].nunique()
        ),
        "dense_missing_rows_restored": int(n_grid - valid.sum()),
        "grid_collisions": collisions,
        "off_grid_rows": off_grid,
        "timestamp_start_ms": start,
        "timestamp_end_ms": end,
    }
    return dense, audit


def rectangle_membership(
    x: np.ndarray,
    y: np.ndarray,
    aoi: pd.DataFrame,
) -> np.ndarray:
    return (
        (x[:, None] >= aoi["x_min"].to_numpy(dtype=float)[None, :])
        & (x[:, None] <= aoi["x_max"].to_numpy(dtype=float)[None, :])
        & (y[:, None] >= aoi["y_min"].to_numpy(dtype=float)[None, :])
        & (y[:, None] <= aoi["y_max"].to_numpy(dtype=float)[None, :])
    )


def zuco_degrees_per_pixel() -> tuple[float, float]:
    horizontal_fov_deg = math.degrees(
        2.0 * math.atan((ZUCO_ACTIVE_WIDTH_MM / 2.0) / ZUCO_VIEWING_DISTANCE_MM)
    )
    vertical_fov_deg = math.degrees(
        2.0 * math.atan((ZUCO_ACTIVE_HEIGHT_MM / 2.0) / ZUCO_VIEWING_DISTANCE_MM)
    )
    return (
        horizontal_fov_deg / ZUCO_WIDTH_PX,
        vertical_fov_deg / ZUCO_HEIGHT_PX,
    )


def official_formula_features(
    fixations: pd.DataFrame,
    aoi: pd.DataFrame,
    reading_speed_seconds: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    n_words = len(aoi)
    if n_words <= 0:
        raise ValueError("Official formula requires at least one word.")

    frame = fixations.copy()
    for column in ("x_px", "y_px", "t_ms", "duration_ms"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = (
        frame["valid_fixation"].fillna(False).astype(bool)
        & np.isfinite(frame["x_px"])
        & np.isfinite(frame["y_px"])
        & np.isfinite(frame["t_ms"])
        & np.isfinite(frame["duration_ms"])
        & frame["duration_ms"].ge(0)
    )
    frame = frame[valid].copy()
    frame = frame[
        frame["x_px"].between(0, SOURCE_BROWSER_WIDTH)
        & frame["y_px"].between(0, SOURCE_BROWSER_HEIGHT)
    ].sort_values(["t_ms", "fixation_index"])

    if frame.empty:
        values = {name: math.nan for name in FORMULA_NAMES}
        values["omission_rate"] = 1.0
        audit = {
            "n_word_boxes": n_words,
            "n_valid_browser_fixations": 0,
            "n_fixations_inside_words": 0,
            "n_unique_fixated_words": 0,
            "n_transferred_saccades": 0,
            "reading_speed_unit": (
                "seconds_per_word" if reading_speed_seconds
                else "milliseconds_per_word"
            ),
        }
        return values, audit

    x = frame["x_px"].to_numpy(dtype=float)
    y = frame["y_px"].to_numpy(dtype=float)
    membership = rectangle_membership(x, y, aoi)
    inside_any = membership.any(axis=1)

    word_assignments = np.full(len(frame), -1, dtype=int)
    if inside_any.any():
        # If HTML boxes overlap, choose the smallest containing box.
        areas = (
            (aoi["x_max"] - aoi["x_min"])
            * (aoi["y_max"] - aoi["y_min"])
        ).to_numpy(dtype=float)
        for index in np.flatnonzero(inside_any):
            candidates = np.flatnonzero(membership[index])
            word_assignments[index] = int(
                candidates[np.argmin(areas[candidates])]
            )

    in_word = word_assignments >= 0
    unique_words = np.unique(word_assignments[in_word])
    omission_rate = 1.0 - len(unique_words) / float(n_words)
    fixation_number = float(np.sum(in_word)) / float(n_words)
    duration_sum_ms = float(
        frame.loc[in_word, "duration_ms"].sum()
    )
    reading_speed_ms = duration_sum_ms / float(n_words)
    reading_speed = (
        reading_speed_ms / 1000.0
        if reading_speed_seconds else reading_speed_ms
    )

    # Approximate inter-fixation transitions. These are not EyeLink events.
    starts = frame["t_ms"].to_numpy(dtype=float)
    durations = frame["duration_ms"].to_numpy(dtype=float)
    ends = starts + durations
    transition_duration_ms = starts[1:] - ends[:-1]

    x_zuco = x / SOURCE_BROWSER_WIDTH * ZUCO_WIDTH_PX
    y_zuco = y / SOURCE_BROWSER_HEIGHT * ZUCO_HEIGHT_PX
    deg_per_px_x, deg_per_px_y = zuco_degrees_per_pixel()
    dx_deg = np.diff(x_zuco) * deg_per_px_x
    dy_deg = np.diff(y_zuco) * deg_per_px_y
    amplitude_deg = np.hypot(dx_deg, dy_deg)

    valid_saccade = (
        np.isfinite(transition_duration_ms)
        & (transition_duration_ms > 0)
        & np.isfinite(amplitude_deg)
    )
    sacc_dur = transition_duration_ms[valid_saccade]
    sacc_amp = amplitude_deg[valid_saccade]
    sacc_velocity = (
        sacc_amp / (sacc_dur / 1000.0)
        if len(sacc_dur) else np.asarray([], dtype=float)
    )

    values = {
        "omission_rate": float(omission_rate),
        "fixation_number": float(fixation_number),
        "reading_speed": float(reading_speed),
        # Published definition: total saccade duration divided by words.
        "mean_sacc_dur": (
            float(np.sum(sacc_dur) / n_words)
            if len(sacc_dur) else math.nan
        ),
        "max_sacc_dur": (
            float(np.max(sacc_dur)) if len(sacc_dur) else math.nan
        ),
        "mean_sacc_velocity": (
            float(np.mean(sacc_velocity))
            if len(sacc_velocity) else math.nan
        ),
        "max_sacc_velocity": (
            float(np.max(sacc_velocity))
            if len(sacc_velocity) else math.nan
        ),
        "mean_sacc_amplitude": (
            float(np.mean(sacc_amp)) if len(sacc_amp) else math.nan
        ),
        "max_sacc_amplitude": (
            float(np.max(sacc_amp)) if len(sacc_amp) else math.nan
        ),
    }
    audit = {
        "n_word_boxes": n_words,
        "n_valid_browser_fixations": len(frame),
        "n_fixations_inside_words": int(np.sum(in_word)),
        "n_unique_fixated_words": len(unique_words),
        "n_transferred_saccades": len(sacc_dur),
        "reading_speed_ms_per_word_unconverted": reading_speed_ms,
        "reading_speed_unit": (
            "seconds_per_word" if reading_speed_seconds
            else "milliseconds_per_word"
        ),
        "zuco_horizontal_deg_per_px": deg_per_px_x,
        "zuco_vertical_deg_per_px": deg_per_px_y,
        "event_definition": (
            "positive gap from current filtered-fixation end to next "
            "filtered-fixation start"
        ),
    }
    return values, audit


def frozen_official_artifact(frozen_dir: Path) -> tuple[dict[str, Any], Path]:
    candidates = [
        frozen_dir / "models" / "official_sent_gaze_sacc.joblib",
        frozen_dir / "official_sent_gaze_sacc.joblib",
    ]
    for path in candidates:
        if path.is_file():
            artifact = joblib.load(path)
            return artifact, path
    matches = list(frozen_dir.rglob("official_sent_gaze_sacc.joblib"))
    if len(matches) == 1:
        return joblib.load(matches[0]), matches[0]
    raise FileNotFoundError(
        f"Could not uniquely locate official_sent_gaze_sacc.joblib in {frozen_dir}"
    )


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
        "--stage3b-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage3b"),
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path("outputs/frozen_zuco_external_validation_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/webqamgaze_external_validation_stage3c"),
    )
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument(
        "--sigma-word-height-ratio",
        type=float,
        default=2.0 / 3.0,
    )
    parser.add_argument("--active-threshold", type=float, default=0.1)
    args = parser.parse_args()

    if args.rate_hz != 30.0:
        raise ValueError(
            "The frozen external protocol requires --rate-hz 30."
        )
    if args.sigma_word_height_ratio <= 0:
        raise ValueError("--sigma-word-height-ratio must be positive.")
    if not (0 < args.active_threshold <= 1):
        raise ValueError("--active-threshold must be in (0, 1].")

    root = args.root.resolve()
    stage2_dir = args.stage2_dir.resolve()
    stage3a_dir = args.stage3a_dir.resolve()
    stage3b_dir = args.stage3b_dir.resolve()
    frozen_dir = args.frozen_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "trials": stage2_dir / "trial_manifest_primary_candidates.csv",
        "samples": stage2_dir / "samples_30hz.parquet",
        "fixations": stage2_dir / "filtered_fixation_events.parquet",
        "rawstat": stage3a_dir / "rawstat_features_primary.csv",
        "stage3b_manifest": stage3b_dir / "stage3b_manifest.json",
        "stage3b_aoi_rows": (
            stage3b_dir / "aoi_rows_normalized_all_candidates.csv"
        ),
        "schema": frozen_dir / "feature_schema.csv",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    trials = pd.read_csv(required["trials"])
    samples = pd.read_parquet(required["samples"])
    fixations = pd.read_parquet(required["fixations"])
    rawstat = pd.read_csv(required["rawstat"])
    stage3b_aoi_rows = pd.read_csv(required["stage3b_aoi_rows"])
    schema_all = pd.read_csv(required["schema"])

    full_schema = schema_for_feature_set(
        schema_all, "soft_full_all", 41
    )
    compact_schema = schema_for_feature_set(
        schema_all,
        "soft_compact_distribution_membership",
        27,
    )
    official_schema = schema_for_feature_set(
        schema_all, "official_sent_gaze_sacc", 9
    )

    full_columns = full_schema["feature_name"].astype(str).tolist()
    compact_columns = compact_schema["feature_name"].astype(str).tolist()
    official_columns = official_schema["feature_name"].astype(str).tolist()

    extractor_columns = list(
        dict.fromkeys(
            list(soft.CONTENT_FEATURES)
            + list(soft.MEMBERSHIP_QUALITY_FEATURES)
        )
    )
    if set(full_columns) != set(extractor_columns):
        raise ValueError(
            "Frozen Full Soft-AOI schema differs from the imported exact "
            f"extractor. Missing from extractor={sorted(set(full_columns)-set(extractor_columns))}; "
            f"extra extractor={sorted(set(extractor_columns)-set(full_columns))}"
        )
    if not set(compact_columns).issubset(set(extractor_columns)):
        raise ValueError("Frozen Compact schema is not a subset of extractor features.")

    artifact, artifact_path = frozen_official_artifact(frozen_dir)
    artifact_columns = list(artifact.get("feature_columns", []))
    if artifact_columns != official_columns:
        raise ValueError(
            "Official artifact feature order differs from feature_schema.csv."
        )
    imputer = artifact["pipeline"].named_steps["imputer"]
    official_training_medians = np.asarray(
        imputer.statistics_, dtype=float
    )
    if len(official_training_medians) != 9:
        raise ValueError("Official artifact imputer does not contain nine medians.")

    # Deterministic unit rule based only on the frozen ZuCo training artifact.
    # A seconds-per-word reading-speed median is expected to be single-digit;
    # a millisecond representation would be in the hundreds or thousands.
    reading_speed_seconds = bool(
        math.isfinite(official_training_medians[2])
        and abs(official_training_medians[2]) < 20.0
    )

    if trials["external_trial_id"].duplicated().any():
        raise ValueError("Primary trial manifest has duplicate IDs.")
    if rawstat["external_trial_id"].duplicated().any():
        raise ValueError("RawStat table has duplicate IDs.")
    if set(rawstat["external_trial_id"]) != set(trials["external_trial_id"]):
        raise ValueError("RawStat and primary trial IDs differ.")

    unique_keys = (
        trials[["set_name", "trial_name"]]
        .drop_duplicates()
        .sort_values(["set_name", "trial_name"])
        .reset_index(drop=True)
    )

    target_paths: dict[str, Path] = {}
    aoi_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    aoi_audit_rows: list[dict[str, Any]] = []
    aoi_frames: list[pd.DataFrame] = []

    for row in unique_keys.itertuples(index=False):
        set_name = str(row.set_name)
        trial_name = str(row.trial_name)
        if set_name not in target_paths:
            target_paths[set_name] = find_target_dataframe(
                root,
                set_name,
            )
        target_path = target_paths[set_name]
        aoi, audit = extract_set_trial_aoi_from_stage3b(
            stage3b_rows=stage3b_aoi_rows,
            set_name=set_name,
            trial_name=trial_name,
            target_path=target_path,
            root=root,
        )
        key = (set_name, trial_name)
        aoi_lookup[key] = aoi
        aoi_frames.append(aoi)
        aoi_audit_rows.append(audit)

    aoi_all = pd.concat(aoi_frames, ignore_index=True)
    aoi_audit = pd.DataFrame(aoi_audit_rows)

    if len(aoi_lookup) != len(unique_keys):
        raise RuntimeError("Set-specific AOI lookup coverage mismatch.")
    if not aoi_audit["within_1280x720"].all():
        bad = aoi_audit.loc[
            ~aoi_audit["within_1280x720"],
            ["set_name", "trial_name", "source_file"],
        ]
        raise ValueError(
            "Some official word boxes lie outside 1280x720:\n"
            + bad.head(30).to_string(index=False)
        )

    sample_groups = samples.groupby("external_trial_id", sort=False)
    fixation_groups = fixations.groupby("external_trial_id", sort=False)

    soft_rows: list[dict[str, Any]] = []
    soft_audit_rows: list[dict[str, Any]] = []
    official_description_rows: list[dict[str, Any]] = []
    official_schema_rows: list[dict[str, Any]] = []
    official_audit_rows: list[dict[str, Any]] = []

    sorted_trials = trials.sort_values(
        ["participant_id", "trial_index"]
    ).reset_index(drop=True)

    for position, trial in enumerate(
        sorted_trials.itertuples(index=False),
        start=1,
    ):
        trial_id = str(trial.external_trial_id)
        if trial_id not in sample_groups.groups:
            raise ValueError(f"Missing 30 Hz sample group: {trial_id}")
        if trial_id not in fixation_groups.groups:
            trial_fixations = pd.DataFrame(
                columns=fixations.columns
            )
        else:
            trial_fixations = fixation_groups.get_group(trial_id)

        key = (str(trial.set_name), str(trial.trial_name))
        if key not in aoi_lookup:
            raise ValueError(f"Missing set-specific AOI key: {key}")
        aoi = aoi_lookup[key]

        dense, dense_audit = build_dense_30hz_samples(
            sample_groups.get_group(trial_id),
            rate_hz=args.rate_hz,
        )

        zuco_task = "NR" if str(trial.task).upper() == "NR" else "TSR"
        trial_row = pd.Series(
            {
                "subject_id": str(trial.participant_id),
                "task": zuco_task,
                "block_id": 1,
                "trial_index_in_block": int(trial.trial_index),
                "trial_id": trial_id,
                # Metadata only; AOI is passed directly.
                "task_sentence_index": int(position - 1),
            }
        )

        soft_record = soft.extract_trial_features(
            samples=dense,
            trial_row=trial_row,
            aoi=aoi,
            rate_hz=int(args.rate_hz),
            sigma_word_height_ratio=float(
                args.sigma_word_height_ratio
            ),
            active_threshold=float(args.active_threshold),
        )
        output_soft = {
            "external_trial_id": trial_id,
            "trial_id": trial_id,
            "subject_id": str(trial.participant_id),
            "participant_id": str(trial.participant_id),
            "set_name": str(trial.set_name),
            "trial_index": int(trial.trial_index),
            "trial_name": str(trial.trial_name),
            "task": str(trial.task).upper(),
            "zuco_task_equivalent": zuco_task,
            "label_nr": int(trial.label_nr),
            **{
                feature: soft_record.get(feature, math.nan)
                for feature in extractor_columns
            },
        }
        soft_rows.append(output_soft)
        soft_audit_rows.append(
            {
                "external_trial_id": trial_id,
                "participant_id": str(trial.participant_id),
                "set_name": str(trial.set_name),
                "trial_name": str(trial.trial_name),
                "task": str(trial.task).upper(),
                "n_word_boxes": len(aoi),
                **dense_audit,
                "soft_aoi_n_samples": soft_record.get(
                    "soft_aoi_n_samples"
                ),
                "soft_aoi_n_valid_samples": soft_record.get(
                    "soft_aoi_n_valid_samples"
                ),
                "soft_aoi_valid_rate": soft_record.get(
                    "soft_aoi_valid_rate"
                ),
                "soft_aoi_active_rate": soft_record.get(
                    "soft_aoi_active_rate"
                ),
                "soft_aoi_word_box_hit_rate": soft_record.get(
                    "soft_aoi_word_box_hit_rate"
                ),
                "soft_aoi_sentence_envelope_hit_rate": soft_record.get(
                    "soft_aoi_sentence_envelope_hit_rate"
                ),
            }
        )

        formula_values, formula_audit = official_formula_features(
            fixations=trial_fixations,
            aoi=aoi,
            reading_speed_seconds=reading_speed_seconds,
        )
        descriptive = {
            "external_trial_id": trial_id,
            "trial_id": trial_id,
            "subject_id": str(trial.participant_id),
            "participant_id": str(trial.participant_id),
            "set_name": str(trial.set_name),
            "trial_index": int(trial.trial_index),
            "trial_name": str(trial.trial_name),
            "task": str(trial.task).upper(),
            "label_nr": int(trial.label_nr),
            **formula_values,
        }
        official_description_rows.append(descriptive)

        schema_ready = {
            key_name: descriptive[key_name]
            for key_name in (
                "external_trial_id",
                "trial_id",
                "subject_id",
                "participant_id",
                "set_name",
                "trial_index",
                "trial_name",
                "task",
                "label_nr",
            )
        }
        for feature_index, schema_name in enumerate(
            official_columns
        ):
            schema_ready[schema_name] = formula_values[
                FORMULA_NAMES[feature_index]
            ]
        official_schema_rows.append(schema_ready)
        official_audit_rows.append(
            {
                "external_trial_id": trial_id,
                "participant_id": str(trial.participant_id),
                "set_name": str(trial.set_name),
                "trial_name": str(trial.trial_name),
                "task": str(trial.task).upper(),
                **formula_audit,
            }
        )

        if position % 100 == 0 or position == len(sorted_trials):
            print(
                f"Extracted features: {position}/{len(sorted_trials)}"
            )

    soft_table = pd.DataFrame(soft_rows)
    soft_audit = pd.DataFrame(soft_audit_rows)
    official_descriptive = pd.DataFrame(
        official_description_rows
    )
    official_ready = pd.DataFrame(official_schema_rows)
    official_audit = pd.DataFrame(official_audit_rows)

    metadata_columns = [
        "external_trial_id", "trial_id", "subject_id",
        "participant_id", "set_name", "trial_index",
        "trial_name", "task", "zuco_task_equivalent", "label_nr",
    ]
    soft_full = soft_table[
        metadata_columns + full_columns
    ].copy()
    soft_compact = soft_table[
        metadata_columns + compact_columns
    ].copy()

    for table_name, table, columns in (
        ("Full Soft-AOI", soft_full, full_columns),
        ("Compact Soft-AOI", soft_compact, compact_columns),
        ("Official transferred formula", official_ready, official_columns),
    ):
        if len(table) != len(trials):
            raise RuntimeError(
                f"{table_name} rows={len(table)}, expected={len(trials)}"
            )
        if table["external_trial_id"].duplicated().any():
            raise RuntimeError(f"{table_name} has duplicate trial IDs.")
        if set(table["external_trial_id"]) != set(
            trials["external_trial_id"]
        ):
            raise RuntimeError(
                f"{table_name} trial IDs differ from primary manifest."
            )
        if np.isinf(table[columns].to_numpy(dtype=float)).any():
            raise RuntimeError(f"{table_name} contains infinity.")

    paths = {
        "aoi": output_dir / "set_specific_word_aoi.csv",
        "aoi_audit": output_dir / "set_specific_word_aoi_audit.csv",
        "soft_full": output_dir / "soft_full_features_primary.csv",
        "soft_compact": output_dir / "soft_compact_features_primary.csv",
        "soft_audit": output_dir / "soft_feature_audit.csv",
        "official_descriptive": output_dir
        / "official_formula_features_descriptive.csv",
        "official_ready": output_dir / "official_features_primary.csv",
        "official_audit": output_dir / "official_feature_audit.csv",
        "schema": output_dir / "frozen_schema_stage3c.csv",
        "readiness": output_dir / "stage3c_readiness.csv",
        "manifest": output_dir / "stage3c_manifest.json",
    }

    aoi_all.to_csv(paths["aoi"], index=False, encoding="utf-8-sig")
    aoi_audit.to_csv(
        paths["aoi_audit"], index=False, encoding="utf-8-sig"
    )
    soft_full.to_csv(
        paths["soft_full"], index=False, encoding="utf-8-sig"
    )
    soft_compact.to_csv(
        paths["soft_compact"], index=False, encoding="utf-8-sig"
    )
    soft_audit.to_csv(
        paths["soft_audit"], index=False, encoding="utf-8-sig"
    )
    official_descriptive.to_csv(
        paths["official_descriptive"],
        index=False,
        encoding="utf-8-sig",
    )
    official_ready.to_csv(
        paths["official_ready"], index=False, encoding="utf-8-sig"
    )
    official_audit.to_csv(
        paths["official_audit"], index=False, encoding="utf-8-sig"
    )
    pd.concat(
        [
            full_schema.assign(stage3c_role="soft_full"),
            compact_schema.assign(stage3c_role="soft_compact"),
            official_schema.assign(stage3c_role="official_transferred"),
        ],
        ignore_index=True,
    ).to_csv(paths["schema"], index=False, encoding="utf-8-sig")

    full_missing = {
        column: int(soft_full[column].isna().sum())
        for column in full_columns
    }
    compact_missing = {
        column: int(soft_compact[column].isna().sum())
        for column in compact_columns
    }
    official_missing = {
        column: int(official_ready[column].isna().sum())
        for column in official_columns
    }

    readiness_rows = [
        {
            "feature_set": "dummy_reference",
            "status": "ready",
            "n_rows": len(trials),
            "n_features": 0,
            "n_missing_cells": 0,
            "note": "no feature input",
        },
        {
            "feature_set": "rawstat_all",
            "status": "ready",
            "n_rows": len(rawstat),
            "n_features": 18,
            "n_missing_cells": int(
                rawstat[
                    schema_for_feature_set(
                        schema_all, "rawstat_all", 18
                    )["feature_name"].tolist()
                ].isna().sum().sum()
            ),
            "note": (
                "pupil-dependent NaN values preserved for frozen ZuCo imputer"
            ),
        },
        {
            "feature_set": "soft_full_all",
            "status": "ready",
            "n_rows": len(soft_full),
            "n_features": len(full_columns),
            "n_missing_cells": int(
                soft_full[full_columns].isna().sum().sum()
            ),
            "note": "exact existing ZuCo Soft-AOI formula, set-specific WebQAmGaze AOIs",
        },
        {
            "feature_set": "soft_compact_distribution_membership",
            "status": "ready",
            "n_rows": len(soft_compact),
            "n_features": len(compact_columns),
            "n_missing_cells": int(
                soft_compact[compact_columns].isna().sum().sum()
            ),
            "note": "exact frozen 27-feature subset",
        },
        {
            "feature_set": "official_sent_gaze_sacc",
            "status": "ready_transferred_nonidentical",
            "n_rows": len(official_ready),
            "n_features": len(official_columns),
            "n_missing_cells": int(
                official_ready[official_columns].isna().sum().sum()
            ),
            "note": (
                "transferred event-based official-feature-formula condition; "
                "inter-fixation transitions are not EyeLink saccades"
            ),
        },
    ]
    pd.DataFrame(readiness_rows).to_csv(
        paths["readiness"], index=False, encoding="utf-8-sig"
    )

    descriptive_external_medians = {
        name: float(
            pd.to_numeric(
                official_descriptive[name], errors="coerce"
            ).median()
        )
        for name in FORMULA_NAMES
    }
    manifest = {
        "stage": "webqamgaze_external_validation_stage3c_feature_generation",
        "version": "1.6",
        "created_at_utc": utc_now_iso(),
        "root": str(root),
        "primary_trials": len(trials),
        "primary_participants": int(
            trials["participant_id"].nunique()
        ),
        "unique_set_trial_aoi_keys": len(aoi_lookup),
        "target_dataframe_sets_loaded": len(target_paths),
        "aoi_source_rule": (
            "Stage 3B content-resolved rows; restrict source_file to "
            "target_dataframe_{set_name}.csv; retain physical source rows "
            "where is_word==True"
        ),
        "browser_coordinate_system": {
            "width": SOURCE_BROWSER_WIDTH,
            "height": SOURCE_BROWSER_HEIGHT,
            "soft_aoi_coordinates": "direct WebQAmGaze browser pixels",
        },
        "soft_aoi": {
            "extractor_module": str(
                Path(soft.__file__).resolve()
            ),
            "rate_hz": args.rate_hz,
            "sigma_word_height_ratio": args.sigma_word_height_ratio,
            "active_threshold": args.active_threshold,
            "valid_basic_definition": "valid_xy because pupil is unavailable",
            "full_feature_order": full_columns,
            "compact_feature_order": compact_columns,
            "full_missing_counts": full_missing,
            "compact_missing_counts": compact_missing,
        },
        "official_transferred": {
            "artifact_path": str(artifact_path),
            "artifact_feature_order": official_columns,
            "formula_order": FORMULA_NAMES,
            "frozen_training_imputer_medians": {
                FORMULA_NAMES[index]: float(
                    official_training_medians[index]
                )
                for index in range(9)
            },
            "reading_speed_output_unit": (
                "seconds_per_word"
                if reading_speed_seconds
                else "milliseconds_per_word"
            ),
            "reading_speed_unit_rule": (
                "use seconds if frozen training median at index 2 is <20; "
                "otherwise use milliseconds"
            ),
            "external_descriptive_medians": descriptive_external_medians,
            "missing_counts": official_missing,
            "saccade_caveat": (
                "Positive inter-fixation gaps from WebQAmGaze filtered events "
                "are approximate transitions, not EyeLink-detected saccades."
            ),
            "amplitude_scale": (
                "1280x720 browser -> 800x600 ZuCo pixels -> degrees using "
                "400x298.9 mm display at 680 mm"
            ),
        },
        "constraints": [
            "No model was fitted.",
            "No model decision score or prediction was produced.",
            "No WebQAmGaze label was used for AOI selection or feature tuning.",
            "All feature orders come from the frozen Common16 schema.",
            "Official condition must be reported as transferred and nonidentical.",
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

    print("=== WEBQAMGAZE EXTERNAL VALIDATION STAGE 3C ===")
    print(f"Primary participants: {manifest['primary_participants']}")
    print(f"Primary trials: {len(trials)}")
    print(
        "Exact set/trial AOI coverage: "
        f"{len(aoi_lookup)}/{len(unique_keys)}"
    )
    print(
        "AOI resolver: Stage 3B direct/parent rows + set-specific source + "
        "shared text_name + is_word filter"
    )
    print(f"Participant set-name variants resolved: {len(target_paths)}")
    print(
        "Unique base target dataframes used: "
        f"{len({path.name for path in target_paths.values()})}"
    )
    print(
        "Word boxes per set/trial: "
        f"min={int(aoi_audit['n_unique_word_boxes'].min())}, "
        f"median={float(aoi_audit['n_unique_word_boxes'].median()):.1f}, "
        f"max={int(aoi_audit['n_unique_word_boxes'].max())}"
    )
    print(
        "AOIs within official 1280x720 browser: "
        f"{int(aoi_audit['within_1280x720'].sum())}/{len(aoi_audit)}"
    )
    print(
        "Full Soft-AOI: "
        f"rows={len(soft_full)}, features={len(full_columns)}, "
        f"missing_cells={int(soft_full[full_columns].isna().sum().sum())}"
    )
    print(
        "Compact Soft-AOI: "
        f"rows={len(soft_compact)}, features={len(compact_columns)}, "
        f"missing_cells={int(soft_compact[compact_columns].isna().sum().sum())}"
    )
    print(
        "Median Soft-AOI valid/active/box-hit rates: "
        f"{soft_audit['soft_aoi_valid_rate'].median():.4f} / "
        f"{soft_audit['soft_aoi_active_rate'].median():.4f} / "
        f"{soft_audit['soft_aoi_word_box_hit_rate'].median():.4f}"
    )
    print(
        "Transferred official formula: "
        f"rows={len(official_ready)}, features={len(official_columns)}, "
        f"missing_cells={int(official_ready[official_columns].isna().sum().sum())}"
    )
    print(
        "Official reading_speed unit selected from frozen artifact: "
        f"{'seconds_per_word' if reading_speed_seconds else 'milliseconds_per_word'}"
    )
    print("Frozen official training medians by formula position:")
    for index, name in enumerate(FORMULA_NAMES):
        print(
            f"  {index}: {name} -> "
            f"{official_training_medians[index]:.6g}"
        )
    print("External transferred-feature medians:")
    for name in FORMULA_NAMES:
        print(
            f"  {name}: "
            f"{descriptive_external_medians[name]:.6g}"
        )
    print(
        "Trials with no derived inter-fixation transition: "
        f"{int(official_audit['n_transferred_saccades'].eq(0).sum())}"
    )
    print("All five frozen feature conditions are now schema-ready.")
    print("No predictions were produced.")
    print("Written:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
