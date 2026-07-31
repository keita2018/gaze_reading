#!/usr/bin/env python
"""Inspect a ZuCo 2.0 MATLAB v7.3/HDF5 result file without loading it all.

The report focuses on:
- top-level HDF5 structure
- sentenceData fields
- the first sentence
- the first sentence's word/rawET nesting
- sentence-level allFixations nesting

Important:
sentenceData/rawData is treated as EEG in the official loader.
The eye-tracking candidate inspected here is sentenceData/word/.../rawET.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def object_summary(obj: h5py.Group | h5py.Dataset) -> str:
    if isinstance(obj, h5py.Group):
        return f"Group(name={obj.name!r}, keys={list(obj.keys())[:30]})"
    return (
        f"Dataset(name={obj.name!r}, shape={obj.shape}, "
        f"dtype={obj.dtype}, matlab_class={obj.attrs.get('MATLAB_class')!r})"
    )


def first_valid_reference(dataset: h5py.Dataset) -> Optional[h5py.Reference]:
    values = np.asarray(dataset[()]).ravel(order="C")
    for value in values:
        if isinstance(value, h5py.Reference) and bool(value):
            return value
    return None


def dereference_first(
    h5: h5py.File, dataset: h5py.Dataset
) -> Optional[h5py.Group | h5py.Dataset]:
    ref = first_valid_reference(dataset)
    return h5[ref] if ref is not None else None


def decode_matlab_char(obj: h5py.Group | h5py.Dataset) -> Optional[str]:
    if not isinstance(obj, h5py.Dataset):
        return None
    try:
        arr = np.asarray(obj[()]).ravel(order="F")
        if arr.size == 0:
            return ""
        if np.issubdtype(arr.dtype, np.integer):
            return "".join(chr(int(x)) for x in arr if int(x) != 0)
    except Exception:
        return None
    return None


def preview_numeric(obj: h5py.Group | h5py.Dataset, rows: int = 5, cols: int = 8) -> str:
    if not isinstance(obj, h5py.Dataset):
        return "<not a dataset>"
    try:
        arr = np.asarray(obj[()])
    except Exception as exc:
        return f"<could not read: {exc}>"

    if arr.dtype == object or h5py.check_dtype(ref=arr.dtype) is not None:
        return "<reference dataset>"

    if arr.ndim == 0:
        return repr(arr.item())
    if arr.ndim == 1:
        return np.array2string(arr[:rows], threshold=50)
    slices = tuple(slice(0, rows if i == 0 else cols) for i in range(arr.ndim))
    return np.array2string(arr[slices], threshold=100, edgeitems=3)


def print_group_fields(group: h5py.Group, indent: str = "  ") -> None:
    for key in sorted(group.keys()):
        obj = group[key]
        print(f"{indent}{key}: {object_summary(obj)}")


def inspect_first_sentence(h5: h5py.File, sentence_data: h5py.Group) -> None:
    print("\n=== FIRST SENTENCE REFERENCES ===")

    for field in ("content", "rawData", "allFixations", "word", "omissionRate"):
        if field not in sentence_data:
            print(f"{field}: MISSING")
            continue
        container = sentence_data[field]
        print(f"{field}: {object_summary(container)}")
        if isinstance(container, h5py.Dataset):
            target = dereference_first(h5, container)
            if target is None:
                print("  first reference: none")
            else:
                print(f"  first reference -> {object_summary(target)}")
                text = decode_matlab_char(target)
                if text is not None:
                    print(f"  decoded text: {text[:500]!r}")
                elif isinstance(target, h5py.Dataset):
                    print(f"  numeric preview: {preview_numeric(target)}")
                elif isinstance(target, h5py.Group):
                    print("  referenced group fields:")
                    print_group_fields(target, indent="    ")


def inspect_all_fixations(h5: h5py.File, sentence_data: h5py.Group) -> None:
    print("\n=== FIRST SENTENCE allFixations ===")
    if "allFixations" not in sentence_data:
        print("allFixations not present.")
        return

    first = dereference_first(h5, sentence_data["allFixations"])
    if first is None:
        print("No valid allFixations reference.")
        return

    print(object_summary(first))
    if isinstance(first, h5py.Group):
        for key in sorted(first.keys()):
            obj = first[key]
            print(f"  {key}: {object_summary(obj)}")
            if isinstance(obj, h5py.Dataset):
                target = dereference_first(h5, obj)
                if target is not None:
                    print(f"    first reference -> {object_summary(target)}")
                    if isinstance(target, h5py.Dataset):
                        print(f"    preview: {preview_numeric(target)}")
                else:
                    print(f"    preview: {preview_numeric(obj)}")


def inspect_raw_et(h5: h5py.File, sentence_data: h5py.Group) -> None:
    print("\n=== FIRST SENTENCE word/rawET ===")
    if "word" not in sentence_data:
        print("sentenceData/word not present.")
        return

    first_word_group = dereference_first(h5, sentence_data["word"])
    if first_word_group is None:
        print("No valid sentenceData/word reference.")
        return
    print(f"First sentence word object: {object_summary(first_word_group)}")

    if not isinstance(first_word_group, h5py.Group):
        print("The first word object is not a group.")
        return

    print("Word-object fields:")
    print_group_fields(first_word_group, indent="  ")

    if "rawET" not in first_word_group:
        print("rawET not present in the first sentence's word object.")
        return

    raw_et_per_word = first_word_group["rawET"]
    print(f"\nrawET per-word container: {object_summary(raw_et_per_word)}")
    if not isinstance(raw_et_per_word, h5py.Dataset):
        return

    word_raw_et_ref = first_valid_reference(raw_et_per_word)
    if word_raw_et_ref is None:
        print("No valid rawET reference for any word in the first sentence.")
        return

    fixation_ref_container = h5[word_raw_et_ref]
    print(
        "First word with rawET -> fixation-reference container: "
        f"{object_summary(fixation_ref_container)}"
    )

    if not isinstance(fixation_ref_container, h5py.Dataset):
        return

    refs = np.asarray(fixation_ref_container[()]).ravel(order="C")
    shown = 0
    for value in refs:
        if not isinstance(value, h5py.Reference) or not bool(value):
            continue
        fixation_array = h5[value]
        print(f"\n  rawET fixation segment {shown}: {object_summary(fixation_array)}")
        if isinstance(fixation_array, h5py.Dataset):
            print(f"  preview:\n{preview_numeric(fixation_array, rows=8, cols=12)}")
        shown += 1
        if shown >= 3:
            break

    if shown == 0:
        print("No fixation-segment references were found.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file")
    args = parser.parse_args()

    path = Path(args.mat_file)
    if not path.is_file():
        raise FileNotFoundError(path)

    if not h5py.is_hdf5(path):
        print(
            "This is not an HDF5/MATLAB v7.3 file. "
            "Use scipy.io.loadmat for older MAT formats."
        )
        return 2

    print(f"FILE: {path.resolve()}")
    print(f"SIZE_MB: {path.stat().st_size / (1024 * 1024):.2f}")

    with h5py.File(path, "r") as h5:
        print("\n=== TOP LEVEL ===")
        for key in sorted(h5.keys()):
            print(f"{key}: {object_summary(h5[key])}")

        if "sentenceData" not in h5:
            print("\nERROR: top-level sentenceData group is missing.")
            return 3

        sentence_data = h5["sentenceData"]
        if not isinstance(sentence_data, h5py.Group):
            print("\nERROR: sentenceData is not an HDF5 group.")
            return 4

        print("\n=== sentenceData FIELDS ===")
        print_group_fields(sentence_data)

        trial_lengths = []
        for key in ("content", "rawData", "allFixations", "word", "omissionRate"):
            if key in sentence_data and isinstance(sentence_data[key], h5py.Dataset):
                trial_lengths.append((key, sentence_data[key].shape))
        print("\n=== SELECTED CONTAINER SHAPES ===")
        for key, shape in trial_lengths:
            print(f"{key}: {shape}")

        inspect_first_sentence(h5, sentence_data)
        inspect_all_fixations(h5, sentence_data)
        inspect_raw_et(h5, sentence_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
