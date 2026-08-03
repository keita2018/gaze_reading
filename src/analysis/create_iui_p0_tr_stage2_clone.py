from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LOCKED_SOURCE_SHA256 = "eafdb132405a9edff54cdfa3e8dc4fa7c22be4fb7fc11420b2753c05b8c8706e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Turkish confirmatory clone of the locked English "
            "WebQAmGaze Stage-2 preparation script without modifying the original."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--source-script",
        default="src/data/prepare_webqamgaze_external_validation_stage2.py",
    )
    parser.add_argument(
        "--output-script",
        default="src/data/prepare_webqamgaze_tr_confirmatory_stage2.py",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/iui_p0_tr_stage2_clone_v1/clone_manifest.json",
    )
    parser.add_argument("--language-code", default="TR")
    parser.add_argument("--language-name", default="Turkish")
    parser.add_argument(
        "--expected-source-sha256",
        default=LOCKED_SOURCE_SHA256,
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    source = (root / args.source_script).resolve()
    output = (root / args.output_script).resolve()
    manifest_path = (root / args.manifest).resolve()

    if not source.exists():
        raise FileNotFoundError(source)

    source_hash_before = sha256(source)
    if source_hash_before != args.expected_source_sha256:
        raise RuntimeError(
            "Locked English Stage-2 source hash does not match.\n"
            f"Expected: {args.expected_source_sha256}\n"
            f"Actual:   {source_hash_before}\n"
            "Do not proceed until the change is explained and recorded."
        )

    text = source.read_text(encoding="utf-8")

    counts_before = {
        "standalone_EN": len(re.findall(r"(?<![A-Za-z])EN(?![A-Za-z])", text)),
        "English": text.count("English"),
        "english": text.count("english"),
    }
    if counts_before["standalone_EN"] < 1:
        raise RuntimeError("No standalone EN token found in the locked script.")

    patched = re.sub(
        r"(?<![A-Za-z])EN(?![A-Za-z])",
        args.language_code,
        text,
    )
    patched = patched.replace("English", args.language_name)
    patched = patched.replace("english", args.language_name.lower())

    counts_after = {
        "standalone_EN": len(re.findall(r"(?<![A-Za-z])EN(?![A-Za-z])", patched)),
        "standalone_TR": len(
            re.findall(
                rf"(?<![A-Za-z]){re.escape(args.language_code)}(?![A-Za-z])",
                patched,
            )
        ),
        "English": patched.count("English"),
        "Turkish": patched.count(args.language_name),
    }

    if counts_after["standalone_EN"] != 0:
        raise RuntimeError("An EN token remains after patching.")

    # Compile before writing the final clone.
    compile(patched, str(output), "exec")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patched, encoding="utf-8")

    source_hash_after = sha256(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Original English Stage-2 script changed during cloning.")

    manifest = {
        "artifact_version": "iui_p0_tr_stage2_clone_v1",
        "source_script": str(source),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "output_script": str(output),
        "output_sha256": sha256(output),
        "language_code": args.language_code,
        "language_name": args.language_name,
        "replacement_counts_before": counts_before,
        "replacement_counts_after": counts_after,
        "original_modified": False,
        "performance_calculated": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Turkish Stage-2 clone created.")
    print(f"Source: {source}")
    print(f"Clone:  {output}")
    print(f"Manifest: {manifest_path}")
    print(f"Clone SHA-256: {manifest['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
