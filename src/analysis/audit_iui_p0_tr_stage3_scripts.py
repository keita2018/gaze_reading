from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_PATHS = [
    "src/data/prepare_webqamgaze_external_validation_stage3a.py",
    "src/data/resolve_webqamgaze_aoi_stage3b.py",
    "src/data/prepare_webqamgaze_external_validation_stage3c.py",
    "src/data/inspect_webqamgaze_official_schema_stage1c.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": "",
        "argparse_options": [],
        "argparse_defaults": {},
        "language_literals": [],
        "path_literals": [],
        "interesting_literals": [],
    }
    if not path.exists():
        return result

    result["sha256"] = sha256(path)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    options: set[str] = set()
    defaults: dict[str, Any] = {}
    languages: set[str] = set()
    paths: set[str] = set()
    interesting: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                option_names = [
                    arg.value for arg in node.args
                    if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("-")
                ]
                for option in option_names:
                    options.add(option)
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(
                            kw.value, ast.Constant
                        ):
                            defaults[option] = kw.value.value

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            lower = value.lower()
            if value.upper() in {"EN", "TR", "ES", "DE", "GR", "RU"}:
                languages.add(value)
            if any(token in lower for token in [
                ".csv", ".parquet", ".json", "stage2", "stage3",
                "webqamgaze_external_validation",
            ]):
                paths.add(value[:500])
            if any(token in lower for token in [
                "language", "participant", "trial", "aoi", "feature",
                "manifest", "label",
            ]):
                interesting.add(value[:500])

    result["argparse_options"] = sorted(options)
    result["argparse_defaults"] = defaults
    result["language_literals"] = sorted(languages)
    result["path_literals"] = sorted(paths)[:200]
    result["interesting_literals"] = sorted(interesting)[:200]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output",
        default="outputs/iui_p0_tr_stage3_script_audit_v1/"
        "stage3_script_audit.json",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    results = [inspect(root / rel) for rel in SCRIPT_PATHS]
    payload = {
        "artifact_version": "iui_p0_tr_stage3_script_audit_v1",
        "scripts": results,
        "model_loaded": False,
        "performance_calculated": False,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Stage-3 script audit completed.")
    print(f"Output: {output}")
    for result in results:
        print("")
        print(result["path"])
        print(f"  exists: {result['exists']}")
        print(f"  options: {result['argparse_options']}")
        print(f"  language literals: {result['language_literals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
