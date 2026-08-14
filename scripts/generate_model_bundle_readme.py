#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Generate a README.md layout for a Compute Flow model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path


EXCLUDED_DIRS = {".git", "__pycache__"}
EXCLUDED_FILES = {".DS_Store", "README.md", "SHA256SUMS"}


def model_bundle_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if relative.name in EXCLUDED_FILES:
            continue
        files.append(relative)
    return sorted(files)


def default_description(path: Path) -> str:
    suffix = path.suffix.lower()
    if path.name.endswith("_manifest.json") or suffix == ".json":
        return "model manifest"
    if path.parts and path.parts[0] == "assets":
        return "model asset"
    if suffix in {".onnx", ".tflite", ".gguf", ".pte", ".bin"}:
        return "model file"
    return "TODO: describe this file"


def render_readme(name: str, root: Path) -> str:
    lines = [
        f"# {name} Model Bundle",
        "",
        f"This repository stores the {name} model bundle.",
        "",
        "## Contents",
        "",
    ]

    files = model_bundle_files(root)
    if files:
        for path in files:
            lines.append(f"- `{path.as_posix()}` - {default_description(path)}")
    else:
        lines.append(
            "- TODO: list all files included in the repo with a brief description"
        )

    lines.extend(
        [
            "",
            "## Checksums",
            "",
            "Generate `SHA256SUMS` from the model bundle root with:",
            "",
            "```sh",
            "find . -type f ! -name SHA256SUMS ! -path './.git/*' -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS",
            "```",
            "",
            "# Model and asset origin",
            "",
            "TODO: describe where the model files and assets came from.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", required=True, help="model bundle name for the README title"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="model bundle root to scan for README contents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write README content to this path instead of stdout",
    )
    args = parser.parse_args()

    content = render_readme(args.name, args.root)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
