#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Generate a SHA256SUMS file for a model bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


_CHECKSUM_FILENAME = "SHA256SUMS"
_READ_SIZE = 1024 * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_READ_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(root: Path) -> list[Path]:
    """Return sorted model file paths, excluding dotfiles and the checksum output."""
    files: list[Path] = []
    for path in root.rglob("*"):
        relative_path = path.relative_to(root)
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        if relative_path == Path(_CHECKSUM_FILENAME) or not path.is_file():
            continue
        files.append(relative_path)
    return sorted(files)


def generate_checksum(root: Path) -> Path:
    """Write checksums for all non-dotfiles in a model bundle."""
    checksum_path = root / _CHECKSUM_FILENAME
    lines = [
        f"{_file_sha256(root / path)}  {path.as_posix()}" for path in model_files(root)
    ]
    checksum_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return checksum_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path, help="model bundle root directory")
    args = parser.parse_args()

    if not args.model_root.is_dir():
        parser.error(f"model bundle root is not a directory: {args.model_root}")

    checksum_path = generate_checksum(args.model_root)
    print(checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
