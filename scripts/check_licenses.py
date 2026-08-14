#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Check SPDX licensing for files changed between HEAD~1 and HEAD."""

from __future__ import annotations

import subprocess
import sys


SPDX_COPYRIGHT = "SPDX-FileCopyrightText:"
SPDX_LICENSE = "SPDX-License-Identifier:"


def _git_bytes(*args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _paths(*, diff_filter: str) -> list[str]:
    output = _git_bytes(
        "diff",
        "--no-renames",
        f"--diff-filter={diff_filter}",
        "--name-only",
        "-z",
        "HEAD~1",
        "HEAD",
        "--",
    )
    return [path.decode("utf-8") for path in output.split(b"\0") if path]


def _content(path: str) -> bytes | None:
    result = subprocess.run(
        ("git", "show", f"HEAD:{path}"),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _has_spdx(content: bytes | None) -> bool:
    if content is None:
        return False
    text = content.decode("utf-8", errors="ignore")
    return SPDX_COPYRIGHT in text and SPDX_LICENSE in text


def _is_licensed(path: str) -> bool:
    if path.startswith("LICENSES/"):
        return True
    if _has_spdx(_content(path)):
        return True
    if path.endswith(".license"):
        return False
    return _has_spdx(_content(f"{path}.license"))


def _paths_to_check() -> list[str]:
    paths = set(_paths(diff_filter="AM"))
    for deleted_path in _paths(diff_filter="D"):
        if not deleted_path.endswith(".license"):
            continue
        licensed_path = deleted_path.removesuffix(".license")
        if _content(licensed_path) is not None:
            paths.add(licensed_path)
    return sorted(paths)


def main() -> int:
    try:
        changed_paths = _paths_to_check()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    missing = [path for path in changed_paths if not _is_licensed(path)]
    if missing:
        print(
            "License check failed; changed files need SPDX copyright and license fields:",
            file=sys.stderr,
        )
        for path in missing:
            print(f"  {path} (add tags in-file or in {path}.license)", file=sys.stderr)
        return 1

    print("License check passed for files changed in HEAD~1..HEAD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
