#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Check trademark usage in added lines and the commit message for HEAD."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ALLOW_MARKER = "trademark-check: allow"
TRADEMARKS_RELATIVE_PATH = Path("TRADEMARKS") / "README.md"
TRADEMARKS_PATH = Path(__file__).resolve().parents[1] / TRADEMARKS_RELATIVE_PATH
MACHINE_VALUES = ("Android", "Linux", "Macos")  # trademark-check: allow
LEGAL_ENTITIES = ("Arm Limited", "Raspberry Pi Ltd")  # trademark-check: allow
LEGAL_NOTICE_PATTERN = re.compile(
    r"\b(?:registered trademark|trademark of)\b", re.IGNORECASE
)
HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class AddedLine:
    path: str
    line_number: int
    text: str


def _load_trademarks() -> tuple[tuple[str, str], ...]:
    try:
        lines = TRADEMARKS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot read {TRADEMARKS_PATH}: {error}") from error

    trademarks: list[tuple[str, str]] = []
    in_table = False
    for line in lines:
        if not line.startswith("|"):
            if in_table:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        if cells[:2] == ["Name", "Body form"]:
            in_table = True
            continue
        if not in_table or set(cells[0]) <= {"-", ":"}:
            continue
        if cells[0] != cells[1]:
            trademarks.append((cells[0], cells[1]))

    if not trademarks:
        raise RuntimeError(f"no trademark rows found in {TRADEMARKS_PATH}")
    return tuple(trademarks)


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def _added_lines() -> list[AddedLine]:
    diff = _git(
        "-c",
        "core.quotepath=false",
        "diff",
        "--no-color",
        "--no-ext-diff",
        "--unified=0",
        "HEAD~1",
        "HEAD",
        "--",
    )
    path = ""
    line_number = 0
    added: list[AddedLine] = []

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            continue
        hunk = HUNK_PATTERN.match(line)
        if hunk:
            line_number = int(hunk.group(1))
            continue
        if not path or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(AddedLine(path, line_number, line[1:]))
            line_number += 1
        elif not line.startswith("-"):
            line_number += 1

    return added


def _mask(pattern: str, text: str) -> str:
    return re.sub(pattern, lambda match: " " * len(match.group(0)), text)


def _mask_allowed_uses(text: str) -> str:
    masked = text
    masked = _mask(r"`[^`]*`", masked)
    masked = _mask(r"\b(?:https?|ssh|git)://\S+", masked)
    masked = _mask(r"\b\S+@\S+", masked)
    masked = _mask(r"(?:[A-Za-z]:)?(?:/[\w.@+~()\-]+)+", masked)

    for entity in LEGAL_ENTITIES:
        masked = masked.replace(entity, " " * len(entity))

    machine_names = "|".join(re.escape(name) for name in MACHINE_VALUES)
    machine_patterns = (
        rf"\bCMAKE_SYSTEM_NAME\s*(?:=|\s+STREQUAL\s+)[\"']?(?:{machine_names})[\"']?",
        rf"\bos=(?:{machine_names})\b",
        rf"(?:system|platform\.system\(\))\s*(?:==|!=)\s*[\"'](?:{machine_names})[\"']",
        rf"set\(CMAKE_SYSTEM_NAME\s+(?:{machine_names})\)",
    )
    for pattern in machine_patterns:
        masked = _mask(pattern, masked)

    return masked


def _violations(
    text: str, trademarks: tuple[tuple[str, str], ...], *, allow_marker: bool
) -> list[str]:
    if allow_marker and ALLOW_MARKER in text:
        return []
    if "SPDX-FileCopyrightText:" in text or LEGAL_NOTICE_PATTERN.search(text):
        return []

    masked = _mask_allowed_uses(text)
    violations: list[str] = []
    for plain, marked in trademarks:
        if marked in (plain + "®", plain + "™"):
            wrong_symbol = "™" if marked.endswith("®") else "®"
            wrong_forms = (plain + wrong_symbol,)
        else:
            wrong_forms = (plain + "®", plain + "™")
        for wrong in wrong_forms:
            if wrong in masked:
                violations.append(f'"{wrong}" uses the wrong form; use "{marked}"')

        pattern = re.compile(rf"\b{re.escape(plain)}\b(?![®™])")
        if pattern.search(masked):
            violations.append(f'"{plain}" is unmarked; use "{marked}"')

    return violations


def main() -> int:
    try:
        trademarks = _load_trademarks()
        added_lines = _added_lines()
        commit_message = _git("show", "-s", "--format=%B", "HEAD")
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for added_line in added_lines:
        if Path(added_line.path) == TRADEMARKS_RELATIVE_PATH:
            continue
        for violation in _violations(added_line.text, trademarks, allow_marker=True):
            failures.append(f"{added_line.path}:{added_line.line_number}: {violation}")

    for line_number, line in enumerate(commit_message.splitlines(), start=1):
        for violation in _violations(line, trademarks, allow_marker=False):
            failures.append(f"HEAD:commit-message:{line_number}: {violation}")

    if failures:
        print("Trademark check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("Trademark check passed for HEAD~1..HEAD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
