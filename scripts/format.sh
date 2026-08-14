#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Run all project formatters.
# Usage:
#   scripts/format.sh          # format in-place
#   scripts/format.sh --check  # check only (for CI)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Markdown files to format - derived from git to naturally exclude build dirs,
# .venv, and other gitignored content.
MD_SOURCES=()
while IFS= read -r f; do
    MD_SOURCES+=("$PROJECT_ROOT/$f")
done < <(git -C "$PROJECT_ROOT" ls-files "*.md")

CHECK=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK=true
fi

if $CHECK; then
    echo "Checking CMake formatting (gersemi)..."
    uv run gersemi --check --line-length 120 --no-warn-about-unknown-commands "$PROJECT_ROOT"
    echo "Checking Python formatting (ruff format)..."
    uv run ruff format --check "$PROJECT_ROOT"
    echo "Checking Markdown formatting (mdformat)..."
    uv run mdformat --number --wrap 100 --check "${MD_SOURCES[@]}"
else
    echo "Formatting CMake files (gersemi)..."
    uv run gersemi -i --line-length 120 --no-warn-about-unknown-commands "$PROJECT_ROOT"
    echo "Formatting Python files (ruff format)..."
    uv run ruff format "$PROJECT_ROOT"
    echo "Formatting Markdown files (mdformat)..."
    uv run mdformat --number --wrap 100 "${MD_SOURCES[@]}"
fi

echo "All formatters passed."
