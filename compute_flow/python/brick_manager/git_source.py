#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Discover tagged brick releases and check out Git registry sources."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from brick_manager.errors import BrickManagerError
from brick_manager.metadata import validate_info_json


@dataclass(frozen=True)
class GitVersion:
    """Metadata and pinned commit for a tagged brick release."""

    tag: str
    commit: str
    info_json: dict[str, Any]


def inspect_git_source(url: str) -> list[GitVersion]:
    """Read the root info.json at every Git tag without installing a brick."""
    with TemporaryDirectory(prefix="compute-flow-git-index-") as temporary_dir:
        repository = Path(temporary_dir) / "source"
        _run_git(
            "clone",
            "--quiet",
            "--filter=blob:none",
            "--no-checkout",
            "--",
            url,
            str(repository),
        )
        _run_git("-C", str(repository), "fetch", "--quiet", "--tags", "origin")
        tags = _run_git("-C", str(repository), "tag", "--list").splitlines()
        if not tags:
            raise BrickManagerError(f"Git registry source has no tags: {url}")

        versions = []
        for tag in tags:
            commit = _run_git(
                "-C",
                str(repository),
                "rev-parse",
                "--verify",
                f"refs/tags/{tag}^{{commit}}",
            ).strip()
            contents = _run_git("-C", str(repository), "show", f"{commit}:info.json")
            try:
                info_json = json.loads(contents)
            except json.JSONDecodeError as error:
                raise BrickManagerError(
                    f"Invalid info.json at Git tag {tag}: {url}"
                ) from error
            if not isinstance(info_json, dict):
                raise BrickManagerError(f"Invalid info.json at Git tag {tag}: {url}")
            validate_info_json(info_json)
            versions.append(GitVersion(tag, commit, info_json))

        return versions


def checkout_git_source(url: str, tag: str, commit: str, destination: Path) -> None:
    """Check out a cached tag only when it still resolves to the cached commit."""
    _run_git("init", "--quiet", str(destination))
    _run_git(
        "-C",
        str(destination),
        "fetch",
        "--quiet",
        "--no-tags",
        "--depth=1",
        url,
        f"refs/tags/{tag}",
    )
    current_commit = _run_git(
        "-C", str(destination), "rev-parse", "FETCH_HEAD^{commit}"
    ).strip()
    if current_commit != commit:
        raise BrickManagerError(
            f"Git tag {tag} changed after it was cached. "
            "Run `compute-flow update` and retry the installation."
        )
    _run_git("-C", str(destination), "checkout", "--quiet", "--detach", commit)


def _run_git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise BrickManagerError(
            "Git is required for remote registry sources"
        ) from error
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise BrickManagerError(f"Git source operation failed: {message}")
    return result.stdout
