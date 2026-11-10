#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Discover bricks from Git registries and check out their recorded commits."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from brick_manager.errors import BrickManagerError
from brick_manager.metadata import load_info_json
from brick_manager.registry import parse_registry


@dataclass(frozen=True)
class GitBrick:
    """Metadata, directory, and pinned commit for a registered brick."""

    subdir: str
    tag: str
    commit: str
    info_json: dict[str, Any]


def inspect_git_source(url: str) -> list[GitBrick]:
    """Read registry.txt and its bricks at every MAJOR.MINOR.PATCH Git tag."""
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
        tags = _run_git(
            "-C", str(repository), "tag", "--list", "--sort=-version:refname"
        ).splitlines()
        if not tags:
            raise BrickManagerError(f"Git registry has no tags: {url}")
        bricks = []
        for tag in tags:
            if not re.fullmatch(
                r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag
            ):
                raise BrickManagerError(
                    f"Invalid registry tag {tag!r} in {url}. "
                    "Expected MAJOR.MINOR.PATCH, for example 1.2.3."
                )
            commit = _run_git(
                "-C", str(repository), "rev-parse", f"refs/tags/{tag}^{{commit}}"
            ).strip()
            _run_git("-C", str(repository), "checkout", "--quiet", "--detach", commit)
            registry_path = repository / "registry.txt"
            if not registry_path.is_file():
                raise BrickManagerError(
                    f"Git registry must contain a root registry.txt: {url}@{tag}"
                )
            if registry_path.is_symlink():
                raise BrickManagerError(
                    f"Registry file must not be a symbolic link: {url}@{tag}"
                )

            for subdir in parse_registry(registry_path):
                source_path = resolve_git_brick(repository, subdir)
                info_json = load_info_json(source_path, version=tag)
                if info_json["type"] != subdir.split("/")[0]:
                    raise BrickManagerError(
                        f"Brick type does not match its registry directory: {subdir}"
                    )
                bricks.append(GitBrick(subdir, tag, commit, info_json))

        return bricks


def resolve_git_brick(repository: Path, subdir: str) -> Path:
    """Resolve a type/variant directory within a Git registry checkout."""
    if not re.fullmatch(r"[a-z0-9_][a-z0-9_-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", subdir):
        raise BrickManagerError(
            f"Invalid registry brick path: {subdir}. Expected <type>/<variant>."
        )
    source_path = (repository / subdir).resolve()
    if not source_path.is_relative_to(repository.resolve()):
        raise BrickManagerError(f"Registry brick path escapes its repository: {subdir}")
    if not (source_path / "info.json").resolve().is_relative_to(source_path):
        raise BrickManagerError(f"Brick metadata escapes its directory: {subdir}")
    return source_path


def checkout_git_source(url: str, commit: str, destination: Path) -> None:
    """Check out the exact commit recorded during registry discovery."""
    _run_git("init", "--quiet", str(destination))
    _run_git(
        "-C",
        str(destination),
        "fetch",
        "--quiet",
        "--no-tags",
        "--depth=1",
        url,
        commit,
    )
    current_commit = _run_git(
        "-C", str(destination), "rev-parse", "FETCH_HEAD^{commit}"
    ).strip()
    if current_commit != commit:
        raise BrickManagerError(
            "Git source does not match its cached commit. "
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
