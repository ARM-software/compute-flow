#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model artifact paths, download, and integrity helpers."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

from brick_manager.errors import BrickManagerError
from brick_manager.identity import BrickSelector
from brick_manager.paths import StatePaths
from brick_manager.tools import git_lfs_environment


ProgressCallback = Callable[[str], None]
_SHA256SUMS_NAME = "SHA256SUMS"
_SHA256SUMS_LINE_PATTERN = re.compile(r"^([0-9a-fA-F]{64})\s+(.+)$")
_GIT_LFS_INSTALL_MESSAGE = (
    "Git LFS is required to download this model artifact because the repository "
    "declares LFS-tracked files."
)


def model_artifact_path(
    state_paths: StatePaths,
    selector: BrickSelector,
    model_name: str,
) -> Path:
    return state_paths.models / selector.brick_type / selector.name / model_name


def artifact_sha256(path: Path) -> str:
    if not path.is_dir():
        raise BrickManagerError(f"Model artifact directory does not exist: {path}")

    sha256sums_path = path / _SHA256SUMS_NAME
    _verify_sha256sums(path, sha256sums_path)
    return _file_sha256(sha256sums_path)


def install_artifact(
    state_paths: StatePaths,
    selector: BrickSelector,
    model_name: str,
    *,
    url: str,
    ref: str,
    expected_sha256: str,
    progress: ProgressCallback | None = None,
) -> Path:
    destination_path = model_artifact_path(state_paths, selector, model_name)
    temporary_parent = destination_path.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{model_name}.",
        dir=temporary_parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / "artifact"
        _download_artifact(state_paths, url, ref, temporary_path, progress)
        _report(progress, "validating")
        actual_sha256 = artifact_sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise BrickManagerError(
                "Downloaded model artifact failed SHA256SUMS validation: "
                f"{model_name} expected {expected_sha256}, got {actual_sha256}"
            )

        _report(progress, "installing")
        remove_artifact(destination_path)
        shutil.move(str(temporary_path), destination_path)

    return destination_path


def remove_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256sums(artifact_path: Path, sha256sums_path: Path) -> None:
    if not sha256sums_path.is_file():
        raise BrickManagerError(f"Missing model checksum manifest: {sha256sums_path}")

    lines = sha256sums_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise BrickManagerError(f"Empty model checksum manifest: {sha256sums_path}")

    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue

        match = _SHA256SUMS_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise BrickManagerError(f"Invalid SHA256SUMS line {line_number}: {line}")

        expected_sha256, relative_name = match.groups()
        if relative_name.startswith("*"):
            relative_name = relative_name[1:]

        file_path = _resolve_manifest_file(
            artifact_path,
            relative_name,
            line_number,
        )
        actual_sha256 = _file_sha256(file_path)
        if actual_sha256 != expected_sha256.lower():
            raise BrickManagerError(
                "Model file failed SHA256SUMS validation: "
                f"{relative_name} expected {expected_sha256.lower()}, "
                f"got {actual_sha256}"
            )


def _resolve_manifest_file(
    artifact_path: Path,
    relative_name: str,
    line_number: int,
) -> Path:
    relative_path = Path(relative_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise BrickManagerError(
            f"Unsafe SHA256SUMS path on line {line_number}: {relative_name}"
        )

    file_path = artifact_path / relative_path
    if not file_path.is_file():
        raise BrickManagerError(f"Missing model file from SHA256SUMS: {relative_name}")
    return file_path


def _download_artifact(
    state_paths: StatePaths,
    url: str,
    ref: str,
    destination_path: Path,
    progress: ProgressCallback | None,
) -> None:
    local_path = _local_path_from_url(url)
    if local_path is not None:
        _report(progress, "copying")
        _copy_local_artifact(local_path, destination_path)
        return

    _report(progress, "downloading")
    _clone_git_artifact(state_paths, url, ref, destination_path)


def _local_path_from_url(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser()
    if parsed.scheme:
        return None

    path = Path(url).expanduser()
    if path.exists():
        return path
    return None


def _copy_local_artifact(source_path: Path, destination_path: Path) -> None:
    if source_path.is_dir():
        shutil.copytree(
            source_path, destination_path, ignore=shutil.ignore_patterns(".git")
        )
        return
    if source_path.is_file():
        destination_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path / source_path.name)
        return
    raise BrickManagerError(f"Model source does not exist: {source_path}")


def _clone_git_artifact(
    state_paths: StatePaths,
    url: str,
    ref: str,
    destination_path: Path,
) -> None:
    clone_env = os.environ.copy()
    clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--no-checkout", url, str(destination_path)],
            env=clone_env,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(destination_path), "checkout", "--quiet", ref],
            env=clone_env,
            check=True,
            capture_output=True,
            text=True,
        )
        if _repository_uses_lfs(destination_path):
            env = _ensure_git_lfs_ready(state_paths, destination_path)
            subprocess.run(
                ["git", "-C", str(destination_path), "lfs", "pull"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip()
        raise BrickManagerError(
            f"Failed to download model artifact from {url}@{ref}: {message}"
        ) from error

    shutil.rmtree(destination_path / ".git")


def _repository_uses_lfs(repository_path: Path) -> bool:
    attributes_path = repository_path / ".gitattributes"
    if not attributes_path.is_file():
        return False
    return "filter=lfs" in attributes_path.read_text(encoding="utf-8")


def _ensure_git_lfs_ready(
    state_paths: StatePaths,
    repository_path: Path,
) -> dict[str, str]:
    env = git_lfs_environment(state_paths)
    try:
        subprocess.run(
            ["git", "-C", str(repository_path), "lfs", "version"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip()
        detail = f" {message}" if message else ""
        raise BrickManagerError(f"{_GIT_LFS_INSTALL_MESSAGE}{detail}") from error

    try:
        subprocess.run(
            ["git", "-C", str(repository_path), "lfs", "install", "--local"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip()
        detail = f" {message}" if message else ""
        raise BrickManagerError(f"{_GIT_LFS_INSTALL_MESSAGE}{detail}") from error

    return env


def _report(progress: ProgressCallback | None, phase: str) -> None:
    if progress is not None:
        progress(phase)
