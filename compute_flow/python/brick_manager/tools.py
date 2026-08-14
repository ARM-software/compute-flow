#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Managed third-party tools for the Compute Flow brick manager."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from brick_manager.errors import BrickManagerError
from brick_manager.paths import StatePaths


CMAKE_VERSION = "4.0.2"
GIT_LFS_VERSION = "3.6.1"
CMAKE_EXECUTABLE_ENV = "COMPUTE_FLOW_CMAKE_EXECUTABLE"
GIT_LFS_EXECUTABLE_ENV = "COMPUTE_FLOW_GIT_LFS_EXECUTABLE"


@dataclass(frozen=True)
class ToolArchive:
    url: str
    filename: str
    executable_name: str
    sha256: str | None = None


def ensure_cmake(state_paths: StatePaths) -> Path:
    configured = configured_cmake()
    if configured is not None:
        return configured
    return _ensure_tool(
        state_paths,
        "cmake",
        CMAKE_VERSION,
        _cmake_archive(),
    )


def ensure_git_lfs(state_paths: StatePaths) -> Path:
    configured = configured_git_lfs()
    if configured is not None:
        return configured
    return _ensure_tool(
        state_paths,
        "git-lfs",
        GIT_LFS_VERSION,
        _git_lfs_archive(),
    )


def git_lfs_environment(state_paths: StatePaths) -> dict[str, str]:
    git_lfs = ensure_git_lfs(state_paths)
    env = os.environ.copy()
    env["PATH"] = f"{git_lfs.parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def configured_cmake() -> Path | None:
    return _configured_executable(CMAKE_EXECUTABLE_ENV)


def configured_git_lfs() -> Path | None:
    executable = _configured_executable(GIT_LFS_EXECUTABLE_ENV)
    if executable is not None and executable.name not in ("git-lfs", "git-lfs.exe"):
        raise BrickManagerError(
            f"{GIT_LFS_EXECUTABLE_ENV} must identify an executable named git-lfs: "
            f"{executable}"
        )
    return executable


def find_managed_cmake(state_paths: StatePaths) -> Path | None:
    return _locate_executable(
        state_paths.home / "tools" / "cmake" / CMAKE_VERSION,
        "cmake",
    )


def find_managed_git_lfs(state_paths: StatePaths) -> Path | None:
    return _locate_executable(
        state_paths.home / "tools" / "git-lfs" / GIT_LFS_VERSION,
        "git-lfs",
    )


def _configured_executable(environment_variable: str) -> Path | None:
    configured = os.environ.get(environment_variable)
    if not configured:
        return None

    expanded = str(Path(configured).expanduser())
    resolved = shutil.which(expanded)
    if resolved is None:
        raise BrickManagerError(
            f"{environment_variable} does not identify an executable: {configured}"
        )
    return Path(resolved)


def _ensure_tool(
    state_paths: StatePaths,
    name: str,
    version: str,
    archive: ToolArchive,
) -> Path:
    tool_root = state_paths.home / "tools" / name / version
    executable = _find_executable(tool_root, archive.executable_name)
    if executable is not None:
        _clear_quarantine(tool_root)
        return executable

    archive_path = state_paths.home / "tools" / "downloads" / archive.filename
    _download(archive.url, archive_path, archive.sha256)
    _extract_tool_archive(archive_path, tool_root)
    _clear_quarantine(tool_root)

    executable = _find_executable(tool_root, archive.executable_name)
    if executable is None:
        raise BrickManagerError(
            f"Downloaded {name} {version}, but did not find {archive.executable_name}."
        )
    return executable


def _cmake_archive() -> ToolArchive:
    version = CMAKE_VERSION
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        filename = f"cmake-{version}-macos-universal.tar.gz"
    elif system == "Linux" and machine in ("x86_64", "amd64"):
        filename = f"cmake-{version}-linux-x86_64.tar.gz"
    elif system == "Linux" and machine in ("aarch64", "arm64"):
        filename = f"cmake-{version}-linux-aarch64.tar.gz"
    else:
        raise BrickManagerError(
            f"Managed CMake is not supported on {system} {machine}."
        )

    return ToolArchive(
        url=(
            f"https://github.com/Kitware/CMake/releases/download/v{version}/{filename}"
        ),
        filename=filename,
        executable_name="cmake",
    )


def _git_lfs_archive() -> ToolArchive:
    version = GIT_LFS_VERSION
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        platform_name = "darwin-arm64"
    elif system == "Darwin":
        platform_name = "darwin-amd64"
    elif system == "Linux" and machine in ("x86_64", "amd64"):
        platform_name = "linux-amd64"
    elif system == "Linux" and machine in ("aarch64", "arm64"):
        platform_name = "linux-arm64"
    else:
        raise BrickManagerError(
            f"Managed Git LFS is not supported on {system} {machine}."
        )

    extension = "zip" if platform_name.startswith("darwin-") else "tar.gz"
    filename = f"git-lfs-{platform_name}-v{version}.{extension}"
    return ToolArchive(
        url=(
            "https://github.com/git-lfs/git-lfs/releases/download/"
            f"v{version}/{filename}"
        ),
        filename=filename,
        executable_name="git-lfs",
        sha256=_git_lfs_sha256(filename),
    )


def _git_lfs_sha256(filename: str) -> str | None:
    return {
        "git-lfs-darwin-amd64-v3.6.1.zip": (
            "b53c361e6c85479507ed39ba99b87ec0888ac52f5afd2084fc68af4103081391"
        ),
        "git-lfs-darwin-arm64-v3.6.1.zip": (
            "83b4ea3b0c72ba19e3bc46e47e92476f4505cc96693333b9fa0a314dddacc4ba"
        ),
        "git-lfs-linux-amd64-v3.6.1.tar.gz": (
            "2138d2e405a12f1a088272e06790b76699b79cb90d0317b77aafaf35de908d76"
        ),
        "git-lfs-linux-arm64-v3.6.1.tar.gz": (
            "1c2720ff53528fbe769633d448d830aa7b682141e3c4f6a9f26b9cf3b2548d0a"
        ),
    }.get(filename)


def _download(url: str, destination: Path, expected_sha256: str | None) -> None:
    if destination.is_file():
        _verify_download(destination, expected_sha256)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}", flush=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        urllib.request.urlretrieve(url, temporary_path)
        _verify_download(temporary_path, expected_sha256)
        os.replace(temporary_path, destination)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise BrickManagerError(f"Failed to download {url}: {error}") from error


def _verify_download(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise BrickManagerError(
            f"Downloaded tool archive failed SHA256 validation: {path.name} "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_tool_archive(archive_path: Path, tool_root: Path) -> None:
    temporary_root = tool_root.parent / f".{tool_root.name}.tmp"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True, exist_ok=True)
    try:
        if archive_path.suffix == ".zip":
            _extract_zip(archive_path, temporary_root)
        else:
            _extract_tar(archive_path, temporary_root)
        _clear_quarantine(temporary_root)
        if tool_root.exists():
            shutil.rmtree(tool_root)
        os.replace(temporary_root, tool_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _extract_zip(archive_path: Path, destination: Path) -> None:
    print(f"Extracting {archive_path}", flush=True)
    with zipfile.ZipFile(archive_path) as zip_file:
        for member in zip_file.infolist():
            _validate_archive_member(destination, member.filename)
        zip_file.extractall(destination)


def _extract_tar(archive_path: Path, destination: Path) -> None:
    print(f"Extracting {archive_path}", flush=True)
    with tarfile.open(archive_path) as tar_file:
        for member in tar_file.getmembers():
            _validate_archive_member(destination, member.name)
        tar_file.extractall(destination)


def _validate_archive_member(destination: Path, member_name: str) -> None:
    member_path = Path(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise BrickManagerError(f"Unsafe managed tool archive path: {member_name}")
    resolved_path = (destination / member_path).resolve()
    if not resolved_path.is_relative_to(destination.resolve()):
        raise BrickManagerError(f"Unsafe managed tool archive path: {member_name}")


def _find_executable(root: Path, executable_name: str) -> Path | None:
    executable = _locate_executable(root, executable_name)
    if executable is not None:
        _make_executable(executable)
    return executable


def _locate_executable(root: Path, executable_name: str) -> Path | None:
    if not root.is_dir():
        return None

    preferred_paths = [
        path
        for path in sorted(root.rglob(executable_name))
        if path.is_file() and "Contents/bin" in path.as_posix()
    ]
    for path in preferred_paths:
        return path

    for path in sorted(root.rglob(executable_name)):
        if path.is_file():
            return path
    return None


def _make_executable(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode | 0o755)
    except OSError:
        pass


def _clear_quarantine(root: Path) -> None:
    if platform.system() != "Darwin" or not root.exists():
        return

    xattr = shutil.which("xattr") or "/usr/bin/xattr"
    if not Path(xattr).is_file():
        return
    subprocess.run(
        [xattr, "-dr", "com.apple.quarantine", str(root)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
