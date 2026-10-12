#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Fetch one pinned dependency archive with the source checkout's Git credentials."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "compute_flow/python"))

from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.paths import resolve_state_paths  # noqa: E402
from brick_manager.tools import git_lfs_environment  # noqa: E402


def fetch_archive(
    source: Path,
    remote: str,
    commit: str,
    profile: str,
    filename: str,
    sha256: str,
    output: Path,
) -> None:
    """Download only the selected archive without checking out the dependency branch."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", remote):
        raise ValueError("Invalid dependency Git remote name")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Dependency Git revision must be a full commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_]+", profile):
        raise ValueError("Invalid dependency profile")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.tar\.gz", filename):
        raise ValueError("Invalid dependency archive filename")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Invalid dependency archive SHA256")

    # Run in the original checkout to retain local credential helpers and SSH settings.
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    git = ["git", "-C", str(source)]
    top = subprocess.run(
        [*git, "rev-parse", "--show-toplevel"],
        capture_output=True,
        env=environment,
        check=True,
        timeout=30,
    )
    if Path(os.fsdecode(top.stdout).strip()).resolve() != source.resolve():
        raise ValueError("LFS dependencies require a Git checkout of Compute Flow")

    available = subprocess.run(
        [*git, "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        env=environment,
        timeout=30,
    )
    if available.returncode:
        subprocess.run(
            [*git, "fetch", "--no-tags", "--no-write-fetch-head", remote, commit],
            capture_output=True,
            env=environment,
            check=True,
            timeout=120,
        )

    archive_path = f"archives/{profile}/{filename}"
    pointer = subprocess.run(
        [*git, "show", f"{commit}:{archive_path}"],
        capture_output=True,
        env=environment,
        check=True,
        timeout=30,
    ).stdout
    expected_pointer = (
        rb"version https://git-lfs.github.com/spec/v1\n"
        + rb"oid sha256:"
        + sha256.encode("ascii")
        + rb"\nsize [0-9]+\n"
    )
    if not re.fullmatch(expected_pointer, pointer):
        raise ValueError("Dependency LFS pointer does not match the pinned SHA256")

    # Reuse the managed, checksum-verified tool used for model downloads.
    environment = git_lfs_environment(resolve_state_paths())
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_LFS_SKIP_SMUDGE"] = "0"
    output.parent.mkdir(parents=True, exist_ok=True)
    lfs = [
        *git,
        "-c",
        "lfs.fetchrecentalways=false",
        "-c",
        f"lfs.fetchinclude={archive_path}",
        "-c",
        "lfs.fetchexclude=",
        "lfs",
    ]

    subprocess.run(
        [*lfs, "fetch", remote, commit],
        capture_output=True,
        env=environment,
        check=True,
        timeout=300,
    )
    # Materialize the fetched object without a worktree or another network request.
    # CMake checks the resulting archive's SHA256 before extraction.
    with output.open("wb") as stream:
        subprocess.run(
            [*lfs, "smudge", archive_path],
            input=pointer,
            stdout=stream,
            stderr=subprocess.PIPE,
            env=environment,
            check=True,
            timeout=60,
        )


def main() -> int:
    """Provide the CMake downloader entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        fetch_archive(
            _ROOT,
            args.remote,
            args.commit,
            args.profile,
            args.filename,
            args.sha256,
            args.output,
        )
    except (OSError, ValueError, BrickManagerError) as error:
        print(f"Dependency LFS download failed: {error}", file=sys.stderr)
        return 1
    except subprocess.SubprocessError:
        # Git output can include credential-bearing remote URLs or HTTP tracing.
        print(
            "Dependency LFS download failed. Check Git remote access, the pinned "
            "dependency commit, and availability of its LFS objects.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
