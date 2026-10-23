#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Package installed dependencies from the dedicated CMake export build."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile


_NOTICE_PREFIXES = ("license", "licence", "copying", "copyright", "notice")


def _copy_notices(source: Path, destination: Path) -> None:
    """Keep upstream and bundled-component notices with their source-relative paths."""
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.name.lower().startswith(_NOTICE_PREFIXES):
            relative = path.relative_to(source)
            if ".git" in relative.parts:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def _archive(prefix: Path, destination: Path) -> None:
    """Produce stable archives and reject links that cannot survive relocation."""

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        if info.issym() and Path(info.linkname).is_absolute():
            raise ValueError(f"Non-relocatable package symlink: {info.name}")
        mode = 0o755 if info.isdir() or info.mode & 0o111 else 0o644
        info = tarfile.data_filter(info, prefix)
        info.mode = mode
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        return info

    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(prefix.iterdir()):
                    archive.add(path, arcname=path.name, filter=normalize)


def export_packages(
    manifest: dict,
    profile: str,
    version: str,
    packages: list[tuple[str, Path, Path, Path]],
    output: Path,
) -> None:
    """Export exactly the selected profile's matrix and write consumer-ready metadata."""
    if version == manifest["version"]:
        raise ValueError("Choose a new bundle version; published bundles are immutable")
    selected = [item for item in manifest["artifacts"] if item["profile"] == profile]
    if not selected or sorted(item["name"] for item in selected) != sorted(
        package[0] for package in packages
    ):
        raise ValueError("Installed packages must match the manifest profile exactly")

    snapshot = copy.deepcopy(manifest)
    snapshot["version"] = version
    snapshot["profiles"] = {profile: snapshot["profiles"][profile]}
    snapshot["artifacts"] = copy.deepcopy(selected)
    build_profile = snapshot["profiles"][profile]
    build_profile["build_type"] = "Release"
    if build_profile["system"] == "darwin":
        build_profile["deployment_target"] = "15.0"
    elif build_profile["system"] == "android":
        build_profile["api_level"] = 24
    installed = {
        name: (prefix, source, build) for name, prefix, source, build in packages
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".prebuilts-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        checksums = []
        for artifact in snapshot["artifacts"]:
            name = artifact["name"]
            prefix, source, build = installed[name]
            if not (prefix / "include").is_dir() or not list(prefix.glob("lib/*.a")):
                raise ValueError(
                    f"Missing installed headers or static libraries for {name}"
                )
            if not list((prefix / "lib/cmake").glob("**/*Config.cmake")) and not list(
                (prefix / "lib/cmake").glob("**/*-config.cmake")
            ):
                raise ValueError(f"Missing installed CMake configuration for {name}")
            if not any(
                path.is_file() and path.name.lower().startswith(("license", "copying"))
                for path in source.iterdir()
            ):
                raise ValueError(f"Missing upstream license for {name}")

            package = staging / name
            package.mkdir()
            for directory in ("include", "lib", "bin", "share", "licenses"):
                if (prefix / directory).is_dir():
                    shutil.copytree(
                        prefix / directory, package / directory, symlinks=True
                    )

            _copy_notices(source, package / "licenses" / name)
            # ExternalProject dependencies may populate their own FetchContent sources.
            for dependency in sorted((build / "_deps").glob("*-src")):
                _copy_notices(dependency, package / "licenses" / dependency.name)

            archive = staging / artifact["filename"]

            _archive(package, archive)

            with archive.open("rb") as stream:
                artifact["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            checksums.append(f"{artifact['sha256']}  {artifact['filename']}\n")

        (staging / "dependencies.json").write_text(
            json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copyfile(
            Path(__file__).resolve().parents[1] / "REUSE.toml", staging / "REUSE.toml"
        )
        (staging / "SHA256SUMS").write_text(
            "".join(sorted(checksums)), encoding="utf-8"
        )
        # Do not expose any new metadata until every archive has passed validation.
        output.mkdir(parents=True, exist_ok=True)
        for path in sorted(
            staging.iterdir(), key=lambda path: path.name == "dependencies.json"
        ):
            if path.is_file():
                path.replace(output / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--package",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "PREFIX", "SOURCE", "BUILD"),
    )
    args = parser.parse_args()
    try:
        export_packages(
            json.loads(args.manifest.read_text(encoding="utf-8")),
            args.profile,
            args.version,
            [
                (name, Path(prefix), Path(source), Path(build))
                for name, prefix, source, build in args.package
            ],
            args.output,
        )
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f"Prebuilt export failed: {error}", file=sys.stderr)
        return 1
    print(f"Exported {args.profile} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
