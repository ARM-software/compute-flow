#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Environment checks for the Compute Flow brick manager."""

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from brick_manager.build import (
    ANDROID_NDK_ROOT_ENV,
    ANDROID_NDK_VERSION,
    ARM_GNU_TOOLCHAIN_VERSION,
    LINUX_ARM64_CC_ENV,
    LINUX_ARM64_CXX_ENV,
    configured_android_ndk,
    configured_linux_arm64_compilers,
    managed_android_ndk_path,
    managed_arm_gnu_compilers,
    path_linux_arm64_compilers,
)
from brick_manager.errors import BrickManagerError
from brick_manager.paths import StatePaths
from brick_manager.tools import (
    CMAKE_EXECUTABLE_ENV,
    CMAKE_VERSION,
    GIT_LFS_EXECUTABLE_ENV,
    GIT_LFS_VERSION,
    configured_cmake,
    configured_git_lfs,
    find_managed_cmake,
    find_managed_git_lfs,
)


@dataclass(frozen=True)
class DependencyCheck:
    name: str
    executable: str
    reason: str
    path: str | None
    install_actions: tuple[str, ...]
    version: str | None = None
    toolchain: str | None = None

    @property
    def installed(self) -> bool:
        return self.path is not None


@dataclass(frozen=True)
class ManagedItemCheck:
    name: str
    version: str | None
    provider: str
    path: str | None
    status: str
    error: str | None = None

    @property
    def installed(self) -> bool:
        return self.status in ("available", "installed")

    @property
    def invalid(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class CheckResult:
    dependencies: list[DependencyCheck]
    native_toolchain: list[DependencyCheck]
    tools: list[ManagedItemCheck]
    toolchains: list[ManagedItemCheck]

    @property
    def missing(self) -> list[DependencyCheck]:
        return [
            dependency
            for dependency in (*self.dependencies, *self.native_toolchain)
            if not dependency.installed
        ]

    @property
    def invalid_overrides(self) -> list[ManagedItemCheck]:
        return [item for item in (*self.tools, *self.toolchains) if item.invalid]

    @property
    def has_errors(self) -> bool:
        return bool(self.missing or self.invalid_overrides)


def check_environment(state_paths: StatePaths) -> CheckResult:
    dependencies = [
        _dependency(
            "Git",
            "git",
            "Required to copy git worktrees and download remote model artifacts.",
        ),
    ]
    return CheckResult(
        dependencies=dependencies,
        native_toolchain=_native_toolchain(),
        tools=_managed_tools(state_paths),
        toolchains=_managed_toolchains(state_paths),
    )


def _dependency(name: str, executable: str, reason: str) -> DependencyCheck:
    return DependencyCheck(
        name=name,
        executable=executable,
        reason=reason,
        path=shutil.which(executable),
        install_actions=_install_actions(executable),
    )


def _install_actions(executable: str) -> tuple[str, ...]:
    system = platform.system()
    if system == "Darwin":
        return _macos_actions(executable)
    if system == "Linux":
        return _linux_actions(executable)
    return (f"Install {executable} and make sure it is available on PATH.",)


def _macos_actions(executable: str) -> tuple[str, ...]:
    package = _package_name(executable)
    return (f"macOS®: brew install {package}",)


def _linux_actions(executable: str) -> tuple[str, ...]:
    package = _package_name(executable)
    return (
        f"Ubuntu/Debian: sudo apt update && sudo apt install {package}",
        f"Fedora: sudo dnf install {package}",
    )


def _package_name(executable: str) -> str:
    return executable


def _native_toolchain() -> list[DependencyCheck]:
    return [
        _native_compiler("C compiler", "CC", "cc"),
        _native_compiler("C++ compiler", "CXX", "c++"),
    ]


def _native_compiler(
    name: str, environment_variable: str, default_executable: str
) -> DependencyCheck:
    executable = _compiler_executable(environment_variable, default_executable)
    path = shutil.which(executable)
    toolchain, version = _compiler_details(path)
    return DependencyCheck(
        name=name,
        executable=executable,
        reason="Required to build Compute Flow for the native target.",
        path=path,
        install_actions=_native_toolchain_install_actions(),
        version=version,
        toolchain=toolchain,
    )


def _compiler_executable(environment_variable: str, default: str) -> str:
    configured = os.environ.get(environment_variable)
    if not configured:
        return default
    try:
        arguments = shlex.split(configured, posix=platform.system() != "Windows")
    except ValueError:
        return configured
    return arguments[0] if arguments else default


def _compiler_details(path: str | None) -> tuple[str | None, str | None]:
    if path is None:
        return None, None
    try:
        result = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    output = result.stdout or result.stderr
    first_line = next(
        (line.strip() for line in output.splitlines() if line.strip()), ""
    )
    lowered_output = output.lower()
    if "clang" in lowered_output:
        toolchain = "Clang"
    elif "free software foundation" in lowered_output or "gcc" in lowered_output:
        toolchain = "GCC"
    else:
        toolchain = Path(path).name

    version_match = re.search(r"\bversion\s+([^\s]+)", first_line, re.IGNORECASE)
    if version_match is None:
        versions = re.findall(r"\d+(?:\.\d+)+", first_line)
        version = versions[-1] if versions else None
    else:
        version = version_match.group(1)
    return toolchain, version


def _native_toolchain_install_actions() -> tuple[str, ...]:
    system = platform.system()
    if system == "Darwin":
        return ("macOS®: xcode-select --install",)
    if system == "Linux":
        return (
            "Ubuntu/Debian: sudo apt update && sudo apt install build-essential",
            "Fedora: sudo dnf install gcc gcc-c++",
        )
    return ("Install C and C++ compilers and make sure they are available on PATH.",)


def _managed_tools(state_paths: StatePaths) -> list[ManagedItemCheck]:
    return [
        _executable_check(
            "CMake",
            CMAKE_VERSION,
            CMAKE_EXECUTABLE_ENV,
            configured_cmake,
            lambda: find_managed_cmake(state_paths),
        ),
        _executable_check(
            "Git LFS",
            GIT_LFS_VERSION,
            GIT_LFS_EXECUTABLE_ENV,
            configured_git_lfs,
            lambda: find_managed_git_lfs(state_paths),
        ),
    ]


def _managed_toolchains(state_paths: StatePaths) -> list[ManagedItemCheck]:
    return [
        _android_ndk_check(state_paths),
        _linux_arm64_toolchain_check(state_paths),
    ]


def _executable_check(
    name: str,
    managed_version: str,
    environment_variable: str,
    configured_executable: Callable[[], Path | None],
    managed_executable: Callable[[], Path | None],
) -> ManagedItemCheck:
    try:
        configured = configured_executable()
    except BrickManagerError as error:
        return _invalid_override(name, (environment_variable,), str(error))

    if configured is not None:
        return ManagedItemCheck(
            name=name,
            version=_executable_version(configured),
            provider="User",
            path=str(configured),
            status="available",
        )

    managed = managed_executable()
    if managed is not None:
        return ManagedItemCheck(
            name=name,
            version=_executable_version(managed) or managed_version,
            provider="Brick Manager",
            path=str(managed),
            status="installed",
        )

    return ManagedItemCheck(
        name=name,
        version=managed_version,
        provider="Brick Manager",
        path=None,
        status="will be downloaded when needed",
    )


def _android_ndk_check(state_paths: StatePaths) -> ManagedItemCheck:
    try:
        configured = configured_android_ndk()
    except BrickManagerError as error:
        return _invalid_override("Android™ NDK", (ANDROID_NDK_ROOT_ENV,), str(error))

    if configured is not None:
        return ManagedItemCheck(
            name="Android™ NDK",
            version=_android_ndk_version(configured),
            provider="User",
            path=str(configured),
            status="available",
        )

    managed = managed_android_ndk_path(state_paths)
    if (managed / "build/cmake/android.toolchain.cmake").is_file():
        return ManagedItemCheck(
            name="Android™ NDK",
            version=_android_ndk_version(managed) or ANDROID_NDK_VERSION,
            provider="Brick Manager",
            path=str(managed),
            status="installed",
        )

    return ManagedItemCheck(
        name="Android™ NDK",
        version=ANDROID_NDK_VERSION,
        provider="Brick Manager",
        path=None,
        status="will be downloaded when needed",
    )


def _linux_arm64_toolchain_check(state_paths: StatePaths) -> ManagedItemCheck:
    try:
        configured = configured_linux_arm64_compilers()
    except BrickManagerError as error:
        return _invalid_override(
            "Linux® Arm® 64-bit toolchain",
            (LINUX_ARM64_CC_ENV, LINUX_ARM64_CXX_ENV),
            str(error),
        )

    if configured is not None:
        return _compiler_pair_check(
            configured,
            "User",
            "available",
        )

    discovered = path_linux_arm64_compilers()
    if discovered is not None:
        return _compiler_pair_check(discovered, "User (PATH)", "available")

    managed = managed_arm_gnu_compilers(state_paths)
    if all(compiler.is_file() for compiler in managed):
        return _compiler_pair_check(managed, "Brick Manager", "installed")

    return ManagedItemCheck(
        name="Linux® Arm® 64-bit toolchain",
        version=ARM_GNU_TOOLCHAIN_VERSION,
        provider="Brick Manager",
        path=None,
        status="will be downloaded when needed",
    )


def _compiler_pair_check(
    compilers: tuple[Path, Path], provider: str, status: str
) -> ManagedItemCheck:
    cc_version = _compiler_details(str(compilers[0]))[1]
    cxx_version = _compiler_details(str(compilers[1]))[1]
    if cc_version == cxx_version:
        version = cc_version
    else:
        version = f"C {cc_version or '-'}, C++ {cxx_version or '-'}"
    return ManagedItemCheck(
        name="Linux® Arm® 64-bit toolchain",
        version=version,
        provider=provider,
        path=f"{compilers[0]} / {compilers[1]}",
        status=status,
    )


def _invalid_override(
    name: str, environment_variables: tuple[str, ...], error: str
) -> ManagedItemCheck:
    configured = " / ".join(
        os.environ.get(variable, "-") for variable in environment_variables
    )
    return ManagedItemCheck(
        name=name,
        version=None,
        provider="User",
        path=configured,
        status="invalid override",
        error=error,
    )


def _executable_version(executable: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout or result.stderr
    match = re.search(r"\d+(?:\.\d+)+(?:[-+.][0-9A-Za-z.-]+)?", output)
    return match.group(0) if match else None


def _android_ndk_version(ndk: Path) -> str | None:
    source_properties = ndk / "source.properties"
    if not source_properties.is_file():
        return None
    text = source_properties.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^Pkg\.Revision\s*=\s*(\S+)", text, re.MULTILINE)
    return match.group(1) if match else None
