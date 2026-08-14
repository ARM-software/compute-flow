#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Build helpers for the Compute Flow brick manager."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from brick_manager.errors import BrickManagerError
from brick_manager.paths import StatePaths
from brick_manager.tools import ensure_cmake


ANDROID_NDK_VERSION = "r27c"
ANDROID_NDK_REVISION = "27.3.13750724"
ARM_GNU_TOOLCHAIN_VERSION = "14.2.rel1"
LINUX_ARM64_TRIPLE = "aarch64-none-linux-gnu"
_SOURCE_BUILD_OPTIONS = (
    "COMPUTE_FLOW_BUILD_PROTOBUF_FROM_SOURCE",
    "COMPUTE_FLOW_BUILD_LITERT_FROM_SOURCE",
    "COMPUTE_FLOW_BUILD_EXECUTORCH_FROM_SOURCE",
    "COMPUTE_FLOW_BUILD_ONNXRUNTIME_FROM_SOURCE",
    "COMPUTE_FLOW_BUILD_MNN_FROM_SOURCE",
    "COMPUTE_FLOW_BUILD_SENTENCEPIECE_FROM_SOURCE",
)
ANDROID_NDK_ROOT_ENV = "COMPUTE_FLOW_ANDROID_NDK_ROOT"
LINUX_ARM64_CC_ENV = "COMPUTE_FLOW_LINUX_ARM64_CC"
LINUX_ARM64_CXX_ENV = "COMPUTE_FLOW_LINUX_ARM64_CXX"


@dataclass(frozen=True)
class BuildTarget:
    name: str
    description: str
    build_dir: str
    install_prefix: str


@dataclass(frozen=True)
class BuildResult:
    build_dir: Path
    sdk_dir: Path


def available_targets() -> list[BuildTarget]:
    return [
        BuildTarget(
            name="native",
            description=f"Build natively for {_native_description()}",
            build_dir="build_native",
            install_prefix=_native_install_prefix(),
        ),
        BuildTarget(
            name="android_arm64",
            description="Cross-compile for Android™ with Arm® 64-bit CPU",
            build_dir="build_android_arm64",
            install_prefix="android_arm64",
        ),
        BuildTarget(
            name="linux_arm64",
            description="Cross-compile for Linux® with Arm® 64-bit CPU",
            build_dir="build_linux_arm64",
            install_prefix="linux_arm64",
        ),
    ]


def format_target_list() -> str:
    return "\n".join(
        f"{target.name} -> {target.description}" for target in available_targets()
    )


def build_target(
    state_paths: StatePaths,
    target_name: str,
    *,
    jobs: int,
    from_source: bool = False,
    streamline: bool = False,
    source_dir: Path | None = None,
) -> BuildResult:
    target = _select_target(target_name)
    if streamline and target.name == "native" and platform.system() != "Linux":
        raise BrickManagerError(
            "Arm® Streamline annotations require a native Linux® target, "
            "linux_arm64, or android_arm64."
        )
    source = (source_dir or Path.cwd()).resolve()
    build_dir = source / target.build_dir
    install_prefix = source / target.install_prefix
    cmake = ensure_cmake(state_paths)
    configure_args = [
        str(cmake),
        "-S",
        str(source),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        *_dependency_cmake_args(from_source),
        f"-DCOMPUTE_FLOW_ENABLE_STREAMLINE={'ON' if streamline else 'OFF'}",
    ]
    cmake_environment = os.environ.copy()
    cmake_environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)

    if target.name == "android_arm64":
        ndk = _ensure_android_ndk(state_paths)
        configure_args.extend(
            [
                f"-DCMAKE_TOOLCHAIN_FILE={ndk / 'build/cmake/android.toolchain.cmake'}",
                "-DANDROID_ABI=arm64-v8a",
                "-DANDROID_PLATFORM=android-28",
            ]
        )
    elif target.name == "linux_arm64":
        configure_args.extend(_linux_arm64_cmake_args(state_paths, build_dir))

    _run(configure_args, environment=cmake_environment)
    _run(
        [str(cmake), "--build", str(build_dir), "--parallel", str(jobs)],
        environment=cmake_environment,
    )
    _run(
        [
            str(cmake),
            "--build",
            str(build_dir),
            "--target",
            "compute_flow_application_sdk",
        ],
        environment=cmake_environment,
    )
    return BuildResult(build_dir=build_dir, sdk_dir=build_dir / "sdk")


def _dependency_cmake_args(from_source: bool) -> list[str]:
    source_setting = "ON" if from_source else "OFF"
    conan_setting = "OFF" if from_source else "ON"
    return [
        f"-DCOMPUTE_FLOW_USE_CONAN={conan_setting}",
        *(f"-D{option}={source_setting}" for option in _SOURCE_BUILD_OPTIONS),
    ]


def _select_target(target_name: str) -> BuildTarget:
    for target in available_targets():
        if target.name == target_name:
            return target
    available = ", ".join(target.name for target in available_targets())
    raise BrickManagerError(
        f"Unknown build target: {target_name}. Available: {available}"
    )


def _native_description() -> str:
    system = platform.system()
    machine = platform.machine() or "unknown"
    if system == "Darwin":
        return f"macOS® {machine}"
    if system == "Linux":
        return f"Linux® {machine}"
    return f"{system or 'native'} {machine}"


def _native_install_prefix() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else machine or "native"
    if system == "Darwin":
        return f"macos_{arch}"
    if system == "Linux":
        return f"linux_{arch}"
    return f"native_{arch}"


def _ensure_android_ndk(state_paths: StatePaths) -> Path:
    configured = configured_android_ndk()
    if configured is not None:
        return configured

    archive_name = _android_ndk_archive_name()
    toolchain_root = state_paths.home / "toolchains"
    ndk_dir = managed_android_ndk_path(state_paths)
    toolchain_file = ndk_dir / "build/cmake/android.toolchain.cmake"
    if toolchain_file.is_file():
        return ndk_dir

    url = (
        "https://dl.google.com/android/repository/"
        f"android-ndk-{ANDROID_NDK_VERSION}-{archive_name}.zip"
    )
    archive = toolchain_root / f"android-ndk-{ANDROID_NDK_VERSION}-{archive_name}.zip"
    toolchain_root.mkdir(parents=True, exist_ok=True)
    _download(url, archive)
    _extract_zip(archive, toolchain_root)
    if not toolchain_file.is_file():
        raise BrickManagerError(
            f"Downloaded Android™ NDK but did not find toolchain file: {toolchain_file}"
        )
    _check_ndk_revision(ndk_dir)
    return ndk_dir


def configured_android_ndk() -> Path | None:
    configured = os.environ.get(ANDROID_NDK_ROOT_ENV)
    if not configured:
        return None

    ndk = Path(configured).expanduser()
    toolchain_file = ndk / "build/cmake/android.toolchain.cmake"
    if not toolchain_file.is_file():
        raise BrickManagerError(
            f"{ANDROID_NDK_ROOT_ENV} does not identify an Android™ NDK root: "
            f"{configured}"
        )
    return ndk


def _android_ndk_archive_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "darwin"
    if system == "Linux":
        return "linux"
    raise BrickManagerError(f"Android™ NDK download is not supported on {system}")


def _check_ndk_revision(ndk_dir: Path) -> None:
    source_properties = ndk_dir / "source.properties"
    if not source_properties.is_file():
        return
    text = source_properties.read_text(encoding="utf-8", errors="ignore")
    if ANDROID_NDK_REVISION not in text:
        raise BrickManagerError(
            f"Expected Android™ NDK {ANDROID_NDK_REVISION}, but {source_properties} differs."
        )


def _linux_arm64_cmake_args(state_paths: StatePaths, build_dir: Path) -> list[str]:
    machine = platform.machine().lower()
    if platform.system() == "Linux" and machine in ("aarch64", "arm64"):
        return []

    configured = configured_linux_arm64_compilers()
    if configured is not None:
        return _write_linux_toolchain(build_dir, *configured)

    discovered = path_linux_arm64_compilers()
    if discovered is not None:
        return _write_linux_toolchain(build_dir, *discovered)

    gcc_path, gxx_path = _ensure_arm_gnu_toolchain(state_paths)
    return _write_linux_toolchain(build_dir, gcc_path, gxx_path)


def configured_linux_arm64_compilers() -> tuple[Path, Path] | None:
    configured_cc = os.environ.get(LINUX_ARM64_CC_ENV)
    configured_cxx = os.environ.get(LINUX_ARM64_CXX_ENV)
    if not configured_cc and not configured_cxx:
        return None
    if not configured_cc or not configured_cxx:
        raise BrickManagerError(
            f"{LINUX_ARM64_CC_ENV} and {LINUX_ARM64_CXX_ENV} must be set together."
        )
    return (
        _configured_compiler(LINUX_ARM64_CC_ENV, configured_cc),
        _configured_compiler(LINUX_ARM64_CXX_ENV, configured_cxx),
    )


def path_linux_arm64_compilers() -> tuple[Path, Path] | None:
    gcc = shutil.which("aarch64-linux-gnu-gcc")
    gxx = shutil.which("aarch64-linux-gnu-g++")
    if gcc and gxx:
        return Path(gcc), Path(gxx)
    return None


def managed_arm_gnu_compilers(state_paths: StatePaths) -> tuple[Path, Path]:
    toolchain = managed_arm_gnu_toolchain_path(state_paths)
    bin_dir = toolchain / "bin"
    return (
        bin_dir / f"{LINUX_ARM64_TRIPLE}-gcc",
        bin_dir / f"{LINUX_ARM64_TRIPLE}-g++",
    )


def _configured_compiler(environment_variable: str, configured: str) -> Path:
    expanded = str(Path(configured).expanduser())
    resolved = shutil.which(expanded)
    if resolved is None:
        raise BrickManagerError(
            f"{environment_variable} does not identify an executable: {configured}"
        )
    return Path(resolved)


def _ensure_arm_gnu_toolchain(state_paths: StatePaths) -> tuple[Path, Path]:
    toolchain_root = state_paths.home / "toolchains"
    archive_name = _arm_gnu_archive_name()
    compilers = managed_arm_gnu_compilers(state_paths)
    if all(compiler.is_file() for compiler in compilers):
        return compilers

    url = (
        "https://developer.arm.com/-/media/Files/downloads/gnu/"
        f"{ARM_GNU_TOOLCHAIN_VERSION}/binrel/{archive_name}.tar.xz"
    )
    archive = toolchain_root / f"{archive_name}.tar.xz"
    toolchain_root.mkdir(parents=True, exist_ok=True)
    _download(url, archive)
    _extract_tar_xz(archive, toolchain_root)
    if not all(compiler.is_file() for compiler in compilers):
        raise BrickManagerError(
            "Downloaded Arm® GNU toolchain but did not find both compilers: "
            f"{compilers[0]} / {compilers[1]}"
        )
    return compilers


def managed_android_ndk_path(state_paths: StatePaths) -> Path:
    return state_paths.home / "toolchains" / f"android-ndk-{ANDROID_NDK_VERSION}"


def managed_arm_gnu_toolchain_path(state_paths: StatePaths) -> Path:
    return state_paths.home / "toolchains" / _arm_gnu_archive_name()


def _arm_gnu_archive_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        host = "darwin-arm64"
    elif system == "Darwin":
        host = "darwin-x86_64"
    elif system == "Linux" and machine in ("x86_64", "amd64"):
        host = "x86_64"
    elif system == "Linux" and machine in ("aarch64", "arm64"):
        host = "aarch64"
    else:
        raise BrickManagerError(
            f"Linux® Arm® 64-bit toolchain download is not supported on {system} {machine}"
        )
    return f"arm-gnu-toolchain-{ARM_GNU_TOOLCHAIN_VERSION}-{host}-{LINUX_ARM64_TRIPLE}"


def _write_linux_toolchain(build_dir: Path, gcc: Path, gxx: Path) -> list[str]:
    if not gcc.is_file() or not gxx.is_file():
        raise BrickManagerError(f"Linux® Arm® 64-bit compiler not found: {gcc} / {gxx}")
    build_dir.mkdir(parents=True, exist_ok=True)
    toolchain = build_dir / "linux_arm64_toolchain.cmake"
    toolchain.write_text(
        "\n".join(
            [
                "set(CMAKE_SYSTEM_NAME Linux)",
                "set(CMAKE_SYSTEM_PROCESSOR aarch64)",
                f"set(CMAKE_C_COMPILER {gcc})",
                f"set(CMAKE_CXX_COMPILER {gxx})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"]


def _download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    print(f"Downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as error:
        destination.unlink(missing_ok=True)
        raise BrickManagerError(f"Failed to download {url}: {error}") from error


def _extract_zip(archive: Path, destination: Path) -> None:
    print(f"Extracting {archive}", flush=True)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(destination)


def _extract_tar_xz(archive: Path, destination: Path) -> None:
    print(f"Extracting {archive}", flush=True)
    with tarfile.open(archive, "r:xz") as tar_file:
        tar_file.extractall(destination)


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, env=environment)
    except FileNotFoundError as error:
        raise BrickManagerError(f"Command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise BrickManagerError(
            f"Build command failed with exit code {error.returncode}: {' '.join(command)}"
        ) from error
