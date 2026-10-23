#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Exercise CMake release selection, verified extraction, fallback, and packaging."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


_ROOT = Path(__file__).resolve().parents[1]
_CMAKE = (
    os.environ.get("CMAKE_COMMAND")
    or shutil.which("cmake")
    or str(_ROOT / ".venv/bin/cmake")
)
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "compute_flow/python"))

from brick_manager.build import _dependency_cmake_args, build_target  # noqa: E402
from brick_manager.paths import StatePaths  # noqa: E402
from package_prebuilts import export_packages  # noqa: E402


def _extract_archive(artifact: dict, source: Path, prefix: Path) -> None:
    """Exercise the same native CMake extraction used by consumers."""
    result = subprocess.run(
        [
            _CMAKE,
            f"-Darchive_url={(source / artifact['filename']).as_uri()}",
            f"-Dsha256={artifact['sha256']}",
            f"-Dprefix={prefix}",
            "-P",
            str(_ROOT / "cmake/deps/FetchRelease.cmake"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


class DependencyDownloadsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "release"
        self.source.mkdir()
        self.modules = self.root / "cmake"
        self.modules.mkdir()
        self.cache = self.root / "deps/prebuilt"
        self.manifest = json.loads((_ROOT / "cmake/deps/dependencies.json").read_text())
        for name in ("Releases.cmake", "FetchRelease.cmake"):
            shutil.copyfile(_ROOT / "cmake/deps" / name, self.modules / name)
        for artifact in self.manifest["artifacts"]:
            self._archive(artifact)

    def _archive(
        self, artifact: dict, *, member_name: str = "lib/library.a", config: bool = True
    ) -> None:
        archive_path = self.source / artifact["filename"]
        with tarfile.open(archive_path, "w:gz") as archive:
            members = {member_name: b"existing binary payload"}
            if config:
                members[
                    f"lib/cmake/{artifact['name']}/{artifact['name']}Config.cmake"
                ] = b"# installed package"
            for name, content in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(content))
        artifact["sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    def _configure(
        self,
        *,
        system: str = "Darwin",
        profile: str = "",
        architecture: str = "arm64",
        names: tuple[str, ...] = ("litert",),
        source_overrides: tuple[str, ...] = (),
        base_url: str | None = None,
    ) -> tuple[bool, list[Path]]:
        (self.modules / "dependencies.json").write_text(json.dumps(self.manifest))
        options = "\n".join(
            f"set(COMPUTE_FLOW_ENABLE_{name.upper()} ON)" for name in names
        )
        options += "\n" + "\n".join(
            f"set(COMPUTE_FLOW_BUILD_{name.upper()}_FROM_SOURCE ON)"
            for name in source_overrides
        )
        script = self.root / "check.cmake"
        script.write_text(f'''cmake_minimum_required(VERSION 3.24)
set(CMAKE_SYSTEM_NAME "{system}")
set(CMAKE_SYSTEM_PROCESSOR "{architecture}")
set(ANDROID_ABI arm64-v8a)
set(COMPUTE_FLOW_DEPENDENCY_PROFILE "{profile}")
set(COMPUTE_FLOW_DEPS_PREFIX_DIR "{self.root / "deps"}")
set(COMPUTE_FLOW_DEPENDENCY_BASE_URL "{str(self.source) if base_url is None else base_url}")
{options}
foreach(package IN ITEMS Protobuf litert executorch onnxruntime MNN sentencepiece)
    set(${{package}}_DIR stale-release CACHE PATH "" FORCE)
endforeach()
function(compute_flow_enable_source_dependencies)
    set(fell_back ON PARENT_SCOPE)
endfunction()
include("{self.modules / "Releases.cmake"}")
if(release_prefixes)
    foreach(package IN ITEMS Protobuf litert executorch onnxruntime MNN sentencepiece)
        if(DEFINED ${{package}}_DIR)
            message(FATAL_ERROR "Stale package configuration survived: ${{package}}")
        endif()
    endforeach()
endif()
file(WRITE "{self.root / "result"}" "${{fell_back}}\n${{CMAKE_PREFIX_PATH}}")
''')
        result = subprocess.run(
            [_CMAKE, "-P", str(script)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        fallback, prefixes = (self.root / "result").read_text().split("\n", 1)
        return fallback == "ON", [
            Path(prefix) for prefix in prefixes.split(";") if prefix
        ]

    def test_default_profiles_preserve_all_six_libraries(self) -> None:
        for system, profile in (
            ("Linux", "linux_aarch64"),  # trademark-check: allow
            ("Darwin", "macos_aarch64"),
            ("Android", "android_aarch64"),  # trademark-check: allow
        ):
            with self.subTest(system=system):
                fallback, prefixes = self._configure(
                    system=system,
                    names=(
                        "litert",
                        "executorch",
                        "onnxruntime",
                        "mnn",
                        "sentencepiece",
                    ),
                )
                self.assertFalse(fallback)
                expected = {
                    self.cache / item["sha256"]
                    for item in self.manifest["artifacts"]
                    if item["profile"] == profile
                }
                self.assertEqual(set(prefixes), expected)
                self.assertEqual(len(prefixes), 6)

    def test_missing_library_falls_back(self) -> None:
        self.manifest["artifacts"] = [
            item for item in self.manifest["artifacts"] if item["name"] != "onnxruntime"
        ]
        fallback, prefixes = self._configure(names=("executorch",))
        self.assertFalse(fallback)
        self.assertEqual(len(prefixes), 1)
        self.assertEqual(self._configure(names=("onnxruntime",)), (True, []))

    def test_unknown_target_or_mismatched_profile_falls_back(self) -> None:
        for options in (
            {"architecture": "x86_64"},
            {"architecture": "arm64;x86_64"},
            {"profile": "linux_aarch64"},
            {"profile": "unknown"},
            {"system": "Unknown"},
        ):
            with self.subTest(options=options):
                self.assertEqual(self._configure(**options), (True, []))

    def test_cached_prefix_is_reused_without_download_host(self) -> None:
        fallback, prefixes = self._configure()
        self.assertFalse(fallback)
        self.assertEqual(self._configure(base_url=""), (False, prefixes))

    def test_file_url_and_source_overrides(self) -> None:
        fallback, prefixes = self._configure(base_url=self.source.as_uri())
        self.assertFalse(fallback)
        self.assertEqual(len(prefixes), 1)
        self.assertEqual(self._configure(source_overrides=("litert",)), (False, []))

    def test_missing_archive_and_checksum_mismatch_fall_back(self) -> None:
        artifact = next(
            item
            for item in self.manifest["artifacts"]
            if item["name"] == "litert" and item["profile"] == "macos_aarch64"
        )
        archive = self.source / artifact["filename"]
        archive.write_bytes(b"corrupt download")
        self.assertEqual(self._configure(), (True, []))
        archive.unlink()
        self.assertEqual(self._configure(), (True, []))
        self.assertFalse(list(self.cache.glob("*/.complete")))

    def test_unsafe_archive_or_missing_configuration_is_rejected(self) -> None:
        artifact = next(
            item
            for item in self.manifest["artifacts"]
            if item["name"] == "litert" and item["profile"] == "macos_aarch64"
        )
        for options in (
            {"member_name": "../outside"},
            {"member_name": "lib/../../outside"},
            {"member_name": str(self.root / "outside")},
            {"config": False},
        ):
            with self.subTest(options=options):
                self._archive(artifact, **options)
                self.assertEqual(self._configure(), (True, []))
                self.assertFalse(list(self.cache.glob("*/.complete")))
                self.assertFalse((self.cache / "outside").exists())
                self.assertFalse((self.root / "outside").exists())

    def test_invalid_mirrors_use_next_source(self) -> None:
        artifact = next(
            item
            for item in self.manifest["artifacts"]
            if item["name"] == "litert" and item["profile"] == "macos_aarch64"
        )
        corrupt = self.root / "corrupt"
        corrupt.mkdir()
        (corrupt / artifact["filename"]).write_bytes(b"corrupt download")

        unavailable = f"{self.root / 'missing'};{corrupt}"
        self.assertEqual(self._configure(base_url=unavailable), (True, []))
        fallback, prefixes = self._configure(
            base_url=f"{unavailable};{self.source.as_uri()}/"
        )
        self.assertFalse(fallback)
        self.assertEqual(prefixes, [self.cache / artifact["sha256"]])

    def test_build_uses_running_python_without_virtualenv_activation(self) -> None:
        environment = {
            key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("brick_manager.build.ensure_cmake", return_value=Path("cmake")),
            patch("brick_manager.build._run") as run,
        ):
            build_target(StatePaths(self.root), "native", jobs=1, source_dir=self.root)
        self.assertIn(
            f"-DPython3_EXECUTABLE={sys.executable}", run.call_args_list[0].args[0]
        )

    def test_android_build_accepts_fresh_r27c_download(self) -> None:
        ndk = self.root / "toolchains/android-ndk-r27c"

        def extract_ndk(_archive: Path, _destination: Path) -> None:
            toolchain_file = ndk / "build/cmake/android.toolchain.cmake"
            toolchain_file.parent.mkdir(parents=True)
            toolchain_file.touch()
            bin_dir = ndk / "toolchains/llvm/prebuilt/linux-x86_64/bin"
            bin_dir.mkdir(parents=True)
            compiler = bin_dir / "clang-18"
            compiler.touch(mode=0o755)
            (bin_dir / "clang").symlink_to(compiler.name)
            (ndk / "source.properties").write_text(
                "Pkg.Desc = Android NDK\n"  # trademark-check: allow
                "Pkg.Revision = 27.2.12479018\n"
                "Pkg.ReleaseName = r27c\n"
            )

        with (
            patch.dict(os.environ, {"COMPUTE_FLOW_ANDROID_NDK_ROOT": ""}),
            patch("brick_manager.build.ensure_cmake", return_value=Path("cmake")),
            patch("brick_manager.build._download"),
            patch("brick_manager.build._extract_zip", side_effect=extract_ndk),
            patch("brick_manager.build._run") as run,
        ):
            build_target(
                StatePaths(self.root), "android_arm64", jobs=1, source_dir=self.root
            )

        self.assertIn(
            f"-DCMAKE_TOOLCHAIN_FILE={ndk / 'build/cmake/android.toolchain.cmake'}",
            run.call_args_list[0].args[0],
        )

    def test_build_modes_explicitly_reset_dependency_options(self) -> None:
        self.assertIn(
            "-DCOMPUTE_FLOW_USE_PREBUILT_DEPS=ON", _dependency_cmake_args(False)
        )
        self.assertIn(
            "-DCOMPUTE_FLOW_USE_PREBUILT_DEPS=OFF", _dependency_cmake_args(True)
        )
        for option in _dependency_cmake_args(True)[1:]:
            self.assertTrue(option.endswith("=ON"))
        for option in _dependency_cmake_args(False)[1:]:
            self.assertTrue(option.endswith("=OFF"))


class PrebuiltExportsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = json.loads((_ROOT / "cmake/deps/dependencies.json").read_text())
        # This fixture installs only LiteRT in its default export profile.
        self.manifest["artifacts"] = [
            item
            for item in self.manifest["artifacts"]
            if item["profile"] != "linux_aarch64" or item["name"] == "litert"
        ]
        self.prefix = self.root / "install"
        self.output = self.root / "exports"
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "LICENSE").write_text("upstream license")
        for name in (
            "include/litert.h",
            "lib/liblitert.a",
            "lib/cmake/litert/litertConfig.cmake",
        ):
            path = self.prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("installed file")

    def _export(self, profile: str = "linux_aarch64") -> None:
        export_packages(
            self.manifest,
            profile,
            "local",
            [("litert", self.prefix, self.source, self.root / "build")],
            self.output,
        )

    def test_export_round_trip_and_stable_archive(self) -> None:
        self._export()
        snapshot = json.loads((self.output / "dependencies.json").read_text())
        (artifact,) = snapshot["artifacts"]
        selected = next(
            item
            for item in self.manifest["artifacts"]
            if item["profile"] == "linux_aarch64" and item["name"] == "litert"
        )
        self.assertEqual(artifact["filename"], selected["filename"])
        self.assertEqual(artifact["version"], selected["version"])
        self.assertRegex(artifact["version"], r"^[0-9a-f]{40}$")
        prefix = self.root / "cache" / artifact["sha256"]
        _extract_archive(artifact, self.output, prefix)
        self.assertEqual((prefix / "lib/liblitert.a").read_text(), "installed file")
        self.assertEqual(
            (prefix / "licenses/litert/LICENSE").read_text(), "upstream license"
        )
        self.assertEqual(
            (self.output / "SHA256SUMS").read_text(),
            f"{artifact['sha256']}  {artifact['filename']}\n",
        )
        before = (self.output / artifact["filename"]).read_bytes()
        self.assertNotIn("size", artifact)
        self.assertNotIn("cmake_config_dir", artifact)
        self.assertFalse(list(self.output.glob("*.license")))
        self.assertEqual(
            (self.output / "REUSE.toml").read_bytes(),
            (_ROOT / "REUSE.toml").read_bytes(),
        )
        os.utime(self.prefix / "lib/liblitert.a", (100, 100))
        self._export()
        self.assertEqual(before, (self.output / artifact["filename"]).read_bytes())

    def test_incomplete_package_is_rejected(self) -> None:
        for missing in (
            self.prefix / "lib/cmake/litert/litertConfig.cmake",
            self.source / "LICENSE",
        ):
            with self.subTest(missing=missing):
                content = missing.read_bytes()
                missing.unlink()
                with self.assertRaises(ValueError):
                    self._export()
                missing.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "match the manifest profile"):
            self._export("macos_aarch64")

    def test_android_export_records_the_publishing_api_level(self) -> None:
        self.manifest["artifacts"] = [
            item
            for item in self.manifest["artifacts"]
            if item["profile"] != "android_aarch64" or item["name"] == "executorch"
        ]
        config = self.prefix / "lib/cmake/ExecuTorch/ExecuTorchConfig.cmake"
        config.parent.mkdir(parents=True)
        config.write_text("# installed configuration")
        export_packages(
            self.manifest,
            "android_aarch64",
            "local",
            [("executorch", self.prefix, self.source, self.root / "build")],
            self.output,
        )
        snapshot = json.loads((self.output / "dependencies.json").read_text())
        self.assertEqual(snapshot["profiles"]["android_aarch64"]["api_level"], 24)
        self.assertNotIn("api_level", self.manifest["profiles"]["android_aarch64"])

    def test_non_relocatable_symlink_is_rejected(self) -> None:
        (self.prefix / "include/external.h").symlink_to(self.source / "LICENSE")
        with self.assertRaisesRegex(ValueError, "Non-relocatable"):
            self._export()


if __name__ == "__main__":
    unittest.main()
