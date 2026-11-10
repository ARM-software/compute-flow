#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Installation checks for brick custom operator packages."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.cache import load_cache, write_cache  # noqa: E402
from brick_manager.custom_ops.validation import validate_custom_ops  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.identity import BrickSelector  # noqa: E402
from brick_manager.install import install_brick  # noqa: E402
from brick_manager.paths import StatePaths  # noqa: E402
from brick_manager.uninstall import uninstall_brick  # noqa: E402
from brick_manager.update import update_cache  # noqa: E402


_SCHEMA = """operators:
  - name: add_fp32
    version: 1
    inputs:
      - {name: lhs, dtype: float32, shape: [N]}
      - {name: rhs, dtype: float32, shape: [N]}
    outputs:
      - {name: sum, dtype: float32, shape: [N]}
"""

_REFERENCE_C = """#include "compute_flow/custom_ops/v1/types.h"

int64_t CF_GET_PERSISTENT_MEMORY_REQUIREMENT(
    const CF_TensorFormatArray* src, CF_MemoryRequirement* requirement) {
    (void)src;
    requirement->size = 0;
    requirement->alignment = 1;
    return 0;
}

int64_t CF_GET_SCRATCH_MEMORY_REQUIREMENT(
    const CF_TensorFormatArray* src, const CF_TensorFormatArray* dst,
    CF_MemoryRequirement* requirement) {
    (void)src;
    (void)dst;
    requirement->size = 0;
    requirement->alignment = 1;
    return 0;
}

int64_t CF_SETUP(const CF_TensorArray* src, CF_Workspace* persistent) {
    (void)src;
    (void)persistent;
    return 0;
}

int64_t CF_RUN(const CF_TensorArray* src, CF_TensorArray* dst,
    const CF_Workspace* persistent, CF_Workspace* scratch,
    const CF_ThreadInfo* thread) {
    (void)src;
    (void)dst;
    (void)persistent;
    (void)scratch;
    (void)thread;
    return 0;
}
"""

_SHAPE_C = """#include "compute_flow/custom_ops/v1/types.h"

int64_t CF_CALCULATE_OUTPUT_SHAPES(
    const CF_TensorShape* inputs, size_t num_inputs,
    CF_TensorShape* outputs, size_t num_outputs) {
    if (num_inputs != 2 || num_outputs != 1 ||
        inputs[0].rank != 1 || inputs[1].rank != 1 ||
        inputs[0].dims[0] != inputs[1].dims[0]) {
        return 1;
    }
    outputs[0].rank = 1;
    outputs[0].dims[0] = inputs[0].dims[0];
    return 0;
}
"""

_OPTIMIZED_ASSEMBLY_C = _REFERENCE_C.replace(
    '#include "compute_flow/custom_ops/v1/types.h"\n',
    '#include "compute_flow/custom_ops/v1/types.h"\n'
    "#ifdef COMPUTE_FLOW_HAS_ASSEMBLY\n"
    "extern int64_t COMPUTE_FLOW_ASM_return_seven(void);\n"
    "#endif\n",
).replace(
    "    (void)thread;\n    return 0;",
    "    (void)thread;\n"
    "#ifdef COMPUTE_FLOW_HAS_ASSEMBLY\n"
    "    return COMPUTE_FLOW_ASM_return_seven();\n"
    "#else\n"
    "    return 3;\n"
    "#endif",
)

_OPTIMIZED_ASSEMBLY_S = """.text
.globl COMPUTE_FLOW_ASM_return_seven
COMPUTE_FLOW_ASM_return_seven:
    mov x0, #7
    ret
"""


class CustomOpsInstallTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.state_paths = StatePaths(self.root / "state")
        self.selector = BrickSelector("tts", "example", "1.0.0")
        self.install_path = self.state_paths.brick_source(self.selector)
        self.info_json = {
            "name": "Example",
            "type": "tts",
            "description": "Example brick.",
            "author": "Example author",
            "version": "1.0.0",
            "models": {},
        }
        (self.source / "info.json").write_text(
            json.dumps(self.info_json), encoding="utf-8"
        )
        (self.source / "Example.cpp").write_text("", encoding="utf-8")
        (self.source / "Example.h").write_text("", encoding="utf-8")
        write_cache(
            self.state_paths.cache,
            {
                "bricks": {
                    self.selector.key: {
                        self.selector.version: {
                            "info_json": self.info_json,
                            "metadata": {
                                "source_type": "local_directory",
                                "source_path": str(self.source),
                                "installed": False,
                                "installed_source_path": None,
                            },
                        }
                    }
                }
            },
        )

    def _write_custom_ops(self, schema: str = _SCHEMA) -> Path:
        custom_ops = self.source / "custom_ops"
        operator = custom_ops / "add_fp32"
        operator.mkdir(parents=True)
        (custom_ops / "schema.yaml").write_text(schema, encoding="utf-8")
        (operator / "reference.c").write_text(
            '#include "compute_flow/custom_ops/v1/types.h"\n', encoding="utf-8"
        )
        (operator / "calculate_output_shapes.c").write_text(_SHAPE_C, encoding="utf-8")
        return custom_ops

    def _build_reference(
        self, *, expect_success: bool
    ) -> subprocess.CompletedProcess[str]:
        if not shutil.which("cmake"):
            self.skipTest("CMake is unavailable")
        project = self.root / "project"
        project.mkdir()
        (project / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(reference_fixture LANGUAGES C CXX)\n"
            "add_library(compute_flow::compute_flow INTERFACE IMPORTED GLOBAL)\n"
            "set_target_properties(compute_flow::compute_flow PROPERTIES\n"
            f'    INTERFACE_INCLUDE_DIRECTORIES "{REPOSITORY_ROOT / "include"}")\n'
            f'add_subdirectory("{self.install_path}" brick)\n',
            encoding="utf-8",
        )
        build_dir = self.root / "build"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(project),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                "compute_flow_brick_tts_example_add_fp32_reference_check",
                "Example_core",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode == 0, expect_success, result.stderr)
        return result

    def _write_optimized_assembly(self, assembly: str) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(_REFERENCE_C, encoding="utf-8")
        cpu = custom_ops / "add_fp32/cpu"
        arm = cpu / "arm"
        arm.mkdir(parents=True)
        (cpu / "registration.yaml").write_text(
            "name: add_fp32\n"
            "version: 1\n"
            "kernels:\n"
            "  - kernel: arm/add_plain\n"
            "    target: arm\n"
            "    hw_features: []\n"
            "    build_features: []\n",
            encoding="utf-8",
        )
        (arm / "add_plain.c").write_text(_OPTIMIZED_ASSEMBLY_C, encoding="utf-8")
        (arm / "add_plain.S").write_text(assembly, encoding="utf-8")

    def _build_optimized_assembly_probe(self, expected: int) -> str:
        if not shutil.which("cmake") or not shutil.which("ctest"):
            self.skipTest("CMake or CTest is unavailable")

        generated = self.install_path / "generated_custom_ops"
        header = next(generated.glob("*_optimized_*.h"))
        project = self.root / "project"
        project.mkdir()
        (project / "probe.cpp").write_text(
            f'#include "{header.name}"\n'
            "int main() {\n"
            "    CF_TensorArray tensors = {0};\n"
            "    CF_Workspace workspace = {0};\n"
            "    CF_ThreadInfo thread = {0};\n"
            f"    return {header.stem}_run(&tensors, &tensors, &workspace, &workspace, &thread) == {expected} ? 0 : 1;\n"
            "}\n",
            encoding="utf-8",
        )
        (project / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(assembly_fixture LANGUAGES C CXX)\n"
            "enable_testing()\n"
            "add_library(compute_flow::compute_flow INTERFACE IMPORTED GLOBAL)\n"
            "set_target_properties(compute_flow::compute_flow PROPERTIES\n"
            f'    INTERFACE_INCLUDE_DIRECTORIES "{REPOSITORY_ROOT / "include"}")\n'
            f'add_subdirectory("{self.install_path}" brick)\n'
            "add_executable(probe probe.cpp)\n"
            "target_link_libraries(probe PRIVATE Example_core)\n"
            f'target_include_directories(probe PRIVATE "{generated}")\n'
            "add_test(NAME assembly_probe COMMAND probe)\n",
            encoding="utf-8",
        )
        build = self.root / "build"
        configure = subprocess.run(
            ["cmake", "-S", str(project), "-B", str(build)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "probe"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["ctest", "--test-dir", str(build), "--output-on-failure"],
            check=True,
            capture_output=True,
            text=True,
        )
        return configure.stderr

    def test_generates_qualified_optimized_assembly(self) -> None:
        self._write_optimized_assembly(_OPTIMIZED_ASSEMBLY_S)

        install_brick(self.state_paths, self.selector.identity)

        generated = self.install_path / "generated_custom_ops"
        c_source = next(generated.glob("*_optimized_*.c")).read_text(encoding="utf-8")
        asm_source = next(generated.glob("*_optimized_*.S")).read_text(encoding="utf-8")
        self.assertNotIn("COMPUTE_FLOW_ASM_return_seven", c_source)
        self.assertNotIn("COMPUTE_FLOW_ASM_return_seven", asm_source)
        self.assertIn("_asm_return_seven", c_source)
        self.assertIn("COMPUTE_FLOW_CUSTOM_OP_ASM_LABEL(", asm_source)

    @unittest.skipUnless(
        platform.machine().lower() in {"aarch64", "arm64"},
        "Requires an AArch64 build host",
    )
    def test_builds_optimized_assembly_and_uses_it(self) -> None:
        self._write_optimized_assembly(_OPTIMIZED_ASSEMBLY_S)

        install_brick(self.state_paths, self.selector.identity)

        warnings = self._build_optimized_assembly_probe(expected=7)
        self.assertNotIn("Using C fallback", warnings)

    @unittest.skipUnless(
        platform.machine().lower() in {"aarch64", "arm64"},
        "Requires an AArch64 build host",
    )
    def test_uses_c_when_assembly_cannot_compile(self) -> None:
        self._write_optimized_assembly("invalid assembly\n")

        install_brick(self.state_paths, self.selector.identity)

        warnings = self._build_optimized_assembly_probe(expected=3)
        self.assertIn("Using C fallback for optimized custom operator", warnings)

    def test_builds_reference_with_empty_registration(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(_REFERENCE_C, encoding="utf-8")
        cpu = custom_ops / "add_fp32/cpu"
        cpu.mkdir()
        (cpu / "registration.yaml").write_text("", encoding="utf-8")
        install_brick(self.state_paths, self.selector.identity)

        generated = self.install_path / "generated_custom_ops"
        headers = list(generated.glob("*_reference_*.h"))
        references = [
            path
            for path in generated.glob("*_reference_*.c")
            if "_check" not in path.stem
        ]
        shapes = list(generated.glob("*_calculate_output_shapes_*.c"))
        shape_headers = list(generated.glob("*_calculate_output_shapes_*.h"))
        checks = list(generated.glob("*_check.cpp"))
        self.assertEqual(len(headers), 1)
        self.assertEqual(len(references), 1)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(len(shape_headers), 1)
        self.assertEqual(len(checks), 1)
        self.assertEqual(set(generated.glob("*.c")), set(references + shapes))
        registry_source = next(generated.glob("*_registry_*.cpp")).read_text(
            encoding="utf-8"
        )
        registry_header = next(generated.glob("*_registry_*.h"))
        brick_cmake = (self.install_path / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn(
            f'COMPUTE_FLOW_CUSTOM_OPS_REGISTRY_HEADER "generated_custom_ops/{registry_header.name}"',
            brick_cmake,
        )
        self.assertIn(
            f'COMPUTE_FLOW_CUSTOM_OPS_REGISTRY_SYMBOL "{registry_header.stem}"',
            brick_cmake,
        )
        self.assertIn('"add_fp32"', registry_source)
        self.assertEqual(headers[0].stem, references[0].stem)
        header = headers[0].read_text(encoding="utf-8")
        reference = references[0].read_text(encoding="utf-8")
        check = checks[0].read_text(encoding="utf-8")
        symbol = f"{headers[0].stem}_run"
        self.assertIn(f"int64_t {symbol}(", header)
        self.assertIn(f"int64_t {symbol}(", reference)
        self.assertIn(f"{symbol}(", check)
        shape_symbol = shape_headers[0].stem
        self.assertIn(
            f"int64_t {shape_symbol}(", shape_headers[0].read_text(encoding="utf-8")
        )
        self.assertIn(f"int64_t {shape_symbol}(", shapes[0].read_text(encoding="utf-8"))
        self.assertIn(f"{shape_symbol}(", check)
        self.assertIn(shape_symbol, registry_source)
        self.assertNotIn("volatile", check)
        self.assertIn(f'#include "{headers[0].name}"', reference)
        self.assertIn(f'#include "{headers[0].name}"', check)
        self.assertIn('#include "compute_flow/custom_ops/v1/types.h"', header)
        self.assertNotIn("typedef struct", header)
        self.assertNotIn('reference.c"', reference)
        for generated_source in (reference, header, check):
            self.assertNotIn("#define CF_RUN", generated_source)
            self.assertNotIn("#define CF_SETUP", generated_source)
            self.assertNotIn("#define CF_GET_", generated_source)

        self._build_reference(expect_success=True)

    @unittest.skipUnless(
        platform.machine().lower() in {"aarch64", "arm64"},
        "Optimized Arm kernels require an AArch64 build host",
    )
    def test_builds_declared_optimized_c_source(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(_REFERENCE_C, encoding="utf-8")
        cpu = custom_ops / "add_fp32/cpu"
        arm = cpu / "arm"
        arm.mkdir(parents=True)
        (cpu / "registration.yaml").write_text(
            "name: add_fp32\n"
            "version: 1\n"
            "kernels:\n"
            "  - kernel: arm/add_plain\n"
            "    target: arm\n"
            "    hw_features: []\n"
            "    build_features: []\n",
            encoding="utf-8",
        )
        source = arm / "add_plain.c"
        source.write_text(_REFERENCE_C, encoding="utf-8")
        install_brick(self.state_paths, self.selector.identity)

        generated = self.install_path / "generated_custom_ops"
        self.assertEqual(len(list(generated.glob("*_optimized_*.c"))), 1)
        registry = next(generated.glob("*_registry_*.cpp")).read_text(encoding="utf-8")
        self.assertIn('}, "arm/add_plain");', registry)
        self.assertLess(
            registry.index("_optimized_"),
            registry.index("_reference_"),
        )
        self.assertIn("#ifdef COMPUTE_FLOW_CUSTOM_OP_", registry)

        self._build_reference(expect_success=True)

        registry_header = next(generated.glob("*_registry_*.h"))
        project = self.root / "project"
        (project / "optimized_registry_probe.cpp").write_text(
            f'#include "{registry_header.name}"\n'
            "namespace custom_ops = compute_flow::custom_ops;\n"
            "int main() {\n"
            f"    const auto& registry = custom_ops::{registry_header.stem}();\n"
            "    if (registry.size() != 2 ||\n"
            "        !registry.data()[0].required_features.empty() ||\n"
            "        !registry.data()[1].required_features.empty()) {\n"
            "        return 1;\n"
            "    }\n"
            "    return 0;\n"
            "}\n",
            encoding="utf-8",
        )
        cmake_path = project / "CMakeLists.txt"
        cmake_path.write_text(
            cmake_path.read_text(encoding="utf-8")
            + "\nenable_testing()\n"
            + "add_executable(optimized_registry_probe optimized_registry_probe.cpp)\n"
            + f'target_include_directories(optimized_registry_probe PRIVATE "{generated}")\n'
            + "target_link_libraries(optimized_registry_probe PRIVATE "
            "compute_flow_brick_tts_example_custom_ops)\n"
            + "add_test(NAME optimized_registry_probe COMMAND optimized_registry_probe)\n",
            encoding="utf-8",
        )
        build_dir = self.root / "build"
        subprocess.run(
            ["cmake", "-S", str(project), "-B", str(build_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                "optimized_registry_probe",
                "--config",
                "Debug",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "--output-on-failure",
                "-C",
                "Debug",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        generated_source = next(generated.glob("*_optimized_*.c"))
        generated_source.write_text(
            '#error "optimized source was compiled"\n', encoding="utf-8"
        )
        timestamp = generated_source.stat()
        os.utime(
            generated_source,
            ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns + 10_000_000_000),
        )
        result = subprocess.run(
            ["cmake", "--build", str(self.root / "build"), "--target", "Example_core"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("optimized source was compiled", result.stderr)

    def test_setup_inputs_are_validated_and_registered(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(_REFERENCE_C, encoding="utf-8")
        cpu = custom_ops / "add_fp32/cpu"
        arm = cpu / "arm"
        arm.mkdir(parents=True)
        (arm / "add_plain.c").write_text(_REFERENCE_C, encoding="utf-8")
        registration = cpu / "registration.yaml"
        declaration = (
            "name: add_fp32\n"
            "version: 1\n"
            "kernels:\n"
            "  - kernel: arm/add_plain\n"
            "    target: arm\n"
            "    hw_features: []\n"
            "    build_features: []\n"
        )
        registration.write_text(
            declaration + "    setup_inputs: [1]\n", encoding="utf-8"
        )
        install_brick(self.state_paths, self.selector.identity)

        registry = next(
            (self.install_path / "generated_custom_ops").glob("*_registry_*.cpp")
        ).read_text(encoding="utf-8")
        self.assertIn('            {1},\n        }, "arm/add_plain"', registry)

        for indices in ("[2]", "[1, 1]", "[true]"):
            with self.subTest(indices=indices):
                registration.write_text(
                    declaration + f"    setup_inputs: {indices}\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(BrickManagerError, "Invalid setup_inputs"):
                    validate_custom_ops(self.source)

    def test_checks_each_optimized_kernel_build_feature(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(_REFERENCE_C, encoding="utf-8")
        cpu = custom_ops / "add_fp32/cpu"
        arm = cpu / "arm"
        arm.mkdir(parents=True)
        (cpu / "registration.yaml").write_text(
            "name: add_fp32\n"
            "version: 1\n"
            "kernels:\n"
            "  - kernel: arm/add_dotprod\n"
            "    target: arm\n"
            "    hw_features: [neon, neon-dotprod]\n"
            "    build_features: [neon-dotprod]\n"
            "  - kernel: arm/add_i8mm\n"
            "    target: arm\n"
            "    hw_features: [neon-i8mm]\n"
            "    build_features: [neon-i8mm]\n",
            encoding="utf-8",
        )
        for stem in ("add_dotprod", "add_i8mm"):
            (arm / f"{stem}.c").write_text(_REFERENCE_C, encoding="utf-8")
        install_brick(self.state_paths, self.selector.identity)

        cmake = (self.install_path / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('check_c_compiler_flag("-march=armv8.2-a+dotprod"', cmake)
        self.assertIn('check_c_compiler_flag("-march=armv8.2-a+i8mm"', cmake)
        self.assertEqual(cmake.count("check_c_compiler_flag("), 2)
        registry = next(
            (self.install_path / "generated_custom_ops").glob("*_registry_*.cpp")
        ).read_text(encoding="utf-8")
        self.assertIn("HardwareFeature::Arm::Neon", registry)
        self.assertIn("HardwareFeature::Arm::NeonDotProd", registry)
        self.assertIn("HardwareFeature::Arm::NeonI8mm", registry)

        self._build_reference(expect_success=True)

    def test_rejects_invalid_optimized_build_feature(self) -> None:
        custom_ops = self._write_custom_ops()
        cpu = custom_ops / "add_fp32/cpu"
        arm = cpu / "arm"
        arm.mkdir(parents=True)
        (arm / "add_plain.c").write_text(_REFERENCE_C, encoding="utf-8")
        (cpu / "registration.yaml").write_text(
            "name: add_fp32\n"
            "version: 1\n"
            "kernels:\n"
            "  - kernel: arm/add_plain\n"
            "    target: arm\n"
            "    hw_features: []\n"
            "    build_features: [unknown]\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(BrickManagerError, "Invalid build_features"):
            install_brick(self.state_paths, self.selector.identity)

    def test_generated_header_uses_generation_year(self) -> None:
        self._write_custom_ops()
        with patch("brick_manager.custom_ops.cmake_generation.date") as generated_date:
            generated_date.today.return_value.year = 2030
            install_brick(self.state_paths, self.selector.identity)

        header = next(
            (self.install_path / "generated_custom_ops").glob("*_reference_*.h")
        )
        self.assertIn(
            "SPDX-FileCopyrightText: Copyright 2030 Arm Limited",
            header.read_text(encoding="utf-8"),
        )

    def test_reference_registry_is_sorted_and_linkable(self) -> None:
        second = _SCHEMA.replace("add_fp32", "subtract_fp32").removeprefix(
            "operators:\n"
        )
        first = _SCHEMA.removeprefix("operators:\n")
        custom_ops = self._write_custom_ops("operators:\n" + second + first)
        subtract = custom_ops / "subtract_fp32"
        subtract.mkdir()
        for name in ("add_fp32", "subtract_fp32"):
            (custom_ops / name / "reference.c").write_text(
                _REFERENCE_C, encoding="utf-8"
            )
            (custom_ops / name / "calculate_output_shapes.c").write_text(
                _SHAPE_C, encoding="utf-8"
            )
        install_brick(self.state_paths, self.selector.identity)

        generated = self.install_path / "generated_custom_ops"
        registry_header = next(generated.glob("*_registry_*.h"))
        registry_source = next(generated.glob("*_registry_*.cpp"))
        source = registry_source.read_text(encoding="utf-8")
        self.assertLess(source.index('"add_fp32"'), source.index('"subtract_fp32"'))

        if not shutil.which("cmake") or not shutil.which("ctest"):
            self.skipTest("CMake or CTest is unavailable")
        project = self.root / "registry_probe"
        project.mkdir()
        (project / "probe.cpp").write_text(
            f'#include "{registry_header.name}"\n'
            '#include "compute_flow/custom_ops/v1/kernel.h"\n'
            "namespace custom_ops = compute_flow::custom_ops;\n"
            "int main() {\n"
            f"    const custom_ops::Registry& registry = custom_ops::{registry_header.stem}();\n"
            "\n"
            '    if (registry.brick_variant_name() != "tts:Example" ||\n'
            "        registry.size() != 2) {\n"
            "        return 1;\n"
            "    }\n"
            "\n"
            '    if (registry.data()[0].operator_name != "add_fp32" ||\n'
            '        registry.data()[1].operator_name != "subtract_fp32" ||\n'
            "        registry.data()[0].schema_version != 1 ||\n"
            "        registry.data()[0].interface_version != custom_ops::v1::Kernel::k_interface_version ||\n"
            "        !registry.data()[0].required_features.empty()) {\n"
            "        return 2;\n"
            "    }\n"
            "\n"
            "    CF_TensorFormatArray formats = {0};\n"
            "    CF_MemoryRequirement requirement = {0};\n"
            "\n"
            "    const auto* kernel = registry.data()[0].as<custom_ops::v1::Kernel>();\n"
            '    if (kernel == nullptr || registry.data()[0].kernel_name != "reference") {\n'
            "        return 3;\n"
            "    }\n"
            "    CF_TensorShape inputs[2] = {{{2}, 1}, {{2}, 1}};\n"
            "    CF_TensorShape outputs[1] = {};\n"
            "    if (kernel->calculate_output_shapes(inputs, 2, outputs, 1) != 0 ||\n"
            "        outputs[0].rank != 1 || outputs[0].dims[0] != 2) {\n"
            "        return 4;\n"
            "    }\n"
            "    inputs[0].dims[0] = 3;\n"
            "    inputs[1].dims[0] = 3;\n"
            "    if (kernel->calculate_output_shapes(inputs, 2, outputs, 1) != 0 ||\n"
            "        outputs[0].dims[0] != 3) {\n"
            "        return 5;\n"
            "    }\n"
            "    inputs[1].dims[0] = 4;\n"
            "    if (kernel->calculate_output_shapes(inputs, 2, outputs, 1) == 0) {\n"
            "        return 6;\n"
            "    }\n"
            "    return kernel->get_persistent_memory_requirement(\n"
            "        &formats, &requirement) != 0;\n"
            "}\n",
            encoding="utf-8",
        )
        (project / "lookup_probe.cpp").write_text(
            f'''#include <string>
#include <vector>
#include "{registry_header.name}"
#include "compute_flow/custom_ops/v1/kernel.h"

namespace custom_ops = compute_flow::custom_ops;

struct FutureKernel {{
    static constexpr uint64_t k_interface_version = 2;

    std::string operator_name;
    uint64_t schema_version;
    std::vector<custom_ops::CpuFeature> required_features;
    int32_t implementation_id;
}};

int main() {{
    const custom_ops::Registry& registry = custom_ops::{registry_header.stem}();
    custom_ops::Registry other_brick("tts:other@1.0.0");
    other_brick.add_kernel(registry.data()[0]);
    other_brick.add_kernel(registry.data()[1]);
    const std::vector<custom_ops::Registry> registries = {{registry, other_brick}};
    const std::vector<custom_ops::Registry> duplicates = {{registry, registry}};
    FutureKernel future_kernel{{"add_fp32", 1, {{custom_ops::HardwareFeature::Arm::Sve2}}, 42}};
    custom_ops::Registry mixed_brick("tts:mixed@1.0.0");
    mixed_brick.add_kernel(registry.data()[0]);
    mixed_brick.add_kernel(future_kernel);
    const std::vector<custom_ops::Registry> mixed_registries = {{mixed_brick}};
    const std::string selected_brick = "tts:Example";
    const std::string selected_operator = "add_fp32";

    auto result = custom_ops::lookup_kernels(registries, selected_brick,
        selected_operator, 1);
    if (result.status != custom_ops::LookupStatus::Ok ||
        result.kernels != &registries[0].data()[0] || result.num_kernels != 1) {{
        return 1;
    }}
    result = custom_ops::lookup_kernels(registries, "tts:other@1.0.0",
        "add_fp32", 1);
    if (result.status != custom_ops::LookupStatus::Ok ||
        result.kernels != &registries[1].data()[0] || result.num_kernels != 1) {{
        return 2;
    }}
    result = custom_ops::lookup_kernels(registries, "tts:absent@1.0.0",
        "add_fp32", 1);
    if (result.status != custom_ops::LookupStatus::BrickNotFound ||
        result.kernels != nullptr || result.num_kernels != 0) {{
        return 3;
    }}
    result = custom_ops::lookup_kernels(registries, selected_brick,
        "absent", 1);
    if (result.status != custom_ops::LookupStatus::OperatorNotFound ||
        result.kernels != nullptr || result.num_kernels != 0) {{
        return 4;
    }}
    result = custom_ops::lookup_kernels(registries, selected_brick,
        "add_fp32", 2);
    if (result.status != custom_ops::LookupStatus::SchemaMismatch ||
        result.kernels != nullptr || result.num_kernels != 0) {{
        return 5;
    }}
    result = custom_ops::lookup_kernels(duplicates, selected_brick,
        "add_fp32", 1);
    if (result.status != custom_ops::LookupStatus::Ambiguous ||
        result.kernels != nullptr || result.num_kernels != 0) {{
        return 6;
    }}
    custom_ops::Registry invalid_brick("tts:invalid@1.0.0");
    invalid_brick.add_kernel(registry.data()[0]);
    invalid_brick.add_kernel(registry.data()[1]);
    invalid_brick.add_kernel(registry.data()[0]);
    const std::vector<custom_ops::Registry> invalid_registries = {{invalid_brick}};
    result = custom_ops::lookup_kernels(invalid_registries, "tts:invalid@1.0.0",
        selected_operator, 1);
    if (result.status != custom_ops::LookupStatus::InvalidArgument ||
        result.kernels != nullptr || result.num_kernels != 0) {{
        return 7;
    }}
    result = custom_ops::lookup_kernels({{}}, selected_brick, selected_operator, 1);
    if (result.status != custom_ops::LookupStatus::BrickNotFound) {{
        return 8;
    }}
    result = custom_ops::lookup_kernels(mixed_registries, "tts:mixed@1.0.0",
        "add_fp32", 1);
    if (mixed_brick.size() != 2 || result.status != custom_ops::LookupStatus::Ok ||
        result.kernels != mixed_registries[0].data() || result.num_kernels != 2 ||
        result.kernels[0].interface_version != custom_ops::v1::Kernel::k_interface_version ||
        result.kernels[1].interface_version != FutureKernel::k_interface_version ||
        result.kernels[1].required_features !=
            std::vector<custom_ops::CpuFeature>{{custom_ops::HardwareFeature::Arm::Sve2}} ||
        result.kernels[0].as<custom_ops::v1::Kernel>() == nullptr ||
        result.kernels[0].as<FutureKernel>() != nullptr ||
        result.kernels[1].as<custom_ops::v1::Kernel>() != nullptr ||
        result.kernels[1].as<FutureKernel>() == nullptr ||
        result.kernels[1].as<FutureKernel>()->implementation_id != 42) {{
        return 9;
    }}
    return 0;
}}
''',
            encoding="utf-8",
        )
        (project / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(reference_registry_fixture LANGUAGES C CXX)\n"
            "enable_testing()\n"
            "add_library(compute_flow::compute_flow INTERFACE IMPORTED GLOBAL)\n"
            "set_target_properties(compute_flow::compute_flow PROPERTIES\n"
            f'    INTERFACE_INCLUDE_DIRECTORIES "{REPOSITORY_ROOT / "include"}")\n'
            f'add_subdirectory("{self.install_path}" brick)\n'
            "add_executable(registry_probe probe.cpp)\n"
            "target_compile_features(registry_probe PRIVATE cxx_std_17)\n"
            f'target_include_directories(registry_probe PRIVATE "{generated}")\n'
            "target_link_libraries(registry_probe PRIVATE Example_core)\n"
            "add_test(NAME registry_probe COMMAND registry_probe)\n"
            f'add_executable(lookup_probe lookup_probe.cpp "{REPOSITORY_ROOT / "compute_flow/custom_ops/registry.cpp"}")\n'
            "target_compile_features(lookup_probe PRIVATE cxx_std_17)\n"
            f'target_include_directories(lookup_probe PRIVATE "{generated}")\n'
            "target_link_libraries(lookup_probe PRIVATE compute_flow_brick_tts_example_custom_ops)\n"
            "add_test(NAME lookup_probe COMMAND lookup_probe)\n",
            encoding="utf-8",
        )
        build = self.root / "registry_build"
        subprocess.run(
            ["cmake", "-S", str(project), "-B", str(build)],
            check=True,
            capture_output=True,
            text=True,
        )
        build_result = subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--target",
                "registry_probe",
                "lookup_probe",
                "--config",
                "Debug",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(build_result.returncode, 0, build_result.stderr)
        subprocess.run(
            ["ctest", "--test-dir", str(build), "--output-on-failure", "-C", "Debug"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_build_rejects_missing_entry_point(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(
            _REFERENCE_C.replace("int64_t CF_RUN(", "int64_t MISSING_RUN("),
            encoding="utf-8",
        )
        install_brick(self.state_paths, self.selector.identity)

        self._build_reference(expect_success=False)

    def test_build_rejects_wrong_entry_point_signature(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(
            _REFERENCE_C.replace(
                "const CF_ThreadInfo* thread", "CF_ThreadInfo* thread"
            ),
            encoding="utf-8",
        )
        install_brick(self.state_paths, self.selector.identity)

        self._build_reference(expect_success=False)

    def test_build_rejects_missing_shape_function(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(_REFERENCE_C, encoding="utf-8")
        shape_source = custom_ops / "add_fp32/calculate_output_shapes.c"
        shape_source.write_text(
            _SHAPE_C.replace("CF_CALCULATE_OUTPUT_SHAPES(", "MISSING_SHAPE_FUNCTION("),
            encoding="utf-8",
        )
        install_brick(self.state_paths, self.selector.identity)

        self._build_reference(expect_success=False)

    def test_build_rejects_wrong_shape_signature(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").write_text(_REFERENCE_C, encoding="utf-8")
        shape_source = custom_ops / "add_fp32/calculate_output_shapes.c"
        shape_source.write_text(
            _SHAPE_C.replace("const CF_TensorShape* inputs", "CF_TensorShape* inputs"),
            encoding="utf-8",
        )
        install_brick(self.state_paths, self.selector.identity)

        self._build_reference(expect_success=False)

    def test_rejects_inline_and_standalone_reference_assembly(self) -> None:
        custom_ops = self._write_custom_ops()
        reference = custom_ops / "add_fp32/reference.c"
        reference.write_text(
            'void kernel(void) { __asm__("nop"); }\n', encoding="utf-8"
        )

        with self.assertRaisesRegex(BrickManagerError, "Assembly is not allowed"):
            install_brick(self.state_paths, self.selector.identity)

        reference.write_text(
            '#include "compute_flow/custom_ops/v1/types.h"\n'
            "// asm in a comment is allowed\n",
            encoding="utf-8",
        )
        (custom_ops / "add_fp32/reference.S").write_text("", encoding="utf-8")

        with self.assertRaisesRegex(
            BrickManagerError, "Undeclared custom operator entry"
        ):
            install_brick(self.state_paths, self.selector.identity)

        (custom_ops / "add_fp32/reference.S").unlink()
        (custom_ops / "add_fp32/helper.h").write_text(
            '__asm__("nop");\n', encoding="utf-8"
        )

        with self.assertRaisesRegex(
            BrickManagerError, "Undeclared custom operator entry"
        ):
            install_brick(self.state_paths, self.selector.identity)

    def test_rejects_inline_assembly_in_optimized_sources(self) -> None:
        custom_ops = self._write_custom_ops()
        cpu = custom_ops / "add_fp32/cpu"
        arm = cpu / "arm"
        arm.mkdir(parents=True)
        (cpu / "registration.yaml").write_text(
            "name: add_fp32\n"
            "version: 1\n"
            "kernels:\n"
            "  - kernel: arm/optimized\n"
            "    target: arm\n"
            "    hw_features: []\n"
            "    build_features: []\n",
            encoding="utf-8",
        )
        source = arm / "optimized.c"
        source.write_text('void kernel(void) { __asm__("nop"); }\n', encoding="utf-8")

        with self.assertRaisesRegex(
            BrickManagerError, "Inline assembly is not allowed in an optimized source"
        ):
            install_brick(self.state_paths, self.selector.identity)

        source.write_text(
            'const char* note = "__asm__"; // asm is allowed in comments\n',
            encoding="utf-8",
        )
        header = arm / "optimized.h"
        header.write_text('void kernel(void) { asm("nop"); }\n', encoding="utf-8")

        with self.assertRaisesRegex(
            BrickManagerError, "Undeclared custom operator entry"
        ):
            install_brick(self.state_paths, self.selector.identity)

        header.unlink()
        (arm / "optimized.S").write_text("invalid assembly", encoding="utf-8")

        self.assertEqual(
            install_brick(self.state_paths, self.selector.identity).status, "installed"
        )

    def test_rejects_extra_operator_files(self) -> None:
        operator = self._write_custom_ops() / "add_fp32"
        for name in ("helper.h", "extra.c", "notes.txt", "reference.S"):
            with self.subTest(name=name):
                extra = operator / name
                extra.write_text("", encoding="utf-8")
                with self.assertRaisesRegex(
                    BrickManagerError, "Undeclared custom operator entry"
                ):
                    install_brick(self.state_paths, self.selector.identity)
                extra.unlink()

        extra_directory = operator / "extra"
        extra_directory.mkdir()
        with self.assertRaisesRegex(
            BrickManagerError, "Undeclared custom operator entry"
        ):
            install_brick(self.state_paths, self.selector.identity)

        self.assertFalse(self.install_path.exists())

    def test_rejects_extra_cpu_files_without_registered_kernels(self) -> None:
        cpu = self._write_custom_ops() / "add_fp32/cpu"
        cpu.mkdir()
        (cpu / "registration.yaml").write_text("", encoding="utf-8")
        (cpu / "extra.c").write_text("", encoding="utf-8")

        with self.assertRaisesRegex(
            BrickManagerError, "Undeclared custom operator entry"
        ):
            install_brick(self.state_paths, self.selector.identity)

        self.assertFalse(self.install_path.exists())

    def test_rejects_extra_cpu_files_with_registered_kernel(self) -> None:
        cpu = self._write_custom_ops() / "add_fp32/cpu"
        arm = cpu / "arm"
        arm.mkdir(parents=True)
        (cpu / "registration.yaml").write_text(
            "name: add_fp32\n"
            "version: 1\n"
            "kernels:\n"
            "  - kernel: arm/add_plain\n"
            "    target: arm\n"
            "    hw_features: []\n"
            "    build_features: []\n",
            encoding="utf-8",
        )
        (arm / "add_plain.c").write_text(_REFERENCE_C, encoding="utf-8")

        for path in (
            cpu / "helper.h",
            cpu / "extra.c",
            arm / "helper.h",
            arm / "extra.c",
            arm / "extra.S",
            arm / "notes.txt",
        ):
            with self.subTest(path=path):
                path.write_text("", encoding="utf-8")
                with self.assertRaisesRegex(
                    BrickManagerError, "Undeclared custom operator entry"
                ):
                    install_brick(self.state_paths, self.selector.identity)
                path.unlink()

        self.assertEqual(
            install_brick(self.state_paths, self.selector.identity).status, "installed"
        )

    def test_installs_valid_contract(self) -> None:
        self._write_custom_ops()

        result = install_brick(self.state_paths, self.selector.identity)

        self.assertEqual(result.status, "installed")
        self.assertTrue(
            (result.install_path / "custom_ops/add_fp32/reference.c").is_file()
        )
        self.assertTrue(
            (
                result.install_path / "custom_ops/add_fp32/calculate_output_shapes.c"
            ).is_file()
        )
        self.assertTrue((result.install_path / "CMakeLists.txt").is_file())
        inventory = load_cache(self.state_paths.cache)["custom_ops"]
        self.assertEqual(list(inventory), [self.selector.identity])
        self.assertEqual(inventory[self.selector.identity][0]["name"], "add_fp32")
        self.assertEqual(inventory[self.selector.identity][0]["version"], 1)

    def test_accepts_brick_without_custom_ops(self) -> None:
        result = install_brick(self.state_paths, self.selector.identity)

        self.assertEqual(result.status, "installed")
        self.assertEqual(load_cache(self.state_paths.cache)["custom_ops"], {})

    def test_replacement_and_uninstall_update_inventory(self) -> None:
        custom_ops = self._write_custom_ops()
        install_brick(self.state_paths, self.selector.identity)

        (custom_ops / "add_fp32/reference.c").unlink()
        (custom_ops / "add_fp32/calculate_output_shapes.c").unlink()
        (custom_ops / "add_fp32").rmdir()
        (custom_ops / "schema.yaml").write_text(
            _SCHEMA.replace("add_fp32", "subtract_fp32"), encoding="utf-8"
        )
        replacement = custom_ops / "subtract_fp32"
        replacement.mkdir()
        (replacement / "reference.c").write_text(
            '#include "compute_flow/custom_ops/v1/types.h"\n', encoding="utf-8"
        )
        (replacement / "calculate_output_shapes.c").write_text(
            _SHAPE_C, encoding="utf-8"
        )

        install_brick(
            self.state_paths,
            self.selector.identity,
            force=True,
            confirm=lambda _: True,
        )

        inventory = load_cache(self.state_paths.cache)["custom_ops"]
        self.assertEqual(
            [operator["name"] for operator in inventory[self.selector.identity]],
            ["subtract_fp32"],
        )

        uninstall_brick(self.state_paths, self.selector.key, confirm=lambda _: True)

        self.assertEqual(load_cache(self.state_paths.cache)["custom_ops"], {})

    def test_new_version_replaces_old_inventory_entry(self) -> None:
        self._write_custom_ops()
        install_brick(self.state_paths, self.selector.identity)
        new_source = self.root / "source_v2"
        shutil.copytree(self.source, new_source)
        new_info_json = {**self.info_json, "version": "2.0.0"}
        (new_source / "info.json").write_text(
            json.dumps(new_info_json), encoding="utf-8"
        )
        new_selector = BrickSelector("tts", "example", "2.0.0")
        cache = load_cache(self.state_paths.cache)
        cache["bricks"][self.selector.key][new_selector.version] = {
            "info_json": new_info_json,
            "metadata": {
                "source_type": "local_directory",
                "source_path": str(new_source),
                "installed": False,
                "installed_source_path": None,
            },
        }
        write_cache(self.state_paths.cache, cache)

        install_brick(self.state_paths, new_selector.identity)

        inventory = load_cache(self.state_paths.cache)["custom_ops"]
        self.assertEqual(list(inventory), [new_selector.identity])
        self.assertFalse(self.install_path.exists())

    def test_same_operator_name_in_separate_bricks(self) -> None:
        self._write_custom_ops()
        second_source = self.root / "second_source"
        shutil.copytree(self.source, second_source)
        second_info_json = {**self.info_json, "name": "Second"}
        (second_source / "info.json").write_text(
            json.dumps(second_info_json), encoding="utf-8"
        )
        (second_source / "Example.cpp").rename(second_source / "Second.cpp")
        (second_source / "Example.h").rename(second_source / "Second.h")
        second_selector = BrickSelector("tts", "second", "1.0.0")
        cache = load_cache(self.state_paths.cache)
        cache["bricks"][second_selector.key] = {
            second_selector.version: {
                "info_json": second_info_json,
                "metadata": {
                    "source_type": "local_directory",
                    "source_path": str(second_source),
                    "installed": False,
                    "installed_source_path": None,
                },
            }
        }
        write_cache(self.state_paths.cache, cache)

        install_brick(self.state_paths, self.selector.identity)
        install_brick(self.state_paths, second_selector.identity)

        first_header = next(
            (self.install_path / "generated_custom_ops").glob("*_reference_*.h")
        )
        second_header = next(
            (
                self.state_paths.brick_source(second_selector) / "generated_custom_ops"
            ).glob("*_reference_*.h")
        )
        first_registry_header = next(
            (self.install_path / "generated_custom_ops").glob("*_registry_*.h")
        )
        second_registry_header = next(
            (
                self.state_paths.brick_source(second_selector) / "generated_custom_ops"
            ).glob("*_registry_*.h")
        )
        self.assertNotEqual(first_header.name, second_header.name)
        self.assertNotEqual(first_registry_header.name, second_registry_header.name)
        if shutil.which("cmake"):
            project = self.root / "combined_headers"
            project.mkdir()
            for header in (
                first_header,
                second_header,
                first_registry_header,
                second_registry_header,
            ):
                shutil.copy2(header, project / header.name)
            kernel_includes = (
                '#include "compute_flow/custom_ops/v1/types.h"\n'
                f'#include "{first_header.name}"\n'
                f'#include "{second_header.name}"\n'
            )
            registry_includes = (
                f'#include "{first_registry_header.name}"\n'
                f'#include "{second_registry_header.name}"\n'
            )
            (project / "headers.c").write_text(kernel_includes, encoding="utf-8")
            (project / "headers.cpp").write_text(
                kernel_includes + registry_includes, encoding="utf-8"
            )
            (project / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n"
                "project(combined_headers LANGUAGES C CXX)\n"
                "add_library(combined_headers OBJECT headers.c)\n"
                "add_library(combined_headers_cpp OBJECT headers.cpp)\n"
                "target_compile_features(combined_headers_cpp PRIVATE cxx_std_17)\n"
                f'target_include_directories(combined_headers PRIVATE "{REPOSITORY_ROOT / "include"}")\n'
                f'target_include_directories(combined_headers_cpp PRIVATE "{REPOSITORY_ROOT / "include"}")\n',
                encoding="utf-8",
            )
            build = self.root / "combined_headers_build"
            subprocess.run(
                ["cmake", "-S", str(project), "-B", str(build)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["cmake", "--build", str(build)],
                check=True,
                capture_output=True,
                text=True,
            )

        inventory = load_cache(self.state_paths.cache)["custom_ops"]
        self.assertEqual(
            list(inventory), [self.selector.identity, second_selector.identity]
        )
        self.assertEqual(
            inventory[self.selector.identity][0]["name"],
            inventory[second_selector.identity][0]["name"],
        )

        uninstall_brick(self.state_paths, self.selector.key, confirm=lambda _: True)

        self.assertEqual(
            list(load_cache(self.state_paths.cache)["custom_ops"]),
            [second_selector.identity],
        )

    def test_registry_update_preserves_but_does_not_rebuild_inventory(self) -> None:
        self._write_custom_ops()
        install_brick(self.state_paths, self.selector.identity)
        registry = self.root / "registry.txt"
        registry.write_text(str(self.source) + "\n", encoding="utf-8")

        update_cache(self.state_paths, str(registry))

        inventory = load_cache(self.state_paths.cache)["custom_ops"]
        self.assertEqual(list(inventory), [self.selector.identity])

        registry.write_text("", encoding="utf-8")
        update_cache(self.state_paths, str(registry))

        inventory = load_cache(self.state_paths.cache)["custom_ops"]
        self.assertEqual(list(inventory), [self.selector.identity])

        cache = load_cache(self.state_paths.cache)
        del cache["custom_ops"]
        write_cache(self.state_paths.cache, cache)

        update_cache(self.state_paths, str(registry))

        self.assertEqual(load_cache(self.state_paths.cache)["custom_ops"], {})

        result = install_brick(self.state_paths, self.selector.identity)

        self.assertEqual(result.status, "already_installed")
        self.assertEqual(load_cache(self.state_paths.cache)["custom_ops"], {})

        install_brick(
            self.state_paths,
            self.selector.identity,
            force=True,
            confirm=lambda _: True,
        )

        inventory = load_cache(self.state_paths.cache)["custom_ops"]
        self.assertEqual(list(inventory), [self.selector.identity])

    def test_accepts_scalar_as_one_element_tensor(self) -> None:
        self._write_custom_ops(_SCHEMA.replace("shape: [N]", "shape: [1]"))

        result = install_brick(self.state_paths, self.selector.identity)

        self.assertEqual(result.status, "installed")

    def test_rejects_zero_rank_tensor_and_scalar_attribute(self) -> None:
        custom_ops = self._write_custom_ops(
            _SCHEMA.replace("shape: [N]", "shape: []", 1)
        )

        with self.assertRaisesRegex(BrickManagerError, "shape must have 1 to 8"):
            install_brick(self.state_paths, self.selector.identity)

        (custom_ops / "schema.yaml").write_text(
            _SCHEMA + "    attributes: [{name: scale, type: float32}]\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(BrickManagerError, "unknown fields.*attributes"):
            install_brick(self.state_paths, self.selector.identity)

    def test_rejects_invalid_contract_before_replacing_installed_brick(self) -> None:
        custom_ops = self._write_custom_ops()
        install_brick(self.state_paths, self.selector.identity)
        (custom_ops / "schema.yaml").write_text(
            _SCHEMA.replace("name: add_fp32", "name: subtract_fp32"), encoding="utf-8"
        )

        with self.assertRaisesRegex(
            BrickManagerError, "Missing custom operator directory"
        ):
            install_brick(
                self.state_paths,
                self.selector.identity,
                force=True,
                confirm=lambda _: True,
            )

        self.assertTrue(
            (self.install_path / "custom_ops/add_fp32/reference.c").is_file()
        )
        cache = load_cache(self.state_paths.cache)
        self.assertEqual(list(cache["custom_ops"]), [self.selector.identity])
        self.assertTrue(
            cache["bricks"][self.selector.key][self.selector.version]["metadata"][
                "installed"
            ]
        )

    def test_rejects_missing_reference_and_invalid_schema(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "add_fp32/reference.c").unlink()

        with self.assertRaisesRegex(
            BrickManagerError, "Missing custom operator reference"
        ):
            install_brick(self.state_paths, self.selector.identity)

        (custom_ops / "add_fp32/reference.c").write_text("", encoding="utf-8")
        (custom_ops / "schema.yaml").write_text(
            _SCHEMA.replace("dtype: float32", "dtype: unknown"), encoding="utf-8"
        )

        with self.assertRaisesRegex(BrickManagerError, "dtype is unsupported"):
            install_brick(self.state_paths, self.selector.identity)

        self.assertFalse(self.install_path.exists())

    def test_requires_output_shape_source_and_types_include(self) -> None:
        custom_ops = self._write_custom_ops()
        shape_source = custom_ops / "add_fp32/calculate_output_shapes.c"
        shape_source.unlink()

        with self.assertRaisesRegex(
            BrickManagerError, "Missing custom operator output shape source"
        ):
            install_brick(self.state_paths, self.selector.identity)

        shape_source.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(
            BrickManagerError, r"must include compute_flow/custom_ops/v1/types\.h"
        ):
            install_brick(self.state_paths, self.selector.identity)

        shape_source.write_text(
            _SHAPE_C + '\nvoid test(void) { __asm__("nop"); }\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(BrickManagerError, "Assembly is not allowed"):
            install_brick(self.state_paths, self.selector.identity)

    def test_requires_types_include_in_each_reference(self) -> None:
        custom_ops = self._write_custom_ops()
        reference = custom_ops / "add_fp32/reference.c"
        for source in (
            "",
            '// #include "compute_flow/custom_ops/v1/types.h"\n',
            '/*\n#include "compute_flow/custom_ops/v1/types.h"\n*/\n',
        ):
            with self.subTest(source=source):
                reference.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(
                    BrickManagerError,
                    r"must include compute_flow/custom_ops/v1/types\.h",
                ):
                    install_brick(self.state_paths, self.selector.identity)

        self.assertFalse(self.install_path.exists())
        reference.write_text(
            "# include <compute_flow/custom_ops/v1/types.h>\n", encoding="utf-8"
        )
        self.assertEqual(
            install_brick(self.state_paths, self.selector.identity).status, "installed"
        )

    def test_rejects_undeclared_operator_and_missing_cpu_manifest(self) -> None:
        custom_ops = self._write_custom_ops()
        (custom_ops / "extra").mkdir()

        with self.assertRaisesRegex(BrickManagerError, "Undeclared custom operator"):
            install_brick(self.state_paths, self.selector.identity)

        (custom_ops / "extra").rmdir()
        (custom_ops / "add_fp32/cpu").mkdir()

        with self.assertRaisesRegex(
            BrickManagerError, "Invalid custom operator CPU layout"
        ):
            install_brick(self.state_paths, self.selector.identity)

        self.assertFalse(self.install_path.exists())


if __name__ == "__main__":
    unittest.main()
