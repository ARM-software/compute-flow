#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Generate brick CMake files from info.json metadata."""

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from pathlib import Path

from brick_manager.dependencies import dependency_metadata
from brick_manager.errors import BrickManagerError
from brick_manager.identity import brick_manager_name
from brick_manager.metadata import JsonObject, load_info_json


@dataclass(frozen=True)
class GeneratedBrickCMake:
    cmake: str
    pybind_source: str
    pybind_source_name: str
    project_name: str
    core_target: str
    alias_target: str
    brick_class: str
    pybind_module: str


def render_brick_cmake(source_path: Path) -> GeneratedBrickCMake:
    source_path = source_path.resolve()
    info_json = load_info_json(source_path)
    _validate_brick_layout(source_path, info_json)

    name = info_json["name"]
    brick_type = info_json["type"]
    manager_name = brick_manager_name(name)
    project_name = _cmake_identifier(f"compute_flow_brick_{brick_type}_{manager_name}")
    core_target = f"{name}_core"
    alias_target = f"{name}::core"
    brick_class = f"{brick_type}::{name}"
    deps = info_json.get("deps", [])
    pybind_module = f"{name}_py"
    pybind_source_name = f"{pybind_module}_generated.cpp"

    values = {
        "COPYRIGHT_YEAR": str(date.today().year),
        "PROJECT_NAME": project_name,
        "BRICK_SOURCE_DIR": source_path.as_posix(),
        "DEPENDENCY_FIND_BLOCKS": _dependency_find_blocks(deps),
        "CORE_TARGET": core_target,
        "BRICK_SOURCES": _brick_sources(source_path, name),
        "ALIAS_TARGET": alias_target,
        "BRICK_CLASS": brick_class,
        "INTERNAL_INCLUDE_BLOCK": _internal_include_block(source_path, core_target),
        "DEPENDENCY_INCLUDE_BLOCKS": _dependency_include_blocks(deps, core_target),
        "DEPENDENCY_LINK_BLOCK": _dependency_link_block(deps, core_target),
        "PYBIND_BLOCK": _pybind_block(pybind_source_name, pybind_module, core_target),
    }
    cmake = _render_template(values)
    return GeneratedBrickCMake(
        cmake=cmake,
        pybind_source=_render_pybind_source(
            copyright_year=values["COPYRIGHT_YEAR"],
            header_name=f"{name}.h",
            brick_type=brick_type,
            brick_name=name,
            pybind_module=pybind_module,
            module_alias=name,
        ),
        pybind_source_name=pybind_source_name,
        project_name=project_name,
        core_target=core_target,
        alias_target=alias_target,
        brick_class=brick_class,
        pybind_module=pybind_module,
    )


def write_brick_cmake(source_path: Path, output_dir: Path) -> Path:
    generated = render_brick_cmake(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "CMakeLists.txt"
    output_path.write_text(generated.cmake, encoding="utf-8")
    (output_dir / generated.pybind_source_name).write_text(
        generated.pybind_source,
        encoding="utf-8",
    )
    return output_path


def _validate_brick_layout(source_path: Path, info_json: JsonObject) -> None:
    cmake_path = source_path / "CMakeLists.txt"
    if cmake_path.is_file() and not _is_generated_cmake(cmake_path):
        raise BrickManagerError(
            f"Managed brick root must not contain handwritten CMake: {cmake_path}"
        )

    name = info_json["name"]
    for suffix in (".cpp", ".h"):
        path = source_path / f"{name}{suffix}"
        if not path.is_file():
            raise BrickManagerError(f"Missing brick implementation file: {path}")

    pybind_sources = sorted(source_path.glob("*_pybind.cpp"))
    if pybind_sources:
        names = ", ".join(source.name for source in pybind_sources)
        raise BrickManagerError(
            f"Managed brick root must not contain handwritten pybind sources in "
            f"{source_path}: {names}"
        )


def _brick_sources(source_path: Path, name: str) -> str:
    sources = [source_path / f"{name}.cpp"]
    internal = source_path / "internal"
    if internal.is_dir():
        sources.extend(sorted(internal.rglob("*.cpp")))
    return "\n".join(
        f'    "${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/{source.relative_to(source_path).as_posix()}"'
        for source in sources
    )


def _dependency_find_blocks(deps: list[str]) -> str:
    blocks = []
    for dep in deps:
        metadata = dependency_metadata(dep)
        lines = ["", f"if(NOT TARGET {metadata.cmake_target})"]
        if metadata.cmake_module is not None:
            lines.extend(
                [
                    f'    set(COMPUTE_FLOW_{dep.upper()}_MODULE "${{CMAKE_SOURCE_DIR}}/{metadata.cmake_module}")',
                    f'    if(EXISTS "${{COMPUTE_FLOW_{dep.upper()}_MODULE}}")',
                    f'        include("${{COMPUTE_FLOW_{dep.upper()}_MODULE}}")',
                    "    else()",
                    f"        find_package({metadata.cmake_package} CONFIG REQUIRED)",
                    "    endif()",
                ]
            )
        else:
            lines.append(f"    find_package({metadata.cmake_package} CONFIG REQUIRED)")
        lines.append("endif()")
        blocks.append("\n".join(lines))
    return "\n".join(blocks).lstrip()


def _internal_include_block(source_path: Path, core_target: str) -> str:
    internal = source_path / "internal"
    if not internal.is_dir():
        return ""

    include_dirs = [
        directory
        for directory in sorted(internal.rglob("*"))
        if directory.is_dir() and any(directory.glob("*.h"))
    ]
    if any(internal.glob("*.h")):
        include_dirs.insert(0, internal)
    if not include_dirs:
        return ""

    lines = [
        "target_include_directories(",
        f"    {core_target}",
        "    PRIVATE",
    ]
    lines.extend(
        f"        $<BUILD_INTERFACE:${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/{directory.relative_to(source_path).as_posix()}>"
        for directory in include_dirs
    )
    lines.append(")")
    return "\n".join(lines)


def _dependency_include_blocks(deps: list[str], core_target: str) -> str:
    blocks = []
    for dep in deps:
        metadata = dependency_metadata(dep)
        if metadata.cmake_source_variable is None:
            continue
        for include_subdir in metadata.extra_include_subdirs:
            include_path = f"${{{metadata.cmake_source_variable}}}/{include_subdir}"
            blocks.append(
                "\n".join(
                    [
                        f'if(DEFINED {metadata.cmake_source_variable} AND EXISTS "{include_path}")',
                        f'    target_include_directories({core_target} PRIVATE "{include_path}")',
                        "endif()",
                    ]
                )
            )
    return "\n".join(blocks)


def _dependency_link_block(deps: list[str], core_target: str) -> str:
    if not deps:
        return ""

    lines = [
        "target_link_libraries(",
        f"    {core_target}",
        "    PRIVATE",
    ]
    lines.extend(f"        {dependency_metadata(dep).cmake_target}" for dep in deps)
    lines.append(")")
    return "\n".join(lines)


def _pybind_block(pybind_source_name: str, pybind_module: str, core_target: str) -> str:
    return "\n".join(
        [
            "if(COMPUTE_FLOW_PYBIND_ENABLED)",
            "    # Disable pybind11's implicit LTO to avoid unresolved GCC .localalias symbols.",
            "    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)",
            f"    pybind11_add_module({pybind_module}",
            f'        "${{CMAKE_CURRENT_LIST_DIR}}/{pybind_source_name}"',
            "    )",
            f"    target_link_libraries({pybind_module} PRIVATE {core_target})",
            f"    compute_flow_install_python_extension({pybind_module})",
            "endif()",
        ]
    )


def _render_pybind_source(
    *,
    copyright_year: str,
    header_name: str,
    brick_type: str,
    brick_name: str,
    pybind_module: str,
    module_alias: str,
) -> str:
    bind_function = f"bind_{pybind_module}"
    namespace = f"compute_flow::bricks::{brick_type}"
    return f"""//
// SPDX-FileCopyrightText: Copyright {copyright_year} Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Generated by the Compute Flow brick manager from info.json.

#include "compute_flow/bindings/python/brick_pybind.h"

#include "{header_name}"

namespace py = pybind11;

namespace {namespace} {{

void {bind_function}(py::module_& m) {{
    m.doc() = "Python bindings for {brick_name}.";
    auto cls = compute_flow::pybind::bind_brick<{brick_name}>(m, "{brick_name}");
    m.attr("{module_alias}") = cls;
}}

}}  // namespace {namespace}

PYBIND11_MODULE({pybind_module}, m) {{
    {namespace}::{bind_function}(m);
}}
"""


def _render_template(values: dict[str, str]) -> str:
    template_path = Path(__file__).parent / "templates" / "brick_CMakeLists.txt.in"
    rendered = template_path.read_text(encoding="utf-8")
    rendered = _remove_template_source_header(rendered)
    for key, value in values.items():
        rendered = rendered.replace(f"@{key}@", value)
    rendered = rendered.replace(
        "        compute_flow::compute_flow\n\n)",
        "        compute_flow::compute_flow\n)",
    )
    lines = []
    previous_blank = False
    for line in rendered.splitlines():
        blank = not line.strip()
        if blank and previous_blank:
            continue
        lines.append(line)
        previous_blank = blank
    return "\n".join(lines).rstrip() + "\n"


def _cmake_identifier(value: str) -> str:
    return value.replace("-", "_")


def _remove_template_source_header(template: str) -> str:
    lines = template.splitlines()
    if not lines or lines[0] != "# Template source file:":
        return template

    while lines and lines[0].strip():
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines) + "\n"


def _is_generated_cmake(cmake_path: Path) -> bool:
    try:
        return (
            "Generated by the Compute Flow brick manager from info.json."
            in cmake_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )
    except FileNotFoundError:
        return False
