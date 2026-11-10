#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Generate C kernels, C++ registries, and CMake targets for brick custom operators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path

from brick_manager.custom_ops import validate_custom_ops
from brick_manager.custom_ops.features import compiler_flags
from brick_manager.custom_ops.validation import load_optimized_kernels
from brick_manager.metadata import JsonObject


_ENTRYPOINT_SUFFIXES = {
    "CF_GET_PERSISTENT_MEMORY_REQUIREMENT": "get_persistent_memory_requirement",
    "CF_GET_SCRATCH_MEMORY_REQUIREMENT": "get_scratch_memory_requirement",
    "CF_SETUP": "setup",
    "CF_RUN": "run",
}
_C_TOKEN = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r"|\b(?P<entry>CF_GET_PERSISTENT_MEMORY_REQUIREMENT|"
    r"CF_GET_SCRATCH_MEMORY_REQUIREMENT|CF_CALCULATE_OUTPUT_SHAPES|CF_SETUP|CF_RUN)\b"
)
_ASM_TOKEN = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r"|\b(?P<symbol>COMPUTE_FLOW_ASM_[A-Za-z_][A-Za-z0-9_]*)\b"
)
_HEADER_GUARD = "COMPUTE_FLOW_CUSTOM_OPS_V1_INTERFACE_H"
_INTERFACE_TEMPLATE = Path(__file__).parent / "templates" / "interface.h.in"
_COMMON_HEADER_GUARD = "COMPUTE_FLOW_CUSTOM_OPS_V1_OUTPUT_SHAPES_H"
_COMMON_INTERFACE_TEMPLATE = (
    Path(__file__).parent / "templates" / "calculate_output_shapes.h.in"
)
_ARM_HW_FEATURE_ENUM = {
    "neon": "Neon",
    "neon-dotprod": "NeonDotProd",
    "neon-fp16": "NeonFp16",
    "neon-i8mm": "NeonI8mm",
    "sve": "Sve",
    "sve2": "Sve2",
    "sme": "Sme",
    "sme2": "Sme2",
}


@dataclass(frozen=True)
class _KernelRegistration:
    operator_name: str
    schema_version: int
    header_name: str
    common_header_name: str
    symbols: dict[str, str]
    common_function: str
    custom_op_kernel_name: str
    hw_features: tuple[str, ...] = ()
    enabled_macro: str | None = None
    setup_inputs: tuple[int, ...] = ()


def render_custom_ops_cmake(
    source_path: Path,
    info_json: JsonObject,
    project_name: str,
    core_target: str,
) -> tuple[str, dict[str, str]]:
    """Render reference, optimized, and common sources and targets."""
    blocks: list[str] = []
    sources: dict[str, str] = {}
    registrations: list[_KernelRegistration] = []
    object_targets: list[str] = []
    optimized_custom_ops: list[str] = []
    reference_c_standard_variable = f"{project_name}_REFERENCE_C_STANDARD"
    operators = validate_custom_ops(source_path)
    if not operators:
        return "", sources

    interface_template = _INTERFACE_TEMPLATE.read_text(encoding="utf-8")
    interface_body = interface_template[
        interface_template.index(f"#ifndef {_HEADER_GUARD}") :
    ]
    common_template = _COMMON_INTERFACE_TEMPLATE.read_text(encoding="utf-8")
    common_body = common_template[
        common_template.index(f"#ifndef {_COMMON_HEADER_GUARD}") :
    ]

    for operator in operators:
        name = str(operator["name"])
        identity = (
            f"{info_json['type']}:{info_json['name']}@{info_json['version']}:{name}"
        )
        digest = sha256(identity.encode("utf-8")).hexdigest()[:12]
        reference_symbol_prefix = (
            f"custom_op_{project_name}_{name}_reference_{digest}".lower()
        )
        common_symbol_prefix = (
            f"custom_op_{project_name}_{name}_calculate_output_shapes_{digest}".lower()
        )
        common_function = common_symbol_prefix
        target = f"{project_name}_{name}_reference"
        object_targets.append(target)
        header_name = f"{reference_symbol_prefix}.h"
        source_name = f"{reference_symbol_prefix}.c"
        check_name = f"{reference_symbol_prefix}_check.cpp"
        common_header_name = f"{common_symbol_prefix}.h"
        common_source_name = f"{common_symbol_prefix}.c"
        reference_symbols = {
            entry: f"{reference_symbol_prefix}_{suffix}"
            for entry, suffix in _ENTRYPOINT_SUFFIXES.items()
        }
        common_symbols = {"CF_CALCULATE_OUTPUT_SHAPES": common_function}
        reference_and_common_symbols = reference_symbols | common_symbols
        reference_registration = _KernelRegistration(
            operator_name=name,
            schema_version=int(operator["version"]),
            header_name=header_name,
            common_header_name=common_header_name,
            symbols=reference_symbols,
            common_function=common_function,
            custom_op_kernel_name="reference",
        )
        guard = f"{reference_symbol_prefix}_H".upper()
        sources[header_name] = _generated_c_header() + _qualify_entrypoints(
            interface_body.replace(_HEADER_GUARD, guard), reference_symbols
        )
        reference_source = (
            source_path / "custom_ops" / name / "reference.c"
        ).read_text(encoding="utf-8")
        sources[source_name] = (
            _generated_c_header()
            + f'#include "{header_name}"\n\n'
            + _qualify_entrypoints(reference_source, reference_and_common_symbols)
        )
        common_guard = f"{common_symbol_prefix}_H".upper()
        sources[common_header_name] = _generated_c_header() + _qualify_entrypoints(
            common_body.replace(_COMMON_HEADER_GUARD, common_guard), common_symbols
        )
        common_source = (
            source_path / "custom_ops" / name / "calculate_output_shapes.c"
        ).read_text(encoding="utf-8")
        sources[common_source_name] = (
            _generated_c_header()
            + f'#include "{common_header_name}"\n\n'
            + _qualify_entrypoints(common_source, common_symbols)
        )
        sources[check_name] = (
            _generated_c_header()
            + f'#include "{header_name}"\n#include "{common_header_name}"\n\n'
            + _qualify_entrypoints(
                _kernel_check_source(include_common=True),
                reference_and_common_symbols,
            )
        )
        # Compile the common source once with the reference target. All
        # optimized registrations point to that same generated function.
        blocks.append(
            f"""add_library({target} OBJECT
    "${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops/{source_name}"
    "${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops/{common_source_name}"
)
set_target_properties({target} PROPERTIES
    C_STANDARD ${{{reference_c_standard_variable}}}
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories({target} PUBLIC
    "${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}"
    "${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/custom_ops/{name}"
    "$<TARGET_PROPERTY:compute_flow::compute_flow,INTERFACE_INCLUDE_DIRECTORIES>"
)
target_compile_options({target} PRIVATE
    "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-pedantic-errors>"
)
add_executable({target}_check
    "${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops/{check_name}"
)
set_target_properties({target}_check PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_link_libraries({target}_check PRIVATE {target})
if(UNIX AND NOT APPLE)
    target_link_libraries({target}_check PRIVATE m)
endif()"""
        )

        # Keep optimized kernels in their own build targets.
        cpu_path = source_path / "custom_ops" / name / "cpu"
        if cpu_path.is_dir():
            for kernel in load_optimized_kernels(
                cpu_path, name, int(operator["version"]), len(operator["inputs"])
            ):
                block, optimized_sources, registration = _render_optimized_kernel(
                    cpu_path,
                    project_name,
                    name,
                    int(operator["version"]),
                    identity,
                    kernel,
                    core_target,
                    reference_c_standard_variable,
                    interface_body,
                    common_header_name,
                    common_function,
                )
                optimized_custom_ops.append(block)
                sources.update(optimized_sources)
                registrations.append(registration)

        # Registry order is dispatch priority; the required reference is the final fallback.
        registrations.append(reference_registration)

    # validate_custom_ops sorts operators by name; preserve each operator's implementation order.
    brick_identity = f"{info_json['type']}:{info_json['name']}"
    registry_block, registry_sources, registry_header, registry_symbol = (
        _render_registry(brick_identity, project_name, registrations)
    )
    blocks.append(registry_block)
    sources.update(registry_sources)
    object_targets.append(f"{project_name}_custom_ops_registry")
    object_sources = "\n".join(
        f"    $<TARGET_OBJECTS:{target}>" for target in object_targets
    )
    archive_target = f"{project_name}_custom_ops"
    blocks.append(
        f"""add_library({archive_target} STATIC
{object_sources}
)
set_target_properties({archive_target} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features({archive_target} PUBLIC cxx_std_17)
target_include_directories({archive_target} INTERFACE
    "$<TARGET_PROPERTY:compute_flow::compute_flow,INTERFACE_INCLUDE_DIRECTORIES>"
)
# The catalog links this archive with compute_flow, so this target must not depend on it.
target_sources({core_target} PRIVATE
{object_sources}
)
set_target_properties({core_target} PROPERTIES
    COMPUTE_FLOW_CUSTOM_OPS_ARCHIVE_TARGET "{archive_target}"
    COMPUTE_FLOW_CUSTOM_OPS_REGISTRY_HEADER "generated_custom_ops/{registry_header}"
    COMPUTE_FLOW_CUSTOM_OPS_REGISTRY_SYMBOL "{registry_symbol}"
)"""
    )
    standard = (
        f"set({reference_c_standard_variable} 99)\n"
        "if(MSVC)\n"
        f"    set({reference_c_standard_variable} 11)\n"
        "endif()"
    )

    return "\n\n".join([standard, *blocks, *optimized_custom_ops]), sources


def _render_optimized_kernel(
    cpu_path: Path,
    project_name: str,
    operator_name: str,
    schema_version: int,
    identity: str,
    kernel: dict[str, object],
    core_target: str,
    c_standard_variable: str,
    interface_body: str,
    common_header_name: str,
    common_function: str,
) -> tuple[str, dict[str, str], _KernelRegistration]:
    """Generate an optimized C kernel with optional standalone assembly."""
    # Give each implementation unique symbols and filenames.
    stem = str(kernel["kernel"])
    target_arch = str(kernel["target"])
    digest = sha256(f"{identity}:{stem}".encode("utf-8")).hexdigest()[:12]
    prefix = f"custom_op_{project_name}_{operator_name}_optimized_{digest}".lower()
    target = f"{project_name}_{operator_name}_optimized_{digest}"
    enabled_macro = f"COMPUTE_FLOW_CUSTOM_OP_{digest.upper()}_ENABLED"

    header_name = f"{prefix}.h"
    source_name = f"{prefix}.c"
    assembly_name = f"{prefix}.S"
    check_name = f"{prefix}_check.cpp"
    guard = f"{prefix}_H".upper()
    optimized_symbols = {
        entry: f"{prefix}_{suffix}" for entry, suffix in _ENTRYPOINT_SUFFIXES.items()
    }
    optimized_and_common_symbols = optimized_symbols | {
        "CF_CALCULATE_OUTPUT_SHAPES": common_function
    }

    # Use the same C interface for the source and its link check.
    source = (cpu_path / f"{stem}.c").read_text(encoding="utf-8")
    source = _qualify_assembly_symbols(source, prefix)
    sources = {
        header_name: _generated_c_header()
        + _qualify_entrypoints(
            interface_body.replace(_HEADER_GUARD, guard), optimized_symbols
        ),
        source_name: _generated_c_header()
        + f'#include "{header_name}"\n\n'
        + _qualify_entrypoints(source, optimized_and_common_symbols),
        check_name: _generated_c_header()
        + f'#include "{header_name}"\n\n'
        + _qualify_entrypoints(
            _kernel_check_source(include_common=False), optimized_symbols
        ),
    }
    assembly_path = cpu_path / f"{stem}.S"
    if assembly_path.is_file():
        assembly = assembly_path.read_text(encoding="utf-8")
        sources[assembly_name] = (
            _generated_c_header()
            + "#if defined(__APPLE__)\n"
            + "#define COMPUTE_FLOW_CUSTOM_OP_ASM_LABEL(name) _##name\n"
            + "#else\n"
            + "#define COMPUTE_FLOW_CUSTOM_OP_ASM_LABEL(name) name\n"
            + "#endif\n\n"
            + _qualify_assembly_symbols(assembly, prefix, assembly=True)
        )

    # Check the requested compiler flag for this kernel only.
    flags = compiler_flags(target_arch, kernel["build_features"], "GNU")
    feature_check = ""

    if flags:
        flag = flags[0]
        check_variable = f"{target}_compiler_supported"
        feature_check = f"""if(CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    include(CheckCCompilerFlag)
    check_c_compiler_flag("{flag}" {check_variable})
else()
    set({check_variable} FALSE)
endif()
if(NOT {check_variable})
    message(WARNING "Skipping optimized custom operator {stem}: compiler does not support {flag}")
endif()"""
        condition = check_variable
        options = (
            f"target_compile_options({target} PRIVATE "
            f'"$<$<COMPILE_LANGUAGE:C>:{flag}>")\n'
        )
    else:
        condition = "TRUE"
        options = ""

    assembly_check = ""
    if assembly_path.is_file():
        assembly_options = f'        "{flags[0]}"\n' if flags else ""
        assembly_compile_options = (
            f'    target_compile_options({target} PRIVATE "$<$<COMPILE_LANGUAGE:ASM>:{flags[0]}>")\n'
            if flags
            else ""
        )
        assembly_check = f"""    include(CheckLanguage)
    check_language(ASM)
    if(CMAKE_ASM_COMPILER)
        enable_language(ASM)
    endif()
    if(CMAKE_ASM_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        set({target}_previous_try_compile_type "${{CMAKE_TRY_COMPILE_TARGET_TYPE}}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
        unset({target}_asm_supported CACHE)
        try_compile({target}_asm_supported
            "${{CMAKE_CURRENT_BINARY_DIR}}/{target}_asm_probe"
            "${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops/{assembly_name}"
            COMPILE_DEFINITIONS
{assembly_options}        "-I${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops"
                "-I${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}"
                "-I${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/custom_ops/{operator_name}"
                "-I${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/custom_ops/{operator_name}/cpu"
                "-I${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/custom_ops/{operator_name}/cpu/{target_arch}"
        )
        set(CMAKE_TRY_COMPILE_TARGET_TYPE "${{{target}_previous_try_compile_type}}")
    endif()
    if({target}_asm_supported)
        target_sources({target} PRIVATE
            "${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops/{assembly_name}"
        )
{assembly_compile_options}        target_compile_definitions({target} PRIVATE COMPUTE_FLOW_HAS_ASSEMBLY=1)
    else()
        message(WARNING "Using C fallback for optimized custom operator {stem}: assembler cannot compile its .S source")
    endif()
"""

    # Build the source and link all four entry points on supported targets.
    block = f"""if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
{feature_check}
if({condition})
    add_library({target} OBJECT
        "${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops/{source_name}"
    )
    set_target_properties({target} PROPERTIES
        C_STANDARD ${{{c_standard_variable}}}
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories({target} PUBLIC
        "${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}"
        "${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/custom_ops/{operator_name}"
        "${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/custom_ops/{operator_name}/cpu"
        "${{COMPUTE_FLOW_BRICK_SOURCE_DIR}}/custom_ops/{operator_name}/cpu/{target_arch}"
        "$<TARGET_PROPERTY:compute_flow::compute_flow,INTERFACE_INCLUDE_DIRECTORIES>"
    )
{assembly_check}    {options}add_executable({target}_check
        "${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops/{check_name}"
    )
    set_target_properties({target}_check PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    target_link_libraries({target}_check PRIVATE {target})
    if(UNIX AND NOT APPLE)
        target_link_libraries({target}_check PRIVATE m)
    endif()
    target_compile_definitions({project_name}_custom_ops_registry PRIVATE {enabled_macro})
    target_sources({project_name}_custom_ops PRIVATE $<TARGET_OBJECTS:{target}>)
    target_sources({core_target} PRIVATE $<TARGET_OBJECTS:{target}>)
    add_dependencies({core_target} {target}_check)
endif()
else()
    message(WARNING "Skipping optimized custom operator {stem}: target {target_arch} requires AArch64")
endif()"""

    registration = _KernelRegistration(
        operator_name=operator_name,
        schema_version=schema_version,
        header_name=header_name,
        common_header_name=common_header_name,
        symbols=optimized_symbols,
        common_function=common_function,
        custom_op_kernel_name=stem,
        hw_features=tuple(str(feature) for feature in kernel["hw_features"]),
        enabled_macro=enabled_macro,
        setup_inputs=tuple(int(index) for index in kernel.get("setup_inputs", [])),
    )

    return block, sources, registration


def _render_registry(
    brick_identity: str,
    project_name: str,
    registrations: list[_KernelRegistration],
) -> tuple[str, dict[str, str], str, str]:
    """Generate the brick registry in implementation priority order."""
    digest = sha256(brick_identity.encode("utf-8")).hexdigest()[:12]
    symbol = f"custom_op_{project_name}_registry_{digest}".lower()
    target = f"{project_name}_custom_ops_registry"
    header_name = f"{symbol}.h"
    source_name = f"{symbol}.cpp"
    guard = f"{symbol}_H".upper()
    header = f"""#ifndef {guard}
#define {guard}

#include "compute_flow/custom_ops/registry.h"

namespace compute_flow::custom_ops {{

const Registry& {symbol}();

}}  // namespace compute_flow::custom_ops

#endif
"""
    includes = "\n".join(
        _render_registry_include(registration) for registration in registrations
    )
    common_includes = "\n".join(
        f'#include "{header_name}"'
        for header_name in sorted(
            {registration.common_header_name for registration in registrations}
        )
    )
    rows = "\n".join(
        _render_registry_row(registration) for registration in registrations
    )
    source = f"""#include "{header_name}"
#include "compute_flow/custom_ops/v1/kernel.h"
{includes}
{common_includes}

namespace compute_flow::custom_ops {{

const Registry& {symbol}() {{
    static const Registry k_registry = [] {{
        Registry registry("{brick_identity}");
{rows}
        return registry;
    }}();
    return k_registry;
}}

}}  // namespace compute_flow::custom_ops
"""
    block = f"""add_library({target} OBJECT
    "${{CMAKE_CURRENT_LIST_DIR}}/generated_custom_ops/{source_name}"
)
set_target_properties({target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories({target} PUBLIC
    "$<TARGET_PROPERTY:compute_flow::compute_flow,INTERFACE_INCLUDE_DIRECTORIES>"
)
target_compile_options({target} PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-pedantic-errors>"
)"""
    return (
        block,
        {
            header_name: _generated_c_header() + header,
            source_name: _generated_c_header() + source,
        },
        header_name,
        symbol,
    )


def _render_registry_include(registration: _KernelRegistration) -> str:
    """Include an optimized header only when its source was built."""
    include = f'#include "{registration.header_name}"'
    if registration.enabled_macro:
        return f"#ifdef {registration.enabled_macro}\n{include}\n#endif"

    return include


def _render_registry_row(registration: _KernelRegistration) -> str:
    """Render one kernel, guarded when its build is optional."""
    features = ", ".join(
        f"::compute_flow::custom_ops::HardwareFeature::Arm::{_ARM_HW_FEATURE_ENUM[feature]}"
        for feature in registration.hw_features
    )
    setup_inputs = ", ".join(str(index) for index in registration.setup_inputs)
    row = (
        "        registry.add_kernel(v1::Kernel{\n"
        f'            "{registration.operator_name}",\n'
        f"            {registration.schema_version},\n"
        f"            {{{features}}},\n"
        + "".join(
            f"            {registration.symbols[entry]},\n"
            for entry in _ENTRYPOINT_SUFFIXES
        )
        + f"            {registration.common_function},\n"
        + f"            {{{setup_inputs}}},\n"
        + "        }, "
        + f'"{registration.custom_op_kernel_name}");'
    )
    if registration.enabled_macro:
        return f"#ifdef {registration.enabled_macro}\n{row}\n#endif"

    return row


def _qualify_entrypoints(source: str, qualified: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        entry = match.group("entry")
        return qualified[entry] if entry else match.group()

    return _C_TOKEN.sub(replace, source)


def _qualify_assembly_symbols(
    source: str, prefix: str, *, assembly: bool = False
) -> str:
    """Qualify authored assembly names without changing comments or literals."""

    def replace(match: re.Match[str]) -> str:
        symbol = match.group("symbol")
        if symbol is None:
            return match.group()
        qualified = f"{prefix}_asm_{symbol[len('COMPUTE_FLOW_ASM_') :]}"
        if assembly:
            return f"COMPUTE_FLOW_CUSTOM_OP_ASM_LABEL({qualified})"
        return qualified

    return _ASM_TOKEN.sub(replace, source)


def _generated_c_header() -> str:
    return (
        "/*\n"
        f" * SPDX-FileCopyrightText: Copyright {date.today().year} Arm Limited and/or "
        "its affiliates <open-source-office@arm.com>\n"
        " * SPDX-"
        "License-Identifier: Apache-2.0\n"
        " */\n\n"
        "/* Generated by the Compute Flow brick manager. */\n\n"
    )


def _kernel_check_source(*, include_common: bool) -> str:
    source = """int main(int argc, char** argv) {
    (void)argv;
    /* Keep each call reachable for the linker without running it normally. */
    if (argc != 0) {
        return 0;
    }

    CF_TensorFormatArray formats = {0};
    CF_TensorArray tensors = {0};
    CF_MemoryRequirement requirement = {0};
    CF_Workspace workspace = {0};
    CF_ThreadInfo thread = {0};

    (void)CF_GET_PERSISTENT_MEMORY_REQUIREMENT(&formats, &requirement);
    (void)CF_GET_SCRATCH_MEMORY_REQUIREMENT(&formats, &formats, &requirement);
    (void)CF_SETUP(&tensors, &workspace);
    (void)CF_RUN(&tensors, &tensors, &workspace, &workspace, &thread);
"""
    if include_common:
        source += "    (void)CF_CALCULATE_OUTPUT_SHAPES(0, 0, 0, 0);\n"
    return (
        source
        + """    return 0;
}
"""
    )
