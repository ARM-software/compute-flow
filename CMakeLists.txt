#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)

project(
    compute_flow
    VERSION 1.0.0
    LANGUAGES C CXX
    DESCRIPTION "Compute Flow is an SDK to unify multiple key EdgeAI frameworks under a single library API."
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_MESSAGE_CONTEXT_SHOW ON CACHE BOOL "Extra context for logging" FORCE)
list(APPEND CMAKE_MESSAGE_CONTEXT "Compute Flow")

# =========================================================================================
# Options
# =========================================================================================

option(COMPUTE_FLOW_ENABLE_LITERT "Enable LiteRT runner" ON)
option(COMPUTE_FLOW_ENABLE_LITERT_GPU "Enable LiteRT GPU acceleration" OFF)
option(COMPUTE_FLOW_ENABLE_ONNXRUNTIME "Enable ONNX Runtime runner" ON)
option(COMPUTE_FLOW_ENABLE_EXECUTORCH "Enable ExecuTorch runner" ON)
option(COMPUTE_FLOW_ENABLE_MNN "Enable MNN runner" ON)
option(COMPUTE_FLOW_ENABLE_MNN_GPU "Enable MNN GPU acceleration on Android™" ${ANDROID})
option(COMPUTE_FLOW_ENABLE_SENTENCEPIECE "Enable SentencePiece tokenizer runner" ON)
option(COMPUTE_FLOW_ENABLE_HF_TOKENIZER "Enable Hugging Face tokenizer runner" ON)
option(COMPUTE_FLOW_ENABLE_LLAMACPP "Enable llama.cpp LM runner" ON)
option(COMPUTE_FLOW_ENABLE_MODEL_TESTS "Install models and enable model-backed CLI smoke tests" OFF)
set(COMPUTE_FLOW_TEST_MODEL_HOME "" CACHE PATH "Compute Flow home containing models used by tests")
option(COMPUTE_FLOW_ENABLE_STREAMLINE "Enable Arm® Streamline timeline annotations" OFF)
if(ANDROID)
    set(COMPUTE_FLOW_STREAMLINE_USE_TCP_DEFAULT ON)
else()
    set(COMPUTE_FLOW_STREAMLINE_USE_TCP_DEFAULT OFF)
endif()
option(
    COMPUTE_FLOW_STREAMLINE_USE_TCP
    "Use TCP instead of Unix-domain sockets for Arm® Streamline annotations"
    ${COMPUTE_FLOW_STREAMLINE_USE_TCP_DEFAULT}
)
option(COMPUTE_FLOW_USE_CONAN "Fetch deps via Conan package manager" ON)
# Per-dependency from-source overrides (force-build from source even if find_package succeeds)
option(COMPUTE_FLOW_BUILD_PROTOBUF_FROM_SOURCE "Force-build Protobuf from source" OFF)
option(COMPUTE_FLOW_BUILD_LITERT_FROM_SOURCE "Force-build LiteRT from source" OFF)
option(COMPUTE_FLOW_BUILD_EXECUTORCH_FROM_SOURCE "Force-build ExecuTorch from source" OFF)
option(COMPUTE_FLOW_BUILD_ONNXRUNTIME_FROM_SOURCE "Force-build ONNX Runtime from source" OFF)
option(COMPUTE_FLOW_BUILD_MNN_FROM_SOURCE "Force-build MNN from source" OFF)
option(COMPUTE_FLOW_BUILD_SENTENCEPIECE_FROM_SOURCE "Force-build SentencePiece from source" OFF)

# =========================================================================================
# Derived settings
# =========================================================================================

# Python bindings are built only for native targets because Python development packages usually
# describe the host environment rather than the cross-compiled target.
if(CMAKE_CROSSCOMPILING)
    set(COMPUTE_FLOW_PYBIND_ENABLED OFF)
else()
    set(COMPUTE_FLOW_PYBIND_ENABLED ON)
endif()

# =========================================================================================
# Global settings
# =========================================================================================

# Keep this enabled as we're linking static deps into shared libs (e.g. pybind module)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(ExternalProject)
include(FetchContent)
include(GNUInstallDirs)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(COMPUTE_FLOW_ENABLE_STREAMLINE)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
        message(FATAL_ERROR "Arm® Streamline annotations are supported only on Linux® and Android™ targets.")
    endif()

    FetchContent_Declare(
        compute_flow_gator
        URL https://github.com/ARM-software/gator/archive/f0774012f36dbdb543e082d3e14ca9db20d0432d.tar.gz
        URL_HASH SHA256=413df392839a431e57cf3a55c543de5f1552e1d9be5d8a4766b337f3e2af8b7d
        SOURCE_SUBDIR
        compute-flow-no-cmake-project
    )
    FetchContent_MakeAvailable(compute_flow_gator)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified. Defaulting to Release.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(APPLE)
    if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS® deployment target for Compute Flow")
    endif()
    if(NOT DEFINED CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "macOS® architecture(s) for Compute Flow")
    endif()
endif()

# =========================================================================================
# Status banner
# =========================================================================================

message(STATUS "--------------------------------")
message(STATUS "Compute Flow v${PROJECT_VERSION}")
message(STATUS "--------------------------------")
message(STATUS "Available runners:")
message(STATUS "\t" "LiteRT:          ${COMPUTE_FLOW_ENABLE_LITERT}")
message(STATUS "\t" "LiteRT GPU:      ${COMPUTE_FLOW_ENABLE_LITERT_GPU}")
message(STATUS "\t" "ExecuTorch:      ${COMPUTE_FLOW_ENABLE_EXECUTORCH}")
message(STATUS "\t" "ONNX-Runtime:    ${COMPUTE_FLOW_ENABLE_ONNXRUNTIME}")
message(STATUS "\t" "MNN:             ${COMPUTE_FLOW_ENABLE_MNN}")
message(STATUS "\t" "MNN GPU:         ${COMPUTE_FLOW_ENABLE_MNN_GPU}")
message(STATUS "\t" "SentencePiece:   ${COMPUTE_FLOW_ENABLE_SENTENCEPIECE}")
message(STATUS "\t" "HF tokenizer:    ${COMPUTE_FLOW_ENABLE_HF_TOKENIZER}")
message(STATUS "\t" "Streamline:      ${COMPUTE_FLOW_ENABLE_STREAMLINE}")
message(STATUS "Available LM runners:")
message(STATUS "\t" "llama.cpp:       ${COMPUTE_FLOW_ENABLE_LLAMACPP}")
message(STATUS "--------------------------------")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
message(STATUS "--------------------------------")

# =========================================================================================
# Python
# =========================================================================================

if(COMPUTE_FLOW_PYBIND_ENABLED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# =========================================================================================
# Include modular cmake files
# =========================================================================================

include(cmake/Utilities.cmake)
include(cmake/Deps.cmake)
include(CTest)

# =========================================================================================
# Path constants
# =========================================================================================

set(COMPUTE_FLOW_PUBLIC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(COMPUTE_FLOW_PRIVATE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/compute_flow")
set(COMPUTE_FLOW_BRICK_MANIFEST_SELECTOR_SOURCE
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/internal/manifest_selector.cpp"
)

# =========================================================================================
# Installed brick cache discovery
# =========================================================================================

function(_compute_flow_resolve_brick_cache_path out_cache_path out_home_dir)
    if(DEFINED ENV{COMPUTE_FLOW_HOME} AND NOT "$ENV{COMPUTE_FLOW_HOME}" STREQUAL "")
        set(home_dir "$ENV{COMPUTE_FLOW_HOME}")
    else()
        set(home_dir "$ENV{HOME}/.compute-flow")
    endif()

    cmake_path(ABSOLUTE_PATH home_dir NORMALIZE)
    set(${out_home_dir} "${home_dir}" PARENT_SCOPE)
    set(${out_cache_path} "${home_dir}/cache.json" PARENT_SCOPE)
endfunction()

function(_compute_flow_json_get out_var json)
    string(JSON value ERROR_VARIABLE json_error GET "${json}" ${ARGN})
    if(json_error)
        message(FATAL_ERROR "Failed to read brick cache field '${ARGN}': ${json_error}")
    endif()
    set(${out_var} "${value}" PARENT_SCOPE)
endfunction()

function(_compute_flow_json_member out_var json)
    string(JSON value ERROR_VARIABLE json_error MEMBER "${json}" ${ARGN})
    if(json_error)
        message(FATAL_ERROR "Failed to read brick cache member '${ARGN}': ${json_error}")
    endif()
    set(${out_var} "${value}" PARENT_SCOPE)
endfunction()

function(_compute_flow_json_length out_var json)
    string(JSON value ERROR_VARIABLE json_error LENGTH "${json}" ${ARGN})
    if(json_error)
        message(FATAL_ERROR "Failed to read brick cache object '${ARGN}': ${json_error}")
    endif()
    set(${out_var} "${value}" PARENT_SCOPE)
endfunction()

function(compute_flow_discover_installed_bricks)
    _compute_flow_resolve_brick_cache_path(cache_path cache_home)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${cache_path}" "${cache_home}")

    set(installed_identities "")
    set(installed_targets "")
    set(installed_types "")
    set(installed_names "")
    set(installed_classes "")
    set(installed_source_dirs "")

    if(NOT EXISTS "${cache_path}")
        message(
            WARNING
            "No Compute Flow brick cache found at: ${cache_path}\n"
            "Run 'uv run compute-flow update' and 'uv run compute-flow brick install <brick>'. "
            "Configuring with no installed bricks."
        )
        set(COMPUTE_FLOW_BRICK_CACHE_PATH "${cache_path}" PARENT_SCOPE)
        set(COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES "" PARENT_SCOPE)
        set(COMPUTE_FLOW_INSTALLED_BRICK_TARGETS "" PARENT_SCOPE)
        set(COMPUTE_FLOW_INSTALLED_BRICK_TYPES "" PARENT_SCOPE)
        set(COMPUTE_FLOW_INSTALLED_BRICK_NAMES "" PARENT_SCOPE)
        set(COMPUTE_FLOW_INSTALLED_BRICK_CLASSES "" PARENT_SCOPE)
        set(COMPUTE_FLOW_INSTALLED_BRICK_SOURCE_DIRS "" PARENT_SCOPE)
        return()
    endif()

    file(READ "${cache_path}" cache_json)
    _compute_flow_json_length(brick_count "${cache_json}" bricks)

    set(brick_index 0)
    while(brick_index LESS brick_count)
        _compute_flow_json_member(brick_key "${cache_json}" bricks ${brick_index})
        _compute_flow_json_length(version_count "${cache_json}" bricks "${brick_key}")

        set(version_index 0)
        while(version_index LESS version_count)
            _compute_flow_json_member(version "${cache_json}" bricks "${brick_key}" ${version_index})
            _compute_flow_json_get(
                installed
                "${cache_json}"
                bricks
                "${brick_key}"
                "${version}"
                metadata
                installed
            )

            if(installed)
                _compute_flow_json_get(type "${cache_json}" bricks "${brick_key}" "${version}" info_json type)
                _compute_flow_json_get(name "${cache_json}" bricks "${brick_key}" "${version}" info_json name)
                _compute_flow_json_get(
                    source_dir
                    "${cache_json}"
                    bricks
                    "${brick_key}"
                    "${version}"
                    metadata
                    installed_source_path
                )

                add_subdirectory("${source_dir}" "${CMAKE_BINARY_DIR}/bricks/${type}/${name}")

                set(core_target "${name}::core")
                if(NOT TARGET "${core_target}")
                    message(
                        FATAL_ERROR
                        "Installed brick '${type}:${name}@${version}' must provide target '${core_target}'."
                    )
                endif()

                get_target_property(property_target "${core_target}" ALIASED_TARGET)
                if(NOT property_target)
                    set(property_target "${core_target}")
                endif()

                get_target_property(brick_class "${property_target}" COMPUTE_FLOW_BRICK_CLASS)
                if(NOT brick_class OR brick_class STREQUAL "brick_class-NOTFOUND")
                    message(
                        FATAL_ERROR
                        "Installed brick '${type}:${name}@${version}' target '${core_target}' "
                        "must set COMPUTE_FLOW_BRICK_CLASS."
                    )
                endif()

                list(APPEND installed_identities "${brick_key}")
                list(APPEND installed_targets "${core_target}")
                list(APPEND installed_types "${type}")
                list(APPEND installed_names "${name}")
                list(APPEND installed_classes "${brick_class}")
                list(APPEND installed_source_dirs "${source_dir}")
            endif()

            math(EXPR version_index "${version_index} + 1")
        endwhile()

        math(EXPR brick_index "${brick_index} + 1")
    endwhile()

    if(NOT installed_targets)
        message(
            WARNING
            "Compute Flow brick cache has no installed bricks: ${cache_path}\n"
            "Run 'uv run compute-flow brick install <brick>'. Configuring with no installed bricks."
        )
    else()
        message(STATUS "Installed bricks from ${cache_path}:")
        foreach(identity IN LISTS installed_identities)
            message(STATUS "\t${identity}")
        endforeach()
    endif()

    set(COMPUTE_FLOW_BRICK_CACHE_PATH "${cache_path}" PARENT_SCOPE)
    set(COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES "${installed_identities}" PARENT_SCOPE)
    set(COMPUTE_FLOW_INSTALLED_BRICK_TARGETS "${installed_targets}" PARENT_SCOPE)
    set(COMPUTE_FLOW_INSTALLED_BRICK_TYPES "${installed_types}" PARENT_SCOPE)
    set(COMPUTE_FLOW_INSTALLED_BRICK_NAMES "${installed_names}" PARENT_SCOPE)
    set(COMPUTE_FLOW_INSTALLED_BRICK_CLASSES "${installed_classes}" PARENT_SCOPE)
    set(COMPUTE_FLOW_INSTALLED_BRICK_SOURCE_DIRS "${installed_source_dirs}" PARENT_SCOPE)
endfunction()

function(compute_flow_configure_brick_factory out_source)
    set(factory_includes "")
    set(factory_branches "")

    list(LENGTH COMPUTE_FLOW_INSTALLED_BRICK_NAMES installed_brick_count)
    if(installed_brick_count GREATER 0)
        math(EXPR last_installed_brick_index "${installed_brick_count} - 1")
        foreach(index RANGE 0 ${last_installed_brick_index})
            list(GET COMPUTE_FLOW_INSTALLED_BRICK_TYPES ${index} brick_type)
            list(GET COMPUTE_FLOW_INSTALLED_BRICK_NAMES ${index} brick_name)
            list(GET COMPUTE_FLOW_INSTALLED_BRICK_CLASSES ${index} brick_class)

            string(APPEND factory_includes "#include \"${brick_name}.h\"\n")
            string(APPEND factory_branches "    if (type == \"${brick_type}\" && name == \"${brick_name}\") {\n")
            string(APPEND factory_branches "        return std::make_unique<${brick_class}>();\n")
            string(APPEND factory_branches "    }\n")
        endforeach()
    endif()

    set(COMPUTE_FLOW_BRICK_FACTORY_INCLUDES "${factory_includes}")
    set(COMPUTE_FLOW_BRICK_FACTORY_BRANCHES "${factory_branches}")

    set(generated_source "${CMAKE_CURRENT_BINARY_DIR}/generated/compute_flow/bricks/brick_factory.cpp")
    configure_file("${CMAKE_SOURCE_DIR}/compute_flow/bricks/brick_factory.cpp.in" "${generated_source}" @ONLY)

    set(${out_source} "${generated_source}" PARENT_SCOPE)
endfunction()

function(compute_flow_configure_brick_type_libraries)
    set(installed_brick_types ${COMPUTE_FLOW_INSTALLED_BRICK_TYPES})
    list(REMOVE_DUPLICATES installed_brick_types)

    set(brick_type_targets "")
    list(LENGTH COMPUTE_FLOW_INSTALLED_BRICK_TYPES installed_brick_count)
    foreach(brick_type IN LISTS installed_brick_types)
        set(concrete_targets "")

        if(installed_brick_count GREATER 0)
            math(EXPR last_installed_brick_index "${installed_brick_count} - 1")
            foreach(index RANGE 0 ${last_installed_brick_index})
                list(GET COMPUTE_FLOW_INSTALLED_BRICK_TYPES ${index} installed_brick_type)
                if(installed_brick_type STREQUAL brick_type)
                    list(GET COMPUTE_FLOW_INSTALLED_BRICK_TARGETS ${index} concrete_target)
                    list(APPEND concrete_targets "${concrete_target}")
                endif()
            endforeach()
        endif()

        compute_flow_collect_static_link_archives(type_archives type_targets ${concrete_targets})

        set(brick_type_target "${brick_type}_core")
        add_library(${brick_type_target} STATIC "${CMAKE_SOURCE_DIR}/compute_flow/bricks/brick_type_library_stub.cpp")
        add_dependencies(${brick_type_target} ${type_targets})

        string(MAKE_C_IDENTIFIER "${brick_type}" brick_type_identifier)
        string(TOUPPER "${brick_type_identifier}" brick_type_prefix)
        set(MONOLITHIC_PREFIX_MAP_${brick_type_identifier} "BRICK_TYPE_${brick_type_prefix}")
        set(COMPUTE_FLOW_BRICK_TYPE_${brick_type_prefix}_MONOLITHIC_ARCHIVES "${type_archives}")

        compute_flow_add_monolithic_archive(
            ${brick_type_target}
            "$<TARGET_FILE:${brick_type_target}>"
            PROVENANCES ${brick_type_identifier}
        )

        list(APPEND brick_type_targets ${brick_type_target})
    endforeach()

    set(COMPUTE_FLOW_INSTALLED_BRICK_TYPE_TARGETS "${brick_type_targets}" PARENT_SCOPE)
endfunction()

function(compute_flow_link_installed_bricks target)
    set(link_targets compute_flow_brick_factory compute_flow::compute_flow ${COMPUTE_FLOW_INSTALLED_BRICK_TYPE_TARGETS})

    target_include_directories(${target} PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)

    if(UNIX AND NOT APPLE AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        list(JOIN link_targets "," link_group_targets)
        target_link_libraries(${target} PRIVATE "$<LINK_GROUP:RESCAN,${link_group_targets}>")
    else()
        target_link_libraries(${target} PRIVATE ${link_targets})
    endif()
endfunction()

# =========================================================================================
# Core library — source files
# =========================================================================================

set(COMPUTE_FLOW_PUBLIC_HEADERS
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/accelerator.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/check.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/compute_flow.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/lm_params.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/manifest.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/memory_planner.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/model_data.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/lm_runners/lm_runner.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/tensor_dict.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/bindings/python/brick_pybind.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/bricks/brick_base_v1.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/bricks/brick_factory.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/bricks/brick_interface_v1.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/bricks/brick_manifest.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/bricks/brick_version.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/profiler/profiler.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/profiler/profiler_view.h"
    "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}/compute_flow/runners/runner.h"
)

# Core sources (always compiled)
set(COMPUTE_FLOW_CORE_SOURCES
    compute_flow/accelerator.cpp
    compute_flow/manifest.cpp
    compute_flow/bricks/brick_manifest.cpp
    compute_flow/bricks/brick_version.cpp
    compute_flow/memory_planner.cpp
    compute_flow/model_data.cpp
    compute_flow/profiler/profiler.cpp
    compute_flow/profiler/timeline_annotations.cpp
    compute_flow/tensor_dict.cpp
    compute_flow/lm_runners/lm_runner.cpp
    compute_flow/runners/runner.cpp
)

# Conditionally add runner sources
if(COMPUTE_FLOW_ENABLE_EXECUTORCH)
    list(APPEND COMPUTE_FLOW_CORE_SOURCES compute_flow/runners/executorch/executorch_runner.cpp)
endif()

if(COMPUTE_FLOW_ENABLE_LITERT)
    list(APPEND COMPUTE_FLOW_CORE_SOURCES compute_flow/runners/litert/litert_runner.cpp)
    if(COMPUTE_FLOW_ENABLE_LITERT_GPU AND NOT ANDROID)
        message(FATAL_ERROR "LiteRT GPU acceleration is currently supported on Android™ only.")
    endif()
endif()

if(COMPUTE_FLOW_ENABLE_ONNXRUNTIME)
    list(APPEND COMPUTE_FLOW_CORE_SOURCES compute_flow/runners/onnxrt/onnxrt_runner.cpp)
endif()

if(COMPUTE_FLOW_ENABLE_MNN)
    list(APPEND COMPUTE_FLOW_CORE_SOURCES compute_flow/runners/mnn/mnn_runner.cpp)
endif()

if(COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    list(APPEND COMPUTE_FLOW_CORE_SOURCES compute_flow/runners/sentencepiece_encode/sentencepiece_encode_runner.cpp)
endif()

if(COMPUTE_FLOW_ENABLE_HF_TOKENIZER)
    list(APPEND COMPUTE_FLOW_CORE_SOURCES compute_flow/runners/huggingface_tokenizer/huggingface_tokenizer_runner.cpp)
endif()

if(COMPUTE_FLOW_ENABLE_LLAMACPP)
    list(APPEND COMPUTE_FLOW_CORE_SOURCES compute_flow/lm_runners/llama_cpp/llama_cpp_lm_runner.cpp)
endif()

# =========================================================================================
# OBJECT library (shared between compute_flow_core and compute_flow targets)
# =========================================================================================

add_library(compute_flow_objs OBJECT ${COMPUTE_FLOW_CORE_SOURCES})

if(COMPUTE_FLOW_ENABLE_STREAMLINE)
    target_sources(compute_flow_objs PRIVATE "${compute_flow_gator_SOURCE_DIR}/annotate/streamline_annotate.c")
endif()

target_include_directories(
    compute_flow_objs
    PRIVATE
        $<BUILD_INTERFACE:${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${COMPUTE_FLOW_PRIVATE_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

if(COMPUTE_FLOW_ENABLE_STREAMLINE)
    target_include_directories(compute_flow_objs PRIVATE "${compute_flow_gator_SOURCE_DIR}/annotate")
    target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_STREAMLINE)
    if(COMPUTE_FLOW_STREAMLINE_USE_TCP)
        target_compile_definitions(compute_flow_objs PRIVATE TCP_ANNOTATIONS)
    endif()
endif()

# --- Compile definitions for framework availability ---
if(COMPUTE_FLOW_ENABLE_LITERT)
    target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_LITERT)
    if(COMPUTE_FLOW_ENABLE_LITERT_GPU)
        target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_LITERT_GPU)
    endif()
endif()
if(COMPUTE_FLOW_ENABLE_ONNXRUNTIME)
    target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_ONNXRT)
endif()
if(COMPUTE_FLOW_ENABLE_MNN)
    target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_MNN)
    if(COMPUTE_FLOW_ENABLE_MNN_GPU)
        target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_MNN_GPU)
    endif()
endif()
if(COMPUTE_FLOW_ENABLE_EXECUTORCH)
    target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_EXECUTORCH)
endif()
if(COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_SENTENCEPIECE)
endif()
if(COMPUTE_FLOW_ENABLE_HF_TOKENIZER)
    target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_HF_TOKENIZER)
endif()
if(COMPUTE_FLOW_ENABLE_LLAMACPP)
    target_compile_definitions(compute_flow_objs PRIVATE COMPUTE_FLOW_HAS_LLAMACPP)
endif()

# --- Link dependencies (always) ---
target_link_libraries(compute_flow_objs PRIVATE nlohmann_json::nlohmann_json libnpy::libnpy)

# --- Conditional link dependencies ---
if(COMPUTE_FLOW_ENABLE_EXECUTORCH)
    target_link_libraries(compute_flow_objs PRIVATE executorch)
    if(COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS)
        target_link_libraries(compute_flow_objs PRIVATE ${COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS})
    endif()
endif()

if(COMPUTE_FLOW_ENABLE_LITERT)
    target_link_libraries(compute_flow_objs PRIVATE litert::litert)
endif()

if(COMPUTE_FLOW_ENABLE_ONNXRUNTIME)
    target_link_libraries(compute_flow_objs PRIVATE onnxruntime::onnxruntime_session)
endif()

if(COMPUTE_FLOW_ENABLE_MNN)
    target_link_libraries(compute_flow_objs PRIVATE ${COMPUTE_FLOW_MNN_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    target_link_libraries(compute_flow_objs PRIVATE ${COMPUTE_FLOW_SENTENCEPIECE_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_HF_TOKENIZER)
    target_link_libraries(compute_flow_objs PRIVATE ${COMPUTE_FLOW_HF_TOKENIZER_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_LLAMACPP)
    target_link_libraries(compute_flow_objs PRIVATE llama)
endif()

# =========================================================================================
# compute_flow_core — thin STATIC from objects (aliased as compute_flow::core)
# =========================================================================================

add_library(compute_flow_core STATIC $<TARGET_OBJECTS:compute_flow_objs>)

target_include_directories(
    compute_flow_core
    PUBLIC $<BUILD_INTERFACE:${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(compute_flow_core PUBLIC nlohmann_json::nlohmann_json libnpy::libnpy)

if(COMPUTE_FLOW_ENABLE_EXECUTORCH)
    target_link_libraries(compute_flow_core PUBLIC executorch)
    if(COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS)
        target_link_libraries(compute_flow_core PUBLIC ${COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS})
    endif()
endif()

if(COMPUTE_FLOW_ENABLE_LITERT)
    target_link_libraries(compute_flow_core PUBLIC litert::litert)
endif()

if(COMPUTE_FLOW_ENABLE_ONNXRUNTIME)
    target_link_libraries(compute_flow_core PUBLIC onnxruntime::onnxruntime_session)
endif()

if(COMPUTE_FLOW_ENABLE_MNN)
    target_link_libraries(compute_flow_core PUBLIC ${COMPUTE_FLOW_MNN_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    target_link_libraries(compute_flow_core PUBLIC ${COMPUTE_FLOW_SENTENCEPIECE_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_HF_TOKENIZER)
    target_link_libraries(compute_flow_core PUBLIC ${COMPUTE_FLOW_HF_TOKENIZER_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_LLAMACPP)
    target_link_libraries(compute_flow_core PUBLIC llama)
endif()

add_library(compute_flow::core ALIAS compute_flow_core)

# =========================================================================================
# compute_flow — monolithic STATIC archive (aliased as compute_flow::compute_flow)
# =========================================================================================

add_library(compute_flow STATIC $<TARGET_OBJECTS:compute_flow_objs>)

target_sources(
    compute_flow
    PUBLIC
        FILE_SET public_headers
            TYPE HEADERS
            BASE_DIRS "${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}"
            FILES ${COMPUTE_FLOW_PUBLIC_HEADERS}
)

target_include_directories(
    compute_flow
    PUBLIC $<BUILD_INTERFACE:${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(compute_flow PUBLIC nlohmann_json::nlohmann_json libnpy::libnpy)

if(COMPUTE_FLOW_ENABLE_EXECUTORCH)
    target_link_libraries(compute_flow PUBLIC executorch)
    if(COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS)
        target_link_libraries(compute_flow PUBLIC ${COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS})
    endif()
endif()

if(COMPUTE_FLOW_ENABLE_LITERT)
    target_link_libraries(compute_flow PUBLIC litert::litert)
endif()

if(COMPUTE_FLOW_ENABLE_ONNXRUNTIME)
    target_link_libraries(compute_flow PUBLIC onnxruntime::onnxruntime_session)
endif()

if(COMPUTE_FLOW_ENABLE_MNN)
    target_link_libraries(compute_flow PUBLIC ${COMPUTE_FLOW_MNN_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    target_link_libraries(compute_flow PUBLIC ${COMPUTE_FLOW_SENTENCEPIECE_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_HF_TOKENIZER)
    target_link_libraries(compute_flow PUBLIC ${COMPUTE_FLOW_HF_TOKENIZER_TARGET})
endif()

if(COMPUTE_FLOW_ENABLE_LLAMACPP)
    target_link_libraries(compute_flow PUBLIC llama)
endif()

add_library(compute_flow::compute_flow ALIAS compute_flow)

# =========================================================================================
# Monolithic archive — merge static deps into compute_flow
# =========================================================================================

# Priority order for dedup: when shared sub-libraries (XNNPACK, pthreadpool, cpuinfo,
# KleidiAI) exist in multiple deps, prefer the copy from the first dep in this list.
set(MONOLITHIC_PRIORITY_ORDER "litert" "executorch")

# Assemble provenance list. Each dep module sets COMPUTE_FLOW_<PREFIX>_MONOLITHIC_ARCHIVES
# (and optionally COMPUTE_FLOW_<PREFIX>_MONOLITHIC_GLOB_DIR). We pass PROVENANCES as a
# flat list of names and a PREFIX_MAP that maps each name to its variable prefix.
set(monolithic_provenances protobuf)
set(MONOLITHIC_PREFIX_MAP_protobuf PROTOBUF)

if(COMPUTE_FLOW_ENABLE_LITERT)
    list(APPEND monolithic_provenances litert)
    set(MONOLITHIC_PREFIX_MAP_litert LITERT)
endif()

if(COMPUTE_FLOW_ENABLE_ONNXRUNTIME)
    list(APPEND monolithic_provenances onnxruntime)
    set(MONOLITHIC_PREFIX_MAP_onnxruntime ONNXRT)
endif()

if(COMPUTE_FLOW_ENABLE_MNN)
    list(APPEND monolithic_provenances mnn)
    set(MONOLITHIC_PREFIX_MAP_mnn MNN)
endif()

if(COMPUTE_FLOW_ENABLE_EXECUTORCH)
    list(APPEND monolithic_provenances executorch)
    set(MONOLITHIC_PREFIX_MAP_executorch EXECUTORCH)
endif()

if(COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    list(APPEND monolithic_provenances sentencepiece)
    set(MONOLITHIC_PREFIX_MAP_sentencepiece SENTENCEPIECE)
endif()

if(COMPUTE_FLOW_ENABLE_HF_TOKENIZER)
    list(APPEND monolithic_provenances hf-tokenizer)
    set(MONOLITHIC_PREFIX_MAP_hf-tokenizer HF_TOKENIZER)
endif()

if(COMPUTE_FLOW_ENABLE_LLAMACPP)
    list(APPEND monolithic_provenances llama-cpp)
    set(MONOLITHIC_PREFIX_MAP_llama-cpp LLAMACPP)
endif()

compute_flow_add_monolithic_archive(
    compute_flow
    "$<TARGET_FILE:compute_flow>"
    OBJECTS compute_flow_objs
    PROVENANCES ${monolithic_provenances}
    PRIORITY_ORDER ${MONOLITHIC_PRIORITY_ORDER}
)

# Strip INTERFACE_LINK_LIBRARIES of static archives already merged into the monolithic archive.
# Before stripping, hoist include directories and compile definitions from header-only
# (INTERFACE_LIBRARY) targets directly onto compute_flow so downstream consumers still
# get them after the targets are removed from the link interface.
#
# Also hoist -force_load / --whole-archive link options from stripped STATIC_LIBRARY targets.
# ExecuTorch targets set INTERFACE_LINK_OPTIONS with -force_load to ensure static constructors
# (backend/kernel registrations) survive linking. When we absorb those archives into the
# monolithic archive, those targets are removed from the link interface, but the original
# individual .a files remain on disk and must still be force-loaded.
set(compute_flow_link_libs "")
get_target_property(compute_flow_link_libs compute_flow INTERFACE_LINK_LIBRARIES)
if(compute_flow_link_libs STREQUAL "INTERFACE_LINK_LIBRARIES-NOTFOUND")
    set(compute_flow_link_libs "")
endif()

set(compute_flow_monolithic_link_libs "")
set(compute_flow_hoisted_link_options "")
set(compute_flow_force_load_targets "")

# _compute_flow_hoist_force_load_opts(<target>)
# If the target is a STATIC_LIBRARY with -force_load / --whole-archive INTERFACE_LINK_OPTIONS,
# append those options to compute_flow_hoisted_link_options.
function(_compute_flow_hoist_force_load_opts target)
    if(NOT TARGET "${target}")
        return()
    endif()
    get_target_property(t_type "${target}" TYPE)
    if(NOT t_type STREQUAL "STATIC_LIBRARY")
        return()
    endif()
    get_target_property(iface_opts "${target}" INTERFACE_LINK_OPTIONS)
    if(NOT iface_opts OR iface_opts STREQUAL "iface_opts-NOTFOUND")
        return()
    endif()
    foreach(opt IN LISTS iface_opts)
        if(opt MATCHES "force_load" OR opt MATCHES "whole-archive")
            list(APPEND compute_flow_hoisted_link_options "${opt}")
            set(compute_flow_hoisted_link_options "${compute_flow_hoisted_link_options}" PARENT_SCOPE)
            list(APPEND compute_flow_force_load_targets "${target}")
            set(compute_flow_force_load_targets "${compute_flow_force_load_targets}" PARENT_SCOPE)
        endif()
    endforeach()
endfunction()

foreach(lib IN LISTS compute_flow_link_libs)
    if(TARGET "${lib}")
        get_target_property(lib_type "${lib}" TYPE)
        if(lib_type STREQUAL "STATIC_LIBRARY")
            # Hoist -force_load / --whole-archive link options before stripping.
            _compute_flow_hoist_force_load_opts("${lib}")
            continue()
        endif()
        if(lib_type STREQUAL "INTERFACE_LIBRARY")
            # Hoist include directories from header-only libs onto compute_flow directly.
            get_target_property(iface_inc "${lib}" INTERFACE_INCLUDE_DIRECTORIES)
            if(iface_inc AND NOT iface_inc STREQUAL "iface_inc-NOTFOUND")
                foreach(inc IN LISTS iface_inc)
                    # Wrap raw paths with BUILD_INTERFACE to avoid install-tree complaints.
                    if(NOT inc MATCHES "^\\$<")
                        target_include_directories(compute_flow PUBLIC "$<BUILD_INTERFACE:${inc}>")
                    else()
                        target_include_directories(compute_flow PUBLIC "${inc}")
                    endif()
                endforeach()
            endif()
            get_target_property(iface_defs "${lib}" INTERFACE_COMPILE_DEFINITIONS)
            if(iface_defs AND NOT iface_defs STREQUAL "iface_defs-NOTFOUND")
                target_compile_definitions(compute_flow PUBLIC ${iface_defs})
            endif()
            # Walk transitive INTERFACE_LINK_LIBRARIES to find STATIC targets with force-load.
            # NOTE: Only walks one level deep. This works because ExecuTorch's umbrella
            # INTERFACE targets (e.g. executorch_backends, executorch_kernels) directly
            # reference the STATIC targets that carry -force_load options. If the dependency
            # chain deepens in the future, this walk must be made recursive.
            get_target_property(iface_link_libs "${lib}" INTERFACE_LINK_LIBRARIES)
            if(iface_link_libs AND NOT iface_link_libs STREQUAL "iface_link_libs-NOTFOUND")
                foreach(transitive_lib IN LISTS iface_link_libs)
                    _compute_flow_hoist_force_load_opts("${transitive_lib}")
                endforeach()
            endif()
            continue()
        endif()
    elseif(lib MATCHES "\\.a$")
        continue()
    endif()

    list(APPEND compute_flow_monolithic_link_libs "${lib}")
endforeach()

set_target_properties(compute_flow PROPERTIES INTERFACE_LINK_LIBRARIES "${compute_flow_monolithic_link_libs}")

# LiteRT's platform and accelerator libraries are not merged into the monolithic archive. Its
# umbrella target is stripped above with the static archive closure, so preserve these link-time
# dependencies explicitly for consumers of compute_flow.
if(COMPUTE_FLOW_ENABLE_LITERT)
    target_link_libraries(compute_flow INTERFACE ${COMPUTE_FLOW_LITERT_RUNTIME_LIBRARIES})
endif()

# MNN's platform libraries are not merged into the monolithic archive. Its imported target is
# stripped above with the static archive closure, so preserve these link-time dependencies for
# consumers of compute_flow.
if(COMPUTE_FLOW_ENABLE_MNN)
    target_link_libraries(compute_flow INTERFACE Threads::Threads)
    if(ANDROID)
        target_link_libraries(compute_flow INTERFACE log m android)
    elseif(UNIX AND NOT APPLE)
        target_link_libraries(compute_flow INTERFACE ${CMAKE_DL_LIBS})
    endif()
endif()

# Apply hoisted force-load link options so static constructors survive linking.
if(compute_flow_hoisted_link_options)
    list(REMOVE_DUPLICATES compute_flow_hoisted_link_options)
    target_link_options(compute_flow INTERFACE ${compute_flow_hoisted_link_options})
endif()

if(ANDROID AND COMPUTE_FLOW_ENABLE_LLAMACPP AND COMPUTE_FLOW_LLAMACPP_OPENMP)
    if(NOT COMPUTE_FLOW_LLAMACPP_OPENMP_LIB)
        message(FATAL_ERROR "COMPUTE_FLOW_LLAMACPP_OPENMP_LIB is required for Android™ OpenMP linking.")
    endif()
    target_link_libraries(compute_flow INTERFACE "${COMPUTE_FLOW_LLAMACPP_OPENMP_LIB}")
    target_link_libraries(compute_flow_core INTERFACE "${COMPUTE_FLOW_LLAMACPP_OPENMP_LIB}")
endif()

# --- Platform-specific link options ---
if(APPLE)
    target_link_options(compute_flow INTERFACE "-framework" "Accelerate")
    target_link_options(compute_flow_core INTERFACE "-framework" "Accelerate")
    target_link_libraries(compute_flow INTERFACE "-framework Foundation" "-framework CoreFoundation")
endif()

if(ANDROID)
    target_link_libraries(compute_flow INTERFACE log)
endif()

# =========================================================================================
# Python bindings
# =========================================================================================

if(COMPUTE_FLOW_PYBIND_ENABLED)
    pybind11_add_module(_compute_flow_py
        ${CMAKE_SOURCE_DIR}/compute_flow/bindings/python/compute_flow_pybind.cpp
        ${COMPUTE_FLOW_BRICK_MANIFEST_SELECTOR_SOURCE}
    )
    target_include_directories(_compute_flow_py PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
    target_link_libraries(_compute_flow_py PRIVATE compute_flow)

    set(COMPUTE_FLOW_PY_BUILD_ROOT "${CMAKE_CURRENT_BINARY_DIR}/python")
    set(COMPUTE_FLOW_PY_PKG_DIR "${COMPUTE_FLOW_PY_BUILD_ROOT}/compute_flow_py")
    set(COMPUTE_FLOW_CLI_EXAMPLES_PY_PKG_DIR "${COMPUTE_FLOW_PY_BUILD_ROOT}/cli_examples")

    set(COMPUTE_FLOW_PY_SRC_PKG_DIR "${CMAKE_SOURCE_DIR}/compute_flow/python/compute_flow_py")
    set(COMPUTE_FLOW_CLI_EXAMPLES_PY_SRC_PKG_DIR "${CMAKE_SOURCE_DIR}/compute_flow/python/cli_examples")

    set_target_properties(
        _compute_flow_py
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${COMPUTE_FLOW_PY_PKG_DIR}"
            RUNTIME_OUTPUT_DIRECTORY
                "${COMPUTE_FLOW_PY_PKG_DIR}" # Windows
    )

    add_custom_target(
        compute_flow_py_pkg
        ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${COMPUTE_FLOW_PY_PKG_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${COMPUTE_FLOW_PY_SRC_PKG_DIR}" "${COMPUTE_FLOW_PY_PKG_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${COMPUTE_FLOW_CLI_EXAMPLES_PY_PKG_DIR}"
        COMMAND
            "${CMAKE_COMMAND}" -E copy_directory "${COMPUTE_FLOW_CLI_EXAMPLES_PY_SRC_PKG_DIR}"
            "${COMPUTE_FLOW_CLI_EXAMPLES_PY_PKG_DIR}"
        COMMENT "Sync Python package files into build tree"
        VERBATIM
    )
    add_dependencies(compute_flow_py_pkg _compute_flow_py)

    set(site_packages_subdir "compute_flow_py")
    compute_flow_install_python_extension(_compute_flow_py site_packages_subdir)
    cmake_path(GET site_packages_subdir PARENT_PATH site_packages_root)
    set(site_packages_cli_examples_subdir "${site_packages_root}/cli_examples")

    add_custom_target(
        install_compute_flow_py_pkg
        ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${site_packages_subdir}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${COMPUTE_FLOW_PY_SRC_PKG_DIR}" "${site_packages_subdir}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${site_packages_cli_examples_subdir}"
        COMMAND
            "${CMAKE_COMMAND}" -E copy_directory "${COMPUTE_FLOW_CLI_EXAMPLES_PY_SRC_PKG_DIR}"
            "${site_packages_cli_examples_subdir}"
        COMMENT "Sync Python package files into ${site_packages_root}"
        VERBATIM
    )

    add_dependencies(install_compute_flow_py_pkg _compute_flow_py compute_flow_py_pkg)
endif()

# =========================================================================================
# Install
# =========================================================================================

install(
    TARGETS compute_flow
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    FILE_SET public_headers DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
    EXPORT ${PROJECT_NAME}Targets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    NAMESPACE compute_flow::
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/compute_flowConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

# =========================================================================================
# Bricks & generic type runners
# =========================================================================================

compute_flow_discover_installed_bricks()

compute_flow_configure_brick_type_libraries()

compute_flow_configure_brick_factory(COMPUTE_FLOW_GENERATED_BRICK_FACTORY_SOURCE)

# --- brick factory target ---
add_library(compute_flow_brick_factory STATIC "${COMPUTE_FLOW_GENERATED_BRICK_FACTORY_SOURCE}")

target_include_directories(
    compute_flow_brick_factory
    PUBLIC $<BUILD_INTERFACE:${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> ${COMPUTE_FLOW_INSTALLED_BRICK_SOURCE_DIRS}
)

target_link_libraries(
    compute_flow_brick_factory
    PUBLIC compute_flow::compute_flow
    PRIVATE ${COMPUTE_FLOW_INSTALLED_BRICK_TYPE_TARGETS}
)

add_library(compute_flow::brick_factory ALIAS compute_flow_brick_factory)

add_custom_target(
    compute_flow_application_sdk
    COMMAND
        "${CMAKE_COMMAND}" "-DCOMPUTE_FLOW_INCLUDE_DIR=${COMPUTE_FLOW_PUBLIC_INCLUDE_DIR}"
        "-DCOMPUTE_FLOW_LIBRARY_MANIFEST=${CMAKE_BINARY_DIR}/compute_flow_sdk_libraries-$<CONFIG>.txt"
        "-DCOMPUTE_FLOW_FORCE_LOAD_LIBRARY_MANIFEST=${CMAKE_BINARY_DIR}/compute_flow_sdk_force_load_libraries-$<CONFIG>.txt"
        "-DCOMPUTE_FLOW_SDK_DIR=${CMAKE_BINARY_DIR}/sdk" -P "${CMAKE_SOURCE_DIR}/cmake/PrepareComputeFlowSDK.cmake"
    DEPENDS compute_flow compute_flow_brick_factory ${COMPUTE_FLOW_INSTALLED_BRICK_TYPE_TARGETS}
    COMMENT "Preparing the Compute Flow application SDK"
    VERBATIM
)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4
)

FetchContent_MakeAvailable(stb)

add_library(stb INTERFACE)
add_library(stb::stb ALIAS stb)

target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

# --- generic type runner executables ---
set(COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES
    ${COMPUTE_FLOW_BRICK_MANIFEST_SELECTOR_SOURCE}
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/internal/thread_config.cpp"
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/internal/wav.cpp"
)
set(COMPUTE_FLOW_BRICK_IMAGE_HELPER_SOURCE "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/internal/image.cpp")
set(COMPUTE_FLOW_BRICK_RUN_CONFIG_SOURCE "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/internal/run_config.cpp")

add_executable(
    audiogen_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/audiogen_cli.cpp"
    ${COMPUTE_FLOW_BRICK_RUN_CONFIG_SOURCE}
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
compute_flow_link_installed_bricks(audiogen_cli)

add_executable(
    audio_detection_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/audio_detection_cli.cpp"
    ${COMPUTE_FLOW_BRICK_RUN_CONFIG_SOURCE}
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
compute_flow_link_installed_bricks(audio_detection_cli)

add_executable(
    audio_enhancement_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/audio_enhancement_cli.cpp"
    ${COMPUTE_FLOW_BRICK_RUN_CONFIG_SOURCE}
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
compute_flow_link_installed_bricks(audio_enhancement_cli)

add_executable(
    image_edit_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/image_edit_cli.cpp"
    ${COMPUTE_FLOW_BRICK_RUN_CONFIG_SOURCE}
    ${COMPUTE_FLOW_BRICK_IMAGE_HELPER_SOURCE}
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
target_link_libraries(image_edit_cli PRIVATE stb::stb)
compute_flow_link_installed_bricks(image_edit_cli)

add_executable(
    monocular_depth_estimation_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/monocular_depth_estimation_cli.cpp"
    ${COMPUTE_FLOW_BRICK_IMAGE_HELPER_SOURCE}
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
target_link_libraries(monocular_depth_estimation_cli PRIVATE stb::stb)
compute_flow_link_installed_bricks(monocular_depth_estimation_cli)

add_executable(
    lm_text_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/lm_text_cli.cpp"
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
compute_flow_link_installed_bricks(lm_text_cli)

add_executable(
    musicgen_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/musicgen_cli.cpp"
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
compute_flow_link_installed_bricks(musicgen_cli)

add_executable(
    stt_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/stt_cli.cpp"
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
compute_flow_link_installed_bricks(stt_cli)

add_executable(
    tts_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/tts_cli.cpp"
    ${COMPUTE_FLOW_BRICK_RUN_CONFIG_SOURCE}
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
compute_flow_link_installed_bricks(tts_cli)

add_executable(
    vad_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/vad_cli.cpp"
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
compute_flow_link_installed_bricks(vad_cli)

add_executable(
    neural_codec_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/neural_codec_cli.cpp"
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
target_link_libraries(neural_codec_cli PRIVATE libnpy::libnpy)
compute_flow_link_installed_bricks(neural_codec_cli)

add_executable(
    object_detection_cli
    "${CMAKE_SOURCE_DIR}/compute_flow/cli_examples/object_detection_cli.cpp"
    ${COMPUTE_FLOW_BRICK_IMAGE_HELPER_SOURCE}
    ${COMPUTE_FLOW_BRICK_CLI_HELPER_SOURCES}
)
target_link_libraries(object_detection_cli PRIVATE stb::stb)
compute_flow_link_installed_bricks(object_detection_cli)

if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(tests)
endif()

function(compute_flow_collect_built_libraries directory out_var)
    set(library_files "")
    get_property(directory_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS directory_targets)
        get_target_property(brick_class "${target}" COMPUTE_FLOW_BRICK_CLASS)
        if(brick_class AND NOT brick_class STREQUAL "brick_class-NOTFOUND")
            continue()
        endif()

        get_target_property(target_type "${target}" TYPE)
        if(target_type STREQUAL "STATIC_LIBRARY" OR target_type STREQUAL "SHARED_LIBRARY")
            list(APPEND library_files "$<TARGET_FILE:${target}>")
        endif()
    endforeach()

    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        compute_flow_collect_built_libraries("${subdirectory}" subdirectory_library_files)
        list(APPEND library_files ${subdirectory_library_files})
    endforeach()

    set(${out_var} "${library_files}" PARENT_SCOPE)
endfunction()

compute_flow_collect_built_libraries("${CMAKE_SOURCE_DIR}" COMPUTE_FLOW_SDK_LIBRARY_FILES)
list(REMOVE_DUPLICATES COMPUTE_FLOW_SDK_LIBRARY_FILES)
list(JOIN COMPUTE_FLOW_SDK_LIBRARY_FILES "\n" COMPUTE_FLOW_SDK_LIBRARY_MANIFEST)
file(
    GENERATE OUTPUT "${CMAKE_BINARY_DIR}/compute_flow_sdk_libraries-$<CONFIG>.txt"
    CONTENT "${COMPUTE_FLOW_SDK_LIBRARY_MANIFEST}\n"
)

list(REMOVE_DUPLICATES compute_flow_force_load_targets)
set(COMPUTE_FLOW_SDK_FORCE_LOAD_LIBRARY_FILES "")
foreach(target IN LISTS compute_flow_force_load_targets)
    list(APPEND COMPUTE_FLOW_SDK_FORCE_LOAD_LIBRARY_FILES "$<TARGET_FILE:${target}>")
endforeach()
list(JOIN COMPUTE_FLOW_SDK_FORCE_LOAD_LIBRARY_FILES "\n" COMPUTE_FLOW_SDK_FORCE_LOAD_LIBRARY_MANIFEST)
file(
    GENERATE OUTPUT "${CMAKE_BINARY_DIR}/compute_flow_sdk_force_load_libraries-$<CONFIG>.txt"
    CONTENT "${COMPUTE_FLOW_SDK_FORCE_LOAD_LIBRARY_MANIFEST}\n"
)
