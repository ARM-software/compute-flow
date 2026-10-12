#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/executorch/ExecuTorch.cmake — ExecuTorch runtime
# =========================================================================================
#
# Creates IMPORTED targets that match the names produced by find_package(executorch CONFIG):
#   executorch                     — main ExecuTorch static library
#   executorch_core                — core runtime
#   extension_module_static        — extension module
#   extension_tensor               — tensor extensions
#   extension_data_loader          — file/mmap data loading
#   extension_flat_tensor          — flat tensor deserialization
#   extension_named_data_map       — named data map for model weights
#   optimized_native_cpu_ops_lib   — optimized CPU kernels
#   xnnpack_backend                — XNNPACK delegate backend
#
# Also populates COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS with the extension/backend targets for the main
# CMakeLists.txt to link against.
#
# Requires Python3 with torch and PyYAML installed (provided by uv sync).
#
# Resolution: find_package → ExternalProject (gated by from-source flag)
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_EXECUTORCH_GIT_REPO "https://github.com/pytorch/executorch.git")
set(COMPUTE_FLOW_EXECUTORCH_GIT_TAG "f7140a46ff38e919c557d45b102d3ff26097c8c9") # v1.5.0

# ExecuTorch v1.5 installs this optional runtime even though the selected targets do not link it.
# Keep it out of the broad archive glob used to assemble the Compute Flow monolithic library.
set(COMPUTE_FLOW_EXECUTORCH_MONOLITHIC_EXCLUDE "libflatccrt.a")

# ---- Try pre-installed (release / system) --------------------------------------------------
if(NOT COMPUTE_FLOW_BUILD_EXECUTORCH_FROM_SOURCE)
    find_package(executorch CONFIG QUIET)

    if(executorch_FOUND)
        message(STATUS "ExecuTorch: found pre-installed")
        add_custom_target(compute_flow_dep_executorch)

        # ExecuTorch is split into multiple targets; prefer umbrella targets when available.
        set(COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS "")
        if(TARGET executorch_extensions)
            list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS executorch_extensions)
        else()
            if(TARGET extension_module_static)
                list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS extension_module_static)
            endif()
            if(TARGET extension_tensor)
                list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS extension_tensor)
            endif()
            if(TARGET extension_data_loader)
                list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS extension_data_loader)
            endif()
            if(TARGET extension_flat_tensor)
                list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS extension_flat_tensor)
            endif()
            if(TARGET extension_named_data_map)
                list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS extension_named_data_map)
            endif()
        endif()

        if(TARGET executorch_kernels)
            list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS executorch_kernels)
        elseif(TARGET optimized_native_cpu_ops_lib)
            list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS optimized_native_cpu_ops_lib)
        endif()

        if(TARGET executorch_backends)
            list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS executorch_backends)
        elseif(TARGET xnnpack_backend)
            list(APPEND COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS xnnpack_backend)
        else()
            message(STATUS "ExecuTorch: xnnpack_backend target not found; XnnpackBackend models may fail to run.")
        endif()

        # Monolithic archive declaration
        compute_flow_extract_target_archives(COMPUTE_FLOW_EXECUTORCH_MONOLITHIC_ARCHIVES
            executorch ${COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS}
        )

        # Derive glob dir from the executorch target's installed location so the
        # build-time script picks up all sub-dep archives (executorch_core, extensions,
        # XNNPACK, pthreadpool, cpuinfo, KleidiAI, etc.) that aren't exposed via
        # target properties.
        get_target_property(_et_loc executorch IMPORTED_LOCATION_RELEASE)
        if(NOT _et_loc)
            get_target_property(_et_loc executorch IMPORTED_LOCATION)
        endif()
        if(_et_loc)
            get_filename_component(COMPUTE_FLOW_EXECUTORCH_MONOLITHIC_GLOB_DIR "${_et_loc}" DIRECTORY)
        endif()

        return()
    endif()
endif()

# ---- Guard: require explicit from-source opt-in -----------------------------------------
if(NOT COMPUTE_FLOW_BUILD_EXECUTORCH_FROM_SOURCE)
    message(
        FATAL_ERROR
        "ExecuTorch not found via find_package and from-source build not enabled.\n"
        "Set -DCOMPUTE_FLOW_BUILD_EXECUTORCH_FROM_SOURCE=ON."
    )
endif()

# ---- Build from source via ExternalProject -----------------------------------------------
message(STATUS "ExecuTorch: building from source via ExternalProject")

compute_flow_dep_prefix_dir(executorch_prefix_dir "executorch")
compute_flow_dep_install_dir(executorch_install_dir "executorch")

# ExecuTorch source directory must be named 'executorch' (expected by its build system)
set(executorch_source_dir "${executorch_prefix_dir}/executorch")

# ---- EP ---------------------------------------------------------------------------------

ExternalProject_Add(
    compute_flow_dep_executorch
    GIT_REPOSITORY "${COMPUTE_FLOW_EXECUTORCH_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_EXECUTORCH_GIT_TAG}"
    PREFIX "${executorch_prefix_dir}"
    SOURCE_DIR "${executorch_source_dir}"
    INSTALL_DIR "${executorch_install_dir}"
    GIT_PROGRESS TRUE
    GIT_SUBMODULES_RECURSE ON
    UPDATE_DISCONNECTED ON
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DPython3_EXECUTABLE:FILEPATH=${Python3_EXECUTABLE}
        # ExecuTorch options
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF
        -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=OFF -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON
        -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=ON
    BUILD_BYPRODUCTS <BINARY_DIR>/libexecutorch.a <BINARY_DIR>/libexecutorch_core.a
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
    # ExecuTorch does not install its BSD-3-Clause license. Binary redistributions must retain the
    # license notice, so copy the exact upstream file into the release prefix.
    COMMAND ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/share/licenses/executorch
    COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/LICENSE <INSTALL_DIR>/share/licenses/executorch/LICENSE
)

set(COMPUTE_FLOW_EXECUTORCH_INSTALL_DIR "${executorch_install_dir}")
set(COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS
    "${executorch_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    "${executorch_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}/executorch/runtime/core/portable_type/c10"
)
set(COMPUTE_FLOW_EXECUTORCH_INTERFACE_COMPILE_DEFINITIONS C10_USING_CUSTOM_GENERATED_MACROS)

# ---- Placeholder IMPORTED targets -------------------------------------------------------
#
# ExecuTorch installs its own CMake config so find_package works after the EP build.
# We create minimal placeholder targets here so that target_link_libraries() calls in the
# main CMakeLists.txt succeed at configure time. The actual libraries appear at build time.

file(MAKE_DIRECTORY ${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS})

# ExecuTorch registers runtimes, backends, and operators through static initializers. Preserve
# those otherwise-unreferenced objects when applications link the Compute Flow archive.
function(_compute_flow_executorch_force_load target)
    if(APPLE)
        set(force_load_option "SHELL:LINKER:-force_load,$<TARGET_FILE:${target}>")
    else()
        set(force_load_option "SHELL:LINKER:--whole-archive $<TARGET_FILE:${target}> LINKER:--no-whole-archive")
    endif()
    set_property(TARGET "${target}" APPEND PROPERTY INTERFACE_LINK_OPTIONS "${force_load_option}")
endfunction()

if(NOT TARGET executorch)
    add_library(executorch STATIC IMPORTED GLOBAL)
    set_target_properties(
        executorch
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libexecutorch.a"
            INTERFACE_COMPILE_DEFINITIONS "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_COMPILE_DEFINITIONS}"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    _compute_flow_executorch_force_load(executorch)
    add_dependencies(executorch compute_flow_dep_executorch)
endif()

if(NOT TARGET executorch_core)
    add_library(executorch_core STATIC IMPORTED GLOBAL)
    set_target_properties(
        executorch_core
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libexecutorch_core.a"
            INTERFACE_COMPILE_DEFINITIONS "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_COMPILE_DEFINITIONS}"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    add_dependencies(executorch_core compute_flow_dep_executorch)
endif()

# Extension/backend targets — ExecuTorch's install layout may vary, but these are the
# common target names the main CMakeLists.txt resolves via COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS.

if(NOT TARGET extension_module_static)
    add_library(extension_module_static STATIC IMPORTED GLOBAL)
    set_target_properties(
        extension_module_static
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_module_static.a"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    add_dependencies(extension_module_static compute_flow_dep_executorch)
endif()

if(NOT TARGET extension_tensor)
    add_library(extension_tensor STATIC IMPORTED GLOBAL)
    set_target_properties(
        extension_tensor
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_tensor.a"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    add_dependencies(extension_tensor compute_flow_dep_executorch)
endif()

if(NOT TARGET optimized_native_cpu_ops_lib)
    add_library(optimized_native_cpu_ops_lib STATIC IMPORTED GLOBAL)
    set_target_properties(
        optimized_native_cpu_ops_lib
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/liboptimized_native_cpu_ops_lib.a"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    _compute_flow_executorch_force_load(optimized_native_cpu_ops_lib)
    add_dependencies(optimized_native_cpu_ops_lib compute_flow_dep_executorch)
endif()

if(NOT TARGET extension_data_loader)
    add_library(extension_data_loader STATIC IMPORTED GLOBAL)
    set_target_properties(
        extension_data_loader
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_data_loader.a"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    add_dependencies(extension_data_loader compute_flow_dep_executorch)
endif()

if(NOT TARGET extension_flat_tensor)
    add_library(extension_flat_tensor STATIC IMPORTED GLOBAL)
    set_target_properties(
        extension_flat_tensor
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_flat_tensor.a"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    add_dependencies(extension_flat_tensor compute_flow_dep_executorch)
endif()

if(NOT TARGET extension_named_data_map)
    add_library(extension_named_data_map STATIC IMPORTED GLOBAL)
    set_target_properties(
        extension_named_data_map
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_named_data_map.a"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    add_dependencies(extension_named_data_map compute_flow_dep_executorch)
endif()

if(NOT TARGET xnnpack_backend)
    add_library(xnnpack_backend STATIC IMPORTED GLOBAL)
    set_target_properties(
        xnnpack_backend
        PROPERTIES
            IMPORTED_LOCATION "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libxnnpack_backend.a"
            INTERFACE_INCLUDE_DIRECTORIES "${COMPUTE_FLOW_EXECUTORCH_INTERFACE_INCLUDE_DIRS}"
    )
    _compute_flow_executorch_force_load(xnnpack_backend)
    add_dependencies(xnnpack_backend compute_flow_dep_executorch)
endif()

# Populate COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS for the main CMakeLists.txt
# (mirrors the logic in the find_package resolution above)
set(COMPUTE_FLOW_EXECUTORCH_EXTRA_TARGETS
    extension_module_static
    extension_tensor
    extension_data_loader
    extension_flat_tensor
    extension_named_data_map
    optimized_native_cpu_ops_lib
    xnnpack_backend
)

# ---- Monolithic archive declaration -----------------------------------------------------
set(COMPUTE_FLOW_EXECUTORCH_MONOLITHIC_ARCHIVES
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libexecutorch.a"
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libexecutorch_core.a"
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_module_static.a"
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_tensor.a"
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_data_loader.a"
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_flat_tensor.a"
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libextension_named_data_map.a"
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/liboptimized_native_cpu_ops_lib.a"
    "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}/libxnnpack_backend.a"
)
# ExecuTorch installs sub-deps (XNNPACK, pthreadpool, cpuinfo, KleidiAI) that aren't
# enumerated — must glob at build time.
set(COMPUTE_FLOW_EXECUTORCH_MONOLITHIC_GLOB_DIR "${executorch_install_dir}/${CMAKE_INSTALL_LIBDIR}")
