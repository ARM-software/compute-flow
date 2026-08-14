#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/Common.cmake — Shared ExternalProject infrastructure for all dependency modules
# =========================================================================================
#
# Provides:
#   COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS      — target-side EP args (toolchain, flags, NDK, etc.)
#   COMPUTE_FLOW_HOST_EP_COMMON_CMAKE_ARGS — host-only EP args (no cross-compile vars)
#   COMPUTE_FLOW_BUILD_HOST_TOOLS          — ON when cross-compiling (auto-detected)
#   COMPUTE_FLOW_DEPS_PREFIX_DIR           — prefix for downloading/building deps
#   COMPUTE_FLOW_DEPS_INSTALL_DIR          — prefix where deps will be installed
#   compute_flow_dep_install_dir()         — helper to get per-dep install path
#   compute_flow_dep_prefix_dir()          — helper to get per-dep prefix path
#
# Must be included before any per-dependency module.
# =========================================================================================

include_guard(GLOBAL)

include(ExternalProject)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ---- Install directories ----------------------------------------------------------------

set(COMPUTE_FLOW_DEPS_PREFIX_DIR
    "${CMAKE_BINARY_DIR}/_deps"
    CACHE PATH
    "Prefix for downloading and building third-party deps"
)
set(COMPUTE_FLOW_DEPS_INSTALL_DIR
    "${COMPUTE_FLOW_DEPS_PREFIX_DIR}/install"
    CACHE PATH
    "Prefix where third-party deps will be installed"
)

# ---- Helper: append a cache variable to an EP CMAKE_ARGS list ---------------------------

function(compute_flow_append_ep_cache_arg list_var cache_var cache_type)
    if(NOT DEFINED ${cache_var})
        return()
    endif()

    if("${${cache_var}}" STREQUAL "")
        return()
    endif()

    set(value "${${cache_var}}")
    # ExternalProject CMAKE_ARGS are CMake lists; escape any semicolons in values.
    string(REPLACE ";" "\\;" value "${value}")
    list(APPEND ${list_var} "-D${cache_var}:${cache_type}=${value}")
    set(${list_var} "${${list_var}}" PARENT_SCOPE)
endfunction()

# ---- Target-side common args ------------------------------------------------------------

set(COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DFETCHCONTENT_UPDATES_DISCONNECTED:BOOL=ON
)

# Stable library names across build types (avoid libprotobufd.a etc.)
foreach(
    postfix_var
    IN
    ITEMS CMAKE_DEBUG_POSTFIX CMAKE_RELEASE_POSTFIX CMAKE_RELWITHDEBINFO_POSTFIX CMAKE_MINSIZEREL_POSTFIX
)
    list(APPEND COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS "-D${postfix_var}:STRING=")
endforeach()

if(NOT CMAKE_CONFIGURATION_TYPES)
    list(APPEND COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()

# Toolchain file
compute_flow_append_ep_cache_arg(COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS CMAKE_TOOLCHAIN_FILE FILEPATH)

# Compiler selection
foreach(
    compiler_var
    IN
    ITEMS
        CMAKE_C_COMPILER
        CMAKE_CXX_COMPILER
        CMAKE_ASM_COMPILER
        CMAKE_AR
        CMAKE_RANLIB
        CMAKE_C_COMPILER_AR
        CMAKE_C_COMPILER_RANLIB
        CMAKE_CXX_COMPILER_AR
        CMAKE_CXX_COMPILER_RANLIB
)
    compute_flow_append_ep_cache_arg(COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS ${compiler_var} FILEPATH)
endforeach()

# Cross-compiling root-path behavior
foreach(
    root_var
    IN
    ITEMS
        CMAKE_SYSROOT
        CMAKE_SYSROOT_COMPILE
        CMAKE_SYSROOT_LINK
        CMAKE_FIND_ROOT_PATH
        CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
        CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
        CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
        CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
)
    compute_flow_append_ep_cache_arg(COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS ${root_var} STRING)
endforeach()

# Apple toolchain inputs
foreach(osx_var IN ITEMS CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_OSX_SYSROOT)
    compute_flow_append_ep_cache_arg(COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS ${osx_var} STRING)
endforeach()

# Android™ NDK toolchain inputs
foreach(android_var IN ITEMS ANDROID_ABI ANDROID_PLATFORM ANDROID_STL ANDROID_NATIVE_API_LEVEL)
    compute_flow_append_ep_cache_arg(COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS ${android_var} STRING)
endforeach()
compute_flow_append_ep_cache_arg(COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS ANDROID_NDK PATH)

# Android™: strip debug info in Release builds to reduce output size
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    option(
        COMPUTE_FLOW_ANDROID_DISABLE_DEBUG_INFO
        "Append -g0 to Release compile flags to override the NDK toolchain default '-g'."
        ON
    )
    if(COMPUTE_FLOW_ANDROID_DISABLE_DEBUG_INFO)
        foreach(var IN ITEMS CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE CMAKE_ASM_FLAGS_RELEASE)
            set(cur "${${var}}")
            if(NOT cur MATCHES "(^| )-g0( |$)")
                string(APPEND cur " -g0")
                string(STRIP "${cur}" cur)
                set(${var} "${cur}" CACHE STRING "Release flags (from-source override)" FORCE)
            endif()
        endforeach()
    endif()
endif()

# Compiler/linker flags
foreach(
    flag_var
    IN
    ITEMS
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS
        CMAKE_MODULE_LINKER_FLAGS
        CMAKE_STATIC_LINKER_FLAGS
)
    compute_flow_append_ep_cache_arg(COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS ${flag_var} STRING)
endforeach()

# Warn early about missing Android™ ABI
if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND (NOT DEFINED ANDROID_ABI OR ANDROID_ABI STREQUAL ""))
    message(
        WARNING
        "CMAKE_SYSTEM_NAME=Android but ANDROID_ABI is not set. "
        "Pass e.g. -DANDROID_ABI=arm64-v8a to configure a valid Android™ target."
    )
endif()

# ---- Host tools (cross-compile only) ----------------------------------------------------

set(COMPUTE_FLOW_BUILD_HOST_TOOLS
    ${CMAKE_CROSSCOMPILING}
    CACHE BOOL
    "Build host-side codegen tools (flatc/protoc) needed for cross-compiling"
)

if(COMPUTE_FLOW_BUILD_HOST_TOOLS)
    set(COMPUTE_FLOW_HOST_EP_COMMON_CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    )

    foreach(
        postfix_var
        IN
        ITEMS CMAKE_DEBUG_POSTFIX CMAKE_RELEASE_POSTFIX CMAKE_RELWITHDEBINFO_POSTFIX CMAKE_MINSIZEREL_POSTFIX
    )
        list(APPEND COMPUTE_FLOW_HOST_EP_COMMON_CMAKE_ARGS "-D${postfix_var}:STRING=")
    endforeach()

    if(NOT CMAKE_CONFIGURATION_TYPES)
        list(APPEND COMPUTE_FLOW_HOST_EP_COMMON_CMAKE_ARGS "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
    endif()
endif()

# ---- Per-dependency install prefix helpers -----------------------------------------------

function(compute_flow_dep_install_dir out_var dep_name)
    set(${out_var} "${COMPUTE_FLOW_DEPS_INSTALL_DIR}/${dep_name}" PARENT_SCOPE)
endfunction()

function(compute_flow_dep_prefix_dir out_var dep_name)
    set(${out_var} "${COMPUTE_FLOW_DEPS_PREFIX_DIR}/${dep_name}" PARENT_SCOPE)
endfunction()
