#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED EXECUTORCH_SOURCE_DIR OR EXECUTORCH_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "EXECUTORCH_SOURCE_DIR must be set (pass -DEXECUTORCH_SOURCE_DIR=/path/to/executorch).")
endif()

set(xnnpack_cmakelists "${EXECUTORCH_SOURCE_DIR}/backends/xnnpack/third-party/XNNPACK/CMakeLists.txt")

if(NOT EXISTS "${xnnpack_cmakelists}")
    message(FATAL_ERROR "Expected XNNPACK CMakeLists.txt at: '${xnnpack_cmakelists}'")
endif()

file(READ "${xnnpack_cmakelists}" content)

if(NOT content MATCHES "FIND_LIBRARY\\(LIBM m\\)")
    message(STATUS "XNNPACK libm lookup already patched (no FIND_LIBRARY(LIBM m) found).")
    return()
endif()

# XNNPACK calls FIND_LIBRARY(LIBM m) which resolves to an absolute path inside the active sysroot/SDK.
#
# If this absolute path is later exported by ExecuTorch's CMake package, it becomes machine-specific and breaks
# consumers on other machines (e.g. different sysroot/SDK location).
#
# Use the link token "m" so the consumer toolchain resolves it from its own sysroot/SDK.
string(REPLACE "FIND_LIBRARY(LIBM m)" "SET(LIBM m)" content "${content}")

file(WRITE "${xnnpack_cmakelists}" "${content}")

# --- flatcc: disable -Werror ---
#
# flatcc enables -Werror by default, which breaks on newer compilers (e.g. AppleClang 21+).
# ExecuTorch builds flatcc twice (ExternalProject + add_subdirectory), so we inject
# FLATCC_ALLOW_WERROR=OFF into both paths.

set(third_party_cmakelists "${EXECUTORCH_SOURCE_DIR}/third-party/CMakeLists.txt")

if(NOT EXISTS "${third_party_cmakelists}")
    message(STATUS "ExecuTorch third-party/CMakeLists.txt not found – skipping flatcc patch.")
    return()
endif()

file(READ "${third_party_cmakelists}" tp_content)

if(NOT tp_content MATCHES "FLATCC_ALLOW_WERROR")
    # Path 1: add -DFLATCC_ALLOW_WERROR=OFF to the flatcc_ep ExternalProject CMAKE_ARGS.
    string(REPLACE "-DFLATCC_RTONLY=OFF" "-DFLATCC_RTONLY=OFF -DFLATCC_ALLOW_WERROR=OFF" tp_content "${tp_content}")

    # Path 2: add FLATCC_ALLOW_WERROR=OFF cache entry before the add_subdirectory(flatcc).
    string(
        REPLACE "set(FLATCC_RTONLY ON CACHE BOOL \"\")"
        "set(FLATCC_ALLOW_WERROR OFF CACHE BOOL \"\")\nset(FLATCC_RTONLY ON CACHE BOOL \"\")"
        tp_content
        "${tp_content}"
    )

    file(WRITE "${third_party_cmakelists}" "${tp_content}")
    message(STATUS "Patched flatcc builds to disable -Werror (FLATCC_ALLOW_WERROR=OFF).")
else()
    message(STATUS "flatcc FLATCC_ALLOW_WERROR already patched – skipping.")
endif()
