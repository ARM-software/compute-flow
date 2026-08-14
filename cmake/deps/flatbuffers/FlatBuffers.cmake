#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/flatbuffers/FlatBuffers.cmake — Host FlatBuffers compiler (cross-compile only)
# =========================================================================================
#
# When cross-compiling, builds the host-side `flatc` binary needed by TFLite's build system.
# On native builds this module is a no-op; TFLite builds its own flatc internally.
#
# Sets:
#   COMPUTE_FLOW_FLATBUFFERS_HOST_INSTALL_DIR — install prefix containing bin/flatc
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_FLATBUFFERS_VERSION "25.9.23")
set(COMPUTE_FLOW_FLATBUFFERS_GIT_REPO "https://github.com/google/flatbuffers.git")
set(COMPUTE_FLOW_FLATBUFFERS_GIT_TAG "v${COMPUTE_FLOW_FLATBUFFERS_VERSION}")

# ---- No-op on native builds -------------------------------------------------------------
if(NOT COMPUTE_FLOW_BUILD_HOST_TOOLS)
    return()
endif()

# ---- Build host-side flatc via ExternalProject -------------------------------------------
message(STATUS "FlatBuffers: building host-side flatc for cross-compilation")

compute_flow_dep_prefix_dir(flatbuffers_host_prefix_dir "flatbuffers-host")
compute_flow_dep_install_dir(flatbuffers_host_install_dir "flatbuffers-host")

ExternalProject_Add(
    compute_flow_dep_flatbuffers
    GIT_REPOSITORY "${COMPUTE_FLOW_FLATBUFFERS_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_FLATBUFFERS_GIT_TAG}"
    PREFIX "${flatbuffers_host_prefix_dir}"
    INSTALL_DIR "${flatbuffers_host_install_dir}"
    GIT_PROGRESS TRUE
    UPDATE_DISCONNECTED ON
    CMAKE_ARGS
        ${COMPUTE_FLOW_HOST_EP_COMMON_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DFLATBUFFERS_BUILD_TESTS:BOOL=OFF -DFLATBUFFERS_BUILD_FLATC:BOOL=ON -DFLATBUFFERS_BUILD_FLATLIB:BOOL=OFF
        -DFLATBUFFERS_INSTALL:BOOL=ON
    BUILD_BYPRODUCTS <INSTALL_DIR>/bin/flatc
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)

set(COMPUTE_FLOW_FLATBUFFERS_HOST_INSTALL_DIR "${flatbuffers_host_install_dir}")
