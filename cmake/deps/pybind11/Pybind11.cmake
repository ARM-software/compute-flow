#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/pybind11/Pybind11.cmake — pybind11 (Python binding library)
# =========================================================================================
#
# pybind11 is the ONE exception that must use FetchContent instead of ExternalProject.
# Reason: pybind11 provides pybind11_add_module(), a CMake function needed at configure
# time by every brick's CMakeLists.txt. ExternalProject builds at build time, so the
# function would not be available.
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_PYBIND11_VERSION "2.12.0")
set(COMPUTE_FLOW_PYBIND11_GIT_REPO "https://github.com/pybind/pybind11.git")
set(COMPUTE_FLOW_PYBIND11_GIT_TAG "v${COMPUTE_FLOW_PYBIND11_VERSION}")

# ---- Try pre-installed (release / system / pip) --------------------------------------------
find_package(pybind11 ${COMPUTE_FLOW_PYBIND11_VERSION} CONFIG QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11: found pre-installed (${pybind11_VERSION})")
    return()
endif()

# ---- Fetch from source (FetchContent — the one exception) --------------------------------
message(STATUS "pybind11: fetching from source via FetchContent")

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY "${COMPUTE_FLOW_PYBIND11_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_PYBIND11_GIT_TAG}"
)
FetchContent_MakeAvailable(pybind11)
