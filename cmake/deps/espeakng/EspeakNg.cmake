#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/espeakng/EspeakNg.cmake — eSpeak-NG phonemization library
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/URL here) ----------------------------------------
set(COMPUTE_FLOW_ESPEAKNG_URL
    "https://github.com/espeak-ng/espeak-ng/archive/4870adfa25b1a32b4361592f1be8a40337c58d6c.tar.gz"
)
set(COMPUTE_FLOW_ESPEAKNG_URL_HASH "SHA256=cd83f84c4e495f281ac14e919aecf2834306ec1ea1b498de8ef6000f3a0f90de")

# ---- Try pre-installed (Conan / system) --------------------------------------------------
find_package(espeakng CONFIG QUIET)
if(espeakng_FOUND AND TARGET espeak-ng)
    message(STATUS "eSpeak-NG: found pre-installed")
    add_custom_target(compute_flow_dep_espeakng)
    return()
endif()

# ---- Build from source via FetchContent --------------------------------------------------
# eSpeak-NG is consumed directly by TTS bricks, so FetchContent is used to expose its CMake
# target during configure.
message(STATUS "eSpeak-NG: building from source via FetchContent")

set(ESPEAKNG_SKIP_DATA ON CACHE BOOL "" FORCE)
set(ESPEAKNG_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    espeakng
    URL "${COMPUTE_FLOW_ESPEAKNG_URL}"
    URL_HASH "${COMPUTE_FLOW_ESPEAKNG_URL_HASH}"
    PATCH_COMMAND
        ${CMAKE_COMMAND} -DESPEAKNG_SOURCE_DIR:PATH=<SOURCE_DIR> -DESPEAKNG_SKIP_DATA:BOOL=ON
        -DESPEAKNG_SKIP_TESTS:BOOL=ON -P ${CMAKE_CURRENT_LIST_DIR}/PatchEspeakNg.cmake
)
FetchContent_MakeAvailable(espeakng)

if(DEFINED espeakng_SOURCE_DIR AND EXISTS "${espeakng_SOURCE_DIR}/src/ucd-tools/src/include")
    target_include_directories(espeak-ng INTERFACE "${espeakng_SOURCE_DIR}/src/ucd-tools/src/include")
endif()

add_custom_target(compute_flow_dep_espeakng)
