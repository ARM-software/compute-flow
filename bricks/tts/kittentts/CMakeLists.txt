#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)

project(compute_flow_brick_kittentts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT TARGET compute_flow::compute_flow)
    find_package(compute_flow REQUIRED)
endif()

if(NOT TARGET espeak-ng)
    find_package(espeakng CONFIG QUIET)
endif()

if(NOT TARGET espeak-ng)
    include(FetchContent)
    set(ESPEAKNG_SKIP_DATA ON CACHE BOOL "" FORCE)
    set(ESPEAKNG_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        espeakng
        URL https://github.com/espeak-ng/espeak-ng/archive/4870adfa25b1a32b4361592f1be8a40337c58d6c.tar.gz
        URL_HASH SHA256=cd83f84c4e495f281ac14e919aecf2834306ec1ea1b498de8ef6000f3a0f90de
        PATCH_COMMAND
            ${CMAKE_COMMAND} -DESPEAKNG_SOURCE_DIR:PATH=<SOURCE_DIR> -DESPEAKNG_SKIP_DATA:BOOL=ON
            -DESPEAKNG_SKIP_TESTS:BOOL=ON -P ${CMAKE_CURRENT_LIST_DIR}/cmake/PatchEspeakNg.cmake
    )
    FetchContent_MakeAvailable(espeakng)
    if(DEFINED espeakng_SOURCE_DIR AND EXISTS "${espeakng_SOURCE_DIR}/src/ucd-tools/src/include")
        target_include_directories(espeak-ng INTERFACE "${espeakng_SOURCE_DIR}/src/ucd-tools/src/include")
    endif()
endif()

if(NOT TARGET libnpy::libnpy)
    find_package(libnpy CONFIG QUIET)
endif()

if(NOT TARGET libnpy::libnpy)
    include(FetchContent)
    FetchContent_Declare(
        libnpy
        URL https://github.com/llohse/libnpy/archive/refs/tags/v1.0.1.zip
        URL_HASH SHA256=3c9864c1bf7af08c88f2056478a2c6d2539674673bab9f9b4a0042a84a857f19
    )
    FetchContent_GetProperties(libnpy)
    if(NOT libnpy_POPULATED)
        FetchContent_Populate(libnpy)
    endif()
    add_library(libnpy INTERFACE)
    target_include_directories(libnpy INTERFACE "${libnpy_SOURCE_DIR}/include")
    add_library(libnpy::libnpy ALIAS libnpy)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json CONFIG QUIET)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/archive/55f93686c01528224f448c19128836e7df245f72.tar.gz
        URL_HASH SHA256=67f4cdd9ca930c9c1e130af4a437c7fc98fab77a2846fc2d2a14b4943831f8ef
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(KittenTTS_core STATIC "KittenTTS.cpp")
add_library(KittenTTS::core ALIAS KittenTTS_core)
set_target_properties(KittenTTS_core PROPERTIES COMPUTE_FLOW_BRICK_CLASS "tts::KittenTTS")

target_include_directories(
    KittenTTS_core
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include>
)

# eSpeak-NG's compat/wctype.h pulls in ucd/ucd.h from ucd-tools; expose that include path
if(DEFINED espeakng_SOURCE_DIR AND EXISTS "${espeakng_SOURCE_DIR}/src/ucd-tools/src/include")
    target_include_directories(KittenTTS_core PUBLIC "${espeakng_SOURCE_DIR}/src/ucd-tools/src/include")
endif()

target_link_libraries(
    KittenTTS_core
    PUBLIC compute_flow::compute_flow espeak-ng nlohmann_json::nlohmann_json libnpy::libnpy
)

if(COMPUTE_FLOW_PYBIND_ENABLED)
    pybind11_add_module(kittentts_py kittentts_pybind.cpp)
    target_link_libraries(kittentts_py PRIVATE KittenTTS_core)
    compute_flow_install_python_extension(kittentts_py)
endif()
