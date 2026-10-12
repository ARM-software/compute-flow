#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)

project(compute_flow_brick_kokoro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

if(NOT TARGET compute_flow::compute_flow)
    find_package(compute_flow REQUIRED)
endif()

if(NOT TARGET espeak-ng)
    find_package(espeakng CONFIG QUIET)
endif()

if(NOT TARGET espeak-ng)
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

if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json CONFIG QUIET)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/archive/55f93686c01528224f448c19128836e7df245f72.tar.gz
        URL_HASH SHA256=67f4cdd9ca930c9c1e130af4a437c7fc98fab77a2846fc2d2a14b4943831f8ef
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(Kokoro_core STATIC "Kokoro.cpp" "internal/zh_ps.cpp")
add_library(Kokoro::core ALIAS Kokoro_core)
set_target_properties(Kokoro_core PROPERTIES COMPUTE_FLOW_BRICK_CLASS "tts::Kokoro")

FetchContent_Declare(
    cppjieba
    GIT_REPOSITORY https://github.com/yanyiwu/cppjieba.git
    GIT_TAG
        9408c1d # we should use a stable tag
)

FetchContent_MakeAvailable(cppjieba)

# Set options for cpp-pinyin before fetching
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CPP_PINYIN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CPP_PINYIN_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    cpp-pinyin
    GIT_REPOSITORY https://github.com/wolfgitpr/cpp-pinyin.git
    GIT_TAG e4dd8310e3bb6b8ac8edc42f2cbd80be92586a8e
    PATCH_COMMAND
        git reset --hard HEAD && git clean -fdx && git apply ${CMAKE_CURRENT_LIST_DIR}/patches/cpp-pinyin.PATCH
)

FetchContent_MakeAvailable(cpp-pinyin)

target_include_directories(
    Kokoro_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/internal
        $<INSTALL_INTERFACE:include>
)

# eSpeak-NG's compat/wctype.h pulls in ucd/ucd.h from ucd-tools; expose that include path
if(DEFINED espeakng_SOURCE_DIR AND EXISTS "${espeakng_SOURCE_DIR}/src/ucd-tools/src/include")
    target_include_directories(Kokoro_core PUBLIC "${espeakng_SOURCE_DIR}/src/ucd-tools/src/include")
endif()

target_link_libraries(
    Kokoro_core
    PUBLIC compute_flow::compute_flow espeak-ng cppjieba cpp-pinyin nlohmann_json::nlohmann_json
)

if(COMPUTE_FLOW_PYBIND_ENABLED)
    pybind11_add_module(kokoro_py kokoro_pybind.cpp)
    target_link_libraries(kokoro_py PRIVATE Kokoro_core)
    compute_flow_install_python_extension(kokoro_py)
endif()
