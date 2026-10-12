#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/llama_cpp/LlamaCpp.cmake — llama.cpp LM inference library
# =========================================================================================
#
# Creates IMPORTED STATIC targets:
#   llama     — main llama library
#   ggml-base — GGML base library
#   ggml      — GGML library
#   ggml-cpu  — GGML CPU backend
#
# Lightweight dep: always builds from EP if not found by find_package.
#
# Resolution: find_package → ExternalProject (always, no from-source gating needed)
# =========================================================================================

include_guard(GLOBAL)

# ---- Android™ NDK version check (require r27+) -------------------------------------------
if(ANDROID)
    set(_llama_ndk_revision "")

    if(DEFINED ANDROID_NDK_REVISION)
        set(_llama_ndk_revision "${ANDROID_NDK_REVISION}")
    elseif(DEFINED ANDROID_NDK)
        set(_llama_ndk_props "${ANDROID_NDK}/source.properties")
        if(EXISTS "${_llama_ndk_props}")
            file(READ "${_llama_ndk_props}" _llama_ndk_props_content)
            string(
                REGEX MATCH "Pkg\\.Revision[ \t]*=[ \t]*[0-9]+(\\.[0-9]+)*"
                _llama_ndk_match
                "${_llama_ndk_props_content}"
            )
            if(_llama_ndk_match)
                string(REGEX REPLACE ".*Pkg\\.Revision[ \t]*=[ \t]*" "" _llama_ndk_revision "${_llama_ndk_match}")
            endif()
        endif()
    endif()

    if(NOT _llama_ndk_revision)
        message(FATAL_ERROR "Android™ NDK revision not detected. Require r27 or newer.")
    endif()

    string(REGEX MATCH "^[0-9]+" _llama_ndk_major "${_llama_ndk_revision}")
    if(NOT _llama_ndk_major)
        message(FATAL_ERROR "Android™ NDK revision parse failed ('${_llama_ndk_revision}'). Require r27 or newer.")
    endif()

    if(_llama_ndk_major LESS 27)
        message(
            FATAL_ERROR
            "Android™ NDK r${_llama_ndk_major} detected (${_llama_ndk_revision}). Require r27 or newer."
        )
    endif()

    if(DEFINED ANDROID_PLATFORM)
        string(REGEX MATCH "[0-9]+" _llama_android_api "${ANDROID_PLATFORM}")
        if(NOT _llama_android_api)
            message(FATAL_ERROR "Android™ platform parse failed ('${ANDROID_PLATFORM}'). Require API 28 or newer.")
        endif()
        if(_llama_android_api LESS 28)
            message(FATAL_ERROR "Android™ platform ${ANDROID_PLATFORM} detected. Require API 28 or newer.")
        endif()
    elseif(DEFINED ANDROID_NATIVE_API_LEVEL)
        if(ANDROID_NATIVE_API_LEVEL LESS 28)
            message(FATAL_ERROR "Android™ API ${ANDROID_NATIVE_API_LEVEL} detected. Require API 28 or newer.")
        endif()
    else()
        message(FATAL_ERROR "Android™ platform not detected. Require API 28 or newer.")
    endif()
endif()

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_LLAMACPP_GIT_REPO "https://github.com/ggerganov/llama.cpp.git")
set(COMPUTE_FLOW_LLAMACPP_GIT_TAG "76f46ad29d61fd8c1401e8221842934bf62a6064")

# ---- Optional build features -------------------------------------------------------------
set(COMPUTE_FLOW_LLAMACPP_OPENMP OFF)
if(ANDROID)
    set(COMPUTE_FLOW_LLAMACPP_OPENMP ON)
endif()

set(_llama_openmp_cmake_args "")
if(COMPUTE_FLOW_LLAMACPP_OPENMP)
    if(NOT ANDROID_NDK)
        message(FATAL_ERROR "Android™ NDK not detected; cannot locate libomp.a for OpenMP.")
    endif()

    file(GLOB _llama_ndk_prebuilt_dirs "${ANDROID_NDK}/toolchains/llvm/prebuilt/*")
    list(LENGTH _llama_ndk_prebuilt_dirs _llama_ndk_prebuilt_count)
    if(_llama_ndk_prebuilt_count EQUAL 0)
        message(FATAL_ERROR "No NDK prebuilt toolchains found under ${ANDROID_NDK}/toolchains/llvm/prebuilt.")
    endif()
    list(GET _llama_ndk_prebuilt_dirs 0 _llama_ndk_prebuilt_dir)

    file(GLOB _llama_clang_lib_dirs "${_llama_ndk_prebuilt_dir}/lib/clang/*/lib/linux")
    list(LENGTH _llama_clang_lib_dirs _llama_clang_lib_count)
    if(_llama_clang_lib_count EQUAL 0)
        message(FATAL_ERROR "No clang lib directories found under ${_llama_ndk_prebuilt_dir}/lib/clang.")
    endif()
    list(SORT _llama_clang_lib_dirs)
    list(GET _llama_clang_lib_dirs -1 _llama_clang_lib_dir)

    set(_llama_openmp_arch "")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(_llama_openmp_arch "aarch64")
    elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(_llama_openmp_arch "arm")
    elseif(ANDROID_ABI STREQUAL "x86")
        set(_llama_openmp_arch "i386")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(_llama_openmp_arch "x86_64")
    else()
        message(FATAL_ERROR "Unsupported ANDROID_ABI for OpenMP: ${ANDROID_ABI}.")
    endif()

    set(COMPUTE_FLOW_LLAMACPP_OPENMP_LIB
        "${_llama_clang_lib_dir}/${_llama_openmp_arch}/libomp.a"
        CACHE FILEPATH
        "Path to static libomp.a used for Android™ OpenMP"
    )

    if(NOT COMPUTE_FLOW_LLAMACPP_OPENMP_LIB OR NOT EXISTS "${COMPUTE_FLOW_LLAMACPP_OPENMP_LIB}")
        message(FATAL_ERROR "libomp.a not found. Set COMPUTE_FLOW_LLAMACPP_OPENMP_LIB to the static OpenMP runtime.")
    endif()

    list(
        APPEND _llama_openmp_cmake_args
        -DGGML_OPENMP:BOOL=ON
        -DOpenMP_omp_LIBRARY:FILEPATH=${COMPUTE_FLOW_LLAMACPP_OPENMP_LIB}
        -DOpenMP_C_FLAGS:STRING=-fopenmp
        -DOpenMP_CXX_FLAGS:STRING=-fopenmp
        -DOpenMP_C_LIB_NAMES:STRING=omp
        -DOpenMP_CXX_LIB_NAMES:STRING=omp
    )
endif()

# ---- Try pre-installed -------------------------------------------------------------------
find_package(llama CONFIG QUIET)

if(llama_FOUND AND TARGET llama)
    message(STATUS "llama.cpp: found pre-installed")
    add_custom_target(compute_flow_dep_llama_cpp)

    # Monolithic archive declaration
    compute_flow_extract_target_archives(COMPUTE_FLOW_LLAMACPP_MONOLITHIC_ARCHIVES
        llama ggml ggml-base ggml-cpu
    )

    return()
endif()

# ---- Build from source via ExternalProject -----------------------------------------------
message(STATUS "llama.cpp: building from source via ExternalProject")

compute_flow_dep_prefix_dir(llamacpp_prefix_dir "llama-cpp")
compute_flow_dep_install_dir(llamacpp_install_dir "llama-cpp")

if(CMAKE_CROSSCOMPILING)
    set(_llama_arch_flags "-march=armv8.2-a+dotprod+fp16")
elseif(APPLE)
    # llama.cpp adds its own native CPU flags on Apple platforms. Keep this baseline to avoid
    # conflicting with its feature detection while preserving the existing macOS® performance.
    set(_llama_arch_flags "-march=armv8.2-a+dotprod+fp16")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|arm|ARM)")
    set(_llama_arch_flags "-mcpu=native")
else()
    set(_llama_arch_flags "-march=native")
endif()

message(STATUS "llama.cpp: compiler architecture flags: ${_llama_arch_flags}")

ExternalProject_Add(
    compute_flow_dep_llama_cpp
    GIT_REPOSITORY "${COMPUTE_FLOW_LLAMACPP_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_LLAMACPP_GIT_TAG}"
    PREFIX "${llamacpp_prefix_dir}"
    INSTALL_DIR "${llamacpp_install_dir}"
    GIT_PROGRESS TRUE
    UPDATE_DISCONNECTED ON
    PATCH_COMMAND
        ${CMAKE_COMMAND} -DLLAMACPP_SOURCE_DIR:PATH=<SOURCE_DIR>
        -DLLAMACPP_PATCH_FILE:PATH=${CMAKE_CURRENT_LIST_DIR}/0001-Add-support-for-int2-per-channel-quantization.patch -P
        ${CMAKE_CURRENT_LIST_DIR}/PatchLlamaCpp.cmake
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_C_FLAGS:STRING=${_llama_arch_flags} -DCMAKE_CXX_FLAGS:STRING=${_llama_arch_flags}
        -DLLAMA_BUILD_COMMON:BOOL=OFF -DLLAMA_BUILD_TESTS:BOOL=OFF -DLLAMA_BUILD_EXAMPLES:BOOL=OFF
        -DLLAMA_BUILD_SERVER:BOOL=OFF -DLLAMA_BUILD_APP:BOOL=OFF -DLLAMA_CURL:BOOL=OFF -DGGML_CPU_KLEIDIAI:BOOL=ON
        -DGGML_METAL:BOOL=OFF -DGGML_BLAS:BOOL=OFF -DGGML_OPENMP:BOOL=OFF ${_llama_openmp_cmake_args}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}/libllama.a
        <INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}/libggml-base.a
        <INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}/libggml.a
        <INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}/libggml-cpu.a
    # KleidiAI is marked EXCLUDE_FROM_ALL so we must explicitly build the target
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target kleidiai
    COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)

set(COMPUTE_FLOW_LLAMACPP_INSTALL_DIR "${llamacpp_install_dir}")

# ---- Placeholder IMPORTED targets -------------------------------------------------------

file(MAKE_DIRECTORY "${llamacpp_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}")

if(NOT TARGET ggml-base)
    add_library(ggml-base STATIC IMPORTED GLOBAL)
    set_target_properties(
        ggml-base
        PROPERTIES
            IMPORTED_LOCATION "${llamacpp_install_dir}/${CMAKE_INSTALL_LIBDIR}/libggml-base.a"
            INTERFACE_INCLUDE_DIRECTORIES "${llamacpp_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    add_dependencies(ggml-base compute_flow_dep_llama_cpp)
endif()

if(NOT TARGET ggml-cpu)
    add_library(ggml-cpu STATIC IMPORTED GLOBAL)
    set_target_properties(
        ggml-cpu
        PROPERTIES
            IMPORTED_LOCATION "${llamacpp_install_dir}/${CMAKE_INSTALL_LIBDIR}/libggml-cpu.a"
            INTERFACE_INCLUDE_DIRECTORIES "${llamacpp_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    add_dependencies(ggml-cpu compute_flow_dep_llama_cpp)
endif()

if(NOT TARGET ggml)
    add_library(ggml STATIC IMPORTED GLOBAL)
    set_target_properties(
        ggml
        PROPERTIES
            IMPORTED_LOCATION "${llamacpp_install_dir}/${CMAKE_INSTALL_LIBDIR}/libggml.a"
            INTERFACE_INCLUDE_DIRECTORIES "${llamacpp_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    target_link_libraries(ggml INTERFACE ggml-base ggml-cpu)
    add_dependencies(ggml compute_flow_dep_llama_cpp)
endif()

if(NOT TARGET llama)
    add_library(llama STATIC IMPORTED GLOBAL)
    set_target_properties(
        llama
        PROPERTIES
            IMPORTED_LOCATION "${llamacpp_install_dir}/${CMAKE_INSTALL_LIBDIR}/libllama.a"
            INTERFACE_INCLUDE_DIRECTORIES "${llamacpp_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    target_link_libraries(llama INTERFACE ggml)
    add_dependencies(llama compute_flow_dep_llama_cpp)
endif()

message(STATUS "llama.cpp: using target 'llama'")

# ---- Monolithic archive declaration -----------------------------------------------------
set(COMPUTE_FLOW_LLAMACPP_MONOLITHIC_ARCHIVES
    "${llamacpp_install_dir}/${CMAKE_INSTALL_LIBDIR}/libllama.a"
    "${llamacpp_install_dir}/${CMAKE_INSTALL_LIBDIR}/libggml.a"
    "${llamacpp_install_dir}/${CMAKE_INSTALL_LIBDIR}/libggml-base.a"
    "${llamacpp_install_dir}/${CMAKE_INSTALL_LIBDIR}/libggml-cpu.a"
)
