#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)

set(COMPUTE_FLOW_PREBUILT_PROFILE "" CACHE STRING "Profile from cmake/deps/dependencies.json")
set(COMPUTE_FLOW_PREBUILT_VERSION local CACHE STRING "Dependency bundle version")
if(export_profile AND NOT export_profile STREQUAL COMPUTE_FLOW_PREBUILT_PROFILE)
    message(FATAL_ERROR "Use a fresh build directory for each dependency profile.")
endif()
set(export_profile "${COMPUTE_FLOW_PREBUILT_PROFILE}" CACHE INTERNAL "Configured dependency profile")
get_filename_component(compute_flow_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(manifest_path "${compute_flow_root}/cmake/deps/dependencies.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${manifest_path}")
file(READ "${manifest_path}" manifest)
string(JSON profile ERROR_VARIABLE profile_error GET "${manifest}" profiles "${export_profile}")
if(profile_error)
    message(FATAL_ERROR "Set COMPUTE_FLOW_PREBUILT_PROFILE to a profile in cmake/deps/dependencies.json.")
endif()
string(JSON system GET "${profile}" system)
string(JSON compiler GET "${profile}" compiler)
string(JSON compiler_version GET "${profile}" compiler_version)
string(JSON cpp_standard GET "${profile}" cpp_standard)

# Set the publishing baseline before CMake initializes the compiler/toolchain.
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CXX_STANDARD "${cpp_standard}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "" FORCE)
set(CMAKE_INSTALL_INCLUDEDIR include CACHE PATH "" FORCE)
foreach(language IN ITEMS C CXX ASM)
    set(CMAKE_${language}_FLAGS "-march=armv8-a" CACHE STRING "" FORCE)
    set(CMAKE_${language}_FLAGS_RELEASE "-O3 -DNDEBUG -g0" CACHE STRING "" FORCE)
endforeach()
if(system STREQUAL "darwin")
    set(CMAKE_OSX_DEPLOYMENT_TARGET 15.0 CACHE STRING "" FORCE)
    set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "" FORCE)
    string(APPEND CMAKE_CXX_FLAGS " -stdlib=libc++")
elseif(system STREQUAL "android")
    if(NOT CMAKE_TOOLCHAIN_FILE)
        message(FATAL_ERROR "Pass CMAKE_TOOLCHAIN_FILE from the matching Android™ NDK.")
    endif()
    set(ANDROID_ABI arm64-v8a CACHE STRING "" FORCE)
    set(ANDROID_PLATFORM android-24 CACHE STRING "" FORCE)
    set(ANDROID_STL c++_shared CACHE STRING "" FORCE)
elseif(system STREQUAL "linux")
    string(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=1")
    if(compiler STREQUAL "clang")
        string(APPEND CMAKE_CXX_FLAGS " -stdlib=libstdc++")
    endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

project(compute_flow_prebuilts LANGUAGES C CXX)
string(TOLOWER "${CMAKE_SYSTEM_NAME}" actual_system)
if(NOT actual_system STREQUAL system OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR "The target platform does not match ${export_profile}.")
endif()
set(compiler_id_clang Clang)
set(compiler_id_apple-clang AppleClang)
set(compiler_id_gcc GNU)
foreach(language IN ITEMS C CXX)
    string(REGEX MATCH "^[0-9]+" actual_version "${CMAKE_${language}_COMPILER_VERSION}")
    if(
        NOT CMAKE_${language}_COMPILER_ID STREQUAL compiler_id_${compiler}
        OR NOT actual_version STREQUAL compiler_version
    )
        message(
            FATAL_ERROR
            "${export_profile} requires ${compiler} ${compiler_version}; ${language} uses ${CMAKE_${language}_COMPILER_ID} ${CMAKE_${language}_COMPILER_VERSION}."
        )
    endif()
endforeach()

find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
include(FetchContent)
include("${compute_flow_root}/cmake/Utilities.cmake")
include("${compute_flow_root}/cmake/deps/Common.cmake")
list(
    APPEND COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS
    -DCMAKE_CONFIGURATION_TYPES:STRING=Release
    -DCMAKE_INSTALL_LIBDIR:STRING=lib
    -DCMAKE_INSTALL_INCLUDEDIR:STRING=include
    -DCMAKE_BUILD_TYPE:STRING=Release
    "-DCMAKE_CXX_STANDARD:STRING=${cpp_standard}"
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=ON
)
set(COMPUTE_FLOW_ENABLE_MNN_GPU "${ANDROID}")

# Select only the library/profile pairs already present in the manifest.
string(JSON num_artifacts LENGTH "${manifest}" artifacts)
math(EXPR last_artifact "${num_artifacts} - 1")
foreach(index RANGE ${last_artifact})
    string(JSON artifact_profile GET "${manifest}" artifacts ${index} profile)
    if(artifact_profile STREQUAL export_profile)
        string(JSON library GET "${manifest}" artifacts ${index} name)
        string(JSON expected_${library} GET "${manifest}" artifacts ${index} version)
        list(APPEND libraries "${library}")
    endif()
endforeach()
set(module_protobuf protobuf/Protobuf)
set(module_litert litert/LiteRT)
set(module_executorch executorch/ExecuTorch)
set(module_onnxruntime onnxruntime/OnnxRuntime)
set(module_mnn mnn/MNN)
set(module_sentencepiece sentencepiece/SentencePiece)
# Protobuf must be available before ONNX Runtime and SentencePiece are configured.
foreach(
    library
    IN
    ITEMS protobuf litert executorch onnxruntime mnn sentencepiece
)
    if(library IN_LIST libraries)
        string(TOUPPER "${library}" library_upper)
        set(COMPUTE_FLOW_BUILD_${library_upper}_FROM_SOURCE ON)
        if(library STREQUAL "litert")
            include("${compute_flow_root}/cmake/deps/flatbuffers/FlatBuffers.cmake")
        endif()
        include("${compute_flow_root}/cmake/deps/${module_${library}}.cmake")
        list(APPEND build_targets compute_flow_dep_${library})
        set(ep_${library} compute_flow_dep_${library})
    endif()
endforeach()
set(ep_onnxruntime onnxruntime_ep)
set(ep_sentencepiece sentencepiece_ep)
foreach(library IN LISTS libraries)
    string(TOUPPER "${library}" library_upper)
    if(NOT COMPUTE_FLOW_${library_upper}_GIT_TAG STREQUAL expected_${library})
        message(FATAL_ERROR "Align the ${library} source pin and manifest commit before exporting.")
    endif()
    if(library STREQUAL "litert")
        set(SOURCE_DIR "${compute_flow_litert_SOURCE_DIR}")
        set(BINARY_DIR "${CMAKE_BINARY_DIR}")
    else()
        ExternalProject_Get_Property(${ep_${library}} SOURCE_DIR BINARY_DIR)
    endif()
    list(
        APPEND package_args
        --package
        "${library}"
        "${COMPUTE_FLOW_DEPS_INSTALL_DIR}/${library}"
        "${SOURCE_DIR}"
        "${BINARY_DIR}"
    )
endforeach()

add_custom_target(
    export-prebuilts
    COMMAND
        "${Python3_EXECUTABLE}" "${compute_flow_root}/scripts/package_prebuilts.py" --manifest "${manifest_path}"
        --profile "${export_profile}" --version "${COMPUTE_FLOW_PREBUILT_VERSION}" --output
        "${CMAKE_BINARY_DIR}/exports" ${package_args}
    DEPENDS ${build_targets}
    USES_TERMINAL
    VERBATIM
)
