#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Download the pinned installed prefixes before the per-dependency find_package calls.
include_guard(GLOBAL)

function(compute_flow_fetch_release_dependencies output result)
    set(${output} "" PARENT_SCOPE)
    set(${result} 0 PARENT_SCOPE)
    if(NOT ARGN)
        return()
    endif()

    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/host.cmake" OPTIONAL)
    if(DEFINED ENV{COMPUTE_FLOW_DEPENDENCY_BASE_URL})
        set(COMPUTE_FLOW_DEPENDENCY_BASE_URL_DEFAULT "$ENV{COMPUTE_FLOW_DEPENDENCY_BASE_URL}")
    endif()
    set(COMPUTE_FLOW_DEPENDENCY_BASE_URL
        "${COMPUTE_FLOW_DEPENDENCY_BASE_URL_DEFAULT}"
        CACHE STRING
        "Semicolon-separated release URLs or local directories, tried in order"
    )
    set(COMPUTE_FLOW_DEPENDENCY_PROFILE "" CACHE STRING "Override the default dependency target profile")
    set(${result} 1 PARENT_SCOPE)

    string(TOLOWER "${CMAKE_SYSTEM_NAME}" system)
    set(architecture "${CMAKE_SYSTEM_PROCESSOR}")
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_ARCHITECTURES)
        set(architecture "${CMAKE_OSX_ARCHITECTURES}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT ANDROID_ABI STREQUAL "arm64-v8a")
        set(architecture "${ANDROID_ABI}")
    endif()
    string(TOLOWER "${architecture}" architecture)
    if(NOT architecture MATCHES "^(arm64|aarch64|armv8)$")
        return()
    endif()
    set(profile "${COMPUTE_FLOW_DEPENDENCY_PROFILE}")
    if(NOT profile)
        set(default_linux linux_aarch64)
        set(default_darwin macos_aarch64)
        set(default_android android_aarch64)
        set(profile "${default_${system}}")
    endif()
    set(manifest_path "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dependencies.json")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${manifest_path}")
    file(READ "${manifest_path}" manifest)
    string(JSON profile_system ERROR_VARIABLE profile_error GET "${manifest}" profiles "${profile}" system)
    if(profile_error OR NOT profile_system STREQUAL system)
        return()
    endif()

    set(cache "${COMPUTE_FLOW_DEPS_PREFIX_DIR}/prebuilt")
    file(MAKE_DIRECTORY "${cache}")
    file(LOCK "${cache}/.lock" GUARD FUNCTION)
    string(JSON num_artifacts LENGTH "${manifest}" artifacts)
    math(EXPR last_artifact "${num_artifacts} - 1")
    set(prefixes "")
    foreach(package IN LISTS ARGN)
        set(matches "")
        foreach(index RANGE ${last_artifact})
            string(JSON name GET "${manifest}" artifacts ${index} name)
            string(JSON artifact_profile GET "${manifest}" artifacts ${index} profile)
            if(name STREQUAL package AND artifact_profile STREQUAL profile)
                list(APPEND matches ${index})
            endif()
        endforeach()
        list(LENGTH matches num_matches)
        if(NOT num_matches EQUAL 1)
            return()
        endif()
        string(JSON filename GET "${manifest}" artifacts ${matches} filename)
        string(JSON sha256 GET "${manifest}" artifacts ${matches} sha256)
        string(LENGTH "${sha256}" hash_length)
        if(
            NOT filename MATCHES "^[A-Za-z0-9_.-]+\\.tar\\.gz$"
            OR NOT sha256 MATCHES "^[0-9a-f]+$"
            OR NOT hash_length EQUAL 64
        )
            return()
        endif()
        set(prefix "${cache}/${sha256}")
        set(complete "")
        if(EXISTS "${prefix}/.complete")
            file(READ "${prefix}/.complete" complete)
            string(STRIP "${complete}" complete)
        endif()
        if(NOT complete STREQUAL sha256)
            if(NOT COMPUTE_FLOW_DEPENDENCY_BASE_URL)
                return()
            endif()
            set(fetch_result 1)
            foreach(base_url IN LISTS COMPUTE_FLOW_DEPENDENCY_BASE_URL)
                if(NOT base_url MATCHES "^[A-Za-z]+://")
                    get_filename_component(base_url "${base_url}" ABSOLUTE)
                    set(base_url "file://${base_url}")
                endif()
                string(REGEX REPLACE "/+$" "" base_url "${base_url}")
                message(STATUS "Fetching ${filename} from ${base_url}")
                execute_process(
                    COMMAND
                        "${CMAKE_COMMAND}" "-Darchive_url=${base_url}/${filename}" "-Dsha256=${sha256}"
                        "-Dprefix=${prefix}" -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/FetchRelease.cmake"
                    RESULT_VARIABLE fetch_result
                    OUTPUT_VARIABLE fetch_output
                    ERROR_VARIABLE fetch_output
                )
                if(fetch_result EQUAL 0)
                    break()
                endif()
                message(VERBOSE "${fetch_output}")
                file(REMOVE_RECURSE "${prefix}.tmp")
                file(REMOVE "${prefix}.tar.gz")
            endforeach()
            if(NOT fetch_result EQUAL 0)
                return()
            endif()
        endif()
        list(APPEND prefixes "${prefix}")
    endforeach()
    # Let the installed configurations supply paths and target properties, including after upgrades.
    foreach(
        package
        IN
        ITEMS Protobuf litert executorch onnxruntime MNN sentencepiece
    )
        unset(${package}_DIR CACHE)
    endforeach()
    set(${output} "${prefixes}" PARENT_SCOPE)
    set(${result} 0 PARENT_SCOPE)
endfunction()

set(release_dependencies "")
if(
    (COMPUTE_FLOW_ENABLE_ONNXRUNTIME OR COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    AND NOT COMPUTE_FLOW_BUILD_PROTOBUF_FROM_SOURCE
)
    list(APPEND release_dependencies protobuf)
endif()
foreach(
    dependency
    IN
    ITEMS litert executorch onnxruntime mnn sentencepiece
)
    string(TOUPPER "${dependency}" option_name)
    if(COMPUTE_FLOW_ENABLE_${option_name} AND NOT COMPUTE_FLOW_BUILD_${option_name}_FROM_SOURCE)
        list(APPEND release_dependencies "${dependency}")
    endif()
endforeach()

compute_flow_fetch_release_dependencies(release_prefixes release_result ${release_dependencies})
if(release_result)
    message(WARNING "Prebuilt release packages are unavailable; building dependencies from source")
    compute_flow_enable_source_dependencies()
else()
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
    endif()
    list(PREPEND CMAKE_PREFIX_PATH ${release_prefixes})
endif()
