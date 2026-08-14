#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)
project(onnxruntime_shim LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ONNXRUNTIME_SOURCE_DIR "" CACHE PATH "ONNX Runtime source dir (contains include/)")
set(ONNXRUNTIME_BUILD_DIR "" CACHE PATH "ONNX Runtime build dir (contains libonnxruntime_*.a and _deps/)")

if(NOT ONNXRUNTIME_SOURCE_DIR OR NOT EXISTS "${ONNXRUNTIME_SOURCE_DIR}")
    message(FATAL_ERROR "ONNXRUNTIME_SOURCE_DIR missing/invalid: '${ONNXRUNTIME_SOURCE_DIR}'")
endif()

if(NOT ONNXRUNTIME_BUILD_DIR OR NOT EXISTS "${ONNXRUNTIME_BUILD_DIR}")
    message(FATAL_ERROR "ONNXRUNTIME_BUILD_DIR missing/invalid: '${ONNXRUNTIME_BUILD_DIR}'")
endif()

# -----------------------
# Install headers
# -----------------------
# ORT public headers live here:
#   <src>/include/onnxruntime/...
install(
    DIRECTORY "${ONNXRUNTIME_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

# Some builds also generate headers under <build>/include (harmless if absent).
if(EXISTS "${ONNXRUNTIME_BUILD_DIR}/include")
    install(
        DIRECTORY "${ONNXRUNTIME_BUILD_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.inc"
    )
endif()

# -----------------------
# Collect required libs
# -----------------------
include("${CMAKE_CURRENT_LIST_DIR}/../Artifacts.cmake")

set(ort_main "${ONNXRUNTIME_BUILD_DIR}/${COMPUTE_FLOW_ONNXRUNTIME_BUILD_MAIN_LIB_RELATIVE_PATH}")

# ORT component archives
set(ort_components "${COMPUTE_FLOW_ONNXRUNTIME_BUILD_COMPONENT_LIB_RELATIVE_PATHS}")
list(TRANSFORM ort_components PREPEND "${ONNXRUNTIME_BUILD_DIR}/")

# Third-party static deps
set(ort_third_party "${COMPUTE_FLOW_ONNXRUNTIME_BUILD_THIRD_PARTY_LIB_RELATIVE_PATHS}")
list(TRANSFORM ort_third_party PREPEND "${ONNXRUNTIME_BUILD_DIR}/")

# KleidiAI is only produced on some platforms/architectures even when onnxruntime_USE_KLEIDIAI=ON.
# Treat it as optional in the shim so host builds don't fail when the ORT build omits it.
set(ort_kleidiai_optional "")
foreach(lib IN LISTS ort_third_party)
    if(lib MATCHES ".*/libkleidiai\\.a$" AND NOT EXISTS "${lib}")
        set(ort_kleidiai_optional "${lib}")
    endif()
endforeach()
if(ort_kleidiai_optional)
    message(STATUS "ONNX Runtime KleidiAI archive not found (optional): ${ort_kleidiai_optional}")
    list(REMOVE_ITEM ort_third_party "${ort_kleidiai_optional}")
endif()

# Auto-discover Abseil libs produced inside the ORT build
file(GLOB_RECURSE ort_absl_libs "${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/**/libabsl_*.a")

# Match your existing exclusion
list(FILTER ort_absl_libs EXCLUDE REGEX ".*/libabsl_failure_signal_handler\\.a$")

list(SORT ort_absl_libs)
list(REMOVE_DUPLICATES ort_absl_libs)

# Validate "core" libs exist (fail early with a clear message)
set(ort_required "${ort_main}" ${ort_components} ${ort_third_party})
foreach(lib IN LISTS ort_required)
    if(NOT EXISTS "${lib}")
        message(FATAL_ERROR "Missing expected ONNX Runtime artifact: ${lib}")
    endif()
endforeach()

# Install everything into <prefix>/lib (flat)
install(FILES "${ort_main}" ${ort_components} ${ort_third_party} ${ort_absl_libs} DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# -----------------------
# Generate a helper file listing "extra" libs by installed name
# -----------------------
# We store basenames so the package is relocatable.
set(extra_names_non_absl "")
foreach(lib IN LISTS ort_components ort_third_party)
    get_filename_component(name "${lib}" NAME)
    list(APPEND extra_names_non_absl "${name}")
endforeach()
list(REMOVE_DUPLICATES extra_names_non_absl)

set(extra_names_absl "")
foreach(lib IN LISTS ort_absl_libs)
    get_filename_component(name "${lib}" NAME)
    list(APPEND extra_names_absl "${name}")
endforeach()
list(REMOVE_DUPLICATES extra_names_absl)

set(ORT_EXTRA_LIB_LINES "")
foreach(name IN LISTS extra_names_non_absl)
    string(APPEND ORT_EXTRA_LIB_LINES "  \"\${ORT_LIBDIR}/${name}\"\n")
endforeach()

if(extra_names_absl)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        string(APPEND ORT_EXTRA_LIB_LINES "  \"-Wl,--start-group\"\n")
    endif()
    foreach(name IN LISTS extra_names_absl)
        string(APPEND ORT_EXTRA_LIB_LINES "  \"\${ORT_LIBDIR}/${name}\"\n")
    endforeach()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        string(APPEND ORT_EXTRA_LIB_LINES "  \"-Wl,--end-group\"\n")
    endif()
endif()

configure_file("${CMAKE_CURRENT_LIST_DIR}/Extras.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/onnxruntimeExtras.cmake" @ONLY)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/onnxruntimeExtras.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnxruntime"
)

# -----------------------
# Generate the package config for find_package(onnxruntime CONFIG)
# -----------------------
set(PACKAGE_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(PACKAGE_LIB_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/onnxruntimeConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnxruntime"
    PATH_VARS PACKAGE_INCLUDE_INSTALL_DIR PACKAGE_LIB_INSTALL_DIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/onnxruntimeConfigVersion.cmake"
    VERSION "0.0.0"
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/onnxruntimeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/onnxruntimeConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnxruntime"
)
