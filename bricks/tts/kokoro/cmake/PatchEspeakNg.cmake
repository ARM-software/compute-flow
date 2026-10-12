#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED ESPEAKNG_SOURCE_DIR OR ESPEAKNG_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "ESPEAKNG_SOURCE_DIR must be set.")
endif()

set(top_cmakelists "${ESPEAKNG_SOURCE_DIR}/CMakeLists.txt")

if(EXISTS "${top_cmakelists}")
    file(READ "${top_cmakelists}" content)

    if(NOT content MATCHES "COMPUTE_FLOW_PATCH_ESPEAKNG")
        string(
            REPLACE "include(cmake/data.cmake)"
            "# COMPUTE_FLOW_PATCH_ESPEAKNG: allow skipping host-executed tools when cross-compiling\nif(ESPEAKNG_SKIP_DATA)\n  message(STATUS \"eSpeak-NG: skipping data generation (ESPEAKNG_SKIP_DATA=ON)\")\nelse()\n  include(cmake/data.cmake)\nendif()"
            content
            "${content}"
        )

        string(
            REPLACE "include(CTest)\nadd_subdirectory(tests)"
            "# COMPUTE_FLOW_PATCH_ESPEAKNG: tests often depend on generated data\nif(ESPEAKNG_SKIP_TESTS)\n  message(STATUS \"eSpeak-NG: skipping tests (ESPEAKNG_SKIP_TESTS=ON)\")\nelse()\n  include(CTest)\n  add_subdirectory(tests)\nendif()"
            content
            "${content}"
        )

        file(WRITE "${top_cmakelists}" "${content}")
        message(STATUS "eSpeak-NG: patched CMakeLists.txt to allow skipping data/tests")
    else()
        message(STATUS "eSpeak-NG: CMakeLists.txt already patched (skip data/tests)")
    endif()
endif()

set(wctype_header "${ESPEAKNG_SOURCE_DIR}/src/include/compat/wctype.h")

if(EXISTS "${wctype_header}")
    file(READ "${wctype_header}" wctype_content)

    string(REPLACE "#define tolower ucd_tolower" "//#define tolower ucd_tolower" wctype_content "${wctype_content}")
    string(REPLACE "#define toupper udc_toupper" "//#define toupper udc_toupper" wctype_content "${wctype_content}")

    file(WRITE "${wctype_header}" "${wctype_content}")
    message(STATUS "eSpeak-NG: patched compat/wctype.h (commented out tolower/toupper macros)")
endif()
