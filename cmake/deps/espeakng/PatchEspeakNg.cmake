#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED ESPEAKNG_SOURCE_DIR)
    message(FATAL_ERROR "ESPEAKNG_SOURCE_DIR is required")
endif()

set(espeakng_cmake "${ESPEAKNG_SOURCE_DIR}/CMakeLists.txt")
if(ESPEAKNG_SKIP_DATA AND EXISTS "${espeakng_cmake}")
    file(READ "${espeakng_cmake}" espeakng_cmake_text)
    string(
        REPLACE "add_subdirectory(data)"
        "if(NOT ESPEAKNG_SKIP_DATA)\n  add_subdirectory(data)\nendif()"
        espeakng_cmake_text
        "${espeakng_cmake_text}"
    )
    string(
        REPLACE "include(cmake/data.cmake)"
        "if(NOT ESPEAKNG_SKIP_DATA)\n  include(cmake/data.cmake)\nendif()"
        espeakng_cmake_text
        "${espeakng_cmake_text}"
    )
    if(ESPEAKNG_SKIP_TESTS)
        string(
            REPLACE "enable_testing()"
            "if(NOT ESPEAKNG_SKIP_TESTS)\n  enable_testing()\nendif()"
            espeakng_cmake_text
            "${espeakng_cmake_text}"
        )
        string(
            REPLACE "add_subdirectory(tests)"
            "if(NOT ESPEAKNG_SKIP_TESTS)\n  add_subdirectory(tests)\nendif()"
            espeakng_cmake_text
            "${espeakng_cmake_text}"
        )
    endif()
    file(WRITE "${espeakng_cmake}" "${espeakng_cmake_text}")
endif()

set(espeakng_wctype "${ESPEAKNG_SOURCE_DIR}/src/include/compat/wctype.h")
if(EXISTS "${espeakng_wctype}")
    file(READ "${espeakng_wctype}" espeakng_wctype_text)
    string(
        REPLACE "#define tolower ucd_tolower"
        "/* #define tolower ucd_tolower */"
        espeakng_wctype_text
        "${espeakng_wctype_text}"
    )
    string(
        REPLACE "#define toupper udc_toupper"
        "/* #define toupper udc_toupper */"
        espeakng_wctype_text
        "${espeakng_wctype_text}"
    )
    string(
        REPLACE "#define towlower(X) towlower((wint_t) X)"
        "/* #define towlower(X) towlower((wint_t) X) */"
        espeakng_wctype_text
        "${espeakng_wctype_text}"
    )
    string(
        REPLACE "#define towupper(X) towupper((wint_t) X)"
        "/* #define towupper(X) towupper((wint_t) X) */"
        espeakng_wctype_text
        "${espeakng_wctype_text}"
    )
    if(espeakng_wctype_text MATCHES "(^|\n)[ \t]*#[ \t]*define[ \t]+(tolower|toupper)[ \t]+")
        message(FATAL_ERROR "Failed to patch eSpeak-NG compat/wctype.h toupper/tolower macros")
    endif()
    file(WRITE "${espeakng_wctype}" "${espeakng_wctype_text}")
endif()
