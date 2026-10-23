#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)

# Run separately so download/extraction errors can trigger the source-build fallback.
# Bound each mirror attempt so a stalled download cannot block the next source.
file(
    DOWNLOAD "${archive_url}"
    "${prefix}.tar.gz"
    EXPECTED_HASH "SHA256=${sha256}"
    TLS_VERIFY ON
    INACTIVITY_TIMEOUT 30
    TIMEOUT 600
)
# CMake versions before 4.3 do not reject absolute or parent-traversal entry paths.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar tf "${prefix}.tar.gz"
    OUTPUT_VARIABLE entries
    COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" entries "${entries}")
foreach(entry IN LISTS entries)
    string(REPLACE "\\" "/" entry "${entry}")
    if(entry MATCHES "^(/|[A-Za-z]:)|(^|/)\\.\\.(/|$)")
        message(FATAL_ERROR "Release archive contains an unsafe entry path: ${entry}")
    endif()
endforeach()
file(REMOVE_RECURSE "${prefix}.tmp")
file(ARCHIVE_EXTRACT INPUT "${prefix}.tar.gz" DESTINATION "${prefix}.tmp")
file(GLOB_RECURSE configs "${prefix}.tmp/lib/cmake/*Config.cmake" "${prefix}.tmp/lib/cmake/*-config.cmake")
if(NOT configs)
    message(FATAL_ERROR "Release archive does not contain an installed CMake package: ${archive_url}")
endif()
file(WRITE "${prefix}.tmp/.complete" "${sha256}\n")
file(REMOVE_RECURSE "${prefix}")
file(RENAME "${prefix}.tmp" "${prefix}")
file(REMOVE "${prefix}.tar.gz")
