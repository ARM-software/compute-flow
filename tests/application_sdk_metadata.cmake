#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

foreach(required_variable IN ITEMS BUILD_DIR INSTALLED_INFO_JSON PACKAGED_INFO_JSON PACKAGED_LIBRARY)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required to test Compute Flow SDK metadata packaging.")
    endif()
endforeach()

set(build_command
    "${CMAKE_COMMAND}"
    --build
    "${BUILD_DIR}"
    --target
    compute_flow_application_sdk
)
if(BUILD_CONFIG)
    list(APPEND build_command --config "${BUILD_CONFIG}")
endif()
execute_process(
    COMMAND ${build_command}
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Compute Flow SDK build failed:\n${build_output}${build_error}")
endif()

if(NOT EXISTS "${PACKAGED_INFO_JSON}")
    message(FATAL_ERROR "Compute Flow SDK did not package installed brick metadata.")
endif()
file(READ "${INSTALLED_INFO_JSON}" installed_metadata)
file(READ "${PACKAGED_INFO_JSON}" packaged_metadata)
if(NOT installed_metadata STREQUAL packaged_metadata)
    message(FATAL_ERROR "Compute Flow SDK changed the installed brick metadata.")
endif()
if(NOT EXISTS "${PACKAGED_LIBRARY}")
    message(FATAL_ERROR "Compute Flow SDK did not package the installed brick type library: ${PACKAGED_LIBRARY}")
endif()
file(SIZE "${PACKAGED_LIBRARY}" packaged_library_size)
if(packaged_library_size EQUAL 0)
    message(FATAL_ERROR "Compute Flow SDK packaged an empty brick type library: ${PACKAGED_LIBRARY}")
endif()
