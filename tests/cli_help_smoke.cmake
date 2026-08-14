#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED CLI_PATH)
    message(FATAL_ERROR "CLI_PATH is required")
endif()

execute_process(
    COMMAND "${CLI_PATH}" --help
    RESULT_VARIABLE cli_result
    OUTPUT_VARIABLE cli_stdout
    ERROR_VARIABLE cli_stderr
    TIMEOUT 10
)

if(NOT cli_result STREQUAL "0" AND NOT cli_result STREQUAL "1")
    message(FATAL_ERROR "${CLI_PATH} --help returned unexpected status '${cli_result}'")
endif()

string(CONCAT cli_output "${cli_stdout}" "${cli_stderr}")
if(NOT cli_output MATCHES "Usage:")
    message(FATAL_ERROR "${CLI_PATH} --help did not print usage information:\n${cli_output}")
endif()
