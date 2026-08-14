#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED LLAMACPP_SOURCE_DIR)
    message(FATAL_ERROR "LLAMACPP_SOURCE_DIR is not set.")
endif()

if(DEFINED LLAMACPP_PATCH_FILE AND NOT LLAMACPP_PATCH_FILE STREQUAL "")
    if(NOT EXISTS "${LLAMACPP_PATCH_FILE}")
        message(FATAL_ERROR "LLAMACPP_PATCH_FILE not found at ${LLAMACPP_PATCH_FILE}.")
    endif()

    execute_process(
        COMMAND git -C "${LLAMACPP_SOURCE_DIR}" apply --check "${LLAMACPP_PATCH_FILE}"
        RESULT_VARIABLE llama_patch_check_result
        OUTPUT_VARIABLE llama_patch_check_output
        ERROR_VARIABLE llama_patch_check_error
    )

    if(llama_patch_check_result EQUAL 0)
        execute_process(
            COMMAND git -C "${LLAMACPP_SOURCE_DIR}" apply "${LLAMACPP_PATCH_FILE}"
            RESULT_VARIABLE llama_patch_apply_result
            OUTPUT_VARIABLE llama_patch_apply_output
            ERROR_VARIABLE llama_patch_apply_error
        )

        if(NOT llama_patch_apply_result EQUAL 0)
            message(
                FATAL_ERROR
                "Failed to apply llama.cpp patch at ${LLAMACPP_PATCH_FILE}:\n${llama_patch_apply_error}"
            )
        endif()

        message(STATUS "Applied llama.cpp patch ${LLAMACPP_PATCH_FILE}.")
    else()
        execute_process(
            COMMAND git -C "${LLAMACPP_SOURCE_DIR}" apply --reverse --check "${LLAMACPP_PATCH_FILE}"
            RESULT_VARIABLE llama_patch_reverse_result
        )

        if(llama_patch_reverse_result EQUAL 0)
            message(STATUS "llama.cpp patch already applied: ${LLAMACPP_PATCH_FILE}.")
        else()
            message(FATAL_ERROR "llama.cpp patch check failed for ${LLAMACPP_PATCH_FILE}:\n${llama_patch_check_error}")
        endif()
    endif()
else()
    message(STATUS "No LLAMACPP_PATCH_FILE specified; skipping llama.cpp patch.")
endif()
