#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

if(EXISTS "${CMAKE_SOURCE_DIR}/bricks")
    add_test(NAME brick_metadata COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/bricks/test_metadata.py")
endif()

if(COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES)
    add_executable(brick_contract_tests "${CMAKE_CURRENT_SOURCE_DIR}/bricks/brick_contract_tests.cpp")
    compute_flow_link_installed_bricks(brick_contract_tests)

    add_test(NAME brick_contract COMMAND brick_contract_tests "${COMPUTE_FLOW_BRICK_CACHE_PATH}")
endif()

set(COMPUTE_FLOW_CLI_SMOKE_TARGETS
    audio_detection_cli
    audio_enhancement_cli
    audiogen_cli
    image_edit_cli
    lm_text_cli
    monocular_depth_estimation_cli
    musicgen_cli
    neural_codec_cli
    object_detection_cli
    stt_cli
    tts_cli
    vad_cli
)

foreach(cli_target IN LISTS COMPUTE_FLOW_CLI_SMOKE_TARGETS)
    add_test(
        NAME "${cli_target}_smoke"
        COMMAND
            "${CMAKE_COMMAND}" "-DCLI_PATH=$<TARGET_FILE:${cli_target}>" -P
            "${CMAKE_CURRENT_SOURCE_DIR}/cli_help_smoke.cmake"
    )
    set_tests_properties("${cli_target}_smoke" PROPERTIES LABELS "cli;cli_help;smoke" TIMEOUT 15)
endforeach()

if(COMPUTE_FLOW_ENABLE_MODEL_TESTS)
    if(NOT COMPUTE_FLOW_TEST_MODEL_HOME)
        if(DEFINED ENV{COMPUTE_FLOW_HOME} AND NOT "$ENV{COMPUTE_FLOW_HOME}" STREQUAL "")
            set(COMPUTE_FLOW_TEST_MODEL_HOME "$ENV{COMPUTE_FLOW_HOME}")
        else()
            set(COMPUTE_FLOW_TEST_MODEL_HOME "$ENV{HOME}/.compute-flow")
        endif()
    endif()

    if(NOT COMPUTE_FLOW_TEST_BRICK_MANAGER)
        unset(COMPUTE_FLOW_TEST_BRICK_MANAGER CACHE)
        find_program(
            COMPUTE_FLOW_TEST_BRICK_MANAGER
            NAMES compute-flow
            HINTS "${CMAKE_SOURCE_DIR}/.venv/bin"
            REQUIRED
            DOC "Compute Flow Brick Manager executable used by tests"
        )
    endif()

    function(compute_flow_add_cli_model_smoke_test brick_type brick_name model_name)
        set(test_name "${brick_type}_${brick_name}_${model_name}_cli_model_smoke")
        string(REGEX REPLACE "[^A-Za-z0-9_]+" "_" test_name "${test_name}")

        add_test(
            NAME "${test_name}"
            COMMAND
                "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cli_model_smoke.py" --brick-type "${brick_type}"
                --brick-name "${brick_name}" --model "${model_name}" --cli "$<TARGET_FILE:${brick_type}_cli>"
                --model-home "${COMPUTE_FLOW_TEST_MODEL_HOME}"
        )
        set_tests_properties(
            "${test_name}"
            PROPERTIES FIXTURES_REQUIRED cli_models LABELS "cli;cli_model;model;smoke" RUN_SERIAL TRUE TIMEOUT 1200
        )
    endfunction()

    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/model_smoke_cases.py" --repository-root
            "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE model_cases_result
        OUTPUT_VARIABLE model_cases_output
        ERROR_VARIABLE model_cases_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT model_cases_result EQUAL 0)
        message(FATAL_ERROR "Failed to load model smoke cases: ${model_cases_error}")
    endif()

    string(REPLACE "\n" ";" model_cases "${model_cases_output}")
    foreach(model_case IN LISTS model_cases)
        string(REPLACE "|" ";" model_case_fields "${model_case}")
        list(GET model_case_fields 0 brick_type)
        list(GET model_case_fields 1 brick_name)
        list(GET model_case_fields 2 model_name)

        set(brick_class "${brick_type}::${brick_name}")
        if(NOT brick_class IN_LIST COMPUTE_FLOW_INSTALLED_BRICK_CLASSES)
            message(
                FATAL_ERROR
                "Model-backed tests require installed brick '${brick_class}'. "
                "Run 'uv run compute-flow update' and 'uv run compute-flow brick install --all' before configuring."
            )
        endif()

        compute_flow_add_cli_model_smoke_test("${brick_type}" "${brick_name}" "${model_name}")
    endforeach()

    add_test(
        NAME cli_model_install
        COMMAND
            "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/install_test_models.py" --brick-manager
            "${COMPUTE_FLOW_TEST_BRICK_MANAGER}" --model-home "${COMPUTE_FLOW_TEST_MODEL_HOME}" --repository-root
            "${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(
        cli_model_install
        PROPERTIES FIXTURES_SETUP cli_models LABELS "cli;cli_model;model;setup" RUN_SERIAL TRUE TIMEOUT 21600
    )
endif()
