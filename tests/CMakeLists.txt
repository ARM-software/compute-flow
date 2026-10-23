#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

set(COMPUTE_FLOW_TEST_SENTENCEPIECE_MODEL "" CACHE FILEPATH "Local tokenizer for decode runner contract tests")
if(COMPUTE_FLOW_ENABLE_SENTENCEPIECE AND COMPUTE_FLOW_TEST_SENTENCEPIECE_MODEL)
    add_executable(sentencepiece_decode_tests runners/sentencepiece_decode_tests.cpp)
    target_link_libraries(sentencepiece_decode_tests PRIVATE compute_flow::compute_flow)
    add_test(NAME sentencepiece_decode COMMAND sentencepiece_decode_tests "${COMPUTE_FLOW_TEST_SENTENCEPIECE_MODEL}")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/bricks")
    add_test(NAME brick_metadata COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/bricks/test_metadata.py")
endif()

add_test(
    NAME brick_manager_catalog
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/bricks/test_catalog.py"
)
add_test(
    NAME brick_manager_model_install
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/bricks/test_model_install.py"
)
if(COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES)
    list(GET COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES 0 application_sdk_test_identity)
    list(GET COMPUTE_FLOW_INSTALLED_BRICK_SOURCE_DIRS 0 application_sdk_test_source_dir)
    string(REPLACE ":" ";" application_sdk_test_identity_parts "${application_sdk_test_identity}")
    list(GET application_sdk_test_identity_parts 0 application_sdk_test_type)
    list(GET application_sdk_test_identity_parts 1 application_sdk_test_name)

    add_test(
        NAME application_sdk_metadata
        COMMAND
            "${CMAKE_COMMAND}" "-DBUILD_DIR=${CMAKE_BINARY_DIR}" "-DBUILD_CONFIG=$<CONFIG>"
            "-DINSTALLED_INFO_JSON=${application_sdk_test_source_dir}/info.json"
            "-DPACKAGED_INFO_JSON=${CMAKE_BINARY_DIR}/sdk/share/compute_flow/bricks/${application_sdk_test_type}/${application_sdk_test_name}/info.json"
            "-DPACKAGED_LIBRARY=${CMAKE_BINARY_DIR}/sdk/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${application_sdk_test_type}_core${CMAKE_STATIC_LIBRARY_SUFFIX}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/application_sdk_metadata.cmake"
    )
endif()

add_test(
    NAME dependency_releases
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_dependency_downloads.py"
)
set_tests_properties(dependency_releases PROPERTIES ENVIRONMENT "CMAKE_COMMAND=${CMAKE_COMMAND}")

if(COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES)
    add_executable(brick_contract_tests "${CMAKE_CURRENT_SOURCE_DIR}/bricks/brick_contract_tests.cpp")
    compute_flow_link_installed_bricks(brick_contract_tests)

    add_test(NAME brick_contract COMMAND brick_contract_tests "${COMPUTE_FLOW_BRICK_CACHE_PATH}")
endif()

set(COMPUTE_FLOW_CLI_SMOKE_TARGETS
    audio_detection_cli
    audio_to_audio_cli
    audio_generation_cli
    image_to_image_cli
    image_generation_cli
    image_classification_cli
    image_segmentation_cli
    lm_text_cli
    monocular_depth_estimation_cli
    object_detection_cli
    audio_embedding_cli
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
