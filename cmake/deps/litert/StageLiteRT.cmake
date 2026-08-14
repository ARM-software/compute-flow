#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

foreach(
    required_var
    LITERT_INSTALL_DIR
    LITERT_C_INCLUDE_DIR
    LITERT_GENERATED_INCLUDE_DIR
    LITERT_SOURCE_DIR
    LITERT_TENSORFLOW_SOURCE_DIR
    LITERT_BUILD_DIR
    LITERT_ARCHIVE_FILELIST
    LITERT_CONFIG
    LITERT_CONFIG_VERSION
)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "StageLiteRT: ${required_var} is required")
    endif()
endforeach()

foreach(
    required_path
    "${LITERT_C_INCLUDE_DIR}/litert/c/litert_common.h"
    "${LITERT_GENERATED_INCLUDE_DIR}/litert/build_common/build_config.h"
    "${LITERT_SOURCE_DIR}/LICENSE"
    "${LITERT_TENSORFLOW_SOURCE_DIR}/LICENSE"
    "${LITERT_ARCHIVE_FILELIST}"
    "${LITERT_CONFIG}"
    "${LITERT_CONFIG_VERSION}"
)
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR "StageLiteRT: expected input is missing: ${required_path}")
    endif()
endforeach()

file(REMOVE_RECURSE "${LITERT_INSTALL_DIR}")
file(
    MAKE_DIRECTORY
        "${LITERT_INSTALL_DIR}/include"
        "${LITERT_INSTALL_DIR}/include/litert/build_common"
        "${LITERT_INSTALL_DIR}/lib/cmake/litert"
        "${LITERT_INSTALL_DIR}/licenses"
)

file(COPY "${LITERT_C_INCLUDE_DIR}/" DESTINATION "${LITERT_INSTALL_DIR}/include")
file(
    COPY "${LITERT_GENERATED_INCLUDE_DIR}/litert/build_common/build_config.h"
    DESTINATION "${LITERT_INSTALL_DIR}/include/litert/build_common"
)

# Two LiteRT internal C headers expose these TFLite C types. Stage only their exact include
# closure rather than installing the legacy TensorFlow Lite API as part of the LiteRT package.
set(tflite_headers
    "tflite/c/c_api_types.h"
    "tflite/c/common.h"
    "tflite/core/c/c_api_types.h"
    "tflite/core/c/common.h"
    "tflite/converter/core/c/tflite_types.h"
    "tflite/delegates/gpu/delegate.h"
    "tflite/delegates/gpu/delegate_options.h"
    "litert/c/internal/litert_accelerator_def.h"
    "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
    "litert/c/internal/litert_delegate_wrapper.h"
    "litert/c/internal/litert_runtime_context.h"
    "litert/runtime/accelerators/gpu_static_registry.h"
)
foreach(relative_header IN LISTS tflite_headers)
    set(source_header "${LITERT_SOURCE_DIR}/${relative_header}")
    if(NOT EXISTS "${source_header}")
        message(FATAL_ERROR "StageLiteRT: required TFLite compatibility header is missing: ${source_header}")
    endif()
    get_filename_component(relative_dir "${relative_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${LITERT_INSTALL_DIR}/include/${relative_dir}")
    file(COPY "${source_header}" DESTINATION "${LITERT_INSTALL_DIR}/include/${relative_dir}")
endforeach()

file(COPY "${LITERT_CONFIG}" "${LITERT_CONFIG_VERSION}" DESTINATION "${LITERT_INSTALL_DIR}/lib/cmake/litert")
file(STRINGS "${LITERT_ARCHIVE_FILELIST}" litert_archives)
foreach(litert_archive IN LISTS litert_archives)
    if(NOT EXISTS "${litert_archive}")
        message(FATAL_ERROR "StageLiteRT: component archive is missing: ${litert_archive}")
    endif()
    get_filename_component(litert_archive_name "${litert_archive}" NAME)
    set(staged_archive "${LITERT_INSTALL_DIR}/lib/${litert_archive_name}")
    file(COPY_FILE "${litert_archive}" "${staged_archive}")
    if(LITERT_OBJCOPY AND litert_archive_name STREQUAL "libabsl_symbolize.a")
        execute_process(
            COMMAND
                "${LITERT_OBJCOPY}" "--redefine-sym=AbslInternalGetFileMappingHint=LiteRtAbslInternalGetFileMappingHint"
                "${staged_archive}"
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()
endforeach()
file(
    REMOVE
    "${LITERT_INSTALL_DIR}/lib/libLiteRtClGlAccelerator.so"
    "${LITERT_INSTALL_DIR}/lib/libLiteRtMetalAccelerator.dylib"
)
file(COPY_FILE "${LITERT_SOURCE_DIR}/LICENSE" "${LITERT_INSTALL_DIR}/licenses/LiteRT-LICENSE")
file(COPY_FILE "${LITERT_TENSORFLOW_SOURCE_DIR}/LICENSE" "${LITERT_INSTALL_DIR}/licenses/TensorFlow-LICENSE")

# Include top-level notices from the dependency sources populated by LiteRT/TFLite.
file(GLOB dependency_source_dirs LIST_DIRECTORIES true "${LITERT_BUILD_DIR}/*" "${LITERT_BUILD_DIR}/_deps/*-src")
foreach(dependency_source_dir IN LISTS dependency_source_dirs)
    if(NOT IS_DIRECTORY "${dependency_source_dir}")
        continue()
    endif()
    get_filename_component(dependency_name "${dependency_source_dir}" NAME)
    file(
        GLOB dependency_notices
        "${dependency_source_dir}/LICENSE*"
        "${dependency_source_dir}/COPYING*"
        "${dependency_source_dir}/NOTICE*"
    )
    foreach(dependency_notice IN LISTS dependency_notices)
        if(IS_DIRECTORY "${dependency_notice}")
            continue()
        endif()
        get_filename_component(notice_name "${dependency_notice}" NAME)
        file(COPY_FILE "${dependency_notice}" "${LITERT_INSTALL_DIR}/licenses/${dependency_name}-${notice_name}")
    endforeach()
endforeach()
