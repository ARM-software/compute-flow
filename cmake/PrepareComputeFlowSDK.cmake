#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

foreach(
    required_variable
    IN
    ITEMS
        COMPUTE_FLOW_INCLUDE_DIR
        COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES
        COMPUTE_FLOW_INSTALLED_BRICK_SOURCE_DIRS
        COMPUTE_FLOW_LIBRARY_MANIFEST
        COMPUTE_FLOW_FORCE_LOAD_LIBRARY_MANIFEST
        COMPUTE_FLOW_SDK_DIR
)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required to prepare the Compute Flow SDK.")
    endif()
endforeach()

set(sdk_staging_dir "${COMPUTE_FLOW_SDK_DIR}.staging")
file(REMOVE_RECURSE "${sdk_staging_dir}")
file(MAKE_DIRECTORY "${sdk_staging_dir}/include" "${sdk_staging_dir}/lib")
file(COPY "${COMPUTE_FLOW_INCLUDE_DIR}/" DESTINATION "${sdk_staging_dir}/include")

list(LENGTH COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES brick_identity_count)
list(LENGTH COMPUTE_FLOW_INSTALLED_BRICK_SOURCE_DIRS brick_source_dir_count)
if(NOT brick_identity_count EQUAL brick_source_dir_count)
    message(FATAL_ERROR "Installed brick identities and source directories must have the same length.")
endif()

if(brick_identity_count GREATER 0)
    math(EXPR last_brick_index "${brick_identity_count} - 1")
    foreach(index RANGE 0 ${last_brick_index})
        list(GET COMPUTE_FLOW_INSTALLED_BRICK_IDENTITIES ${index} brick_identity)
        list(GET COMPUTE_FLOW_INSTALLED_BRICK_SOURCE_DIRS ${index} brick_source_dir)

        if(NOT brick_identity MATCHES "^([^:]+):([^:]+)$")
            message(FATAL_ERROR "Invalid installed brick identity: ${brick_identity}")
        endif()
        set(brick_type "${CMAKE_MATCH_1}")
        set(brick_name "${CMAKE_MATCH_2}")
        set(info_json "${brick_source_dir}/info.json")
        if(NOT EXISTS "${info_json}")
            message(FATAL_ERROR "Installed brick metadata does not exist: ${info_json}")
        endif()

        set(metadata_dir "${sdk_staging_dir}/share/compute_flow/bricks/${brick_type}/${brick_name}")
        file(MAKE_DIRECTORY "${metadata_dir}")
        file(COPY_FILE "${info_json}" "${metadata_dir}/info.json")
    endforeach()
endif()

file(STRINGS "${COMPUTE_FLOW_LIBRARY_MANIFEST}" built_libraries)

set(copied_libraries "")
foreach(library IN LISTS built_libraries)
    if(NOT EXISTS "${library}")
        continue()
    endif()

    get_filename_component(library_name "${library}" NAME)
    set(destination "${sdk_staging_dir}/lib/${library_name}")
    if(EXISTS "${destination}")
        file(SHA256 "${library}" source_sha256)
        file(SHA256 "${destination}" destination_sha256)
        if(NOT source_sha256 STREQUAL destination_sha256)
            message(FATAL_ERROR "Multiple built libraries have the same name: ${library_name}")
        endif()
        continue()
    endif()

    file(COPY_FILE "${library}" "${destination}")
    list(APPEND copied_libraries "${library_name}")
endforeach()

file(STRINGS "${COMPUTE_FLOW_FORCE_LOAD_LIBRARY_MANIFEST}" force_load_libraries)
set(force_load_library_names "")
foreach(library IN LISTS force_load_libraries)
    if(NOT EXISTS "${library}")
        message(FATAL_ERROR "Force-load library does not exist: ${library}")
    endif()

    get_filename_component(library_name "${library}" NAME)
    set(destination "${sdk_staging_dir}/lib/${library_name}")
    if(EXISTS "${destination}")
        file(SHA256 "${library}" source_sha256)
        file(SHA256 "${destination}" destination_sha256)
        if(NOT source_sha256 STREQUAL destination_sha256)
            message(FATAL_ERROR "Multiple built libraries have the same name: ${library_name}")
        endif()
    else()
        file(COPY_FILE "${library}" "${destination}")
        list(APPEND copied_libraries "${library_name}")
    endif()
    list(APPEND force_load_library_names "${library_name}")
endforeach()

list(REMOVE_DUPLICATES force_load_library_names)
set(force_load_library_manifest "")
foreach(library_name IN LISTS force_load_library_names)
    string(APPEND force_load_library_manifest "${library_name}\n")
endforeach()
file(WRITE "${sdk_staging_dir}/force_load_libraries.txt" "${force_load_library_manifest}")

if(NOT copied_libraries)
    message(FATAL_ERROR "No built libraries were listed in ${COMPUTE_FLOW_LIBRARY_MANIFEST}.")
endif()

file(COPY_FILE "${CMAKE_CURRENT_LIST_DIR}/ComputeFlowSDKCMakeLists.txt.in" "${sdk_staging_dir}/CMakeLists.txt")
file(COPY_FILE "${CMAKE_CURRENT_LIST_DIR}/ComputeFlowSDKREADME.md.in" "${sdk_staging_dir}/README.md")

file(REMOVE_RECURSE "${COMPUTE_FLOW_SDK_DIR}")
file(RENAME "${sdk_staging_dir}" "${COMPUTE_FLOW_SDK_DIR}")
list(LENGTH copied_libraries copied_library_count)
message(STATUS "Prepared Compute Flow SDK with ${copied_library_count} libraries: ${COMPUTE_FLOW_SDK_DIR}")
