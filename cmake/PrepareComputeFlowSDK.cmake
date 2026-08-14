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
