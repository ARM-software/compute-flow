#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/litert/LiteRT.cmake — LiteRT runtime
# =========================================================================================
#
# Creates the target used by both dependency paths:
#   litert::litert — LiteRT static C API and its dependency closure
#
# The source build uses FetchContent because LiteRT does not install a complete static C API
# package. It stages a relocatable CMake package for Conan export by preserving the static archive
# closure and collecting the required public headers.
#
# Resolution: find_package → FetchContent (gated by from-source flag)
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_LITERT_VERSION "2.1.6")
set(COMPUTE_FLOW_LITERT_URL
    "https://github.com/google-ai-edge/LiteRT/archive/refs/tags/v${COMPUTE_FLOW_LITERT_VERSION}.tar.gz"
)
set(COMPUTE_FLOW_LITERT_URL_HASH "SHA256=f95fa96332c56b7103db7a02ab4edab845949c196a986db55bddaa70539ee45b")
set(COMPUTE_FLOW_LITERT_TENSORFLOW_VERSION "2.21.0-rc0")
set(COMPUTE_FLOW_LITERT_TENSORFLOW_URL
    "https://github.com/tensorflow/tensorflow/archive/refs/tags/v${COMPUTE_FLOW_LITERT_TENSORFLOW_VERSION}.tar.gz"
)
set(COMPUTE_FLOW_LITERT_TENSORFLOW_URL_HASH "SHA256=91b2c2d12f0ebd78d84e5e1401269d0e6e36db5b74e19cc929e0f012b380af39")

set(COMPUTE_FLOW_LITERT_RUNTIME_LIBRARIES "")
if(APPLE)
    list(APPEND COMPUTE_FLOW_LITERT_RUNTIME_LIBRARIES "-framework Metal" "-framework Foundation" "-framework OpenGL")
elseif(ANDROID)
    list(
        APPEND COMPUTE_FLOW_LITERT_RUNTIME_LIBRARIES
        android
        log
        EGL
        GLESv3
        GLESv2
    )
elseif(UNIX)
    list(APPEND COMPUTE_FLOW_LITERT_RUNTIME_LIBRARIES "${CMAKE_DL_LIBS}" m)
endif()

# ---- Try pre-installed (Conan / system) --------------------------------------------------
if(NOT COMPUTE_FLOW_BUILD_LITERT_FROM_SOURCE)
    find_package(litert ${COMPUTE_FLOW_LITERT_VERSION} CONFIG QUIET)
    if(litert_FOUND)
        message(STATUS "LiteRT: found pre-installed")
        if(TARGET litert::gpu_accelerator)
            get_target_property(litert_link_libraries litert::litert INTERFACE_LINK_LIBRARIES)
            if(litert_link_libraries)
                list(REMOVE_ITEM litert_link_libraries litert::gpu_accelerator)
                set_property(TARGET litert::litert PROPERTY INTERFACE_LINK_LIBRARIES "${litert_link_libraries}")
            endif()
        endif()
        if(COMPUTE_FLOW_ENABLE_LITERT_GPU AND NOT LITERT_STATIC_GPU_SUPPORTED)
            message(
                FATAL_ERROR
                "The installed LiteRT package does not contain static GPU support. "
                "Install a package built with the current Compute Flow LiteRT integration."
            )
        endif()
        add_custom_target(compute_flow_dep_litert)
        if(DEFINED LITERT_STATIC_ARCHIVES)
            set(COMPUTE_FLOW_LITERT_MONOLITHIC_ARCHIVES "${LITERT_STATIC_ARCHIVES}")
        else()
            compute_flow_extract_target_archives(COMPUTE_FLOW_LITERT_MONOLITHIC_ARCHIVES litert::litert)
        endif()
        if(DEFINED LITERT_LIBDIR)
            set(COMPUTE_FLOW_LITERT_MONOLITHIC_GLOB_DIR "${LITERT_LIBDIR}")
        endif()
        return()
    endif()
endif()

# ---- Guard: require explicit from-source opt-in -----------------------------------------
if(NOT COMPUTE_FLOW_BUILD_LITERT_FROM_SOURCE)
    message(
        FATAL_ERROR
        "LiteRT not found via find_package and from-source build not enabled.\n"
        "Set -DCOMPUTE_FLOW_BUILD_LITERT_FROM_SOURCE=ON."
    )
endif()

if(CMAKE_VERSION VERSION_LESS 4.0.1)
    message(FATAL_ERROR "Building LiteRT from source requires CMake 4.0.1 or newer.")
endif()

message(STATUS "LiteRT: building ${COMPUTE_FLOW_LITERT_VERSION} from source")
set(LITERT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(COMPUTE_FLOW_LITERT_STATIC_GPU_SUPPORTED OFF)
if(ANDROID)
    set(COMPUTE_FLOW_LITERT_STATIC_GPU_SUPPORTED ON)
    set(LITERT_ENABLE_GPU ON CACHE BOOL "" FORCE)
else()
    set(LITERT_ENABLE_GPU OFF CACHE BOOL "" FORCE)
endif()
set(LITERT_ENABLE_NPU OFF CACHE BOOL "" FORCE)
set(LITERT_ENABLE_QUALCOMM OFF CACHE BOOL "" FORCE)
set(LITERT_ENABLE_SAMSUNG OFF CACHE BOOL "" FORCE)
set(LITERT_DISABLE_KLEIDIAI OFF CACHE BOOL "" FORCE)
# Compute Flow's statically registered GPU accelerator uses the TensorFlow Lite GPU delegate.
set(TFLITE_ENABLE_GPU ${LITERT_ENABLE_GPU} CACHE BOOL "" FORCE)

FetchContent_Declare(
    compute_flow_litert_tensorflow
    URL "${COMPUTE_FLOW_LITERT_TENSORFLOW_URL}"
    URL_HASH "${COMPUTE_FLOW_LITERT_TENSORFLOW_URL_HASH}"
    SOURCE_SUBDIR
    compute_flow_no_subdirectory
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(compute_flow_litert_tensorflow)
set(TENSORFLOW_SOURCE_DIR "${compute_flow_litert_tensorflow_SOURCE_DIR}" CACHE PATH "" FORCE)

FetchContent_Declare(
    compute_flow_litert
    URL "${COMPUTE_FLOW_LITERT_URL}"
    URL_HASH "${COMPUTE_FLOW_LITERT_URL_HASH}"
    PATCH_COMMAND
        "${CMAKE_COMMAND}" -DLITERT_SOURCE_DIR:PATH=<SOURCE_DIR> -P "${CMAKE_CURRENT_LIST_DIR}/PatchLiteRT.cmake"
        SOURCE_SUBDIR compute_flow_no_subdirectory
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(compute_flow_litert)
add_subdirectory("${compute_flow_litert_SOURCE_DIR}/litert" "${compute_flow_litert_BINARY_DIR}" EXCLUDE_FROM_ALL)

# LiteRT creates its OpenCL-dependent TensorFlow Lite GPU GL core target unconditionally on Android™.
# Keep it out of CPU-only builds without modifying the fetched LiteRT sources.
if(NOT LITERT_ENABLE_GPU AND TARGET litert_tflite_gpu_gl_core)
    set_property(TARGET litert_tflite_gpu_gl_core PROPERTY EXCLUDE_FROM_ALL TRUE)
    foreach(litert_runtime_target IN ITEMS litert_runtime_c_api_static litert_runtime_c_api_shared_lib)
        foreach(litert_link_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(litert_link_libraries "${litert_runtime_target}" "${litert_link_property}")
            if(litert_link_libraries)
                list(REMOVE_ITEM litert_link_libraries litert_tflite_gpu_gl_core)
                set_property(
                    TARGET "${litert_runtime_target}"
                    PROPERTY "${litert_link_property}" "${litert_link_libraries}"
                )
            endif()
        endforeach()
    endforeach()
endif()

# LiteRT adds its OpenCL sources unconditionally, even when GPU support is disabled. Remove them
# from CPU-only builds so platforms without OpenCL headers can build the runtime.
if(NOT LITERT_ENABLE_GPU AND TARGET litert_runtime)
    get_target_property(litert_runtime_sources litert_runtime SOURCES)
    list(REMOVE_ITEM litert_runtime_sources open_cl_memory.cc open_cl_sync.cc)
    set_property(TARGET litert_runtime PROPERTY SOURCES "${litert_runtime_sources}")
endif()

if(APPLE AND TARGET xnnpack-microkernels-prod)
    # The macOS® archiver silently truncates XNNPACK's unusually large microkernel archive. Rebuild it with
    # libtool's file-list interface so normal and monolithic downstream links see every object.
    find_program(COMPUTE_FLOW_APPLE_LIBTOOL NAMES libtool REQUIRED)
    set(xnnpack_microkernel_filelist "${compute_flow_litert_BINARY_DIR}/xnnpack_microkernels.filelist")
    file(
        GENERATE OUTPUT "${xnnpack_microkernel_filelist}"
        CONTENT "$<JOIN:$<TARGET_OBJECTS:xnnpack-microkernels-prod>,\n>\n"
    )
    set(xnnpack_microkernel_stamp "${compute_flow_litert_BINARY_DIR}/xnnpack_microkernels.complete")
    add_custom_command(
        OUTPUT "${xnnpack_microkernel_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E rm -f "$<TARGET_FILE:xnnpack-microkernels-prod>"
        COMMAND
            "${COMPUTE_FLOW_APPLE_LIBTOOL}" -static -o "$<TARGET_FILE:xnnpack-microkernels-prod>" -filelist
            "${xnnpack_microkernel_filelist}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${xnnpack_microkernel_stamp}"
        DEPENDS xnnpack-microkernels-prod "${xnnpack_microkernel_filelist}"
        VERBATIM
    )
    add_custom_target(compute_flow_xnnpack_microkernels DEPENDS "${xnnpack_microkernel_stamp}")
endif()

function(
    _compute_flow_collect_litert_archives
    target
    archives_in
    visited_in
    out_archives
    out_visited
)
    set(archives "${archives_in}")
    set(visited "${visited_in}")
    get_target_property(aliased_target "${target}" ALIASED_TARGET)
    if(aliased_target)
        set(target "${aliased_target}")
    endif()
    list(FIND visited "${target}" visited_index)
    if(NOT visited_index EQUAL -1)
        set(${out_archives} "${archives}" PARENT_SCOPE)
        set(${out_visited} "${visited}" PARENT_SCOPE)
        return()
    endif()
    list(APPEND visited "${target}")

    get_target_property(target_type "${target}" TYPE)
    if(target_type STREQUAL "STATIC_LIBRARY")
        list(APPEND archives "$<TARGET_FILE:${target}>")
    endif()

    foreach(link_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(link_items "${target}" "${link_property}")
        if(NOT link_items)
            continue()
        endif()
        foreach(link_item IN LISTS link_items)
            if(link_item MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
                set(link_item "${CMAKE_MATCH_1}")
            elseif(link_item MATCHES "^\\$<BUILD_INTERFACE:([^>]+)>$")
                set(link_item "${CMAKE_MATCH_1}")
            elseif(link_item MATCHES "^\\$<")
                continue()
            endif()
            if(TARGET "${link_item}")
                _compute_flow_collect_litert_archives(
                    "${link_item}" "${archives}" "${visited}" archives visited
                )
            elseif(link_item MATCHES "\\.(a|lib)$")
                list(APPEND archives "${link_item}")
            endif()
        endforeach()
    endforeach()

    list(REMOVE_DUPLICATES archives)
    set(${out_archives} "${archives}" PARENT_SCOPE)
    set(${out_visited} "${visited}" PARENT_SCOPE)
endfunction()

add_library(compute_flow_litert INTERFACE)
add_library(litert::litert ALIAS compute_flow_litert)
target_link_libraries(compute_flow_litert INTERFACE litert_runtime_c_api_static)
target_link_libraries(compute_flow_litert INTERFACE ${COMPUTE_FLOW_LITERT_RUNTIME_LIBRARIES})
if(TARGET compute_flow_xnnpack_microkernels)
    add_dependencies(compute_flow_litert compute_flow_xnnpack_microkernels)
endif()
target_include_directories(
    compute_flow_litert
    INTERFACE $<BUILD_INTERFACE:${compute_flow_litert_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)
add_custom_target(compute_flow_dep_litert DEPENDS litert_c_api litert_runtime)
if(TARGET compute_flow_xnnpack_microkernels)
    add_dependencies(compute_flow_dep_litert compute_flow_xnnpack_microkernels)
endif()
_compute_flow_collect_litert_archives(
    litert_runtime_c_api_static "" "" COMPUTE_FLOW_LITERT_MONOLITHIC_ARCHIVES litert_archive_targets
)

# Stage a self-contained, relocatable package for Conan export. LiteRT does not currently install
# its static C API and dependency closure. Preserve the component archives so Compute Flow can
# deduplicate shared dependencies such as XNNPACK when assembling its monolithic archive.
compute_flow_dep_install_dir(litert_install_dir "litert")
set(COMPUTE_FLOW_LITERT_INSTALL_DIR "${litert_install_dir}")

set(litert_package_build_dir "${CMAKE_CURRENT_BINARY_DIR}/litert-package")
set(litert_package_config_dir "${litert_package_build_dir}/lib/cmake/litert")
file(MAKE_DIRECTORY "${litert_package_config_dir}")

set(litert_package_dependencies "")
set(COMPUTE_FLOW_LITERT_COMPONENT_ARCHIVES "")
foreach(litert_archive_target IN LISTS litert_archive_targets)
    get_target_property(litert_archive_target_type "${litert_archive_target}" TYPE)
    if(litert_archive_target_type STREQUAL "STATIC_LIBRARY")
        list(APPEND litert_package_dependencies "${litert_archive_target}")
        string(
            APPEND COMPUTE_FLOW_LITERT_COMPONENT_ARCHIVES
            "        \"\${LITERT_LIBDIR}/$<TARGET_FILE_NAME:${litert_archive_target}>\"\n"
        )
    elseif(litert_archive_target_type STREQUAL "OBJECT_LIBRARY")
        list(APPEND litert_package_dependencies "${litert_archive_target}")
    endif()
endforeach()

set(PACKAGE_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(PACKAGE_LIB_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/Config.cmake.in"
    "${litert_package_config_dir}/litertConfig.cmake.in"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/litert"
    PATH_VARS PACKAGE_INCLUDE_INSTALL_DIR PACKAGE_LIB_INSTALL_DIR
)
file(
    GENERATE OUTPUT "${litert_package_config_dir}/litertConfig.cmake"
    INPUT "${litert_package_config_dir}/litertConfig.cmake.in"
)
write_basic_package_version_file(
    "${litert_package_config_dir}/litertConfigVersion.cmake"
    VERSION "${COMPUTE_FLOW_LITERT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

set(litert_package_stamp "${litert_package_build_dir}/litert-package.complete")
set(litert_package_filelist "${litert_package_build_dir}/litert-package.filelist")
file(GENERATE OUTPUT "${litert_package_filelist}" CONTENT "$<JOIN:${COMPUTE_FLOW_LITERT_MONOLITHIC_ARCHIVES},\n>\n")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(litert_package_objcopy NAMES llvm-objcopy objcopy REQUIRED)
else()
    set(litert_package_objcopy "")
endif()
if(TARGET compute_flow_xnnpack_microkernels)
    list(APPEND litert_package_dependencies compute_flow_xnnpack_microkernels)
endif()
add_custom_command(
    OUTPUT "${litert_package_stamp}"
    COMMAND
        "${CMAKE_COMMAND}" "-DLITERT_INSTALL_DIR=${litert_install_dir}"
        "-DLITERT_C_INCLUDE_DIR=${compute_flow_litert_BINARY_DIR}/c/include"
        "-DLITERT_GENERATED_INCLUDE_DIR=${CMAKE_BINARY_DIR}/include"
        "-DLITERT_SOURCE_DIR=${compute_flow_litert_SOURCE_DIR}"
        "-DLITERT_TENSORFLOW_SOURCE_DIR=${compute_flow_litert_tensorflow_SOURCE_DIR}"
        "-DLITERT_ARCHIVE_FILELIST=${litert_package_filelist}" "-DLITERT_OBJCOPY=${litert_package_objcopy}"
        "-DLITERT_BUILD_DIR=${CMAKE_BINARY_DIR}" "-DLITERT_CONFIG=${litert_package_config_dir}/litertConfig.cmake"
        "-DLITERT_CONFIG_VERSION=${litert_package_config_dir}/litertConfigVersion.cmake" -P
        "${CMAKE_CURRENT_LIST_DIR}/StageLiteRT.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${litert_package_stamp}"
    DEPENDS
        ${litert_package_dependencies}
        "${CMAKE_CURRENT_LIST_DIR}/Config.cmake.in"
        "${CMAKE_CURRENT_LIST_DIR}/PatchLiteRT.cmake"
        "${CMAKE_CURRENT_LIST_DIR}/StageLiteRT.cmake"
        "${litert_package_config_dir}/litertConfig.cmake"
        "${litert_package_config_dir}/litertConfigVersion.cmake"
    VERBATIM
)
add_custom_target(compute_flow_package_litert DEPENDS "${litert_package_stamp}")
add_dependencies(compute_flow_dep_litert compute_flow_package_litert)
