#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/mnn/MNN.cmake — MNN runtime
# =========================================================================================
#
# Creates/uses target:
#   MNN::MNN — MNN static runtime library.
#
# Resolution: find_package -> ExternalProject (gated by from-source flag).
# =========================================================================================

include_guard(GLOBAL)

set(COMPUTE_FLOW_MNN_GIT_REPO "https://github.com/alibaba/MNN.git")
set(COMPUTE_FLOW_MNN_GIT_TAG "3.6.1")

if(COMPUTE_FLOW_ENABLE_MNN_GPU AND NOT ANDROID)
    message(FATAL_ERROR "MNN GPU acceleration is supported on Android™ only.")
endif()

if(NOT COMPUTE_FLOW_BUILD_MNN_FROM_SOURCE)
    find_package(MNN CONFIG QUIET)

    if(MNN_FOUND)
        if(TARGET MNN::MNN)
            set(COMPUTE_FLOW_MNN_TARGET MNN::MNN)
        elseif(TARGET MNN)
            set(COMPUTE_FLOW_MNN_TARGET MNN)
        else()
            message(FATAL_ERROR "MNN was found, but it did not provide an MNN or MNN::MNN target.")
        endif()

        message(STATUS "MNN: found pre-installed")
        add_custom_target(compute_flow_dep_mnn)
        compute_flow_extract_target_archives(COMPUTE_FLOW_MNN_MONOLITHIC_ARCHIVES ${COMPUTE_FLOW_MNN_TARGET})
        return()
    endif()

    message(
        FATAL_ERROR
        "MNN not found via find_package and from-source build not enabled.\n"
        "Set -DCOMPUTE_FLOW_BUILD_MNN_FROM_SOURCE=ON."
    )
endif()

message(STATUS "MNN: building from source via ExternalProject")

compute_flow_dep_prefix_dir(mnn_prefix_dir "mnn")
compute_flow_dep_install_dir(mnn_install_dir "mnn")

# MNN bundles a private FlatBuffers copy whose inline symbols otherwise collide with the copies in
# other monolithic runtimes. Give that private implementation a distinct C++ namespace.
set(mnn_cxx_flags "${CMAKE_CXX_FLAGS}")
string(APPEND mnn_cxx_flags " -Dflatbuffers=mnn_flatbuffers")
string(STRIP "${mnn_cxx_flags}" mnn_cxx_flags)

ExternalProject_Add(
    compute_flow_dep_mnn
    GIT_REPOSITORY "${COMPUTE_FLOW_MNN_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_MNN_GIT_TAG}"
    PREFIX "${mnn_prefix_dir}"
    INSTALL_DIR "${mnn_install_dir}"
    GIT_PROGRESS TRUE
    GIT_SUBMODULES_RECURSE ON
    UPDATE_DISCONNECTED ON
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DMNN_BUILD_SHARED_LIBS:BOOL=OFF
        "-DCMAKE_CXX_FLAGS:STRING=${mnn_cxx_flags}" -DMNN_SEP_BUILD:BOOL=OFF -DMNN_BUILD_FOR_ANDROID_COMMAND:BOOL=ON
        -DMNN_BUILD_TOOLS:BOOL=OFF -DMNN_BUILD_CONVERTER:BOOL=OFF -DMNN_BUILD_TEST:BOOL=OFF
        -DMNN_BUILD_BENCHMARK:BOOL=OFF -DMNN_BUILD_DEMO:BOOL=OFF -DMNN_BUILD_LLM:BOOL=OFF -DMNN_BUILD_OPENCV:BOOL=OFF
        -DMNN_KLEIDIAI:BOOL=ON -DMNN_KLEIDIAI_DEFAULT_ON:BOOL=ON -DMNN_LOW_MEMORY:BOOL=ON -DMNN_METAL:BOOL=OFF
        -DMNN_OPENCL:BOOL=${COMPUTE_FLOW_ENABLE_MNN_GPU} -DMNN_SME2:BOOL=ON -DMNN_VULKAN:BOOL=OFF
    BUILD_BYPRODUCTS <INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}/libMNN.a
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)

set(COMPUTE_FLOW_MNN_INSTALL_DIR "${mnn_install_dir}")
set(COMPUTE_FLOW_MNN_TARGET MNN::MNN)

file(MAKE_DIRECTORY "${mnn_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}")

if(NOT TARGET MNN::MNN)
    add_library(MNN::MNN STATIC IMPORTED GLOBAL)
    set_target_properties(
        MNN::MNN
        PROPERTIES
            IMPORTED_LOCATION "${mnn_install_dir}/${CMAKE_INSTALL_LIBDIR}/libMNN.a"
            INTERFACE_INCLUDE_DIRECTORIES "${mnn_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    add_dependencies(MNN::MNN compute_flow_dep_mnn)
endif()

find_package(Threads REQUIRED)
set_property(TARGET MNN::MNN APPEND PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads)
if(ANDROID)
    set_property(TARGET MNN::MNN APPEND PROPERTY INTERFACE_LINK_LIBRARIES log m android)
elseif(UNIX AND NOT APPLE)
    set_property(TARGET MNN::MNN APPEND PROPERTY INTERFACE_LINK_LIBRARIES dl)
endif()

set(COMPUTE_FLOW_MNN_MONOLITHIC_ARCHIVES "${mnn_install_dir}/${CMAKE_INSTALL_LIBDIR}/libMNN.a")
