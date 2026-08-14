#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/sentencepiece/SentencePiece.cmake — SentencePiece tokenizer
# =========================================================================================
#
# Creates IMPORTED targets:
#   sentencepiece-static                — main sentencepiece static library
#   sentencepiece::sentencepiece        — alias for sentencepiece-static
#   sentencepiece::sentencepiece-static — alias for sentencepiece-static
#
# Also populates COMPUTE_FLOW_SENTENCEPIECE_TARGET with the resolved target name for the main
# CMakeLists.txt to link against.
#
# Depends on: compute_flow_dep_protobuf (and protobuf_host_ep when cross-compiling)
#
# Resolution: find_package → ExternalProject (gated by from-source flag)
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_SENTENCEPIECE_VERSION "0.2.0")
set(COMPUTE_FLOW_SENTENCEPIECE_GIT_REPO "https://github.com/google/sentencepiece.git")
set(COMPUTE_FLOW_SENTENCEPIECE_GIT_TAG "v${COMPUTE_FLOW_SENTENCEPIECE_VERSION}")

# ---- Try pre-installed (Conan / system) --------------------------------------------------
if(NOT COMPUTE_FLOW_BUILD_SENTENCEPIECE_FROM_SOURCE)
    find_package(sentencepiece CONFIG QUIET)

    if(sentencepiece_FOUND)
        message(STATUS "SentencePiece: found pre-installed")
        add_custom_target(compute_flow_dep_sentencepiece)

        # Target resolution cascade
        set(COMPUTE_FLOW_SENTENCEPIECE_TARGET "" CACHE STRING "CMake target name to link for SentencePiece")
        if(COMPUTE_FLOW_SENTENCEPIECE_TARGET AND TARGET "${COMPUTE_FLOW_SENTENCEPIECE_TARGET}")
            set(COMPUTE_FLOW_SENTENCEPIECE_TARGET "${COMPUTE_FLOW_SENTENCEPIECE_TARGET}")
        elseif(TARGET sentencepiece-static)
            set(COMPUTE_FLOW_SENTENCEPIECE_TARGET sentencepiece-static)
        elseif(TARGET sentencepiece::sentencepiece-static)
            set(COMPUTE_FLOW_SENTENCEPIECE_TARGET sentencepiece::sentencepiece-static)
        elseif(TARGET sentencepiece)
            set(COMPUTE_FLOW_SENTENCEPIECE_TARGET sentencepiece)
        elseif(TARGET sentencepiece::sentencepiece)
            set(COMPUTE_FLOW_SENTENCEPIECE_TARGET sentencepiece::sentencepiece)
        else()
            message(
                FATAL_ERROR
                "Found SentencePiece via find_package(sentencepiece CONFIG), but no known CMake target exists. "
                "Set -DCOMPUTE_FLOW_SENTENCEPIECE_TARGET=<target> (e.g. sentencepiece-static)."
            )
        endif()

        # Monolithic archive declaration
        compute_flow_extract_target_archives(COMPUTE_FLOW_SENTENCEPIECE_MONOLITHIC_ARCHIVES
            ${COMPUTE_FLOW_SENTENCEPIECE_TARGET}
        )

        return()
    endif()
endif()

# ---- Guard: require explicit from-source opt-in -----------------------------------------
if(NOT COMPUTE_FLOW_BUILD_SENTENCEPIECE_FROM_SOURCE)
    message(
        FATAL_ERROR
        "SentencePiece not found via find_package and from-source build not enabled.\n"
        "Set -DCOMPUTE_FLOW_BUILD_SENTENCEPIECE_FROM_SOURCE=ON."
    )
endif()

# ---- Build from source via ExternalProject -----------------------------------------------
message(STATUS "SentencePiece: building from source via ExternalProject")

compute_flow_dep_prefix_dir(sentencepiece_prefix_dir "sentencepiece")
compute_flow_dep_install_dir(sentencepiece_install_dir "sentencepiece")

# ---- EP cmake args ----------------------------------------------------------------------

set(sentencepiece_extra_cmake_args
    -DSPM_ENABLE_SHARED:BOOL=OFF
    -DSPM_ENABLE_TCMALLOC:BOOL=OFF
    -DSPM_TCMALLOC_STATIC:BOOL=OFF
    -DSPM_USE_BUILTIN_PROTOBUF:BOOL=OFF
    -DSPM_PROTOBUF_PROVIDER:STRING=package
    # Wire in Protobuf explicitly to avoid picking up system Protobuf during cross-compiles.
    -DProtobuf_INCLUDE_DIR:PATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR}
    -DProtobuf_INCLUDE_DIRS:PATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR}
    -DPROTOBUF_INCLUDE_DIR:PATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR}
    -DProtobuf_LIBRARY:FILEPATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libprotobuf.a
    -DProtobuf_LITE_LIBRARY:FILEPATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libprotobuf-lite.a
    -DPROTOBUF_LITE_LIBRARY:FILEPATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libprotobuf-lite.a
)

# Android™: Protobuf static libs need -llog for __android_log_write
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(sentencepiece_exe_ldflags "${CMAKE_EXE_LINKER_FLAGS} -llog")
    string(REPLACE ";" "\\;" sentencepiece_exe_ldflags "${sentencepiece_exe_ldflags}")
    list(APPEND sentencepiece_extra_cmake_args "-DCMAKE_EXE_LINKER_FLAGS:STRING=${sentencepiece_exe_ldflags}")
endif()

# ---- EP dependency + build command configuration ----------------------------------------

set(sentencepiece_ep_depends compute_flow_dep_protobuf)
set(sentencepiece_ep_build_command_arg "")

if(COMPUTE_FLOW_BUILD_HOST_TOOLS)
    list(
        APPEND sentencepiece_extra_cmake_args
        -DProtobuf_PROTOC_EXECUTABLE:FILEPATH=${COMPUTE_FLOW_PROTOBUF_HOST_INSTALL_DIR}/bin/protoc
        -DPROTOBUF_PROTOC_EXECUTABLE:FILEPATH=${COMPUTE_FLOW_PROTOBUF_HOST_INSTALL_DIR}/bin/protoc
        -Dprotobuf_PROTOC_EXECUTABLE:FILEPATH=${COMPUTE_FLOW_PROTOBUF_HOST_INSTALL_DIR}/bin/protoc
    )
    list(APPEND sentencepiece_ep_depends protobuf_host_ep)
    # Ensure host protoc is on PATH during build
    set(sentencepiece_ep_build_command_arg
        BUILD_COMMAND
        ${CMAKE_COMMAND}
        -E
        env
        "PATH=${COMPUTE_FLOW_PROTOBUF_HOST_INSTALL_DIR}/bin:$ENV{PATH}"
        ${CMAKE_COMMAND}
        --build
        <BINARY_DIR>
        --parallel
    )
else()
    list(
        APPEND sentencepiece_extra_cmake_args
        -DProtobuf_PROTOC_EXECUTABLE:FILEPATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/bin/protoc
        -DPROTOBUF_PROTOC_EXECUTABLE:FILEPATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/bin/protoc
        -Dprotobuf_PROTOC_EXECUTABLE:FILEPATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/bin/protoc
    )
endif()

# ---- Main EP ----------------------------------------------------------------------------

ExternalProject_Add(
    sentencepiece_ep
    GIT_REPOSITORY "${COMPUTE_FLOW_SENTENCEPIECE_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_SENTENCEPIECE_GIT_TAG}"
    PREFIX "${sentencepiece_prefix_dir}"
    INSTALL_DIR "${sentencepiece_install_dir}"
    GIT_PROGRESS TRUE
    UPDATE_DISCONNECTED ON
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        ${sentencepiece_extra_cmake_args} ${sentencepiece_ep_build_command_arg}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
    DEPENDS ${sentencepiece_ep_depends}
)

# ---- Shim EP: generate sentencepieceConfig.cmake ----------------------------------------

set(sentencepiece_shim_prefix_dir "${COMPUTE_FLOW_DEPS_PREFIX_DIR}/sentencepiece-shim")

ExternalProject_Add(
    sentencepiece_shim_ep
    PREFIX "${sentencepiece_shim_prefix_dir}"
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/shim"
    DEPENDS sentencepiece_ep
    CMAKE_ARGS ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX:PATH=${sentencepiece_install_dir}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)

add_custom_target(compute_flow_dep_sentencepiece DEPENDS sentencepiece_shim_ep)

set(COMPUTE_FLOW_SENTENCEPIECE_INSTALL_DIR "${sentencepiece_install_dir}")

# ---- Placeholder IMPORTED targets -------------------------------------------------------

file(MAKE_DIRECTORY "${sentencepiece_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}")

if(NOT TARGET sentencepiece-static)
    add_library(sentencepiece-static STATIC IMPORTED GLOBAL)
    set_target_properties(
        sentencepiece-static
        PROPERTIES
            IMPORTED_LOCATION "${sentencepiece_install_dir}/${CMAKE_INSTALL_LIBDIR}/libsentencepiece.a"
            INTERFACE_INCLUDE_DIRECTORIES "${sentencepiece_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    target_link_libraries(sentencepiece-static INTERFACE protobuf::libprotobuf-lite)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        target_link_libraries(sentencepiece-static INTERFACE atomic log)
    endif()
    add_dependencies(sentencepiece-static sentencepiece_shim_ep)
endif()

if(NOT TARGET sentencepiece::sentencepiece)
    add_library(sentencepiece::sentencepiece ALIAS sentencepiece-static)
endif()

if(NOT TARGET sentencepiece::sentencepiece-static)
    add_library(sentencepiece::sentencepiece-static ALIAS sentencepiece-static)
endif()

# ---- Target resolution for main CMakeLists.txt ------------------------------------------
set(COMPUTE_FLOW_SENTENCEPIECE_TARGET "" CACHE STRING "CMake target name to link for SentencePiece")
if(COMPUTE_FLOW_SENTENCEPIECE_TARGET AND TARGET "${COMPUTE_FLOW_SENTENCEPIECE_TARGET}")
    set(COMPUTE_FLOW_SENTENCEPIECE_TARGET "${COMPUTE_FLOW_SENTENCEPIECE_TARGET}")
elseif(TARGET sentencepiece-static)
    set(COMPUTE_FLOW_SENTENCEPIECE_TARGET sentencepiece-static)
elseif(TARGET sentencepiece::sentencepiece-static)
    set(COMPUTE_FLOW_SENTENCEPIECE_TARGET sentencepiece::sentencepiece-static)
elseif(TARGET sentencepiece)
    set(COMPUTE_FLOW_SENTENCEPIECE_TARGET sentencepiece)
elseif(TARGET sentencepiece::sentencepiece)
    set(COMPUTE_FLOW_SENTENCEPIECE_TARGET sentencepiece::sentencepiece)
else()
    message(
        FATAL_ERROR
        "SentencePiece from-source build did not create a known target. "
        "Set -DCOMPUTE_FLOW_SENTENCEPIECE_TARGET=<target> (e.g. sentencepiece-static)."
    )
endif()

# ---- Monolithic archive declaration -----------------------------------------------------
set(COMPUTE_FLOW_SENTENCEPIECE_MONOLITHIC_ARCHIVES
    "${sentencepiece_install_dir}/${CMAKE_INSTALL_LIBDIR}/libsentencepiece.a"
)
