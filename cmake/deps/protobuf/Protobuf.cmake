#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/protobuf/Protobuf.cmake — Google Protocol Buffers
# =========================================================================================
#
# Creates IMPORTED targets:
#   protobuf::libprotobuf       — full protobuf static library
#   protobuf::libprotobuf-lite  — lite protobuf static library
#   protobuf::protoc            — protoc compiler executable
#
# When cross-compiling (COMPUTE_FLOW_BUILD_HOST_TOOLS=ON), also builds a host-side
# protoc binary for code generation during dependent EP builds.
#
# Resolution: find_package → ExternalProject (gated by from-source flag)
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_PROTOBUF_VERSION "3.21.12")
set(COMPUTE_FLOW_PROTOBUF_GIT_REPO "https://github.com/protocolbuffers/protobuf.git")
set(COMPUTE_FLOW_PROTOBUF_GIT_TAG "v${COMPUTE_FLOW_PROTOBUF_VERSION}")

# ---- Try pre-installed (Conan / system) --------------------------------------------------
if(NOT COMPUTE_FLOW_BUILD_PROTOBUF_FROM_SOURCE)
    set(Protobuf_USE_STATIC_LIBS ON CACHE BOOL "" FORCE)
    find_package(Protobuf CONFIG QUIET)

    if(Protobuf_FOUND)
        message(STATUS "Protobuf: found pre-installed (${Protobuf_VERSION})")
        add_custom_target(compute_flow_dep_protobuf)

        # Monolithic archive declaration
        compute_flow_extract_target_archives(COMPUTE_FLOW_PROTOBUF_MONOLITHIC_ARCHIVES
            protobuf::libprotobuf protobuf::libprotobuf-lite
        )

        # Compatibility bridge: set FindProtobuf variables for downstream deps
        if(TARGET protobuf::libprotobuf-lite)
            get_target_property(pb_inc protobuf::libprotobuf-lite INTERFACE_INCLUDE_DIRECTORIES)

            get_target_property(pb_lite protobuf::libprotobuf-lite IMPORTED_LOCATION_RELEASE)
            if(NOT pb_lite)
                get_target_property(pb_lite protobuf::libprotobuf-lite IMPORTED_LOCATION)
            endif()

            set(Protobuf_INCLUDE_DIR "${pb_inc}" CACHE PATH "" FORCE)
            set(Protobuf_INCLUDE_DIRS "${pb_inc}" CACHE PATH "" FORCE)
            set(Protobuf_LITE_LIBRARY "${pb_lite}" CACHE FILEPATH "" FORCE)
            set(Protobuf_LITE_LIBRARIES "${pb_lite}" CACHE FILEPATH "" FORCE)
        endif()

        if(TARGET protobuf::libprotobuf)
            get_target_property(pb_full protobuf::libprotobuf IMPORTED_LOCATION_RELEASE)
            if(NOT pb_full)
                get_target_property(pb_full protobuf::libprotobuf IMPORTED_LOCATION)
            endif()
            set(Protobuf_LIBRARY "${pb_full}" CACHE FILEPATH "" FORCE)
            set(Protobuf_LIBRARIES "${pb_full}" CACHE FILEPATH "" FORCE)
        endif()

        return()
    endif()
endif()

# ---- Guard: require explicit from-source opt-in -----------------------------------------
if(NOT COMPUTE_FLOW_BUILD_PROTOBUF_FROM_SOURCE)
    message(
        FATAL_ERROR
        "Protobuf not found via find_package and from-source build not enabled.\n"
        "Set -DCOMPUTE_FLOW_BUILD_PROTOBUF_FROM_SOURCE=ON."
    )
endif()

# ---- Build from source via ExternalProject -----------------------------------------------
message(STATUS "Protobuf: building from source via ExternalProject")

compute_flow_dep_prefix_dir(protobuf_prefix_dir "protobuf")
compute_flow_dep_install_dir(protobuf_install_dir "protobuf")

# ---- Host protoc (cross-compile only) ---------------------------------------------------
if(COMPUTE_FLOW_BUILD_HOST_TOOLS)
    compute_flow_dep_prefix_dir(protobuf_host_prefix_dir "protobuf-host")
    compute_flow_dep_install_dir(protobuf_host_install_dir "protobuf-host")

    ExternalProject_Add(
        protobuf_host_ep
        GIT_REPOSITORY "${COMPUTE_FLOW_PROTOBUF_GIT_REPO}"
        GIT_TAG "${COMPUTE_FLOW_PROTOBUF_GIT_TAG}"
        PREFIX "${protobuf_host_prefix_dir}"
        INSTALL_DIR "${protobuf_host_install_dir}"
        GIT_PROGRESS TRUE
        UPDATE_DISCONNECTED ON
        CMAKE_ARGS
            ${COMPUTE_FLOW_HOST_EP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS:BOOL=OFF
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -Dprotobuf_BUILD_TESTS:BOOL=OFF
            -Dprotobuf_BUILD_SHARED_LIBS:BOOL=OFF -Dprotobuf_BUILD_PROTOC_BINARIES:BOOL=ON
            -Dprotobuf_DEBUG_POSTFIX:STRING= -Dprotobuf_WITH_ZLIB:BOOL=OFF
        BUILD_BYPRODUCTS <INSTALL_DIR>/bin/protoc
        INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
    )

    set(COMPUTE_FLOW_HOST_PROTOC "${protobuf_host_install_dir}/bin/protoc")
    set(COMPUTE_FLOW_PROTOBUF_HOST_INSTALL_DIR "${protobuf_host_install_dir}")
endif()

# ---- Target-side protobuf ---------------------------------------------------------------
set(protobuf_build_protoc ON)
if(CMAKE_CROSSCOMPILING AND COMPUTE_FLOW_BUILD_HOST_TOOLS)
    set(protobuf_build_protoc OFF)
endif()

ExternalProject_Add(
    compute_flow_dep_protobuf
    GIT_REPOSITORY "${COMPUTE_FLOW_PROTOBUF_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_PROTOBUF_GIT_TAG}"
    PREFIX "${protobuf_prefix_dir}"
    INSTALL_DIR "${protobuf_install_dir}"
    GIT_PROGRESS TRUE
    UPDATE_DISCONNECTED ON
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -Dprotobuf_BUILD_TESTS:BOOL=OFF -Dprotobuf_BUILD_SHARED_LIBS:BOOL=OFF
        -Dprotobuf_BUILD_PROTOC_BINARIES:BOOL=${protobuf_build_protoc} -Dprotobuf_DEBUG_POSTFIX:STRING=
        -Dprotobuf_WITH_ZLIB:BOOL=OFF
    BUILD_BYPRODUCTS
        <BINARY_DIR>/libprotobuf.a
        <BINARY_DIR>/libprotobuf-lite.a
        <BINARY_DIR>/libprotoc.a
        <BINARY_DIR>/protoc
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)

set(COMPUTE_FLOW_PROTOBUF_INSTALL_DIR "${protobuf_install_dir}")

# ---- Placeholder IMPORTED targets -------------------------------------------------------
file(MAKE_DIRECTORY "${protobuf_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}")

if(NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf STATIC IMPORTED GLOBAL)
    set_target_properties(
        protobuf::libprotobuf
        PROPERTIES
            IMPORTED_LOCATION "${protobuf_install_dir}/${CMAKE_INSTALL_LIBDIR}/libprotobuf.a"
            INTERFACE_INCLUDE_DIRECTORIES "${protobuf_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    add_dependencies(protobuf::libprotobuf compute_flow_dep_protobuf)
endif()

if(NOT TARGET protobuf::libprotobuf-lite)
    add_library(protobuf::libprotobuf-lite STATIC IMPORTED GLOBAL)
    set_target_properties(
        protobuf::libprotobuf-lite
        PROPERTIES
            IMPORTED_LOCATION "${protobuf_install_dir}/${CMAKE_INSTALL_LIBDIR}/libprotobuf-lite.a"
            INTERFACE_INCLUDE_DIRECTORIES "${protobuf_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    add_dependencies(protobuf::libprotobuf-lite compute_flow_dep_protobuf)
endif()

if(NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED GLOBAL)
    if(COMPUTE_FLOW_BUILD_HOST_TOOLS)
        set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION "${protobuf_host_install_dir}/bin/protoc")
        add_dependencies(protobuf::protoc protobuf_host_ep)
    else()
        set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION "${protobuf_install_dir}/bin/protoc")
        add_dependencies(protobuf::protoc compute_flow_dep_protobuf)
    endif()
endif()

# ---- Compatibility bridge (FindProtobuf variables) --------------------------------------
set(Protobuf_INCLUDE_DIR "${protobuf_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}" CACHE PATH "" FORCE)
set(Protobuf_INCLUDE_DIRS "${protobuf_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}" CACHE PATH "" FORCE)
set(Protobuf_LIBRARY "${protobuf_install_dir}/${CMAKE_INSTALL_LIBDIR}/libprotobuf.a" CACHE FILEPATH "" FORCE)
set(Protobuf_LIBRARIES "${protobuf_install_dir}/${CMAKE_INSTALL_LIBDIR}/libprotobuf.a" CACHE FILEPATH "" FORCE)
set(Protobuf_LITE_LIBRARY "${protobuf_install_dir}/${CMAKE_INSTALL_LIBDIR}/libprotobuf-lite.a" CACHE FILEPATH "" FORCE)
set(Protobuf_LITE_LIBRARIES
    "${protobuf_install_dir}/${CMAKE_INSTALL_LIBDIR}/libprotobuf-lite.a"
    CACHE FILEPATH
    ""
    FORCE
)

if(COMPUTE_FLOW_BUILD_HOST_TOOLS)
    set(Protobuf_PROTOC_EXECUTABLE "${protobuf_host_install_dir}/bin/protoc" CACHE FILEPATH "" FORCE)
else()
    set(Protobuf_PROTOC_EXECUTABLE "${protobuf_install_dir}/bin/protoc" CACHE FILEPATH "" FORCE)
endif()

# ---- Monolithic archive declaration -----------------------------------------------------
set(COMPUTE_FLOW_PROTOBUF_MONOLITHIC_ARCHIVES
    "${protobuf_install_dir}/${CMAKE_INSTALL_LIBDIR}/libprotobuf.a"
    "${protobuf_install_dir}/${CMAKE_INSTALL_LIBDIR}/libprotobuf-lite.a"
)
