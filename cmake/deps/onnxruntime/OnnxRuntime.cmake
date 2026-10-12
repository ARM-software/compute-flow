#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/onnxruntime/OnnxRuntime.cmake — ONNX Runtime
# =========================================================================================
#
# Creates IMPORTED targets:
#   onnxruntime::onnxruntime_session — main ONNX Runtime static library (with all deps)
#   onnxruntime::onnxruntime         — convenience umbrella (INTERFACE, links to session)
#
# When building from source, uses a shim ExternalProject to install headers, libraries,
# and an onnxruntimeConfig.cmake package config.
#
# Depends on: compute_flow_dep_protobuf
#
# Resolution: find_package → ExternalProject (gated by from-source flag)
# =========================================================================================

include_guard(GLOBAL)

# ---- Monolithic archive exclude list -----------------------------------------------------
# KleidiAI is also provided by LiteRT/ExecuTorch (prioritized). Exclude ORT's copy to avoid
# duplicate symbols on Linux® (ar -M extracts all object files from each archive).
if(COMPUTE_FLOW_ENABLE_LITERT OR COMPUTE_FLOW_ENABLE_EXECUTORCH)
    set(COMPUTE_FLOW_ONNXRT_MONOLITHIC_EXCLUDE "libkleidiai.a")
endif()

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_ONNXRUNTIME_GIT_REPO "https://github.com/microsoft/onnxruntime.git")
set(COMPUTE_FLOW_ONNXRUNTIME_GIT_TAG "e6e048e073fd9e787195d809a66904990d48b082")

# ---- Try pre-installed (release / system) --------------------------------------------------
if(NOT COMPUTE_FLOW_BUILD_ONNXRUNTIME_FROM_SOURCE)
    find_package(onnxruntime CONFIG QUIET)

    if(onnxruntime_FOUND)
        message(STATUS "ONNX Runtime: found pre-installed")
        add_custom_target(compute_flow_dep_onnxruntime)

        # Monolithic archive declaration
        compute_flow_extract_target_archives(COMPUTE_FLOW_ONNXRT_MONOLITHIC_ARCHIVES
            onnxruntime::onnxruntime_session
        )

        return()
    endif()
endif()

# ---- Guard: require explicit from-source opt-in -----------------------------------------
if(NOT COMPUTE_FLOW_BUILD_ONNXRUNTIME_FROM_SOURCE)
    message(
        FATAL_ERROR
        "ONNX Runtime not found via find_package and from-source build not enabled.\n"
        "Set -DCOMPUTE_FLOW_BUILD_ONNXRUNTIME_FROM_SOURCE=ON."
    )
endif()

# ---- Build from source via ExternalProject -----------------------------------------------
message(STATUS "ONNX Runtime: building from source via ExternalProject")

compute_flow_dep_prefix_dir(onnxrt_prefix_dir "onnxruntime")
compute_flow_dep_install_dir(onnxrt_install_dir "onnxruntime")

# ---- Build byproducts from artifacts list ------------------------------------------------

include("${CMAKE_CURRENT_LIST_DIR}/Artifacts.cmake")

set(onnxrt_build_byproducts "<BINARY_DIR>/${COMPUTE_FLOW_ONNXRUNTIME_BUILD_MAIN_LIB_RELATIVE_PATH}")
foreach(
    rel
    IN
    LISTS
        COMPUTE_FLOW_ONNXRUNTIME_BUILD_COMPONENT_LIB_RELATIVE_PATHS
        COMPUTE_FLOW_ONNXRUNTIME_BUILD_THIRD_PARTY_LIB_RELATIVE_PATHS
)
    list(APPEND onnxrt_build_byproducts "<BINARY_DIR>/${rel}")
endforeach()

# ---- Main EP ----------------------------------------------------------------------------

ExternalProject_Add(
    onnxruntime_ep
    GIT_REPOSITORY "${COMPUTE_FLOW_ONNXRUNTIME_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_ONNXRUNTIME_GIT_TAG}"
    PREFIX "${onnxrt_prefix_dir}"
    SOURCE_SUBDIR cmake
    GIT_PROGRESS TRUE
    UPDATE_DISCONNECTED ON
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS:BOOL=OFF
        # ONNX Runtime options
        -Donnxruntime_BUILD_SHARED_LIB:BOOL=OFF -Donnxruntime_ENABLE_PYTHON:BOOL=OFF
        -Donnxruntime_BUILD_UNIT_TESTS:BOOL=ON # Needed for libre2
        -Donnxruntime_ENABLE_TRAINING:BOOL=OFF -Donnxruntime_USE_KLEIDIAI:BOOL=ON -Donnxruntime_USE_XNNPACK:BOOL=OFF
        # Help ORT find the protobuf package installed by compute_flow_dep_protobuf
        -DProtobuf_DIR:PATH=${COMPUTE_FLOW_PROTOBUF_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/protobuf
    BUILD_BYPRODUCTS ${onnxrt_build_byproducts}
    # Skip install; the shim handles it
    INSTALL_COMMAND ""
    DEPENDS compute_flow_dep_protobuf
)

# ---- Shim EP: install headers/libs + onnxruntimeConfig.cmake ----------------------------

set(onnxrt_shim_prefix_dir "${COMPUTE_FLOW_DEPS_PREFIX_DIR}/onnxruntime-shim")
set(onnxrt_source_dir "${onnxrt_prefix_dir}/src/onnxruntime_ep")
set(onnxrt_build_dir "${onnxrt_prefix_dir}/src/onnxruntime_ep-build")

ExternalProject_Add(
    onnxruntime_shim_ep
    PREFIX "${onnxrt_shim_prefix_dir}"
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/shim"
    DEPENDS onnxruntime_ep
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX:PATH=${onnxrt_install_dir}
        -DONNXRUNTIME_SOURCE_DIR:PATH=${onnxrt_source_dir} -DONNXRUNTIME_BUILD_DIR:PATH=${onnxrt_build_dir}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)

add_custom_target(compute_flow_dep_onnxruntime DEPENDS onnxruntime_shim_ep)

set(COMPUTE_FLOW_ONNXRT_INSTALL_DIR "${onnxrt_install_dir}")

# ---- Placeholder IMPORTED targets -------------------------------------------------------

file(MAKE_DIRECTORY "${onnxrt_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}")
file(MAKE_DIRECTORY "${onnxrt_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/session")

if(NOT TARGET onnxruntime::onnxruntime_session)
    add_library(onnxruntime::onnxruntime_session STATIC IMPORTED GLOBAL)
    set_target_properties(
        onnxruntime::onnxruntime_session
        PROPERTIES
            IMPORTED_LOCATION "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_session.a"
            INTERFACE_INCLUDE_DIRECTORIES
                "${onnxrt_install_dir}/${CMAKE_INSTALL_INCLUDEDIR};${onnxrt_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/session"
    )

    # The full set of component + third-party libs is discovered at shim-install time
    # and written into onnxruntimeExtras.cmake. At configure time we can only declare the
    # known fixed set. The monolithic archive step (which runs GLOB at build time) handles the rest.
    set(onnxrt_known_component_libs
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_optimizer.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_providers.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_util.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_framework.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_graph.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_lora.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_mlas.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_common.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_flatbuffers.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnx.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnx_proto.a"
        "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libre2.a"
    )

    find_package(Threads QUIET)
    set_property(
        TARGET onnxruntime::onnxruntime_session
        PROPERTY INTERFACE_LINK_LIBRARIES "${onnxrt_known_component_libs};protobuf::libprotobuf;Threads::Threads"
    )

    if(APPLE)
        set_property(
            TARGET onnxruntime::onnxruntime_session
            APPEND
            PROPERTY INTERFACE_LINK_LIBRARIES "-framework Foundation" "-framework CoreFoundation"
        )
    endif()

    if(UNIX AND NOT APPLE)
        set_property(TARGET onnxruntime::onnxruntime_session APPEND PROPERTY INTERFACE_LINK_LIBRARIES dl m)
    endif()

    add_dependencies(onnxruntime::onnxruntime_session onnxruntime_shim_ep)
endif()

if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime INTERFACE IMPORTED GLOBAL)
    target_link_libraries(onnxruntime::onnxruntime INTERFACE onnxruntime::onnxruntime_session)
endif()

# ---- Monolithic archive declaration -----------------------------------------------------
set(COMPUTE_FLOW_ONNXRT_MONOLITHIC_ARCHIVES
    "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime_session.a"
    ${onnxrt_known_component_libs}
)
# Abseil archive names vary by ORT version (~67 libs) — must glob at build time.
set(COMPUTE_FLOW_ONNXRT_MONOLITHIC_GLOB_DIR "${onnxrt_install_dir}/${CMAKE_INSTALL_LIBDIR}")
