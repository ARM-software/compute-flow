#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/nlohmann_json/NlohmannJson.cmake — nlohmann/json (header-only JSON library)
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/URL here) ----------------------------------------
set(COMPUTE_FLOW_NLOHMANN_JSON_VERSION "3.12.0")
set(COMPUTE_FLOW_NLOHMANN_JSON_URL
    "https://github.com/nlohmann/json/archive/55f93686c01528224f448c19128836e7df245f72.tar.gz"
)
set(COMPUTE_FLOW_NLOHMANN_JSON_URL_HASH "SHA256=67f4cdd9ca930c9c1e130af4a437c7fc98fab77a2846fc2d2a14b4943831f8ef")

# ---- Try pre-installed (release / system) --------------------------------------------------
find_package(nlohmann_json ${COMPUTE_FLOW_NLOHMANN_JSON_VERSION} CONFIG QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "nlohmann_json: found pre-installed")
    add_custom_target(compute_flow_dep_nlohmann_json)
    return()
endif()

# ---- Build from source via ExternalProject -----------------------------------------------
message(STATUS "nlohmann_json: building from source via ExternalProject")

compute_flow_dep_install_dir(nlohmann_install nlohmann_json)

ExternalProject_Add(
    compute_flow_dep_nlohmann_json
    URL "${COMPUTE_FLOW_NLOHMANN_JSON_URL}"
    URL_HASH "${COMPUTE_FLOW_NLOHMANN_JSON_URL_HASH}"
    INSTALL_DIR "${nlohmann_install}"
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DJSON_BuildTests=OFF
        -DJSON_Install=ON
)

# ---- IMPORTED target (placeholder until EP builds) ---------------------------------------
file(MAKE_DIRECTORY "${nlohmann_install}/include")
add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
set_target_properties(
    nlohmann_json::nlohmann_json
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_install}/include"
)
add_dependencies(nlohmann_json::nlohmann_json compute_flow_dep_nlohmann_json)
