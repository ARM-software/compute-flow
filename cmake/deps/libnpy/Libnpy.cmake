#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/libnpy/Libnpy.cmake — libnpy (header-only NumPy file I/O library)
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/URL here) ----------------------------------------
set(COMPUTE_FLOW_LIBNPY_VERSION "1.0.1")
set(COMPUTE_FLOW_LIBNPY_URL "https://github.com/llohse/libnpy/archive/refs/tags/v${COMPUTE_FLOW_LIBNPY_VERSION}.zip")
set(COMPUTE_FLOW_LIBNPY_URL_HASH "SHA256=3c9864c1bf7af08c88f2056478a2c6d2539674673bab9f9b4a0042a84a857f19")

# ---- Try pre-installed (release / system) --------------------------------------------------
find_package(libnpy CONFIG QUIET)
if(libnpy_FOUND AND TARGET libnpy::libnpy)
    message(STATUS "libnpy: found pre-installed")
    add_custom_target(compute_flow_dep_libnpy)
    return()
endif()

# ---- Build from source via ExternalProject -----------------------------------------------
# libnpy is header-only with no CMake install support, so we skip configure/build
# and just copy the include directory.
message(STATUS "libnpy: installing headers from source via ExternalProject")

compute_flow_dep_install_dir(libnpy_install libnpy)

ExternalProject_Add(
    compute_flow_dep_libnpy
    URL "${COMPUTE_FLOW_LIBNPY_URL}"
    URL_HASH "${COMPUTE_FLOW_LIBNPY_URL_HASH}"
    INSTALL_DIR "${libnpy_install}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "${CMAKE_COMMAND}" -E copy_directory <SOURCE_DIR>/include <INSTALL_DIR>/include
)

# ---- IMPORTED targets (placeholder until EP installs) ------------------------------------
file(MAKE_DIRECTORY "${libnpy_install}/include")
add_library(libnpy INTERFACE IMPORTED GLOBAL)
set_target_properties(libnpy PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${libnpy_install}/include")
add_dependencies(libnpy compute_flow_dep_libnpy)

add_library(libnpy::libnpy ALIAS libnpy)
