#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# deps/huggingface_tokenizer/HuggingfaceTokenizer.cmake — Hugging Face tokenizer (tokenizer.cpp)
# =========================================================================================
#
# Creates IMPORTED targets:
#   tokenizer_lib                          — main tokenizer static library
#   huggingface_tokenizer::huggingface_tokenizer — alias for tokenizer_lib
#   huggingface_tokenizer_onig             — Oniguruma static library dependency
#
# Also populates COMPUTE_FLOW_HF_TOKENIZER_TARGET with the resolved target name for the
# main CMakeLists.txt to link against.
#
# Resolution: find_package → ExternalProject fallback
# =========================================================================================

include_guard(GLOBAL)

# ---- Source coordinates (change version/tag here) ----------------------------------------
set(COMPUTE_FLOW_HF_TOKENIZER_GIT_REPO "https://github.com/wangzhaode/tokenizer.cpp.git")
set(COMPUTE_FLOW_HF_TOKENIZER_GIT_TAG "8638367b9ae48928561f9212477b7be2260ca7a7")

# ---- Try pre-installed (release / system) --------------------------------------------------
find_package(huggingface_tokenizer CONFIG QUIET)
if(huggingface_tokenizer_FOUND)
    message(STATUS "HuggingFace tokenizer: found pre-installed")
    add_custom_target(compute_flow_dep_huggingface_tokenizer)

    set(COMPUTE_FLOW_HF_TOKENIZER_TARGET "" CACHE STRING "CMake target name to link for HuggingFace tokenizer")
    if(COMPUTE_FLOW_HF_TOKENIZER_TARGET AND TARGET "${COMPUTE_FLOW_HF_TOKENIZER_TARGET}")
        set(COMPUTE_FLOW_HF_TOKENIZER_TARGET "${COMPUTE_FLOW_HF_TOKENIZER_TARGET}")
    elseif(TARGET huggingface_tokenizer::huggingface_tokenizer)
        set(COMPUTE_FLOW_HF_TOKENIZER_TARGET huggingface_tokenizer::huggingface_tokenizer)
    elseif(TARGET tokenizer_lib)
        set(COMPUTE_FLOW_HF_TOKENIZER_TARGET tokenizer_lib)
    elseif(TARGET tokenizer_lib::tokenizer_lib)
        set(COMPUTE_FLOW_HF_TOKENIZER_TARGET tokenizer_lib::tokenizer_lib)
    elseif(TARGET tokenizer)
        set(COMPUTE_FLOW_HF_TOKENIZER_TARGET tokenizer)
    elseif(TARGET tokenizer::tokenizer)
        set(COMPUTE_FLOW_HF_TOKENIZER_TARGET tokenizer::tokenizer)
    else()
        message(
            FATAL_ERROR
            "Found HuggingFace tokenizer via find_package(huggingface_tokenizer CONFIG), but no known CMake target exists. "
            "Set -DCOMPUTE_FLOW_HF_TOKENIZER_TARGET=<target> (e.g. tokenizer_lib)."
        )
    endif()

    # Monolithic archive declaration
    compute_flow_extract_target_archives(COMPUTE_FLOW_HF_TOKENIZER_MONOLITHIC_ARCHIVES
        ${COMPUTE_FLOW_HF_TOKENIZER_TARGET}
    )
    if(TARGET huggingface_tokenizer_onig)
        compute_flow_extract_target_archives(_hf_onig_archives huggingface_tokenizer_onig)
        list(APPEND COMPUTE_FLOW_HF_TOKENIZER_MONOLITHIC_ARCHIVES ${_hf_onig_archives})
    endif()

    return()
endif()

# ---- Build from source via ExternalProject -----------------------------------------------
message(STATUS "HuggingFace tokenizer: building from source via ExternalProject")

compute_flow_dep_prefix_dir(hf_tokenizer_prefix_dir "huggingface_tokenizer")
compute_flow_dep_install_dir(hf_tokenizer_install_dir "huggingface_tokenizer")

set(hf_tokenizer_include_dir "${hf_tokenizer_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}")
set(hf_tokenizer_lib_dir "${hf_tokenizer_install_dir}/${CMAKE_INSTALL_LIBDIR}")
set(hf_tokenizer_tokenizer_lib
    "${hf_tokenizer_lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}tokenizer_lib${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
set(hf_tokenizer_onig_lib "${hf_tokenizer_lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}onig${CMAKE_STATIC_LIBRARY_SUFFIX}")

ExternalProject_Add(
    compute_flow_dep_huggingface_tokenizer
    GIT_REPOSITORY "${COMPUTE_FLOW_HF_TOKENIZER_GIT_REPO}"
    GIT_TAG "${COMPUTE_FLOW_HF_TOKENIZER_GIT_TAG}"
    PREFIX "${hf_tokenizer_prefix_dir}"
    INSTALL_DIR "${hf_tokenizer_install_dir}"
    GIT_PROGRESS TRUE
    UPDATE_DISCONNECTED ON
    CMAKE_ARGS
        ${COMPUTE_FLOW_EP_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=<BINARY_DIR>/lib -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=<BINARY_DIR>/lib
    INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory "${hf_tokenizer_include_dir}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${hf_tokenizer_lib_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/include/tokenizer.hpp "${hf_tokenizer_include_dir}/tokenizer.hpp"
    COMMAND
        ${CMAKE_COMMAND} -E copy
        <BINARY_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tokenizer_lib${CMAKE_STATIC_LIBRARY_SUFFIX}
        "${hf_tokenizer_lib_dir}"
    COMMAND
        ${CMAKE_COMMAND} -E copy <BINARY_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}onig${CMAKE_STATIC_LIBRARY_SUFFIX}
        "${hf_tokenizer_lib_dir}"
)

# ---- IMPORTED targets (placeholder until EP builds) ---------------------------------------
file(MAKE_DIRECTORY "${hf_tokenizer_include_dir}")

if(NOT TARGET huggingface_tokenizer_onig)
    add_library(huggingface_tokenizer_onig STATIC IMPORTED GLOBAL)
    set_target_properties(
        huggingface_tokenizer_onig
        PROPERTIES
            IMPORTED_LOCATION "${hf_tokenizer_onig_lib}"
            INTERFACE_INCLUDE_DIRECTORIES "${hf_tokenizer_include_dir}"
    )
    add_dependencies(huggingface_tokenizer_onig compute_flow_dep_huggingface_tokenizer)
endif()

if(NOT TARGET tokenizer_lib)
    add_library(tokenizer_lib STATIC IMPORTED GLOBAL)
    set_target_properties(
        tokenizer_lib
        PROPERTIES
            IMPORTED_LOCATION "${hf_tokenizer_tokenizer_lib}"
            INTERFACE_INCLUDE_DIRECTORIES "${hf_tokenizer_include_dir}"
    )
    target_link_libraries(tokenizer_lib INTERFACE huggingface_tokenizer_onig)
    add_dependencies(tokenizer_lib compute_flow_dep_huggingface_tokenizer)
endif()

if(NOT TARGET huggingface_tokenizer::huggingface_tokenizer)
    add_library(huggingface_tokenizer::huggingface_tokenizer ALIAS tokenizer_lib)
endif()

# ---- Target resolution for main CMakeLists.txt --------------------------------------------
set(COMPUTE_FLOW_HF_TOKENIZER_TARGET "" CACHE STRING "CMake target name to link for HuggingFace tokenizer")
if(COMPUTE_FLOW_HF_TOKENIZER_TARGET AND TARGET "${COMPUTE_FLOW_HF_TOKENIZER_TARGET}")
    set(COMPUTE_FLOW_HF_TOKENIZER_TARGET "${COMPUTE_FLOW_HF_TOKENIZER_TARGET}")
elseif(TARGET tokenizer_lib)
    set(COMPUTE_FLOW_HF_TOKENIZER_TARGET tokenizer_lib)
elseif(TARGET huggingface_tokenizer::huggingface_tokenizer)
    set(COMPUTE_FLOW_HF_TOKENIZER_TARGET huggingface_tokenizer::huggingface_tokenizer)
else()
    message(
        FATAL_ERROR
        "HuggingFace tokenizer target not found. "
        "Set -DCOMPUTE_FLOW_HF_TOKENIZER_TARGET=<target> (e.g. tokenizer_lib) to proceed."
    )
endif()

# ---- Monolithic archive declaration -----------------------------------------------------
set(COMPUTE_FLOW_HF_TOKENIZER_MONOLITHIC_ARCHIVES "${hf_tokenizer_tokenizer_lib}" "${hf_tokenizer_onig_lib}")
