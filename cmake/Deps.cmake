#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# Deps.cmake — Dependency orchestrator for Compute Flow
# =========================================================================================
#
# Includes the modular per-dependency modules under cmake/deps/ in the correct order.
#
# Resolution strategy (per dependency):
#   1. find_package() — discovers pre-installed packages (e.g. via Conan or system)
#   2. ExternalProject fallback — builds from source when find_package fails
#
# Conan (COMPUTE_FLOW_USE_CONAN) and per-dep from-source flags can coexist: Conan populates
# CMAKE_PREFIX_PATH, then each dep's find_package() discovers Conan packages. Per-dep
# COMPUTE_FLOW_BUILD_<DEP>_FROM_SOURCE flags can override individual deps.
#
# Expected variables (set by the root CMakeLists.txt):
#   COMPUTE_FLOW_USE_CONAN
#   COMPUTE_FLOW_ENABLE_LITERT
#   COMPUTE_FLOW_ENABLE_ONNXRUNTIME
#   COMPUTE_FLOW_ENABLE_EXECUTORCH
#   COMPUTE_FLOW_ENABLE_MNN
#   COMPUTE_FLOW_ENABLE_SENTENCEPIECE
#   COMPUTE_FLOW_ENABLE_HF_TOKENIZER
#   COMPUTE_FLOW_ENABLE_LLAMACPP
#   COMPUTE_FLOW_PYBIND_ENABLED
# =========================================================================================

# ---- Shared EP infrastructure (must come first) -----------------------------------------
include(cmake/deps/Common.cmake)

function(compute_flow_enable_source_dependencies)
    set(COMPUTE_FLOW_USE_CONAN OFF CACHE BOOL "Fetch deps via Conan package manager" FORCE)
    foreach(
        source_option
        IN
        ITEMS
            COMPUTE_FLOW_BUILD_PROTOBUF_FROM_SOURCE
            COMPUTE_FLOW_BUILD_LITERT_FROM_SOURCE
            COMPUTE_FLOW_BUILD_EXECUTORCH_FROM_SOURCE
            COMPUTE_FLOW_BUILD_ONNXRUNTIME_FROM_SOURCE
            COMPUTE_FLOW_BUILD_MNN_FROM_SOURCE
            COMPUTE_FLOW_BUILD_SENTENCEPIECE_FROM_SOURCE
    )
        set(${source_option} ON CACHE BOOL "Force-build dependency from source" FORCE)
    endforeach()
endfunction()

# ---- Conan setup (optional — populates CMAKE_PREFIX_PATH) --------------------------------
if(COMPUTE_FLOW_USE_CONAN)
    if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/Conan.cmake")
        include(cmake/deps/Conan.cmake)
    else()
        message(STATUS "Conan packages are unavailable; building dependencies from source")
        compute_flow_enable_source_dependencies()
    endif()
endif()

# ---- From-source banner ------------------------------------------------------------------
if(
    COMPUTE_FLOW_BUILD_PROTOBUF_FROM_SOURCE
    OR COMPUTE_FLOW_BUILD_LITERT_FROM_SOURCE
    OR COMPUTE_FLOW_BUILD_EXECUTORCH_FROM_SOURCE
    OR COMPUTE_FLOW_BUILD_ONNXRUNTIME_FROM_SOURCE
    OR COMPUTE_FLOW_BUILD_MNN_FROM_SOURCE
    OR COMPUTE_FLOW_BUILD_SENTENCEPIECE_FROM_SOURCE
)
    message(STATUS "")
    message(STATUS "================================================================")
    message(STATUS "Compute Flow: building one or more dependencies from source")
    message(STATUS "")
    message(STATUS "This may take a while (20-30 mins) depending on your network")
    message(STATUS "connection and system performance.")
    message(STATUS "")
    message(STATUS "  cmake --build <build-dir> --parallel 8")
    message(STATUS "================================================================")
    message(STATUS "")
endif()

# =========================================================================================
# Per-dependency modules (order matters for inter-dependency relationships)
# =========================================================================================

# ---- Always-on lightweight deps ----------------------------------------------------------
include(cmake/deps/nlohmann_json/NlohmannJson.cmake)
include(cmake/deps/libnpy/Libnpy.cmake)

# ---- pybind11 (when building Python bindings, FetchContent exception) --------------------
if(COMPUTE_FLOW_PYBIND_ENABLED)
    include(cmake/deps/pybind11/Pybind11.cmake)
endif()

# ---- Protobuf (needed by ONNX Runtime and SentencePiece — must come before both) ---------
if(COMPUTE_FLOW_ENABLE_ONNXRUNTIME OR COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    include(cmake/deps/protobuf/Protobuf.cmake)
endif()

# ---- FlatBuffers host tools (cross-compile only, needed by LiteRT — must come before it) -
if(COMPUTE_FLOW_ENABLE_LITERT)
    include(cmake/deps/flatbuffers/FlatBuffers.cmake)
endif()

# ---- Heavy framework deps (gated by per-framework toggles) -------------------------------
if(COMPUTE_FLOW_ENABLE_LITERT)
    include(cmake/deps/litert/LiteRT.cmake)
endif()

if(COMPUTE_FLOW_ENABLE_EXECUTORCH)
    include(cmake/deps/executorch/ExecuTorch.cmake)
endif()

if(COMPUTE_FLOW_ENABLE_ONNXRUNTIME)
    include(cmake/deps/onnxruntime/OnnxRuntime.cmake)
endif()

if(COMPUTE_FLOW_ENABLE_MNN)
    include(cmake/deps/mnn/MNN.cmake)
endif()

if(COMPUTE_FLOW_ENABLE_SENTENCEPIECE)
    include(cmake/deps/sentencepiece/SentencePiece.cmake)
endif()

if(COMPUTE_FLOW_ENABLE_HF_TOKENIZER)
    include(cmake/deps/huggingface_tokenizer/HuggingfaceTokenizer.cmake)
endif()

# ---- Lightweight framework deps ---------------------------------------------------------
if(COMPUTE_FLOW_ENABLE_LLAMACPP)
    include(cmake/deps/llama_cpp/LlamaCpp.cmake)
endif()
