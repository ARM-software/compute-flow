#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Central list of ONNX Runtime static libraries the shim installs.
#
# Paths are relative to the ORT build directory (${ONNXRUNTIME_BUILD_DIR}).

set(COMPUTE_FLOW_ONNXRUNTIME_BUILD_MAIN_LIB_RELATIVE_PATH "libonnxruntime_session.a")

set(COMPUTE_FLOW_ONNXRUNTIME_BUILD_COMPONENT_LIB_RELATIVE_PATHS
    "libonnxruntime_optimizer.a"
    "libonnxruntime_providers.a"
    "libonnxruntime_util.a"
    "libonnxruntime_framework.a"
    "libonnxruntime_graph.a"
    "libonnxruntime_lora.a"
    "libonnxruntime_mlas.a"
    "libonnxruntime_common.a"
    "libonnxruntime_flatbuffers.a"
)

set(COMPUTE_FLOW_ONNXRUNTIME_BUILD_THIRD_PARTY_LIB_RELATIVE_PATHS
    "_deps/onnx-build/libonnx.a"
    "_deps/onnx-build/libonnx_proto.a"
    "_deps/re2-build/libre2.a"
    # Required when onnxruntime_USE_KLEIDIAI=ON (mlas calls into this archive).
    "_deps/kleidiai-build/libkleidiai.a"
)
