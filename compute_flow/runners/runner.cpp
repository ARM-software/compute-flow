//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/runners/runner.h"

#include <stdexcept>
#include <string>

#ifdef COMPUTE_FLOW_HAS_EXECUTORCH
#include "runners/executorch/executorch_runner.h"
#endif
#ifdef COMPUTE_FLOW_HAS_LITERT
#include "runners/litert/litert_runner.h"
#endif
#ifdef COMPUTE_FLOW_HAS_ONNXRT
#include "runners/onnxrt/onnxrt_runner.h"
#endif
#ifdef COMPUTE_FLOW_HAS_MNN
#include "runners/mnn/mnn_runner.h"
#endif
#ifdef COMPUTE_FLOW_HAS_SENTENCEPIECE
#include "runners/sentencepiece_encode/sentencepiece_encode_runner.h"
#endif
#ifdef COMPUTE_FLOW_HAS_HF_TOKENIZER
#include "runners/huggingface_tokenizer/huggingface_tokenizer_runner.h"
#endif

namespace compute_flow {

namespace {
[[noreturn]] void throw_disabled_runner(const char* name) {
    throw std::runtime_error(
        std::string("Runner '") + name +
        "' is not available because the corresponding framework was disabled at build time.");
}
}  // namespace

std::unique_ptr<IRunner> make_runner(RunnerType type) {
    switch (type) {
        case RunnerType::ExecuTorch:
#ifdef COMPUTE_FLOW_HAS_EXECUTORCH
            return std::make_unique<RunnerAdapter<ExecuTorchRunner>>();
#else
            throw_disabled_runner("ExecuTorch");
#endif
        case RunnerType::LiteRT:
#ifdef COMPUTE_FLOW_HAS_LITERT
            return std::make_unique<RunnerAdapter<LiteRTRunner>>();
#else
            throw_disabled_runner("LiteRT");
#endif
        case RunnerType::SentencePieceEncode:
#ifdef COMPUTE_FLOW_HAS_SENTENCEPIECE
            return std::make_unique<RunnerAdapter<SentencePieceEncodeRunner>>();
#else
            throw_disabled_runner("SentencePiece");
#endif
        case RunnerType::HuggingFaceTokenizer:
#ifdef COMPUTE_FLOW_HAS_HF_TOKENIZER
            return std::make_unique<RunnerAdapter<HuggingFaceTokenizerRunner>>();
#else
            throw_disabled_runner("HuggingFace tokenizer");
#endif
        case RunnerType::ONNXRT:
#ifdef COMPUTE_FLOW_HAS_ONNXRT
            return std::make_unique<RunnerAdapter<ONNXRTRunner>>();
#else
            throw_disabled_runner("ONNX Runtime");
#endif
        case RunnerType::MNN:
#ifdef COMPUTE_FLOW_HAS_MNN
            return std::make_unique<RunnerAdapter<MNNRunner>>();
#else
            throw_disabled_runner("MNN");
#endif
        default:
            return nullptr;
    }
}

}  // namespace compute_flow
