//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/lm_runners/lm_runner.h"

#include <stdexcept>
#include <string>

#ifdef COMPUTE_FLOW_HAS_LLAMACPP
#include "compute_flow/lm_runners/llama_cpp/llama_cpp_lm_runner.h"
#endif

namespace compute_flow {

std::unique_ptr<ILMRunner> make_lm_runner(LMRunnerType type) {
    switch (type) {
        case LMRunnerType::Llama_cpp:
#ifdef COMPUTE_FLOW_HAS_LLAMACPP
            return std::make_unique<LMRunnerAdapter<LlamaCppLMRunner>>();
#else
            throw std::runtime_error(
                "LM runner 'llama.cpp' is not available because the corresponding "
                "framework was disabled at build time.");
#endif
        default:
            return nullptr;
    }
}

}  // namespace compute_flow
