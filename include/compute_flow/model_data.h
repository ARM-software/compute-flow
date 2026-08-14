//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compute_flow/accelerator.h"
#include "compute_flow/lm_params.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace compute_flow {

enum class RunnerType {
    ExecuTorch = 0,
    LiteRT = 1,
    SentencePieceEncode = 2,
    HuggingFaceTokenizer = 3,
    ONNXRT = 4,
    MNN = 5,
    Unknown = 6
};

enum class LMRunnerType { Llama_cpp, Unknown };

struct ModelData {
    AcceleratorConfig                                            acc_config;
    RunnerType                                                   framework_type { RunnerType::Unknown };
    LMRunnerType                                                 lm_runner_type { LMRunnerType::Unknown };
    std::string                                                  model_name;
    std::string                                                  model_path;
    std::unordered_map<std::string, int64_t>                     inputs;
    std::unordered_map<std::string, int64_t>                     outputs;
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> states;

    ModelData& set_input(std::string name, int64_t id);
    ModelData& set_output(std::string name, int64_t id);
    ModelData& set_state(std::string name, int64_t id_in, int64_t id_out);
    bool     has_input(std::string_view name)  const noexcept;
    bool     has_output(std::string_view name) const noexcept;
    bool     has_state(std::string_view name) const noexcept;
    int64_t  input(std::string_view name)  const; // throws std::out_of_range if missing
    int64_t  output(std::string_view name) const; // throws std::out_of_range if missing
    std::pair<int64_t, int64_t> state(std::string_view name) const;

    // Just for LM runners
    LMParams lm_params;
};

using ConstantValue = std::variant<int64_t, int32_t, double, float, bool, std::string>;
using ConfigValue = ConstantValue;

}  // namespace compute_flow
