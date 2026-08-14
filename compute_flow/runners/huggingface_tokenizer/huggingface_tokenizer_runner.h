//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef COMPUTE_FLOW_HAS_HF_TOKENIZER

#include "compute_flow/accelerator.h"
#include "compute_flow/model_data.h"
#include "compute_flow/profiler/profiler_view.h"
#include "compute_flow/tensor_dict.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tokenizer.hpp>

namespace compute_flow {

class HuggingFaceTokenizerRunner {
public:
    HuggingFaceTokenizerRunner() = default;
    void setup(const ModelData& model, ProfilerView profiler = {});
    const std::string& model_name() const;
    void map_input(std::string_view key, size_t idx);
    void map_input(std::string_view key, std::string_view tensor_name);
    void map_output(std::string_view key, size_t idx);
    void map_output(std::string_view key, std::string_view tensor_name);
    void set_input(const std::string& key, const TensorData& tensor);
    TensorData get_output(const std::string& key);
    TensorFormat get_input_format(const std::string& key);
    TensorFormat get_output_format(const std::string& key);
    void reset_states();
    void run();

private:
    std::shared_ptr<tokenizer::PreTrainedTokenizer> _tokenizer;
    std::string                                     _model_name;
    ProfilerView                                    _profiler;
    std::vector<TensorData>                         _inputs;
    std::vector<TensorData>                         _outputs;
    std::unordered_map<std::string, size_t>         _in_tensor_idx;
    std::unordered_map<std::string, size_t>         _out_tensor_idx;
    std::string                                     _in;
    std::vector<int32_t>                            _out;
};

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_HF_TOKENIZER
