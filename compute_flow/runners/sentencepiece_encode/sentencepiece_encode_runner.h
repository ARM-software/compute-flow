//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef COMPUTE_FLOW_HAS_SENTENCEPIECE

#include "compute_flow/accelerator.h"
#include "compute_flow/model_data.h"
#include "compute_flow/profiler/profiler_view.h"
#include "compute_flow/tensor_dict.h"

#include <memory>
#include <string>
#include <functional>

#include <sentencepiece_processor.h>

#define SENTENCEPIECE_ENCODE_CHECK(x)                                 \
    if (!(x)) {                                                 \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);\
        exit(1);                                                \
    }

namespace compute_flow {

class SentencePieceEncodeRunner {
public:
    SentencePieceEncodeRunner();
    void setup(const std::string& model_name,
                  const std::string& model_path,
                  const AcceleratorConfig& accelerator,
                  ProfilerView profiler = {});
    void setup(const ModelData& model,
               ProfilerView profiler = {});
    void run();
    const std::string& model_name() const;
    void map_input(std::string_view key, size_t idx);
    void map_input(std::string_view key, std::string_view tensor_name);
    void map_output(std::string_view key, size_t idx);
    void map_output(std::string_view key, std::string_view tensor_name);
    void set_input(const std::string& key, const TensorData& tensor);
    void reset_states();
    TensorData get_output(const std::string& key);
    TensorFormat get_input_format(const std::string& key);
    TensorFormat get_output_format(const std::string& key);

private:
    sentencepiece::SentencePieceProcessor   _sp;
    std::string                             _model_name;
    ProfilerView                            _profiler;
    std::vector<TensorData>                 _inputs;
    std::vector<TensorData>                 _outputs;
    std::unordered_map<std::string, size_t> _in_tensor_idx;
    std::unordered_map<std::string, size_t> _out_tensor_idx;
    std::string                             _in;
    std::vector<int32_t>                    _out;
};

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_SENTENCEPIECE
