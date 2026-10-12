//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef COMPUTE_FLOW_HAS_LITERT

#include "compute_flow/model_data.h"
#include "compute_flow/profiler/profiler_view.h"
#include "compute_flow/tensor_dict.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace compute_flow {

class LiteRTRunner {
public:
    LiteRTRunner();
    ~LiteRTRunner();
    LiteRTRunner(const LiteRTRunner&) = delete;
    LiteRTRunner& operator=(const LiteRTRunner&) = delete;
    LiteRTRunner(LiteRTRunner&&) noexcept;
    LiteRTRunner& operator=(LiteRTRunner&&) noexcept;

    void setup(const ModelData& model, ProfilerView profiler = {});
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
    enum class ExecutionMode { CompiledModel, Interpreter };
    struct CompiledModelState;
    struct InterpreterState;

    void setup_compiled_model(const ModelData& model, ProfilerView profiler);
    void setup_interpreter(const ModelData& model, ProfilerView profiler);
    void run_compiled_model();
    void run_interpreter();
    void set_input_compiled_model(const std::string& key, const TensorData& tensor);
    void set_input_interpreter(const std::string& key, const TensorData& tensor);
    void reset_states_compiled_model();
    void reset_states_interpreter();
    TensorData get_output_compiled_model(const std::string& key);
    TensorData get_output_interpreter(const std::string& key);
    void lock_output_compiled_model(size_t idx);
    void unlock_outputs_compiled_model();
    size_t find_input_idx_by_name_compiled_model(std::string_view name) const;
    size_t find_output_idx_by_name_compiled_model(std::string_view name) const;
    size_t find_input_idx_by_name_interpreter(std::string_view name) const;
    size_t find_output_idx_by_name_interpreter(std::string_view name) const;

    std::string _model_name;
    ModelData _compiled_model_data;
    ProfilerView _profiler;
    ExecutionMode _execution_mode {ExecutionMode::CompiledModel};
    std::unique_ptr<CompiledModelState> _compiled_model;
    std::unique_ptr<InterpreterState> _interpreter;
    std::vector<TensorData> _inputs;
    std::vector<TensorData> _outputs;
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> _states;
    std::unordered_map<std::string, size_t> _in_tensor_idx;
    std::unordered_map<std::string, size_t> _out_tensor_idx;
};

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_LITERT
