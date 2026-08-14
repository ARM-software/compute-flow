//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// ADD MACRO TO PREVENT BUILDING THIS FILE IF LITERT MACRO IS NOT PASSED

#pragma once

#ifdef COMPUTE_FLOW_HAS_ONNXRT

#include "compute_flow/accelerator.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/model_data.h"
#include "compute_flow/profiler/profiler_view.h"
#include "compute_flow/tensor_dict.h"

#include <memory>
#include <optional>
#include <string>
#include <functional>

#include <onnxruntime_cxx_api.h>

#define ONNXRT_CHECK(x)                                 \
    if (!(x)) {                                                 \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);\
        exit(1);                                                \
    }

namespace compute_flow {

class ONNXRTRunner {
public:
    ONNXRTRunner();
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
    struct StateInfo {
        std::vector<int64_t> initial_shape;  // original tensor shape (may include -1)
        std::vector<int64_t> shape;          // current tensor shape
        void* data;                          // pointer to the tensor data
        DataType dt;                         // tensor element type
        std::optional<Ort::Value> value;     // persistent storage for state inputs
    };

    void initialize_cpu_delegate(const CpuOptions& cpu_options);
    void set_initial_state_info();
    static inline DataType convert_onnxrt_dt_to_internal_dt(ONNXTensorElementDataType type);
    static inline int64_t get_num_elems(const std::vector<int64_t>& values);
    static inline int64_t get_elem_size(DataType t);
    static inline std::vector<int64_t> convert_onnxrt_shape_to_internal_shape(const std::vector<int64_t>& dims);
    inline size_t find_input_idx_by_name(std::string_view name) const;
    inline size_t find_output_idx_by_name(std::string_view name) const;
    inline bool is_state(const std::string& key);

    std::string                                            _model_name;
    ProfilerView                                           _profiler;
    Ort::Env                                               _env;
    std::unique_ptr<Ort::Session>                          _model;
    Ort::SessionOptions                                    _session_opts;
    Ort::MemoryInfo                                        _memory_info;

    std::vector<TensorData>                                      _inputs;
    std::vector<TensorData>                                      _outputs;
    std::unordered_map<std::string, size_t>                      _in_tensor_idx;
    std::unordered_map<std::string, size_t>                      _out_tensor_idx;
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> _states;
    std::unordered_map<std::string, StateInfo>                   _states_info;
    std::vector<Ort::Value>                                      _last_outputs;
};

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_ONNXRT
