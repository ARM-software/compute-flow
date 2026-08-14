//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef COMPUTE_FLOW_HAS_EXECUTORCH

#include "compute_flow/accelerator.h"
#include "compute_flow/model_data.h"
#include "compute_flow/profiler/profiler_view.h"
#include "compute_flow/tensor_dict.h"

#include <memory>
#include <string>
#include <functional>

#if defined(ET_USE_THREADPOOL)
#include "executorch/extension/threadpool/cpuinfo_utils.h"
#include "executorch/extension/threadpool/threadpool.h"
#endif
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/platform/log.h"

#define EXECUTORCH_CHECK(x)                                 \
    if (!(x)) {                                                 \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);\
        exit(1);                                                \
    }

namespace compute_flow {

class ExecuTorchRunner {
public:
    ExecuTorchRunner();
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

    void initialize_cpu_delegate(const CpuOptions& cpu_options);
    static inline DataType convert_et_dt_to_internal_dt(executorch::aten::ScalarType t);
    static inline executorch::aten::ScalarType convert_internal_dt_to_et_dt(DataType t);
    static inline std::vector<int64_t>
    convert_et_shape_to_internal_shape(const std::vector<executorch::aten::SizesType>& dims);
    inline size_t find_input_idx_by_name(std::string_view name) const;
    inline size_t find_output_idx_by_name(std::string_view name) const;

    std::string                                            _model_name;
    ProfilerView                                           _profiler;
    std::unique_ptr<executorch::extension::Module>         _model;
    std::vector<TensorData>                                _inputs;
    std::vector<TensorData>                                _outputs;
    std::unordered_map<std::string, size_t>                _in_tensor_idx;
    std::unordered_map<std::string, size_t>                _out_tensor_idx;
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> _states;
    std::vector<std::shared_ptr<executorch::aten::Tensor>> _aten_inputs;
    std::vector<executorch::runtime::EValue>               _et_inputs;
    std::vector<executorch::runtime::EValue>               _et_outputs;
    std::vector<std::vector<uint8_t>>                      _raw_inputs;
    std::vector<std::vector<uint8_t>>                      _raw_outputs;
};

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_EXECUTORCH
