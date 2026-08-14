//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "compute_flow/manifest.h"
#include "compute_flow/profiler/profiler_view.h"
#include "compute_flow/tensor_dict.h"

namespace compute_flow {

struct AcceleratorConfig;

/**
 * Common interface for AI runners
 */
class IRunner {
public:
    virtual ~IRunner() = default;

    virtual void setup(const ModelData& model,
                       ProfilerView profiler = {}) = 0;

    virtual const std::string& model_name() const = 0;

    virtual void map_input(std::string_view key, size_t idx) = 0;
    virtual void map_input(std::string_view key, std::string_view tensor_name) = 0;
    virtual void map_output(std::string_view key, size_t idx) = 0;
    virtual void map_output(std::string_view key, std::string_view tensor_name) = 0;
    virtual void set_input(const std::string& key, const TensorData& tensor) = 0;
    virtual TensorData get_output(const std::string& key) = 0;
    virtual TensorFormat get_input_format(const std::string& key) = 0;
    virtual TensorFormat get_output_format(const std::string& key) = 0;
    virtual void reset_states() = 0;
    virtual void run() = 0;
};

// Lightweight adapter that wraps an existing implementation
template <class Impl>
class RunnerAdapter : public virtual IRunner {
public:
    RunnerAdapter() = default;

    RunnerAdapter(const RunnerAdapter&) = delete;
    RunnerAdapter& operator=(const RunnerAdapter&) = delete;
    RunnerAdapter(RunnerAdapter&&) noexcept = default;
    RunnerAdapter& operator=(RunnerAdapter&&) noexcept = default;

    void setup(const ModelData& model, ProfilerView profiler = {}) override { impl_.setup(model, profiler); }
    const std::string& model_name() const override { return impl_.model_name(); }

    void map_input(std::string_view key, size_t idx) override { impl_.map_input(key, idx); }
    void map_input(std::string_view key, std::string_view tensor_name) override { impl_.map_input(key, tensor_name); }

    void map_output(std::string_view key, size_t idx) override { impl_.map_output(key, idx); }
    void map_output(std::string_view key, std::string_view tensor_name) override { impl_.map_output(key, tensor_name); }

    void set_input(const std::string& key, const TensorData& tensor) override { impl_.set_input(key, tensor); }
    TensorData get_output(const std::string& key) override { return impl_.get_output(key); }

    TensorFormat get_input_format(const std::string& key) override { return impl_.get_input_format(key); }
    TensorFormat get_output_format(const std::string& key) override { return impl_.get_output_format(key); }

    void reset_states() override { impl_.reset_states(); }

    void run() override { this->impl_.run(); }

    Impl& impl() { return impl_; }
    const Impl& impl() const { return impl_; }

protected:
    Impl impl_;
};

std::unique_ptr<IRunner> make_runner(RunnerType type);

}  // namespace compute_flow
