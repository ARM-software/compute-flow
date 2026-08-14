//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compute_flow/bricks/brick_interface_v1.h"
#include "compute_flow/profiler/profiler_view.h"

#include <string>

namespace compute_flow::bricks {

/**
 * @brief Convenience base class for brick implementations.
 *
 * This class enforces the brick lifecycle and manages profiling automatically.
 * Derived classes implement the protected *_impl() methods.
 */
class BrickBaseV1 : public IBrickV1 {
public:
    std::string name() const final {
        return _name;
    }

    void setup(const ComputeFlowManifest& manifest,
               const BrickOptions& options = {}) final {
        _name.clear();
        _options = options;

        _profiler.reset();
        _profiler.configure(options.profiling);

        auto scope = _profiler.measure("setup");
        const BrickInfo& info = manifest.get_brick_info();

        // Validate the manifest before loading models or preparing brick state
        COMPUTE_FLOW_THROW_IF_ERROR(validate(manifest),
                                    "Manifest validation failed for brick " + info.name);
        // Load all the models
        load_models(manifest);
        // Load all the metadata and allocate intermediate buffers
        prepare(manifest);
        // Make sure the state is reset before the first call
        reset_states_preserving_profile();

        if(_options.runtime.enable_warmup) {
            warmup();

            // Make sure the state is reset after warmup
            reset_states_preserving_profile();
        }

        _name = info.name;
    }

    void update_config(const ComputeFlowManifest& manifest) final {
        _profiler.reset();
        auto scope = _profiler.measure("update_config");
        update_config_impl(manifest, ProfilerView{_profiler});
    }

    TensorDict run(const TensorDict& input) final {
        auto scope = _profiler.measure("run");
        return run_impl(input, ProfilerView{_profiler});
    }

    void reset_states() final {
        reset_states_impl(ProfilerView{_profiler});
        _profiler.reset();
    }

    void write_profile(std::ostream& os) const final {
        _profiler.write(os);
    }

protected:
    /**
     * @brief Load and initialize the models required by the brick.
     *
     * This method is called during `setup()` after the profiler has been
     * configured and reset.
     *
     * Implementers should load model assets, initialize runners, and
     * allocate model-related resources.
     *
     * @param manifest Manifest describing the brick configuration and assets.
     * @param profiler Restricted profiler access for measuring internal scopes.
     */
    virtual void load_models_impl(const ComputeFlowManifest& manifest,
                                  ProfilerView profiler) = 0;

    /**
     * @brief Perform one-time brick preparation.
     *
     * This method is called during `setup()` after models have been loaded.
     *
     * Implementers should perform initialization steps that must happen once
     * before execution, such as tensor preparation.
     *
     * @param manifest Manifest describing the brick configuration.
     * @param profiler Restricted profiler access for measuring internal scopes.
     */
    virtual void prepare_impl(const ComputeFlowManifest& manifest,
                              ProfilerView profiler) = 0;

    /**
     * @brief Execute optional warmup runs.
     *
     * This method is called during `setup()` only when warmup is enabled
     * through `RuntimeOptions`.
     *
     * Implementers may execute one or more dummy runs to reduce first-run
     * latency.
     *
     * @param profiler Restricted profiler access for measuring internal scopes.
     */
    virtual void warmup_impl(ProfilerView profiler) = 0;

    /**
     * @brief Update brick configuration from the provided manifest.
     *
     * This method is called by `update_config()` and allows lightweight
     * configuration updates without reloading models or rebuilding the
     * entire brick state.
     *
     * Implementers should update only runtime-configurable fields.
     *
     * @param manifest Manifest containing the updated configuration.
     * @param profiler Restricted profiler access for measuring internal scopes.
     */
    virtual void update_config_impl(const ComputeFlowManifest& manifest,
                                    ProfilerView profiler) = 0;

    /**
     * @brief Execute the brick on the provided input tensors.
     *
     * This method contains the core execution logic of the brick.
     *
     * @param input Input tensors.
     * @param profiler Restricted profiler access for measuring internal scopes.
     *
     * @return Output tensors produced by the brick.
     */
    virtual TensorDict run_impl(const TensorDict& input,
                                ProfilerView profiler) = 0;

    /**
     * @brief Reset the internal state of the brick.
     *
     * This method is called by `reset_states()` before the profiler
     * lifecycle is reset.
     *
     * Implementers should restore the brick to its initial execution state
     * without releasing persistent resources allocated during `setup()`.
     *
     * @param profiler Restricted profiler access for measuring internal scopes.
     */
    virtual void reset_states_impl(ProfilerView profiler) = 0;

private:
    void load_models(const ComputeFlowManifest& manifest) {
        auto scope = _profiler.measure("load_models");
        load_models_impl(manifest, ProfilerView{_profiler});
    }

    void prepare(const ComputeFlowManifest& manifest) {
        auto scope = _profiler.measure("prepare");
        prepare_impl(manifest, ProfilerView{_profiler});
    }

    void warmup() {
        auto scope = _profiler.measure("warmup");
        warmup_impl(ProfilerView{_profiler});
    }

    void reset_states_preserving_profile() {
        reset_states_impl(ProfilerView{_profiler});
    }

    Profiler     _profiler;
    BrickOptions _options;
    std::string  _name;
};

} // namespace compute_flow::bricks
