//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "profiler.h"

#include <string_view>

namespace compute_flow {

/**
 * @brief Restricted profiler access for framework components.
 *
 * ProfilerView provides limited access to a Profiler instance.
 * It allows measuring profiling scopes and inspecting profiling
 * state, while intentionally preventing lifecycle management
 * operations such as reset, configure, or write.
 *
 * This class is designed to be propagated across framework layers
 * (e.g. bricks, runners, helper utilities) without exposing the
 * full Profiler API.
 */
class ProfilerView final {
public:
    /**
     * @brief Construct a restricted profiler view.
     */
    ProfilerView() = default;

    /**
     * @brief Construct a restricted profiler view.
     *
     * @param profiler Reference to the underlying profiler.
     */
    explicit ProfilerView(Profiler& profiler)
        : _profiler(&profiler) {}

    /**
     * @brief Measure a named profiling scope.
     *
     * The returned scope object automatically stops profiling
     * when destroyed (RAII).
     *
     * @param name Scope name.
     *
     * @return Profiling scope handle.
     */
    [[nodiscard]] Profiler::Scope measure(std::string_view name) {
        if (_profiler == nullptr) {
            return {};
        }
        return _profiler->measure(name);
    }

    /**
     * @brief Measure one named model invocation.
     */
    [[nodiscard]] Profiler::Scope measure_model(std::string_view runner, std::string_view name) {
        if (_profiler == nullptr) {
            return {};
        }
        return _profiler->measure_model(runner, name);
    }

    /**
     * @brief Check whether profiling is enabled.
     */
    bool is_enabled() const {
        return _profiler != nullptr && _profiler->is_enabled();
    }

    /**
     * @brief Check whether operation-level profiling is enabled.
     */
    bool is_per_op_profiling_enabled() const {
        return _profiler != nullptr && _profiler->is_per_op_profiling_enabled();
    }

    /**
     * @brief Check whether external timeline annotations are enabled.
     */
    bool is_timeline_annotations_enabled() const {
        return _profiler != nullptr && _profiler->is_timeline_annotations_enabled();
    }

    /**
     * @brief Start a named runner timer.
     */
    void start_timer(std::string_view runner, std::string_view name) {
        if (_profiler != nullptr) {
            _profiler->start_timer(runner, name);
        }
    }

    /**
     * @brief End a named runner timer.
     */
    void end_timer(std::string_view runner, std::string_view name) {
        if (_profiler != nullptr) {
            _profiler->end_timer(runner, name);
        }
    }

    /**
     * @brief Add a nested profiling sample under a runner timer.
     */
    void add_nested_sample(std::string_view runner,
                           std::string_view parent,
                           std::string_view name,
                           long long elapsed_ns) {
        if (_profiler != nullptr) {
            _profiler->add_nested_sample(runner, parent, name, elapsed_ns);
        }
    }

private:
    Profiler* _profiler = nullptr;
};

} // namespace compute_flow
