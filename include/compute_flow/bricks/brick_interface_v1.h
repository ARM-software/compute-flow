//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <string>
#include <string_view>

#include "compute_flow/check.h"
#include "compute_flow/manifest.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runtime.h"
#include "compute_flow/tensor_dict.h"

/**
 * @file
 * @brief Common interface for Compute Flow bricks.
 *
 * @section design_goals Design Goals
 * - Minimize misuse by keeping the API simple and limiting configuration knobs
 * - Each brick includes a built-in profiler for easy performance evaluation,
 *   especially during development
 * - The profiler should be straightforward to use with minimal setup and hard to misuse
 */

namespace compute_flow::bricks {

/**
 * @brief Aggregated configuration options for a brick.
 */
struct BrickOptions {
    /**
     * @brief Runtime execution options.
     */
    RuntimeOptions runtime;

    /**
     * @brief Profiling options.
     */
    ProfilingOptions profiling;
};

/**
 * @brief Common interface for Compute Flow bricks.
 *
 * A brick represents a self-contained computational unit that:
 * - is configured via a manifest and runtime options,
 * - processes input tensors to produce output tensors,
 * - may maintain internal state across invocations.
 *
 * Implementations are expected to be efficient by default and
 * require minimal configuration.
 */
class IBrickV1 {
public:
    virtual ~IBrickV1() = default;

    /**
     * @brief Get the brick API implemented by the brick.
     *
     * @return Brick API identifier declared by the brick metadata, such as `tts.v1`.
     */
    virtual std::string_view api() const noexcept = 0;

    /**
     * @brief Get the unique name of the brick.
     *
     * @return Human-readable identifier for the brick instance.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Validate the provided manifest.
     *
     * Implementations should verify that the manifest is compatible
     * with the brick (e.g. required assets, models, metadata).
     *
     * @param manifest The manifest describing the brick configuration.
     * @return ValidationResult describing success or the first validation error.
     *
     */
    virtual ValidationResult validate(const ComputeFlowManifest& manifest) const = 0;

    /**
     * @brief Prepare the brick for execution.
     *
     * This method initializes internal resources based on the provided
     * manifest and options. It may also perform optional warmup runs.
     *
     * @param manifest The manifest describing the brick configuration.
     * @param options Configuration options for runtime and profiling.
     *
     * @warning Must be called before run(). Calling run() prior to setup()
     *          results in undefined behavior.
     */
    virtual void setup(const ComputeFlowManifest& manifest, const BrickOptions& options = {}) = 0;

    /**
     * @brief Update configuration fields from the provided manifest.
     *
     * This method performs a lightweight update (a.k.a. fast setup) by overriding
     * the brick configuration fields after `setup()` has been called.
     *
     * If only configuration fields in the manifest have changed, this method avoids
     * reloading models and enables faster reinitialization.
     *
     * @param manifest The manifest describing the brick configuration.
     *
     * @warning Must be called after `setup()` and before `run()`. Calling `run()`
     *          without applying configuration updates result in using the configuration from the previous run.
     */
    virtual void update_config(const ComputeFlowManifest& manifest) = 0;

    /**
     * @brief Execute the brick on the provided inputs.
     *
     * @param input Input tensors.
     * @return Output tensors produced by the brick.
     *
     * @warning The input tensors must conform to the expected shapes and
     *          types defined by the manifest. Violations may lead to
     *          runtime errors or incorrect results.
     * @warning Keep the input tensors valid and unmodified until
     *          is_generation_complete() returns true.
     * @warning Repeated calls for the same generation must provide the same
     *          input values, but their storage addresses may change between calls.
     */
    virtual TensorDict run(const TensorDict& input) = 0;

    /**
     * @brief Reset internal state to initial conditions.
     *
     * After calling this method, the brick behaves as if it has not
     * processed any inputs yet (excluding setup).
     *
     * @note Does not release resources allocated during setup().
     */
    virtual void reset_states() = 0;

    /**
     * @brief Check whether the brick has completed its output generation.
     *
     * This is primarily relevant for streaming or iterative bricks.
     *
     * @return true if all output has been generated, false otherwise.
     *
     * @note For non-streaming bricks, this typically returns true
     *       immediately after run().
     */
    virtual bool is_generation_complete() const = 0;

    /**
     * @brief Write profiling information for the last execution.
     *
     * The output format is determined by the configured ProfilingOptions.
     *
     * @param os Output stream to write profiling data to.
     *
     * @warning If profiling is disabled, the output may be empty.
     */
    virtual void write_profile(std::ostream& os) const = 0;
};

}  // namespace compute_flow::bricks
