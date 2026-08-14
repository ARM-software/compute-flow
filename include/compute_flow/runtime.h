//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace compute_flow {

/**
 * @brief Options controlling weights caching behavior.
 */
struct WeightsCachingOptions {
    /**
     * @brief Enable caching of model weights across runs.
     *
     * When enabled, implementations may persist weights to reduce
     * initialization overhead on subsequent executions.
     *
     * @warning Enabling caching may increase disk usage and requires
     *          implementations to manage cache consistency.
     */
    bool enabled = false;
};

/**
 * @brief Runtime execution options for a brick.
 */
struct RuntimeOptions {
    /**
     * @brief Enable warmup execution during setup.
     *
     * When enabled, the brick may perform one or more warmup runs
     * to reduce first-inference latency.
     *
     * @warning Warmup increases setup time and may execute the model
     *          multiple times before actual usage.
     */
    bool enable_warmup = true;

    /**
     * @brief Options controlling weights caching.
     */
    WeightsCachingOptions weights_caching;
};
} // namespace compute_flow