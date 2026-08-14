//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace compute_flow {

/**
 * @brief LM sampling policy.
 *
 * - `Greedy`: selects the token with the highest probability.
 * - `Distribution`: samples from the model probability distribution.
 * - `Disabled`: no sampling performed by the runner.
 */
enum class LMSamplingPolicy {
    Greedy,
    Distribution,
    Disabled
};

/**
 * @brief RNG strategy for stochastic sampling.
 *
 * - `Mt19937`: Mersenne Twister (`std::mt19937`).
 * - `Lcg`: linear congruential generator (LCG).
 * - `Unknown`: unspecified (runner may treat this as an error).
 */
enum class LMRNGStrategy {
    Mt19937,
    Lcg,
    Unknown
};

/**
 * @brief Top-k tie handling strategy.
 *
 * - `StrictK`: keep exactly `k` candidates.
 * - `ThresholdTies`: keep all candidates with `logit >= kth_logit`.
 * - `Unknown`: unspecified (runner may treat this as an error).
 */
enum class LMTopKStrategy {
    StrictK,
    ThresholdTies,
    Unknown
};

/**
 * @brief Position layout used when embeddings are passed directly to an LM runner.
 *
 * - `Single`: one position per embedding.
 * - `MRoPEBroadcast`: four identical position planes per embedding, as required by
 *   models using M-RoPE/IMRoPE with embedding rather than token input.
 */
enum class LMEmbeddingPositionMode {
    Single,
    MRoPEBroadcast
};

/**
 * @brief Parameters for distribution sampling.
 *
 * - `top_k`: maximum number of candidates to keep (0 disables top-k).
 * - `top_p`: nucleus sampling threshold (1.0 disables top-p).
 * - `min_p`: minimum probability floor (0.0 disables min-p).
 * - `temperature`: softmax temperature (must be > 0).
 * - `seed`: RNG seed for stochastic sampling.
 * - `rng_strategy`: RNG strategy to use.
 * - `top_k_strategy`: tie handling strategy for top-k.
 */
struct LMDistributionParams {
    int32_t  top_k       { 40 };     // 0 = disabled
    float    top_p       { 0.95F };  // 1.0 = disabled
    float    min_p       { 0.0F };   // 0.0 = disabled
    float    temperature { 0.8F };   // Must be > 0
    uint32_t seed        { 4568 };
    LMRNGStrategy rng_strategy { LMRNGStrategy::Unknown };
    LMTopKStrategy top_k_strategy { LMTopKStrategy::Unknown };
};

/**
 * @brief Aggregated sampling parameters for LM runners.
 *
 * - `sampling_policy`: selects the sampling policy.
 * - `distribution_params`: parameters used when `sampling_policy` is `Distribution`.
 * - `max_context_tokens`: maximum context length for KV cache management.
 * - `batch_size`: decode batch size.
 * - `repetition_penalty`: penalize repeated tokens (> 1.0 increases penalty).
 * - `suppress_token_ranges`: inclusive token ID ranges suppressed during sampling.
 * - `min_generated_tokens`: minimum non-EOG tokens to generate before allowing EOG.
 * - `max_tokens_to_gen`: maximum non-EOG tokens to generate (`-1` means unlimited).
 * - `append_generated_token`: whether to append sampled tokens to the context.
 * - `embedding_position_mode`: position layout for direct embedding input.
 */
struct LMParams {
    LMSamplingPolicy sampling_policy { LMSamplingPolicy::Disabled };

    // Used only if sampling_policy == Distribution
    LMDistributionParams distribution_params;

    // Used only when embeddings are passed directly to the model.
    LMEmbeddingPositionMode embedding_position_mode { LMEmbeddingPositionMode::Single };

    int32_t max_context_tokens { 2048 };
    int32_t batch_size         { 256 };

    // Additional sampling controls.
    float repetition_penalty { 1.0F };
    std::vector<std::pair<int32_t, int32_t>> suppress_token_ranges;
    int32_t min_generated_tokens { 1 };
    int32_t max_tokens_to_gen { -1 };

    // Execution behavior flags.
    bool append_generated_token { true };
};

}  // namespace compute_flow
