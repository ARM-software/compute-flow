<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Sampling Policies

This document describes sampling policy knobs used by Compute Flow LM runners, with a focus on top-k
tie handling and RNG strategy selection. These options are typically configured under
`runner_params.sampler` in a model's manifest entry.

**Top-K Policy** Top-k filtering defines the candidate set before sampling. When logits contain ties
around the k-th value, different policies produce different candidate sets and can lead to divergent
samples.

Available policies:

1. `strict_k` Keeps exactly `k` candidates, even when multiple tokens share the k-th logit value.
   This matches llama.cpp's default top-k behavior.
2. `threshold_ties` Keeps all tokens with `logit >= kth_logit`. This matches the common Hugging Face
   Transformers implementation and can yield more than `k` candidates when ties occur.

Recommended usage:

1. Use `strict_k` for consistency with llama.cpp defaults.
2. Use `threshold_ties` to match Hugging Face Transformers behavior.

Manifest field:

1. `runner_params.sampler.distribution.top_k_strategy`

**RNG Strategy** RNG strategy controls how random numbers are generated for stochastic sampling.

Available strategies:

1. `mt19937` Uses the Mersenne Twister (`std::mt19937`). This is the standard RNG in many C++
   sampling implementations.
2. `lcg` Uses a linear congruential generator (LCG).

Recommended usage:

1. Use `lcg` when you need deterministic LCG sampling behavior.
2. Use `mt19937` when you want Mersenne Twister behavior.

Manifest field:

1. `runner_params.sampler.distribution.rng_strategy`

**Suppress Token Ranges** This optional list excludes token ID ranges from sampling. When provided,
all tokens in the specified ranges are treated as ineligible (their logits are effectively masked)
before greedy selection or distribution sampling. This is useful for large suppression sets without
enumerating thousands of individual IDs.

Format:

1. `[[start, end], ...]` (inclusive ranges).
2. A single token can be expressed as `[id, id]`.

Recommended usage:

1. Omit the field (or use an empty list) when no suppression is needed.
2. Prefer ranges for large contiguous blocks of IDs to keep manifests compact.

Manifest field (optional):

1. `runner_params.sampler.suppress_token_ranges`

**Append Generated Token** This flag controls whether the runner appends the sampled token to the
context by decoding it. When `true`, each `run()` advances the model state and produces fresh logits
for the next step. When `false`, the runner will not advance context after sampling, which is useful
when the LM runner is embedded as a backbone and a higher-level component manages token advancement.

Recommended usage:

1. Set to `true` for the generic `lm_text` brick. If it is `false`, generation will stall because
   logits never advance between `run()` calls.
2. Set to `false` when the LM runner is used as a backbone and another component controls context
   advancement.

Manifest field (required for LM runner manifests):

1. `runner_params.sampler.append_generated_token`

**Maximum Generated Tokens** This optional limit stops generation after a response produces the
configured number of non-EOG tokens. The default value, `-1`, leaves generation unlimited so that it
stops on EOG or another runtime constraint.

Valid values:

1. `-1` for unlimited generation.
2. `0` to evaluate the prompt without generating a token.
3. A positive integer for a finite generation limit.

Manifest field (optional):

1. `runner_params.sampler.max_tokens_to_gen`
