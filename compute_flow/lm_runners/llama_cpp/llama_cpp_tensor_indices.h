//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace compute_flow {

// Input tensor indices
constexpr size_t k_input_token_ids_idx = 0;
constexpr size_t k_input_embeds_idx    = 1;

// Output tensor indices
constexpr size_t k_output_token_id_idx = 0;
constexpr size_t k_output_hidden_state_idx = 1;
constexpr size_t k_output_logits_idx   = 2;
constexpr size_t k_output_hidden_state_seq_idx = 3; // full sequence of hidden states [num_tokens, n_embd]
constexpr size_t k_output_pooled_embedding_idx = 4;


}  // namespace compute_flow
