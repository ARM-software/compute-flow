<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Llama.cpp Tensor Index Mapping

Llama.cpp does not expose named tensors for inputs or outputs. Compute Flow therefore assigns
meaning to tensors based on the index in the manifest `model.inputs` and `model.outputs` maps,
rather than the key string. This keeps manifests flexible while ensuring the runtime can interpret
tensors consistently.

**Inputs**

Index mapping for `model.inputs`:

- `0` → `token_ids`
- `1` → `inputs_embeds`

**Outputs**

Index mapping for `model.outputs`:

- `0` → `token_id`
- `1` → `hidden_state`
- `2` → `logits`
- `3` → `hidden_state_seq`
- `4` → `pooled_embedding`

`pooled_embedding` is the model-configured sequence-level embedding returned by
`llama_get_embeddings_seq()`. It has `float32` data type and shape `[embedding_dimension]`. The
model GGUF metadata must define a pooling type. For example, BGE embedding models commonly use mean
pooling. Mapping this output for a model without configured pooling causes `run()` to fail.

Example: if the manifest contains `"my_input": 1` under `model.inputs`, Compute Flow will interpret
that tensor as `inputs_embeds`, regardless of the key name.
