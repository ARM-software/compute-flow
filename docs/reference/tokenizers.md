<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Tokenizers

This document summarizes the tokenizers available as runners and how they are implemented.

## SentencePiece

- **Runner:** `sentencepiece_enc`
- **Implementation:** Native SentencePiece C++ library.
- **Input/Output:** `String` → `I32` token IDs.
- **Notes:** Suitable when you have a `.model` SentencePiece file and want token IDs directly in
  C++.

### Decoding token IDs

The `sentencepiece_dec` runner converts a one-dimensional `int64` token history to `string` UTF-8
bytes using the matching `.model` file. Declare one input and one output, both at index `0`, and
`runner_params: {}`. No recurrent `states` are used. Enable it with
`COMPUTE_FLOW_ENABLE_SENTENCEPIECE`, the same build option as the encoder.

- `set_input()` copies the IDs and checks their range against the tokenizer vocabulary.
- `run()` decodes the complete supplied history, replacing the previous output. An empty history
  produces empty text.
- `get_output()` returns runner-owned bytes, valid until the next `run()`, `reset_states()`,
  `setup()`, or destruction. Copy any output that must outlive these operations.
- `reset_states()` clears the history and text without reloading the tokenizer.

Decode the complete history when exact SentencePiece normalization, special-token handling, and
byte-fallback behavior matter.

Concatenating independently decoded tokens is not equivalent.

For example, with a tokenizer containing pieces `▁we` and `▁are` (`▁` marks a space), decoding their
IDs together produces `we are`. Decoding each ID separately produces `we` and `are`; concatenating
those results gives `weare`, losing the space.

Bricks returning incremental text must retain unfinished UTF-8 sequences and any suffix that a later
token can change before emitting append-only text.

## Hugging Face Tokenizer (`tokenizer.json`)

- **Runner:** `hf_tokenizer`
- **Implementation:** Based on `wangzhaode/tokenizer.cpp`, a C++ implementation that loads Hugging
  Face `tokenizer.json` files. We use this implementation instead of the official Hugging Face
  tokenizer to avoid the Rust dependency.
- **Input/Output:** `String` → `I32` token IDs.
- **Notes:** Set `model_path` to the `tokenizer.json` file. When `model_path` points to
  `tokenizer.json`, the runner will use its parent directory to initialize the tokenizer.

## Usage Notes

- Tokenizer runners are declared in the manifest `model` section via `framework_type`.
- The tokenizer must match the downstream model vocabulary (token IDs are assumed to align with
  model embeddings).
