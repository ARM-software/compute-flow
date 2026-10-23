<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# NemotronASR

Transcribe speech with language-prompted Nemotron 3.5.

## Documentation

- [Architecture and streaming behaviour](docs/architecture.md)

## Configuration

| Manifest key                          | Meaning                                                                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `configs.language.value`              | Selects a locale string, or `auto` for automatic language detection, using the package's `assets.languages_map_file`.                 |
| `configs.max_symbols_per_frame.value` | Bounds greedy decoding work per encoder frame. Reaching the cap without a blank raises an error rather than silently truncating text. |

The manifest requires `language`, a `languages_map_file` asset, `max_symbols_per_frame`, and the
encoder's `prompt_id` input. Nemotron 3.0 and numeric `configs.prompt_id` are unsupported. The
values are applied during `setup()`. Call `update_config()` between utterances, then
`reset_states()` before the next generation. Reset after an inference error before reusing the
brick. A new model package, frontend constant, or tokenizer requires `setup()` again.
