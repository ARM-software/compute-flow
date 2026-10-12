<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Updating manifest config

Use `update_config()` to apply manifest `configs` to a brick. It can be called before `setup()` to
select the initial configuration, or after `setup()` to change runtime configuration for the next
request.

When `update_config()` is called after `setup()`, do not call `setup()` again. The brick keeps the
loaded models and applies only the updated config values.

`update_config()` is not used for model files, assets, constants, or accelerator choices. Those are
part of the setup lifecycle.

## Qwen3-TTS speaker example

The Qwen3-TTS 0.6B Custom Voice manifest exposes the speaker as a config value:

```json
"configs": {
  "language": {
    "type": "string",
    "value": "english"
  },
  "speaker": {
    "type": "string",
    "value": "aiden"
  },
  "num_active_code_pred_groups": {
    "type": "int32",
    "value": 16
  }
}
```

Update the manifest config and call `update_config()` before the request that should use it:

```cpp
#include "compute_flow/compute_flow.h"

#include <string>

namespace bricks = compute_flow::bricks;

int main() {
    auto manifest = bricks::load_manifest("qwen3_tts_custom_voice_en_manifest.json");
    auto brick    = bricks::load_brick(manifest);

    bricks::BrickOptions options;

    brick->setup(manifest, options);

    manifest.set_config("speaker", std::string{"serena"});
    brick->update_config(manifest);

    // brick->run();
}
```

## Language-model translation example

A translation model package can expose source and destination languages as configs without making
them part of the model-agnostic `lm_text` brick API. Set the initial languages before setup:

```cpp
auto manifest = bricks::load_manifest("hy_mt2_1_8b_manifest.json");
manifest.set_config("src_lang", std::string{"English"});
manifest.set_config("dst_lang", std::string{"French"});

auto brick = bricks::load_brick(manifest);
brick->setup(manifest);
```

To change languages for a later request, update the same manifest and apply it to the loaded brick:

```cpp
manifest.set_config("src_lang", std::string{"French"});
manifest.set_config("dst_lang", std::string{"Chinese"});
brick->update_config(manifest);
brick->reset_states();
```

The `lm_text` brick reloads its `prompt_template`, `system_prompt`, and `assistant_prefix` configs.
Manifest template expansion resolves model-specific references such as `{{ src_lang }}` and
`{{ dst_lang }}`; the brick does not interpret those language configs itself.
