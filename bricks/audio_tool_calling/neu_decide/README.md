<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# NeuDecide

Generates structured tool calls from spoken requests using the tool definitions in the selected
model package. The `calls` output is one UTF-8 string containing a JSON array in generation order.
Each call has a tool `name` and an `arguments` object. An empty completed response returns `[]`.

See the [official NeuDecide documentation](https://huggingface.co/neuphonic/neudecide) for tool
schemas and model limitations.

The manifest's `tools` asset defines the available tools. Changes to this asset take effect after
setup is run again.

The required `max_output_tokens` config (`int64`) limits decoding, including the initial tool-call
marker. It accepts values from `2` to the manifest's `max_answer_tokens` constant, inclusive. The
model manifest supplies its initial value, which is `128` in the provided package. Set it before
`setup()`, or apply it with `update_config()` before the next request, calling `reset_states()` as
usual. For example, this manifest config limits decoding to 64 tokens:

```json
"configs": {
  "max_output_tokens": { "type": "int64", "value": 64 }
}
```

A smaller limit can truncate the response. If decoding reaches the limit before the full JSON array
is complete, the brick returns `[]`; it does not expose partial tool calls. A complete response that
finishes exactly at the limit is accepted.
