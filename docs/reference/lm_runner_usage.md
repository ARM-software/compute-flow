<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# LM Runner Usage Example

This example shows the basic flow for using an LM runner with Compute Flow. The runner encodes new
input when you call `set_input()` and then `run()`. It preserves the model context across runs
unless you explicitly call `reset_states()`.

## Generate Token IDs Until EOG

Use a single input prompt, then call `run()` repeatedly to sample new token IDs. The generation ends
when `is_end_of_generation()` reports true.

```cpp
using compute_flow::TensorData;
using compute_flow::DataType;

auto lm_runner = make_lm_runner(manifest["talker"].lm_runner_type);
lm_runner->setup(manifest["talker"], &profiler);

TensorData input{};
input.data = tokens.data();
input.format.dt = DataType::I32;
input.format.shape = { static_cast<int64_t>(tokens.size()) };

// set_input() enforces encoding ONLY on the next first run().
lm_runner->set_input("token_ids", input);

while (!lm_runner->is_end_of_generation()) {
    lm_runner->run();

    const auto token_out = lm_runner->get_output("token_id");
    const int32_t token_id = *static_cast<const int32_t*>(token_out.data);
    generated.push_back(token_id);
}
```

## Encode On Every Run With New Input

If you call `set_input()` every step, the next `run()` will encode the new input and continue from
the existing context.

```cpp
using compute_flow::TensorData;
using compute_flow::DataType;

auto lm_runner = make_lm_runner(manifest["talker"].lm_runner_type);
lm_runner->setup(manifest["talker"], &profiler);

TensorData input{};
input.data = tokens.data();
input.format.dt = DataType::I32;
input.format.shape = { static_cast<int64_t>(tokens.size()) };

// set_input() is called, so run() encodes.
lm_runner->set_input("token_ids", input);
lm_runner->run();

while (!lm_runner->is_end_of_generation()) {
    TensorData step_input{};
    step_input.data = next_tokens.data();
    step_input.format.dt = DataType::I32;
    step_input.format.shape = { static_cast<int64_t>(next_tokens.size()) };

    // set_input() is called each step, so run() encodes every time.
    // The context is preserved
    lm_runner->set_input("token_ids", step_input);
    lm_runner->run();
}

lm_runner->reset_states();
```

## Notes

- `set_input()` flags that new input has been provided. The following `run()` will encode it and
  preserve the existing context.
- To discard the context and start a fresh sequence, call `reset_states()` before the next
  `set_input()` / `run()` cycle.
