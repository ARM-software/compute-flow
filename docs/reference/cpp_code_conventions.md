<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# C/C++ Code Conventions

These conventions apply to all C and C++ code across Compute Flow, including public APIs, core
implementation, runners, bindings, bricks, command-line tools, tests, and examples.

**Naming**

- Classes and structs use PascalCase, for example `Qwen3TTS`.
- Functions use snake_case.
- Variables use lower snake_case with no capital letters, for example `qwen_tts_`.
- Counting variables use `num_<name>` instead of `n_<name>`.
- Constants use the `k_` prefix and should be `constexpr` whenever possible.
- Private class members use a leading underscore: `_<name>`.

**Types**

- Use fixed-width integer types from `cstdint` to make intent explicit.
- Examples: `int32_t`, `int64_t`.
- Do not qualify fixed-width integer types with `std::`; use `uint32_t` instead of `std::uint32_t`.
- Use `int32_t` instead of `int`.
- Prefer `int64_t` or `uint64_t` instead of `size_t` whenever possible.
- Use `float` instead of `float32`.
- Do not use `float16` to keep compatibility across supported operating systems.

**Const And Immutability**

- If a variable is declared and never changed, declare it `const`.
- Prefer `constexpr` for compile-time constants.
- At interfaces, primitive parameters should not be `const`. Pass them by value.
- Pass classes or structs by reference, and use `const` references for inputs.

**Functions And Namespaces**

- File-local `static` functions should live in an anonymous namespace.
- Keep variable definitions close to their first use.
- Add an empty line after a completed group of variable definitions before subsequent checks,
  function calls, or control flow.
- Add an empty line before and after a standalone function call. Calls that form one logical block
  may remain together.

**Control Flow**

- Always use braces for control-flow bodies, even when the body contains only one statement.

**Casting**

- Use `static_cast` for basic primitive conversions instead of C-style casts.

**Includes**

- Include only the headers you need.

**Examples**

```cpp
namespace {
int32_t count_tokens(const std::vector<int32_t>& token_ids) {
  const int32_t num_tokens = static_cast<int32_t>(token_ids.size());
  return num_tokens;
}
}  // namespace

class MyClass {
 public:
  explicit MyClass(const Config& config);

 private:
  static constexpr int32_t k_max_tokens = 8192;
  int32_t _num_speakers = 0;
};
```
