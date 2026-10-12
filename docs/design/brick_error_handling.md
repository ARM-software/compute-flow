<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Brick error handling design

## Goal

The goal is to provide one shared Compute Flow mechanism for:

- always-on runtime checks
- manifest and model-package validation
- converting validation failures into setup errors
- internal programmer assertions

## Design principles

- Validation failures should preserve the failed expression, file, and line.
- Runtime check failures should preserve the failed expression, file, and line.
- Runtime checks should remain active in release builds and throw instead of terminating the process
  directly.
- Validation should not terminate the process directly.
- `setup()` should fail fast when validation fails.
- Error messages should be visible in current CLI examples, even if they do not catch exceptions.
- Internal invariants should use standard C++ assertions.
- Brick code should not define brick-specific check macros.

## Runtime checks

Use `COMPUTE_FLOW_CHECK()` for conditions that must be checked in every build and where execution
cannot continue after failure. Examples include failed file operations, invalid file structures,
backend errors, and runtime bounds derived from external data.

The macro is part of the public C++ API in `<compute_flow/check.h>` and is also available through
`<compute_flow/compute_flow.h>`.

```cpp
#define COMPUTE_FLOW_CHECK(expr)                                        \
    do {                                                                \
        if (!(expr)) {                                                   \
            const std::string message =                                  \
                std::string("Check failed: ") + #expr +                  \
                " at " + __FILE__ + ":" + std::to_string(__LINE__);     \
            std::cerr << "Error: " << message << std::endl;              \
            throw std::runtime_error(message);                           \
        }                                                               \
    } while (false)
```

The expression is evaluated once. On failure, the macro prints the error and throws
`std::runtime_error`, allowing applications to catch the failure. Do not use `COMPUTE_FLOW_ASSERT()`
for these checks because assertions may be disabled in release builds.

## Validation result

Change brick validation from a raw `bool` to a result object:

```cpp
struct ValidationResult {
    bool ok = true;
    std::string error;

    explicit operator bool() const {
        return ok;
    }
};
```

This keeps validation non-throwing while allowing the failure reason to travel back to `setup()`.

## Validation macro

Use `COMPUTE_FLOW_VALIDATE()` inside brick `validate()` implementations:

```cpp
#define COMPUTE_FLOW_VALIDATE(expr)                                     \
    do {                                                               \
        if (!(expr)) {                                                  \
            return {                                                    \
                false,                                                  \
                std::string("Validation failed: ") + #expr +            \
                    " at " + __FILE__ + ":" + std::to_string(__LINE__)  \
            };                                                         \
        }                                                              \
    } while (false)
```

Example:

```cpp
ValidationResult Qwen3TTS::validate(const ComputeFlowManifest& manifest) const {
    COMPUTE_FLOW_VALIDATE(manifest.has_config("language"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("speaker"));
    return {};
}
```

The returned error includes the failed expression and source location:

```text
Validation failed: manifest.has_config("speaker") at bricks/tts/qwen3-tts/Qwen3TTS.cpp:548
```

## Throwing validation errors from setup

Use `COMPUTE_FLOW_THROW_IF_ERROR()` when a caller needs to convert a `ValidationResult` into an
exception:

```cpp
#define COMPUTE_FLOW_THROW_IF_ERROR(result_expr, context)          \
    do {                                                          \
        const auto result = (result_expr);                         \
        if (!result) {                                             \
            const std::string message =                            \
                std::string(context) + ": " + result.error;        \
            std::cerr << "Error: " << message << std::endl;        \
            throw std::runtime_error(message);                     \
        }                                                         \
    } while (false)
```

For now, this macro always prints to `std::cerr` before throwing. This keeps current CLI examples
readable even if they do not catch exceptions. A future refinement can add a build option to disable
printing when applications want full control over logging.

`BrickBaseV1::setup()` should use it like this:

```cpp
COMPUTE_FLOW_THROW_IF_ERROR(
    validate(manifest),
    "Manifest validation failed for brick " + name()
);
```

This gives prototype-friendly behavior: setup fails immediately, the error is printed, and the
exception is still catchable by C++ applications or translated by Python bindings.

## Assertions

Use `COMPUTE_FLOW_ASSERT()` for programmer invariants and internal assumptions:

```cpp
#define COMPUTE_FLOW_ASSERT(expr) assert(expr)
```

Assertions are separate from validation:

- `COMPUTE_FLOW_CHECK()` is for always-on runtime conditions where execution cannot continue.
- `COMPUTE_FLOW_VALIDATE()` is for manifest, model-package, or user-provided configuration checks.
- `COMPUTE_FLOW_THROW_IF_ERROR()` converts a returned `ValidationResult` into an exception.
- `COMPUTE_FLOW_ASSERT()` is for states that should be impossible if the implementation is correct.
