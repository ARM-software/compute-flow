<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Brick Runtime Interface V1 Design

A brick exposes two separate contracts:

1. The **brick runtime interface**, which defines how the runtime drives execution through functions
   such as `setup()`, `run()`, and `teardown()`.

2. The **brick API**, which defines the public input and output tensors exchanged with the brick.

This document focuses on the brick runtime interface, which defines how the brick is executed.

Brick runtime interfaces follow the Compute Flow
[compatibility principle](../PRODUCT.md#compatibility-principle). New interface versions may be
introduced for new bricks, but bricks built against previous interface versions must continue to
work with newer releases of Compute Flow.

## Table of Contents

- [Overview](#overview)
- [Design Goals](#design-goals)
- [Runtime Interface: `IBrickV1` and `BrickBaseV1`](#runtime_interface-ibrickv1-and-brickbasev1)
  - [`IBrickV1` methods](#ibrickv1-methods)
    - [`ValidationResult`](#validationresult)
  - [Execution lifecycle](#execution-lifecycle)
    - [The `setup()` method](#the-setup-method)
  - [Configuration](#configuration)
    - [`BrickOptions`](#brickoptions)
    - [`RuntimeOptions`](#runtimeoptions)
    - [`ProfilingOptions`](#profilingoptions)
  - [`BrickBaseV1` methods](#brickbasev1-methods)
    - [Rationale behind `BrickBaseV1`](#rationale-behind-the-brickbasev1)
- [Profiling Design (detailed)](#profiling-design-detailed)
  - [Enabling high-performance brick development](#enabling-high-performance-brick-development)
  - [Minimal runtime interface](#minimal-runtime-interface)
  - [Goals](#goals)
  - [Core Principles](#core-principles)
  - [What is exposed to users](#what-is-exposed-to-users)
  - [What is NOT exposed](#what-is-not-exposed)
  - [The controlled lifecycle](#the-controlled-lifecycle)
  - [Resource Acquisition Is Initialization (RAII) Profiling](#resource-acquisition-is-initialization-raii-profiling)
  - [Restricted Access](#restricted-access)
- [Loading a brick](#loading-a-brick)
  - [Loading the manifest (`load_manifest()`)](#loading-the-manifest-load_manifest)
  - [Loading the brick (`load_brick()`)](#loading-the-brick-load_brick)
- [C++ Usage](#c-usage)
  - [Enabling the profiler](#enabling-the-profiler)
  - [Streaming Example](#streaming-example)
  - [Changing the config fields at runtime](#changing-the-config-fields-at-runtime)
  - [Setting the accelerator type](#setting-the-accelerator-type)

______________________________________________________________________

## Overview

The **Brick Interface V1** defines the common contract for compute bricks in the Compute Flow SDK.

A *brick* is a self-contained computational unit that:

- is configured via a manifest
- processes input tensors to produce output tensors
- may maintain internal state to support streaming generation

To define the design goals of the runtime interface, it is important to consider Compute Flow’s
vision:

> Compute Flow is designed for easy prototyping, rapid development, and education, making it an
> ideal platform to demonstrate and profile edge AI solutions on Arm®.

Based on this vision, the runtime interface is designed to be:

- **simple to use**
- **hard to misuse**

______________________________________________________________________

## Design Goals

- Minimal configuration
- Built-in profiling to support development and performance evaluation
- Predictable and well-defined lifecycle
- Low overhead by default

______________________________________________________________________

## Runtime Interface: `IBrickV1` and `BrickBaseV1`

| Runtime Interface | Description                                      |
| ----------------- | ------------------------------------------------ |
| `IBrickV1`        | Public brick API (contract)                      |
| `BrickBaseV1`     | Convenience base class for brick implementations |

______________________________________________________________________

### `IBrickV1` methods

`IBrickV1` defines the set of virtual methods that must be implemented by all bricks:

| Method                     | Description                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `api()`                    | Returns the brick API identifier declared by the implementation                           |
| `name()`                   | Returns the human-readable brick name                                                     |
| `validate()`               | Returns a validation result for manifest compatibility                                    |
| `setup()`                  | Validates the manifest, initializes the brick, allocates resources, and resets the states |
| `update_config()`          | Update configuration fields from the provided manifest                                    |
| `run()`                    | Executes the brick and returns output tensors                                             |
| `reset_states()`           | Resets internal state                                                                     |
| `is_generation_complete()` | Indicates whether output generation is complete for the given input                       |
| `write_profile()`          | Writes profiling results to the provided stream                                           |

By design, all bricks support streaming semantics, even if they do not perform iterative generation.

- For non-streaming bricks, `is_generation_complete()` returns `true` immediately after the first
  call to `run()`.
- For streaming bricks, `is_generation_complete()` may return `false`, and `run()` must be called
  repeatedly until it returns `true`.

Keep the input tensors valid and unmodified until `is_generation_complete()` returns `true`.
Repeated calls for the same generation must provide the same input values, but their storage
addresses may change between calls.

After `is_generation_complete()` returns `true`, the user must call `reset_states()` before starting
a new session.

`api()` returns the exact `info.json` `api` value implemented by the brick, such as `tts.v1`. The
value identifies the public input/output tensor contract and is available immediately after
construction, before `setup()`.

#### `ValidationResult`

`validate()` returns a `ValidationResult` instead of a raw boolean:

```cpp
struct ValidationResult {
    bool ok = true;
    std::string error;

    explicit operator bool() const;
};

virtual ValidationResult validate(const ComputeFlowManifest& manifest) const = 0;
```

Validation is non-throwing. A successful validation returns the default result. A failed validation
returns `ok == false` and an error message describing the first failed validation expression and its
source location.

For bricks derived from `BrickBaseV1`, `name()` returns the `brick.name` value from the manifest
after a successful `setup()`. Before setup, or after a failed setup, `name()` returns an empty
string. Brick-specific `validate()` implementations must inspect `manifest.get_brick_info()` instead
of calling `name()`, because validation runs before the base class stores the manifest-backed name.

______________________________________________________________________

### Execution lifecycle

The execution lifecycle is as follows:

```
setup() → run() → (repeat run() if generation is not completed) → reset_states() → (optionally update the config with update_config())
```

`setup()` performs manifest validation before loading models or preparing brick state. Applications
may still call `validate()` explicitly when they want to check manifest compatibility before setup.
`BrickBaseV1::setup()` converts validation failures into runtime errors before loading models.
Validation must not be duplicated inside brick-specific model-loading code.

#### The `setup()` method

The `setup()` method takes two inputs:

| Input      | Description                                           |
| ---------- | ----------------------------------------------------- |
| `manifest` | Defines the brick configuration and associated assets |
| `options`  | Specifies runtime and profiling configuration         |

______________________________________________________________________

### Configuration

#### `BrickOptions`

| Option             | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `RuntimeOptions`   | Controls runtime behavior (e.g. enabling warmup)      |
| `ProfilingOptions` | Controls profiling behavior (e.g. enabling profiling) |

______________________________________________________________________

#### `RuntimeOptions`

- Enable warmup execution
- Configure weights caching

______________________________________________________________________

#### `ProfilingOptions`

- Statistics and external timeline annotations can be enabled independently
- Optional operation-level profiling
- Output formats: Pretty (human-readable) or CSV (machine-readable)

### `BrickBaseV1` methods

`BrickBaseV1` provides default implementations for common functionality of `IBrickV1` and enforces
correct lifecycle and profiling behavior. The following methods are implemented in the base class
and marked as `final`:

| Method                     | Description                                                                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `name() final`             | Returns the manifest-backed brick name after successful setup, or an empty string before setup                                     |
| `setup(...) final`         | Resets/configures profiler, validates the manifest, and calls `load_models_impl(...)`, `prepare_impl(...)`, and `warmup_impl(...)` |
| `update_config(...) final` | Resets profiler and calls `update_config_impl(...)`                                                                                |
| `run(...) final`           | Profiles execution and calls `run_impl(...)`                                                                                       |
| `reset_states() final`     | Resets state via `reset_states_impl(...)` and resets profiler                                                                      |
| `write_profile(...) final` | Outputs profiling results                                                                                                          |

Most bricks are expected to inherit from `BrickBaseV1`. Implementers provide the brick-specific
logic by overriding the following methods:

| Method                    | Description                               |
| ------------------------- | ----------------------------------------- |
| `load_models_impl(...)`   | Brick-specific model loading logic        |
| `prepare_impl(...)`       | Brick-specific preparation logic          |
| `warmup_impl(...)`        | Optional warmup logic                     |
| `update_config_impl(...)` | Brick-specific configuration update logic |
| `run_impl(...)`           | Core execution logic                      |
| `reset_states_impl(...)`  | Brick-specific state reset logic          |

#### Rationale behind the `BrickBaseV1`

This design of the `BrickBaseV1` follows the **Template Method pattern**, where the base class
controls the execution flow while delegating brick-specific behavior to derived classes.

The main benefits are:

- **Correctness by construction** Lifecycle management and profiling are handled in the base class,
  preventing misuse (e.g. forgetting to reset or measure the profiler).

- **Consistency across bricks** All bricks follow the same execution and profiling behavior, making
  them easier to reason about and compare.

- **Reduced boilerplate** Implementers only need to focus on core logic, without reimplementing
  common patterns.

- **Minimal API surface** The public runtime interface remains simple, while internal complexity
  (e.g. profiling lifecycle) is encapsulated.

This approach ensures that bricks are both **easy to implement** and **hard to misuse**, in line
with the overall design goals.

______________________________________________________________________

## Profiling Design (detailed)

Profiling is an intrinsic property of a brick, to ensure the following:

- Performance can be assessed from the earliest stages of brick development
- Performance is evaluated at the same level where computation happens
- Profiling is consistent across all bricks
- No external or ad-hoc instrumentation is required

By making profiling part of the brick itself, performance becomes a first-class aspect of the
design, alongside functionality.

### Enabling high-performance brick development

Profiling is essential to develop high-performance bricks:

- It provides immediate feedback during development
- It allows developers to identify bottlenecks quickly
- It ensures that performance considerations are not an afterthought

Without built-in profiling, performance evaluation would be:

- Inconsistent
- Error-prone
- Often neglected.

Embedding profiling directly into every brick, we allow a consistent way to measure and compare
brick performance.

### Minimal runtime interface

Profiling is embedded into the brick and its runtime interface is intentionally minimal:

- Users can enable/disable profiling
- Users can retrieve results via `write_profile()`

The underlying profiler is not exposed, ensuring:

- Controlled lifecycle management
- Consistent behavior
- Safe usage in parallel environments

### Goals

The profiling system is designed to:

- Provide clear and immediate performance insights during brick development
- Be easy to use with minimal configuration
- Avoiding incorrect usage
- Introduce minimal overhead when disabled

### Core Principles

- One profiler per brick. Therefore, each brick owns its profiler instance
- The profiler is not exposed through the public runtime interface
- No shared/global state
- Safe for parallel execution

### What is exposed to users

- `write_profile(std::ostream&)`: Only exposed outside the brick
- `profile_scope(std::string_view name)`: Only exposed inside the brick

### What is NOT exposed

The following profiler capabilities are intentionally not exposed to users or brick implementations:

- reset()
- start() / stop()
- configuration after setup()
- direct access to the profiler object

Rationale:

- Prevent inconsistent profiling states
- Avoid lifecycle misuse
- Keep the API minimal

### The controlled lifecycle

The profiler lifecycle is fully managed by the framework. Therefore, users and implementers do not
need to manually manage profiling state:

| Brick API call    | Behavior                                                                 |
| ----------------- | ------------------------------------------------------------------------ |
| `setup()`         | Reset the profiler and measure the setup scope and its subscopes         |
| `update_config()` | Reset the profiler and measure the update_config scope and its subscopes |
| `run()`           | Measure the run scope and its subscopes                                  |
| `reset_states()`  | Reset the profiler                                                       |

At this point, it should be clear why `setup()`, `update_config()`, `run()`, and `reset_states()`
are marked as `final` in `BrickBaseV1`. The primary reason is to prevent misuse of the profiler and
ensure a consistent and correct profiling lifecycle.

Implementers override:

- setup_impl(...)
- update_config_impl(...)
- run_impl(...)
- reset_states_impl(...)

to ensure the following:

- Profiling reset is never forgotten
- Consistent behavior across all bricks

### Resource Acquisition Is Initialization (RAII) Profiling

RAII ensures resources are automatically acquired during object construction and released during
destruction. Therefore, scopes automatically measure execution:

- start on construction
- stop on destruction

### Restricted Access

A limited `ProfilerView` is used internally:

- allows `measure()`
- prevents reset/config misuse

______________________________________________________________________

## Loading a brick

Bricks are created in two steps:

```cpp
bricks::BrickManifest manifest = bricks::load_manifest(path);
bricks::Brick         brick    = bricks::load_brick(manifest);
```

The loading process is intentionally split into two stages because some manifest fields can be
modified at runtime.

### Loading the manifest (`load_manifest()`)

It loads the brick manifest from disk and returns a user-facing `BrickManifest`.

The returned manifest does NOT expose the full internal manifest representation. Only the fields
intended for user interaction are exposed through the public API.

Some manifest fields are immutable and cannot be modified after loading. The MUTABLE fields are:

- runtime-configurable values (`config`) that may be updated after `brick->setup()` through
  `brick->update_config(...)`
- accelerator configuration for each model required by the brick before calling `brick->setup()`.

All other exposed fields are read-only and cannot be modified by the user.

> ℹ️ If accelerator configuration is not explicitly provided before calling `setup()`, the default
> configuration defined in the manifest is used. By default, the CPU acceleration with 1 thread only
> is used.

### Loading the brick (`load_brick()`)

It creates a brick instance from the loaded manifest.

The returned brick implements the `IBrickV1` runtime interface and can then be validated,
configured, set up, and executed through the standard brick lifecycle.

______________________________________________________________________

## C++ Usage

### Enabling the profiler

```cpp
bricks::BrickManifest manifest = bricks::load_manifest(path);
bricks::Brick         brick    = bricks::load_brick(manifest);

bricks::BrickOptions options;
options.profiling.enabled = true;

brick->setup(manifest, options);

TensorDict inputs;
auto outputs = brick->run(inputs);

brick->write_profile(std::cout);
```

### Streaming Example

```cpp
bricks::BrickManifest manifest = bricks::load_manifest(path);
bricks::Brick         brick    = bricks::load_brick(manifest);

brick->setup(manifest, options);

while (!brick->is_generation_complete()) {
    auto outputs = brick->run(inputs);

    // Consume outputs
}
```

### Changing the config fields at runtime

```cpp
bricks::BrickManifest manifest = bricks::load_manifest(path);
bricks::Brick         brick    = bricks::load_brick(manifest);

brick->setup(manifest, options);

TensorDict inputs;
auto outputs0 = brick->run(inputs);

manifest.config["config_field"] = 3;

// Lightweight setup
brick->update_config(manifest);

auto outputs1 = brick->run(inputs);
```

### Setting the accelerator type

```cpp
bricks::BrickManifest manifest = bricks::load_manifest(path);
bricks::Brick         brick    = bricks::load_brick(manifest);

// It must be set before calling setup()
manifest["model_name"].acc_config = AcceleratorConfig::cpu_only(num_threads);

brick->setup(manifest, options);

TensorDict inputs;
auto outputs = brick->run(inputs);
```

______________________________________________________________________
