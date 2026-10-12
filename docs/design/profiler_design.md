<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Profiler Design

## Table of Contents

- [Overview](#overview)
- [Design Goals](#design-goals)
- [Core Concepts](#core-concepts)
  - [Profiler ownership](#profiler-ownership)
  - [Scope-based profiling](#scope-based-profiling)
  - [Model and operation profiling](#model-and-operation-profiling)
- [Public API](#public-api)
  - [`ProfilingOptions`](#profilingoptions)
  - [`Profiler`](#profiler)
  - [`Profiler::Scope`](#profilerscope)
  - [`ProfilerView`](#profilerview)
- [Lifecycle](#lifecycle)
  - [Configuration](#configuration)
  - [Reset](#reset)
  - [Disabled profiling](#disabled-profiling)
- [Scope Reporting](#scope-reporting)
  - [Single-call scopes](#single-call-scopes)
  - [Repeated scopes](#repeated-scopes)
  - [Streaming mode](#streaming-mode)
- [Output Formats](#output-formats)
  - [Pretty output](#pretty-output)
  - [CSV output](#csv-output)
  - [Units](#units)
- [Integration With Bricks](#integration-with-bricks)
  - [`BrickBaseV1`](#brickbasev1)
  - [Legacy bricks](#legacy-bricks)
- [Usage](#usage)
  - [C++ user example](#c-user-example)
  - [Brick implementation example](#brick-implementation-example)
  - [Runner integration example](#runner-integration-example)
- [Rationale](#rationale)

______________________________________________________________________

## Overview

The profiler provides built-in performance measurement for bricks.

Profiling is a first-class part of the brick execution model. It is designed to answer three
questions:

- How long did a high-level brick scope take?
- Which models or runners contributed to that scope?
- Which operations inside a model were the dominant costs, when per-operation profiling is enabled?

The profiler is intentionally simple to use. A scope starts when a `Profiler::Scope` object is
created and ends when that object is destroyed. This makes profiling robust in the presence of early
returns and exceptions.

______________________________________________________________________

## Design Goals

- Provide useful performance information with minimal user configuration
- Make common profiling behavior consistent across bricks
- Support streaming workloads where `run()` is called repeatedly
- Report first-call and all-call performance for repeated scopes
- Support model-level and operation-level analysis
- Support human-readable and machine-readable outputs
- Avoid exposing lifecycle operations such as reset through restricted brick views
- Keep profiling cheap when disabled

______________________________________________________________________

## Core Concepts

### Profiler ownership

Each brick owns its profiler instance. There is no global profiler and no shared profiling state
between bricks.

This keeps profiling:

- local to the brick being measured
- deterministic across independent brick instances
- safe for parallel execution of different brick objects

______________________________________________________________________

### Scope-based profiling

A scope is a high-level execution phase.

Examples:

| Scope           | Meaning                                     |
| --------------- | ------------------------------------------- |
| `setup`         | Brick initialization and warmup             |
| `update_config` | Runtime configuration update                |
| `run`           | One call to brick execution                 |
| `decode`        | A brick-defined internal phase, if measured |

Scopes are measured with RAII:

```cpp
auto scope = profiler.measure("run");
```

The scope starts immediately and ends when `scope` is destroyed.

Top-level scopes are also used as report sections. If a scope is called multiple times, the report
contains:

- one table for the first call
- one table for all calls aggregated

______________________________________________________________________

### Model and operation profiling

Inside a scope, runners record model-level timings:

```cpp
profiler.start_timer("litert", "speech_decoder");
runner.run();
profiler.end_timer("litert", "speech_decoder");
```

When supported by a backend, operation timings are attached to a parent model:

```cpp
profiler.add_nested_sample("litert",
                           "speech_decoder",
                           "CONV_2D",
                           elapsed_ns);
```

The llama.cpp runner appends graph-visible tensor types to operation names. Homogeneous operations
use one type, such as `RMS_NORM[F32]`, while mixed-type operations show inputs and output, such as
`GET_ROWS[Q2_0C,I32->F32]`. `MUL_MAT` also identifies `GEMV` or `GEMM`. These labels do not describe
internal backend packing or the selected microkernel.

The report treats these as:

| Record type | Description                                   |
| ----------- | --------------------------------------------- |
| scope       | Time spent in the measured scope              |
| subscope    | Time spent in a nested scope                  |
| model       | Time spent in a runner/model                  |
| op          | Time spent in a backend operation for a model |
| overhead    | Scope time not attributed to contributor rows |

Direct child scopes are treated as contributors in the same way as model rows. This allows
`BrickBaseV1` to report `setup` contributors such as `load_models`, `prepare`, and `warmup` without
requiring a backend runner.

Scope time is reported outside the contributor table. It is not added to the table total. The
performance summary reports scope-level overhead as:

```text
scope time - sum(contributors)
```

______________________________________________________________________

## Public API

### `ProfilingOptions`

`ProfilingOptions` controls profiler behavior:

- `enabled` controls collected statistics and textual reports.
- `enable_timeline_annotations` independently controls external timeline annotations when Compute
  Flow was built with an annotation backend. It is otherwise a no-op.
- `enable_op_profiling` controls optional operation-level statistics. Operation samples are not
  emitted as timeline annotations. Backend instrumentation must only be installed when both
  `enabled` and `enable_op_profiling` are true because it can add runtime overhead.

| Option                        | Description                                             | Default |
| ----------------------------- | ------------------------------------------------------- | ------- |
| `enabled`                     | Enables or disables statistics and reports              | `true`  |
| `enable_op_profiling`         | Enables backend operation-level profiling, if supported | `false` |
| `enable_timeline_annotations` | Enables compiled external timeline annotations          | `true`  |
| `format`                      | Output format                                           | Pretty  |
| `unit`                        | Time unit used in reports                               | ms      |

Options are applied through `Profiler::configure()`.

There are no public setter methods. Configuration is updated as one coherent value:

```cpp
ProfilingOptions options;
options.enabled = true;
options.format = ProfilingFormat::Csv;
options.unit = ProfilingUnit::Milliseconds;

profiler.configure(options);
```

______________________________________________________________________

### `Profiler`

`Profiler` is the concrete profiling object owned by a brick.

The public API supports:

| Method                              | Description                                  |
| ----------------------------------- | -------------------------------------------- |
| `configure(options)`                | Applies all profiling options                |
| `is_enabled()`                      | Returns whether profiling is enabled         |
| `is_per_op_profiling_enabled()`     | Returns whether per-op profiling is enabled  |
| `is_timeline_annotations_enabled()` | Returns whether timeline output is active    |
| `measure(name)`                     | Starts a RAII scope                          |
| `measure_model(runner, name)`       | Starts a RAII model invocation scope         |
| `start_timer(...)`                  | Starts model timing                          |
| `end_timer(...)`                    | Ends model timing                            |
| `cancel_active(...)`                | Cancels active timing for a model            |
| `add_nested_sample(...)`            | Adds an operation sample under a model       |
| `reset()`                           | Clears collected profiling state             |
| `report(...)`                       | Returns table-style pretty output            |
| `report_visual(...)`                | Returns visual pretty output                 |
| `report_csv()`                      | Returns CSV output                           |
| `write(os)`                         | Writes output according to configured format |

`write(os)` is the preferred user-facing output path because it honors `ProfilingOptions::format`.

______________________________________________________________________

### `Profiler::Scope`

`Profiler::Scope` is a move-only RAII object.

It has the following behavior:

- starts timing when constructed
- ends timing when destroyed
- records scope statistics automatically
- supports moving
- cannot be copied

This prevents common mistakes such as starting a timer and forgetting to stop it.

______________________________________________________________________

### `ProfilerView`

`ProfilerView` is a restricted view used by `BrickBaseV1`.

It intentionally lives with the brick base API because it defines what brick implementations are
allowed to do with profiling:

- measure scopes
- measure model invocations
- inspect whether profiling is enabled
- inspect whether per-op profiling is enabled
- inspect whether external timeline annotations are enabled

It does not expose:

- `reset()`
- `configure()`
- direct ownership of the profiler lifecycle

This prevents derived brick implementations from invalidating the lifecycle owned by `BrickBaseV1`.

______________________________________________________________________

## Lifecycle

### Configuration

The profiler is configured through `ProfilingOptions`.

For `BrickBaseV1`, configuration happens during `setup()`:

```cpp
_profiler.reset();
_profiler.configure(options.profiling);
```

Legacy bricks that do not use `BrickBaseV1` must still route profile output through
`Profiler::write()` so that configured output format and units are honored.

______________________________________________________________________

### Reset

Reset clears:

- model statistics
- operation statistics
- scope statistics
- active scope state

`BrickBaseV1` owns reset behavior:

| Brick method      | Profiler behavior                               |
| ----------------- | ----------------------------------------------- |
| `setup()`         | reset, configure, measure `setup`               |
| `update_config()` | reset, measure `update_config`                  |
| `run()`           | measure `run`                                   |
| `reset_states()`  | reset after implementation state has been reset |

______________________________________________________________________

### Disabled profiling

When statistics are disabled:

- statistics timing calls return early
- reports are empty
- `write(os)` writes nothing

Timeline scopes continue to emit when `enable_timeline_annotations` is active. If statistics and
timeline annotations are both disabled, scope creation returns an empty scope.

This avoids showing empty tables when profiling is disabled.

______________________________________________________________________

## Scope Reporting

### Single-call scopes

If a scope is called once, the report contains one section for that scope:

```text
Scope: setup
Scope time | calls: 1 | total ms: 125.40 | avg ms: 125.40 | min ms: 125.40 | max ms: 125.40

Model Performance Summary
...
```

The table reports the work performed inside the scope. The scope time is displayed outside the
table.

______________________________________________________________________

### Repeated scopes

If the same scope is called multiple times, the report contains two tables:

```text
Scope: run
First Call
Scope time | calls: 1 | total ms: 350.00 | avg ms: 350.00 | min ms: 350.00 | max ms: 350.00

Model Performance Summary
...

Scope: run
All Calls
Scope time | calls: 3 | total ms: 820.00 | avg ms: 273.33 | min ms: 130.00 | max ms: 350.00

Model Performance Summary
...
```

The model names are not modified. The report uses section labels instead of appending text such as
`(first run)` or `(all runs)` to model names.

______________________________________________________________________

### Streaming mode

Streaming bricks call `run()` repeatedly until generation is complete.

For streaming, this means the `run` scope is usually called more than once. The profiler reports:

- first `run()` call
- all `run()` calls aggregated

This captures both:

- first-chunk latency
- full streaming execution cost

If `run()` is called only once, only a single `run` scope table is emitted.

______________________________________________________________________

## Output Formats

### Pretty output

Pretty output is intended for terminal inspection.

It includes:

- scope name
- scope timing
- model table
- optional operation breakdown
- overhead, computed as scope time minus contributor time
- grand total for contributor rows plus overhead

Scope time is a container duration, so it is displayed separately. The performance summary
reconciles back to that container duration by adding an `Overhead` row.

______________________________________________________________________

### CSV output

CSV output is intended for tooling.

The CSV schema is:

| Column         | Description                                       |
| -------------- | ------------------------------------------------- |
| `scope`        | Scope name, such as `run` or `setup`              |
| `table`        | Empty, `First Call`, or `All Calls`               |
| `record_type`  | `scope`, `subscope`, `model`, `op`, or `overhead` |
| `runner`       | Runner/backend name for model and op rows         |
| `parent`       | Parent model for op rows                          |
| `name`         | Scope, model, or operation name                   |
| `calls`        | Number of samples                                 |
| `total_<unit>` | Total time in the configured unit                 |
| `avg_<unit>`   | Average time in the configured unit               |
| `min_<unit>`   | Minimum sample time in the configured unit        |
| `max_<unit>`   | Maximum sample time in the configured unit        |
| `scope_calls`  | Number of times the scope was called              |

Example:

```csv
scope,table,record_type,runner,parent,name,calls,total_ms,avg_ms,min_ms,max_ms,scope_calls
run,First Call,scope,,,run,1,350.000,350.000,350.000,350.000,3
run,First Call,subscope,n.a.,run,text_encoder,1,90.000,90.000,90.000,90.000,3
run,First Call,model,litert,,speech_decoder,1,130.000,130.000,130.000,130.000,3
run,First Call,overhead,,,Overhead,1,130.000,130.000,,,3
run,All Calls,model,litert,,speech_decoder,3,390.000,130.000,129.000,131.000,3
```

______________________________________________________________________

### Units

The output unit is controlled by `ProfilingOptions::unit`.

Supported units:

| Unit         | Suffix |
| ------------ | ------ |
| Seconds      | `s`    |
| Milliseconds | `ms`   |
| Microseconds | `us`   |
| Nanoseconds  | `ns`   |

The default unit is milliseconds.

______________________________________________________________________
