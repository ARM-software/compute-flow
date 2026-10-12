<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Compute Flow product overview

Compute Flow is an SDK that helps developers build AI applications from pre-built building blocks
called bricks. A brick provides a specific AI capability, such as text-to-speech or speech-to-text.
Each brick type has a well-defined API, available from C/C++ and Python. This makes it easier to
combine bricks into larger AI pipelines.

Bricks can have different variants. For example, Qwen3-TTS and Pocket-TTS are text-to-speech bricks.
Because they follow the same API, developers can swap one variant for another without changing the
application code or rebuilding the application.

To make bricks easier to find and use, Compute Flow includes a friendly command-line tool called the
Brick Manager. The Brick Manager helps developers discover and install bricks, and build Compute
Flow for the target device they want to run on.

## Compatibility principle

Backward compatibility is a core principle of Compute Flow: a brick that works today should continue
to work with future releases. Compute Flow brick and runner APIs must remain backward compatible.
New bricks can adopt newer API versions, while bricks built against previous API versions must
continue to work with newer releases of Compute Flow.

## Goals

- Make bricks discoverable and installable through the Compute Flow Brick Manager.
- Keep applications that consume bricks independent from model-specific file layouts, manifests, and
  runner setup.
- Support native and cross-compiled builds through the same installed-brick workflow.
- Keep examples simple enough for smoke testing and profiling on real devices.
- Keep C++ APIs portable across supported targets.
- Require model manifests to declare model files, tensor formats, runner settings, assets, and
  constants explicitly.

## Constraints

- Code uses the `compute_flow` namespace and paths.
- User-facing prose uses Compute Flow.
- Bricks should be validated with real model artifacts and representative sample inputs when
  possible.
- Model packages should record source, license, conversion, and validation details before they are
  published.
- Ideally, performance work should keep model quality artifacts, profiling logs, target hardware,
  thread count, model precision, and runtime framework visible.

## Non-goals

- Compute Flow is not a general model zoo. Model repositories hold the artifacts needed by supported
  bricks.
- Compute Flow does not require every brick to expose every backend supported by core. A brick
  should expose only the frameworks that are implemented and validated.
