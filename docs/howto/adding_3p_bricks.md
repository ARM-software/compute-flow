<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# How to add a 3rd party brick

Run these commands from the Compute Flow project root. The Git repository of the 3rd party brick
must have at least one tag. Each tagged release must contain a root `info.json` and the brick
implementation files named by that metadata.

## Set up the Brick Manager

```bash
uv sync && source .venv/bin/activate
```

## Add the Git source

Replace the example URL with the brick repository you want to use:

```bash
compute-flow registry add https://example.com/team/example-tts.git
```

The command saves the URL in your local registry. It does not access the repository, discover
versions, or install the brick or its models. Refresh the local cache to discover tagged releases:

```bash
compute-flow update
```

List configured sources to confirm the URL and its origin:

```bash
compute-flow registry list
```

Inspect the available versions:

```bash
compute-flow brick list --type=tts
compute-flow brick info tts:example-tts@1.0.0
```

## Install a version

Select a version, then inspect its declared models:

```bash
compute-flow brick install tts:example-tts@1.0.0
compute-flow model list tts:example-tts
```

If the brick declares a model you need, install it with
`compute-flow model install tts:example-tts <model-name>`.

Run `compute-flow update` later to discover newly tagged versions. Updating the cache leaves the
installed brick version unchanged; install a new version explicitly when you are ready to switch.

To stop discovering versions from this repository, remove its source:

```bash
compute-flow registry remove https://example.com/team/example-tts.git
```

Removing the source leaves any installed brick and models in place. Run `compute-flow update` to
remove its available versions from the cache.
