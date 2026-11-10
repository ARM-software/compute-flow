<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# How to add a 3rd party brick

Run these commands from the Compute Flow project root. A third-party Git repository can contain one
or more bricks.

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
versions, or install bricks or models. Refresh the local cache to fetch all tags from every
registered repository, including the official Arm® registry:

```bash
compute-flow update
```

List configured sources to confirm the URL and its origin:

```bash
compute-flow registry list
```

Inspect the available versions. A brick's versions are the tags whose registries contain it:

```bash
compute-flow brick list --type=tts
compute-flow brick info tts:example-tts@1.0.0
```

## Install a version

Omit `@version` to install the latest version of a brick:

```bash
compute-flow brick install tts:example-tts
```

Select a version, then inspect its declared models:

```bash
compute-flow brick install tts:example-tts@1.0.0
compute-flow model list tts:example-tts
```

If the brick declares a model you need, install it with
`compute-flow model install tts:example-tts <model-name>`.

Installation copies the selected brick from the tagged commit recorded by `update`. Run
`compute-flow update` later to discover new registry tags. It reports newer versions of installed
bricks and shows the command to install each update. Your installed brick version stays unchanged
until you choose to install another version.

To stop discovering versions from this repository, remove its source:

```bash
compute-flow registry remove https://example.com/team/example-tts.git
```

Removing the source leaves any installed brick and models in place. Run `compute-flow update` to
remove its available versions from the cache.
