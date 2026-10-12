<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Capturing Arm® Streamline traces on Linux® and Android™

## Dependencies

- Arm® Streamline Performance Analyzer, installed as part of Arm® Performance Studio.
- The `gatord` binary from the same Arm® Performance Studio installation as Streamline.
- `ssh` and `scp` for a remote Linux® target, or `adb` for an Android™ target.
- A Compute Flow application and its runtime files deployed to the target.

Running `gatord` as root provides access to system-wide hardware counters. Non-root capture is also
supported, but the available counters are more limited.

## Install Streamline

Streamline is installed on the host workstation as part of Arm® Performance Studio:

1. Open the [Arm® Performance Studio downloads][performance-studio-downloads].
2. Select the package for the host workstation and follow the installation guide linked from the
   download page.
3. Use the Performance Studio Hub to launch Streamline once and confirm that the installation works.
4. Locate the installation's `streamline` directory. The capture steps below use its matching
   `gatord` binaries.

Keep the installation directory available while following this guide. Do not copy `gatord` from a
different Streamline version because its capture protocol might be incompatible.

## Goal

This guide shows how to build Compute Flow with Streamline annotations, run `gatord` on an Arm®
64-bit target, capture an application through the Streamline graphical interface, and inspect the
Compute Flow timeline tracks.

Use the `gatord` supplied with the installed Streamline version. A different version might use an
incompatible capture protocol.

## Prepare the application

Run these commands from the Compute Flow project root:

```bash
uv sync
source .venv/bin/activate
compute-flow check
```

Build for the target with Streamline annotations:

```bash
# Cross-compile for an Arm® 64-bit Linux® target
compute-flow build linux_arm64 --streamline

# Or cross-compile for an Android™ target
compute-flow build android_arm64 --streamline
```

Use `compute-flow build native --streamline` when building directly on a Linux® host.

Before capturing, deploy all files needed to run the application:

- the application or CLI binary from `build_linux_arm64` or `build_android_arm64`
- the selected manifest JSON
- every model, tokenizer, asset, and sample input referenced by the manifest

Keep the manifest and its referenced files in the same relative layout on the target. Run the
application from the directory used when validating it without Streamline.

## Capture on Linux®

### Step 1: Copy and start `gatord`

Set `STREAMLINE_DIR` to the `streamline` directory in the Arm® Performance Studio installation:

```bash
STREAMLINE_DIR=/path/to/Arm_Performance_Studio/streamline
```

Copy the Arm® 64-bit Linux® daemon to the target and make it executable:

```bash
scp "$STREAMLINE_DIR/bin/linux/arm64/gatord" <user>@<target>:/tmp/gatord
ssh <user>@<target> chmod +x /tmp/gatord
```

Run it in the foreground. Keep this terminal open during the capture:

```bash
ssh -t <user>@<target> sudo /tmp/gatord --allow-command
```

Run `/tmp/gatord --allow-command` without `sudo` if root access is unavailable.

### Step 2: Connect and capture

In Streamline:

1. Open the **Start** view and select **TCP** as the device type.
2. Enter the target hostname or IP address. `gatord` listens on port `8080` by default.
3. In **Configure application**, enter the absolute application command and its arguments.
4. Set **Working directory** to the target directory containing the application and runtime files.
5. Select **Stop on exit** if the capture should end when the application exits.
6. Select **Start capture**.
7. When the application completes, select **Stop capture and analyze** if capture is still running.

If the target is reachable only through SSH, forward the `gatord` port:

```bash
ssh -L 8080:localhost:8080 -N <user>@<target>
```

Then connect Streamline to `127.0.0.1:8080`.

## Capture on Android™

### Step 1: Copy and start `gatord`

Set `STREAMLINE_DIR` to the `streamline` directory in the Arm® Performance Studio installation:

```bash
STREAMLINE_DIR=/path/to/Arm_Performance_Studio/streamline
```

Copy the Arm® 64-bit Android™ daemon to the device and make it executable:

```bash
adb push "$STREAMLINE_DIR/bin/android/arm64/gatord" /data/local/tmp/gatord
adb shell chmod +x /data/local/tmp/gatord
```

Run it in the foreground. Keep this terminal open during the capture:

```bash
adb shell /data/local/tmp/gatord --allow-command
```

### Step 2: Connect and capture

In Streamline:

1. Open the **Start** view and select **TCP** as the device type.
2. Select the `adb <serial-number>` target for the device.
3. In **Configure application**, enter the absolute application command and its arguments.
4. Set **Working directory** to the device directory containing the application and runtime files.
5. Select **Stop on exit** if the capture should end when the application exits.
6. Select **Start capture**.
7. When the application completes, select **Stop capture and analyze** if capture is still running.

## Inspect and reopen the trace

After analysis, Streamline opens the **Timeline** view automatically. Expand the captured
application and its thread, then expand the **Compute Flow** annotation group. Depending on the
application, it contains:

- **Lifecycle** intervals such as `setup`, `run`, and `update_config`
- **Stage** intervals created by nested brick scopes
- **Model** intervals labelled with the runner and model name

Streamline assigns interval colours from its built-in rotating palette to help distinguish adjacent
annotations. These colours are visual aids; they do not encode the track type, runner, model status,
or another stable identity.

Streamline stores the capture in its **Streamline Data** view. Double-click a completed capture to
open it again. If its report needs to be regenerated, right-click it and select **Analyze**.

## Troubleshooting

- If no Compute Flow tracks appear, confirm that the deployed binary came from a build using
  `--streamline`.
- If Streamline cannot connect, confirm that `gatord` is still running and that TCP port `8080` is
  reachable, or select the correct `adb` target.
- If Streamline reports a protocol mismatch, replace the target daemon with `gatord` from the same
  Streamline installation.
- If hardware counters are missing, rerun `gatord` with the privileges required by the target.

For local or headless capture, advanced counter configuration, and target-specific troubleshooting,
see the official [Streamline user guide][streamline-user-guide],
[Linux® target setup guide][linux-target-guide], and
[Android™ target setup guide][android-target-guide].

[android-target-guide]: https://documentation-service.arm.com/static/695d3143d62b0164052767ac
[linux-target-guide]: https://documentation-service.arm.com/static/68921f45e7f7ce6150e87a7a
[performance-studio-downloads]: https://developer.arm.com/tools-and-software/arm-performance-studio#Downloads
[streamline-user-guide]: https://documentation-service.arm.com/static/68921f50e7f7ce6150e87cc2
