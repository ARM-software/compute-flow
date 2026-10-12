<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Running a minimal Compute Flow application on native platforms and Android™

## Dependencies

- A Mac® with Apple silicon, or a Linux® system with an Arm®-based CPU, such as a Raspberry Pi®.
- `uv`, `git`, and a native C and C++ compiler.
- `adb` and an Android™ device for the Android™ steps.

## Goal

This guide installs the Qwen3-TTS 0.6B Custom Voice brick and model, builds the Compute Flow SDK for
native and Android™ targets, and links it to a minimal C++ application. The application runs one
text-to-speech prompt and reports the number of generated audio samples.

Run all commands from the Compute Flow project root.

## Install the brick and model

### Step 1: Set up Compute Flow

Create and activate the project environment:

```bash
uv sync
source .venv/bin/activate
```

Check the required system dependencies:

```bash
compute-flow check
```

If the check reports a missing dependency, follow the suggested action and run the check again.

### Step 2: Download the brick and model

Update the local brick registry, then install the Qwen3-TTS 0.6B Custom Voice model:

```bash
compute-flow update
compute-flow model install tts:qwen3-tts qwen3-tts-0-6b-custom-voice
```

The model command installs both the `tts:qwen3-tts` brick source and the
`qwen3-tts-0-6b-custom-voice` model package. Inspect the installation and save the model package
path:

```bash
compute-flow model list tts:qwen3-tts
compute-flow model info tts:qwen3-tts qwen3-tts-0-6b-custom-voice
PACKAGE_DIR="$(compute-flow model path tts:qwen3-tts qwen3-tts-0-6b-custom-voice)"
```

## Build the native SDK

### Step 3: Build Compute Flow

Build Compute Flow and the installed brick for the current machine:

```bash
compute-flow build native
```

The `native` target uses the host compiler on both macOS® and Linux®.

After the build completes, the generated SDK is available in the build directory. For a native
build, it is located at `build_native/sdk`. This folder contains everything needed to build a C++
application with the bricks that were installed when Compute Flow was built: public headers, brick
and runtime libraries, CMake integration, and installed brick metadata.

The SDK also contains an `info.json` file for each installed brick. In `info.json`, you can find the
links to download the available models for that brick at runtime.

> ℹ️ Model packages remain separate and are provided to the application at runtime through their
> manifests.

Copy the SDK to a new application directory:

```bash
mkdir minimal_example
cp -R build_native/sdk/. minimal_example/
```

The copied directory already contains `CMakeLists.txt`, `include/`, and `lib/`.

## Create the application

### Step 4: Add the source file

Add `main.cpp` to the new `minimal_example` directory:

```cpp
#include <compute_flow/compute_flow.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace bricks = compute_flow::bricks;

int32_t main(int32_t argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: minimal_example <manifest.json>\n";
        return 1;
    }

    const std::string manifest_path = argv[1];

    // Load the model configuration, then create the installed brick implementation.
    auto manifest                  = bricks::load_manifest(manifest_path);
    auto brick                     = bricks::load_brick(manifest);

    // Prepare the brick and warm it up to improve the first inference.
    bricks::BrickOptions options;
    options.runtime.enable_warmup = true;
    brick->setup(manifest, options);

    std::string prompt = "Hello, I am Aiden, your personal AI assistant.";

    // Describe the text prompt with the brick's public tensor interface.
    compute_flow::TensorDict input;
    input.declare_tensor("text_prompt");
    input["text_prompt"].data         = prompt.data();
    input["text_prompt"].format.dt    = compute_flow::DataType::String;
    input["text_prompt"].format.shape = {static_cast<int64_t>(prompt.size())};

    int64_t total_num_samples = 0;
    int64_t sample_rate       = 0;

    // Continue running until the brick reports that audio generation is complete.
    while (!brick->is_generation_complete()) {
        const auto output = brick->run(input);

        // Read this audio chunk through the public TTS output tensors.
        total_num_samples += output["left_channel"].format.shape[0];
        sample_rate = *static_cast<const int64_t*>(output["sample_rate"].data);
    }

    std::cout << "Generated " << total_num_samples << " samples at " << sample_rate << " Hz.\n";

    // Clear request state while keeping the loaded models ready for reuse.
    brick->reset_states();
    return 0;
}
```

### Step 5: Link the SDK with CMake

The copied `minimal_example/CMakeLists.txt` already sets the project name:

```cmake
set(PROJ_NAME minimal_example)
```

The generated CMake file uses `${PROJ_NAME}` to create the executable, links it to
`compute_flow::sdk`, and configures the required C++ standard and platform link dependencies. Change
`PROJ_NAME` to use another application name.

### Step 6: Build and run the application natively

Configure and build the application from the Compute Flow project root:

```bash
cmake -S minimal_example -B minimal_example/build
cmake --build minimal_example/build --parallel 4
```

Run it with the English Qwen3-TTS 0.6B Custom Voice manifest:

```bash
./minimal_example/build/minimal_example "$PACKAGE_DIR/qwen3_tts_custom_voice_en_manifest.json"
```

The application prints the number of generated samples per audio channel and their sample rate. It
does not write the generated audio to a file.

## Build and run on Android™

### Step 7: Build the Android™ SDK

Build Compute Flow and the installed brick for Android™:

```bash
compute-flow build android_arm64
```

### Step 8: Cross-compile the application

Copy the Android™ SDK and the same `main.cpp` into a separate application directory. Use the NDK
toolchain recorded by the Compute Flow build so the application targets the same Android™ ABI and
API level:

```bash
mkdir minimal_example_android
cp -R build_android_arm64/sdk/. minimal_example_android/
cp minimal_example/main.cpp minimal_example_android/main.cpp
ANDROID_TOOLCHAIN="$(sed -n 's/^CMAKE_TOOLCHAIN_FILE:[^=]*=//p' build_android_arm64/CMakeCache.txt)"
cmake -S minimal_example_android -B minimal_example_android/build \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_TOOLCHAIN" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build minimal_example_android/build --parallel 4
```

### Step 9: Deploy and run on the device

Push the application executable and the full model package, including its manifest, model files,
tokenizers, and other referenced assets. Run from the device directory that contains the package so
the manifest's relative paths resolve:

```bash
PACKAGE_DIR="$(compute-flow model path tts:qwen3-tts qwen3-tts-0-6b-custom-voice)"
adb shell mkdir -p /data/local/tmp/compute-flow/minimal-example/model
adb push minimal_example_android/build/minimal_example /data/local/tmp/compute-flow/minimal-example/
adb push "$PACKAGE_DIR"/* /data/local/tmp/compute-flow/minimal-example/model/
adb shell 'cd /data/local/tmp/compute-flow/minimal-example && ./minimal_example model/qwen3_tts_custom_voice_en_manifest.json'
```

The application reports the generated sample count and sample rate on the device.
