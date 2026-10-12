<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Developing a brick

A Compute Flow brick consists of two parts:

- a model package
- C++ inference code

These usually live in separate repositories. For example, for the Qwen3-TTS brick:

- The [model package](https://huggingface.co/Arm/qwen3-tts-0-6b-custom-voice-mix-precision) is
  hosted on Hugging Face
- [`bricks/tts/qwen3_tts/`](../../bricks/tts/qwen3_tts/) contains the C++ inference code in Compute
  Flow

A model package can also be hosted on ModelScope or another model repository, and inference code can
be hosted in another Git repository, such as GitHub or GitLab.

## Before you start - choose the brick API

The first question to answer is: which API does this brick need? Check
[`brick_apis/`](../../brick_apis/) for an existing type. If one fits, follow its input and output
contract so applications based on the same brick type can switch variants without changing a single
line of code. If you need a new API or brick type, define the [brick API](../design/brick_api.md)
and its [C++ and Python CLI examples](../design/brick_cli_examples.md) before continuing.

If the brick type already exists, your implementation must comply with the existing brick interface.
The brick type interface defines the input tensors an application supplies and the output tensors it
consumes. It does not depend on the model used by a brick variant.

By adhering to the existing interface:

- Your brick can be executed using the existing Brick Type CLI.
- Applications can adopt your brick without requiring code changes.
- Users can switch between brick variants transparently.

## Step 1 - Model package requirements

Before writing inference code, export the original model, such as a PyTorch model, to one or more
runners supported by Compute Flow.

Use an exporter compatible with each runner. For PyTorch models, consider:

- LiteRT: [`litert_torch`](https://github.com/google-ai-edge/litert-torch)
- ExecuTorch: its [export pipeline](https://docs.pytorch.org/executorch/stable/getting-started.html)
- ONNX Runtime: PyTorch's
  [`torch.onnx.export`](https://onnxruntime.ai/docs/tutorials/accelerate-pytorch/pytorch.html)

Check the runner versions in the [third-party notices](../../THIRD_PARTY_NOTICES.md). You can mix
runners when one cannot export every component or meet the performance target.

For example, the current Qwen3-TTS package splits more than 20 components across LiteRT, ONNX
Runtime, and llama.cpp.

Once exported, compare every component with the original model on representative inputs before
integration.

Next, create a [manifest](../design/brick_manifest.md) for the package. It declares the brick
identity, model files, runners, input and output tensors, configuration, constants, and required
assets. A package might look like this:

```text
<model-package>/
  model0.tflite
  model1.onnx
  model2.gguf
  <name>_manifest.json
  assets/
  samples/
  README.md
  SHA256SUMS
```

Keep runtime assets in `assets/` and declare their paths in the manifest. Keep
`<name>_manifest.json` as the only root JSON file; put JSON assets such as `tokenizer.json` under
`assets/`.

If a CLI example needs sample audio, images, or text, put inputs in `samples/` so users can select
them with paths such as `@sample/example.wav` when using the CLI example. Prefer samples with the
same license as the model. Samples are included in the package checksums but are not runtime assets.

Generally, the README.md should record model, asset, and sample sources and licenses, quality and
performance results, and how to add the package to the Brick Manager registry.

Once the models, manifest, assets, and samples are ready, generate `SHA256SUMS` for all non-dotfiles
in the package. For this step, you can use the following script available in Compute Flow:

```bash
scripts/generate_checksum.py path/to/model-package
```

## Step 2 - Brick implementation requirements

The brick implementation is C++ inference code. It reads the brick manifest to learn which models to
execute, which runners to use, which constants to initialize, and which optional assets to load. It
contains at least a `.cpp` file, a `.h` file, and an `info.json` file.

For an `ExampleDetector` variant, the brick source might look like this:

```text
bricks/audio_detection/example_detector/
  ExampleDetector.cpp
  ExampleDetector.h
  info.json
```

An `internal/` subdirectory may contain additional source files. The Brick Manager generates build
and binding files from this layout. If useful, create starter files with:

```bash
python3 scripts/create_brick_template.py audio_detection ExampleDetector
```

Replace the generated placeholders, implement the inference code with the Compute Flow Runner API,
and read model details from the manifest.

- See [brick structure](../design/brick_structure.md) for more information about the brick structure
  and the `info.json` fields.
- See the [guidelines for developing bricks](#guidelines-for-developing-bricks) to learn good
  practices for brick development.

In `info.json`, declare each model source. Get its `sha256` value by hashing the generated
`SHA256SUMS` file itself:

```bash
shasum -a 256 path/to/model-package/SHA256SUMS
```

Copy the first value in the output into `info.json`. The `-c` option checks the files listed inside
`SHA256SUMS` instead of producing this digest.

A model package can be hosted in multiple repositories, but every source must have the same SHA256
hash for its `SHA256SUMS` file.

## Step 3 - Test the brick

Compute Flow automates brick tests so developers can run them with less setup:

- Brick authors decide which models, manifests, and inputs to test and how to validate the outputs.
- Compute Flow generates a script to run the test cases through the CLI examples. Developers can run
  it locally or call it from CI.

This workflow covers native builds. Android™ testing is not covered at this stage.

### Test requirements

Keep the test cases inside a `test/` folder in the brick, beside its implementation:

```text
bricks/vad/silero/
  SileroVAD.cpp
  SileroVAD.h
  info.json
  test/
    test_cases.yaml
```

When the brick is installed, Brick Manager generates `test/run_tests.sh` in its installed source
copy from `info.json` and `test/test_cases.yaml`. List the manifests to test and the CLI arguments
for each test case:

```yaml
models:
  silero-fp32:
    silero_manifest.json:
      - args: ["-a", "@sample/beckett.wav"]
      - args: ["-a", "@sample/beckett.wav", "-t", "2"]
```

The first key is a model name from `info.json`; the next key is a manifest filename in that model's
installed package. Each manifest has one or more test cases. Each case has an `args` list of CLI
arguments, excluding `-m`. The arguments can differ between cases and manifests. Use `@sample/...`
for input files in the model package's `samples/` directory.

The native build and Python CLI environment must be ready before running the generated script.

For each model named in `test_cases.yaml`, the script:

1. Installs the model with `compute-flow model install <type:name> <model>`. Each model must be
   declared in the installed brick's `info.json`; installation verifies the model package checksums.
2. Checks that each selected manifest exists in the installed model package.
3. Runs
   `python -m cli_examples.<type>_cli -m @manifest/<brick-name>/<model-name>/<manifest-file> <args>`
   once per test case, passing each `args` item as a separate argument.
4. Uninstalls the model after its test cases pass, then moves to the next model.

A failed command or missing manifest stops the script with a nonzero exit status. An exit trap
attempts to uninstall the active model before the script exits. The native nightly CI jobs run
generated scripts for installed bricks that provide test cases.

For the example above, the generated script could look like this:

```bash
#!/usr/bin/env bash
set -euo pipefail

compute-flow model install vad:silero-vad silero-fp32
trap 'compute-flow model uninstall vad:silero-vad silero-fp32' EXIT
model_dir_0=$(compute-flow model path vad:silero-vad silero-fp32)
[[ -f "$model_dir_0/silero_manifest.json" ]] || exit 1
python -m cli_examples.vad_cli \
  -m @manifest/silero-vad/silero-fp32/silero_manifest.json \
  -a @sample/beckett.wav
python -m cli_examples.vad_cli \
  -m @manifest/silero-vad/silero-fp32/silero_manifest.json \
  -a @sample/beckett.wav -t 2
compute-flow model uninstall vad:silero-vad silero-fp32
trap - EXIT
```

The first version checks that installation, CLI invocation, and uninstallation succeed. It does not
check whether the output is correct. A future version may add a brick-local output validation
function to each test case. Bricks without `test/test_cases.yaml` do not get a generated test
script. To run the script after building for a native target:

```bash
bash "$(compute-flow brick path vad:silero-vad)/test/run_tests.sh"
```

The Brick Manager copies brick sources during installation. After changing a local brick's tests,
reinstall it with `compute-flow brick install <type:name> --force` before generating the script
again.

## Guidelines for developing bricks

### API guidelines

Every brick must implement the virtual methods defined by `BrickBaseV1`. See the
[brick runtime interface](../design/brick_runtime_interface_v1.md) for the authoritative contract.

#### Public APIs

Bricks must not expose additional public APIs beyond the defined brick interface. All functionality
should be accessible through the standard Compute Flow contract.

### Runtime memory guidelines

#### Use the MemoryPlanner

Intermediate tensors should be stored in the **MemoryPlanner** whenever possible. Future releases
may use it to:

- Reduce memory fragmentation
- Reuse memory more effectively
- Improve runtime performance

#### Exception for strings

String-based intermediate data may be stored using:

```cpp
std::vector<std::string>
```

when appropriate.

#### Allocate early

Whenever tensor shapes are known during setup, allocate intermediate buffers during:

```cpp
setup()
```

rather than:

```cpp
run()
```

This minimizes runtime overhead.

### Performance guidelines

#### Dynamic shapes

Some models require dynamic tensor shapes that depend on user input. In these cases:

- Dynamic allocation may be unavoidable.
- Additional runtime buffers may be created as needed.
- Assertions should be added when appropriate to validate expected shape behaviour.

Whenever possible, prefer static shapes for maximum performance.

#### Precompute during setup

Any computation that is identical across invocations should be performed during:

```cpp
setup()
```

rather than repeatedly in:

```cpp
run()
```

Precomputed tensors should be stored in the MemoryPlanner.

### Code quality guidelines

Prioritise:

- Readability
- Maintainability
- Clear separation of responsibilities

over aggressive micro-optimisations.

#### Use the Runner API

Model execution should always be performed through the **Compute Flow Runner API**. The Runner API
is runner-agnostic and allows bricks to operate with different execution backends without code
changes.

For example, switching between:

- LiteRT
- ExecuTorch

should only require updates to the manifest rather than modifications to brick code. This separation
keeps bricks portable and simplifies experimentation with different runtimes.

#### Metadata

Avoid hard-coding constants directly in brick implementations. Instead:

- Store constants in the manifest as metadata.
- Retrieve them through the appropriate configuration mechanisms.

This allows one brick implementation to support model variants that differ only in input and output
shapes or the number of layers.

#### Prefer simpler control flow

If an optional feature introduces significant branching complexity, consider moving it into
dedicated helper functions. For example, optional voice cloning support may be easier to maintain
when separated from the core inference path.

#### Use descriptive names

Variable names should clearly communicate intent and ownership. Avoid abbreviations unless they are
widely understood within the domain.

#### Follow project conventions

All brick implementations must comply with the
[C++ coding standards](../reference/cpp_code_conventions.md).

### Streaming and multi-step generation

Some brick types generate outputs incrementally, including:

- Large Language Models (LLMs)
- Streaming Text-to-Speech systems
- Future streaming-capable brick variants

For these scenarios, the brick must correctly report generation status through:

```cpp
is_generation_complete()
```

Applications rely on this status to determine whether output generation has completed.

Even if a current variant produces outputs in a single step, future variants of the same brick type
may support streaming. Correctly implementing generation status ensures compatibility across all
variants.
