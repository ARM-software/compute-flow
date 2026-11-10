<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Custom operator package contract

A brick with custom operators includes `custom_ops/schema.yaml` and one folder per operator. The
brick author supplies these sources:

| Role      | Location                    | Purpose                                                                                                      |
| --------- | --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Reference | `reference.c`               | Required portable C99 fallback kernel.                                                                       |
| Optimized | `cpu/`                      | Optional architecture-specific C kernels and paired assembly declared in `cpu/registration.yaml`.            |
| Common    | `calculate_output_shapes.c` | Required portable C99 function that calculates output dimensions from current input shapes for every kernel. |

Each operator folder may contain only `reference.c`, `calculate_output_shapes.c`, and an optional
`cpu/` folder. Within `cpu/`, only `registration.yaml`, each declared kernel's `.c` source, and its
optional matching `.S` source are allowed. Custom operator headers, undeclared sources, and other
files or folders cause installation to fail. Inline assembly in the portable C sources and declared
optimized C sources is prohibited. Brick installation checks the schema, folder layout, required
includes, and assembly use; it does not compile or register kernels. The Brick Manager records
declarations that pass these installation checks in the `custom_ops` inventory in `cache.json` and
removes them when the brick is uninstalled.

```text
custom_ops/
  schema.yaml
  add_fp32/
    reference.c
    calculate_output_shapes.c
```

The schema declares the names seen by a model, with tensor order, types, and shapes:

```yaml
operators:
  - name: add_fp32
    version: 1
    inputs:
      - {name: lhs, dtype: float32, shape: [N]}
      - {name: rhs, dtype: float32, shape: [N]}
    outputs:
      - {name: sum, dtype: float32, shape: [N]}
```

The schema has these requirements:

- Operator names use lowercase letters, digits, and underscores, start with a letter, and match
  their folder names exactly. Versions are positive integers.
- Tensor names are identifiers unique within their input or output list. Inputs may be empty; at
  least one output is required.
- Shape dimensions are nonnegative integers or symbolic identifiers, with one to eight dimensions.
  Scalar values must be passed as one-element tensors with shape `[1]`; the interface has no scalar
  arguments.
- Tensor ranks are fixed by the schema. Input and output dimensions may vary between calls; weights
  and constants keep fixed shapes. Output dimensions must be derivable from input shapes alone.
- Supported tensor dtypes are `bool`, `float16`, `bfloat16`, `float32`, `float64`, `int8`, `int32`,
  `int64`, `uint8`, `uint32`, and `uint64`.

Dimensions keep the schema's shape order. For a contiguous `float32` image with shape
`[3, height, width]`, `CF_TensorFormat` has `rank = 3`, `dims = [3, height, width]`, and byte
`strides = [4 * height * width, 4 * width, 4]`. Width is the fastest-varying dimension, and the
three channels occupy separate image planes. Implementations must use the supplied strides for other
layouts.

## Generated source contract

The source contract is defined by the Brick Manager's
[interface template](../../compute_flow/python/brick_manager/custom_ops/templates/interface.h.in).
Each reference and optimized implementation must define the four entry points declared there:
`CF_GET_PERSISTENT_MEMORY_REQUIREMENT`, `CF_GET_SCRATCH_MEMORY_REQUIREMENT`, `CF_SETUP`, and
`CF_RUN`. Each operator's `calculate_output_shapes.c` defines `CF_CALCULATE_OUTPUT_SHAPES` with the
signature in the
[shape declaration template](../../compute_flow/python/brick_manager/custom_ops/templates/calculate_output_shapes.h.in):

```c
int64_t CF_CALCULATE_OUTPUT_SHAPES(
    const CF_TensorShape* input_shapes,
    size_t num_inputs,
    CF_TensorShape* output_shapes,
    size_t num_outputs);
```

The function fills the fixed-rank output shapes and returns zero on success. It cannot inspect
tensor values. Both authored C files include
[`compute_flow/custom_ops/v1/types.h`](../../include/compute_flow/custom_ops/v1/types.h) for their
shared argument types. Installation fails if either file omits this include. The Brick Manager
generates separate, uniquely named headers and C sources for the reference and shape functions,
writing entry point names qualified by brick version and operator directly into both files. Each
generated header includes the shared types header, so multiple operator headers can be included in
one source file.

The next Compute Flow build:

- Compiles the generated reference and shape sources and builds a small executable that references
  all five entry points. The executable links against their object. Wrong signatures fail
  compilation; missing entry points fail linking. Both sources are included in the brick core
  library.
- Compiles each declared optimized C source under `cpu/` as a separate target on AArch64 when the
  compiler accepts its `build_features` flags. Each source has the same link check and is built with
  the brick core. An unsupported flag produces a warning and skips that target.
- Compiles a matching preprocessed `.S` file into the same target when the assembler accepts the
  source and its `build_features` flag. If it cannot, the build warns and uses the C implementation.
- Adds each compiled optimized kernel to the brick registry in declaration order, followed by the
  required reference implementation. Kernel selection uses this order and falls back to the
  reference when no earlier implementation's hardware requirements are met.

The interface passes tensor dimensions, byte strides, buffers, workspace, and thread IDs. Tensor
dtypes are declared in the schema. Each entry point returns zero on success and a nonzero value on
failure.

## Workspace ownership

`compute_flow::custom_ops::MemoryManager` owns one aligned workspace buffer. It is declared in
[`compute_flow/custom_ops/memory_manager.h`](../../include/compute_flow/custom_ops/memory_manager.h).

Use the manager as follows:

- Keep a manager for each distinct prepared state. No registration is needed in `MemoryManager`.
  Query the selected kernel's persistent requirement and call `allocate(bytes, alignment)` before
  `setup()`. The LiteRT adapter prepares a state once per constant input buffer listed in
  `setup_inputs` and reuses it for later calls with that buffer, even when other input shapes
  change. LiteRT's `Init` callback receives initialization data but no input tensor buffers, so it
  cannot pack an RHS tensor there. The adapter calls the Compute Flow kernel's `setup()` on the
  first `Run` that receives the tensor. List every constant input read by `setup()` or used to size
  persistent memory in `cpu/registration.yaml`. Its shape and contents must remain unchanged. Read
  changing input values in `run()`. An empty `setup_inputs` list uses all input formats to identify
  prepared state. Recreating the compiled model starts a new adapter and prepares a new state.
- Share one scratch manager across custom operator calls that do not overlap. Query the selected
  kernel's scratch requirement with the current tensor formats and call `allocate()` before each
  `run()`. Sequential calls reuse capacity; requests are not added together.
- Construct `CF_Workspace{manager.data(), manager.size()}` for the kernel call. A zero-byte request
  yields a null, zero-sized workspace while retaining capacity for a later call.
- Give calls that can overlap, including concurrent model executions, separate scratch managers.

The manager releases its buffer when destroyed or reset. It does not select a kernel, query its
memory requirement, call its entry points, or allocate separate scratch storage for each thread.

## Compiled registry

[`compute_flow::custom_ops::Registry`](../../include/compute_flow/custom_ops/registry.h) stores
kernels from any interface version. The Brick Manager generates one registry per brick. Each
registration contains:

- `operator_name` and `schema_version` to match a model operator;
- `interface_version` to identify the kernel API;
- `required_features`, a list of `CpuFeature` values used to select an implementation in registry
  order;
- `kernel_name`, the implementation name;
- a typed kernel, such as `v1::Kernel`.

Each v1 kernel registration also points to its operator's single `CF_CALCULATE_OUTPUT_SHAPES`
function, including optimized kernels. It calculates output dimensions from current input shapes;
the schema fixes output ranks. The scratch requirement depends on current input and output formats.

The Brick Manager names a portable registration `reference` and an optimized registration after its
`registration.yaml` source stem.

The build generates a catalog of installed brick types, names, and registry accessors.
`find_compiled_registry()` searches this catalog by brick type and name; the brick version does not
affect lookup.

`detect_cpu_features()` reports available features. `select_kernel()` chooses the first compatible
implementation in registry order. The required reference implementation is last and has no feature
requirements, so it is selected when no earlier entry matches. An optimized entry may also have no
feature requirements; the build only includes it on its declared CPU target.

An installed brick declares one schema version per operator name. Input shapes alone cannot
determine outputs that depend on tensor values or node attributes.

Given a successful `lookup_kernels()` result, check the interface version before accessing a kernel:

```cpp
namespace custom_ops = compute_flow::custom_ops;

const auto& registration = result.kernels[0];

switch (registration.interface_version) {
    case custom_ops::v1::Kernel::k_interface_version: {
        const auto* kernel = registration.as<custom_ops::v1::Kernel>();

        if (kernel == nullptr) {
            // Report a mismatched registration.
            break;
        }

        // Use the v1 kernel.
        break;
    }
    default:
        // Skip or report an unsupported interface version.
        break;
}
```

`as<T>()` returns `nullptr` if the stored kernel has a different C++ type. An unknown interface
version can be skipped or reported as unsupported. `schema_version` describes the model operator; it
is independent of `interface_version`. Keep implementations for each operator contiguous and at one
schema version.

The v1 kernel type is defined in
[`compute_flow/custom_ops/v1/kernel.h`](../../include/compute_flow/custom_ops/v1/kernel.h). The
`HardwareFeature` enums and `CpuFeature` variant are defined in
[`compute_flow/custom_ops/registry_types.h`](../../include/compute_flow/custom_ops/registry_types.h).

## Thread pool

`compute_flow::custom_ops::ThreadPool` is a reusable C++17 thread executor declared in
[`compute_flow/custom_ops/thread_pool.h`](../../include/compute_flow/custom_ops/thread_pool.h). Its
callback interface does not depend on a custom operator kernel version:

- Thread zero runs on the caller; the pool owns the background threads.
- Calls on the same pool are serialized and finish before `run()` returns.
- A caller can vary the requested thread count between calls and reuse one pool for sequential
  custom operators. Separate pools allow calls to overlap.

## CPU feature catalog

The version 1 feature catalog is in
[`brick_manager/custom_ops/features.py`](../../compute_flow/python/brick_manager/custom_ops/features.py).
These names can be used in a kernel's `cpu/registration.yaml` and map to `HardwareFeature::Arm`
values in the compiled registry. `CpuFeature` currently contains only the Arm® feature enum. The
`arm` target describes AArch64 kernels. `hw_features` names the capabilities needed to run a kernel;
`build_features` names the extensions needed to compile its C source. Build features stay in build
metadata because each kernel can use different compiler flags. A kernel using encoded instructions
may have `hw_features: [neon-i8mm]` and `build_features: []`. `HardwareFeature::to_string()` returns
target-qualified names such as `arm_neon_i8mm`.

| `target` | `hw_features` value | C++ value          | Hardware requirement              | Build setting              |
| -------- | ------------------- | ------------------ | --------------------------------- | -------------------------- |
| `arm`    | `neon`              | `Arm::Neon`        | AdvSIMD                           | No additional flag         |
| `arm`    | `neon-dotprod`      | `Arm::NeonDotProd` | AdvSIMD dot product               | `-march=armv8.2-a+dotprod` |
| `arm`    | `neon-fp16`         | `Arm::NeonFp16`    | AdvSIMD half-precision arithmetic | `-march=armv8.2-a+fp16`    |
| `arm`    | `neon-i8mm`         | `Arm::NeonI8mm`    | AdvSIMD integer matrix multiply   | `-march=armv8.2-a+i8mm`    |
| `arm`    | `sme`               | `Arm::Sme`         | SME                               | `-march=armv9.2-a+sme`     |
| `arm`    | `sme2`              | `Arm::Sme2`        | SME2                              | `-march=armv9.2-a+sme2`    |
| `arm`    | `sve`               | `Arm::Sve`         | SVE                               | `-march=armv8.2-a+sve`     |
| `arm`    | `sve2`              | `Arm::Sve2`        | SVE2                              | `-march=armv8.2-a+sve2`    |

`neon` is a hardware requirement only; it is not a `build_features` value because AArch64 builds
already support AdvSIMD.

When a kernel declares multiple build features, the catalog combines their extensions into one
`-march` flag using the highest required architecture level. For example, `neon-dotprod` and
`neon-i8mm` produce `-march=armv8.2-a+dotprod+i8mm`. An empty `build_features` list adds no flag.
Compiler support is checked when optimized sources are built.

Declare each optimized C source in `cpu/registration.yaml` using its relative source stem:

```yaml
name: add_fp32
version: 1
kernels:
  - kernel: arm/add_neon
    target: arm
    hw_features: [neon-dotprod]
    build_features: [neon-dotprod]
```

This entry selects `cpu/arm/add_neon.c` and, when present, `cpu/arm/add_neon.S`. The registration
name and version must match the operator schema. The Brick Manager checks the declared source,
target, and feature names during installation.

Use `COMPUTE_FLOW_ASM_<name>` for every assembly function exported by the `.S` file, in both its
assembly definition and C declaration. The Brick Manager qualifies those names for each brick and
kernel. The generated C source defines `COMPUTE_FLOW_HAS_ASSEMBLY` when the `.S` file is compiled;
guard calls to assembly with `#ifdef COMPUTE_FLOW_HAS_ASSEMBLY` and provide a C path for builds
without it. For example:

```c
#ifdef COMPUTE_FLOW_HAS_ASSEMBLY
extern void COMPUTE_FLOW_ASM_add_neon(const float* lhs, const float* rhs, float* sum);
#endif

static void add_neon(const float* lhs, const float* rhs, float* sum) {
#ifdef COMPUTE_FLOW_HAS_ASSEMBLY
    COMPUTE_FLOW_ASM_add_neon(lhs, rhs, sum);
#else
    *sum = *lhs + *rhs;
#endif
}
```
