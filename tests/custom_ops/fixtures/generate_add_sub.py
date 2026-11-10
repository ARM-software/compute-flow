# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Generate a minimal LiteRT model with custom add and subtract nodes."""

import argparse
from pathlib import Path

import flatbuffers
import tflite


_PARSER = argparse.ArgumentParser()
_PARSER.add_argument("--dtype", choices=("float32", "int8"), default="float32")
_PARSER.add_argument(
    "--case",
    choices=("base", "same", "different", "subgraphs", "shared_constant"),
    default="base",
)
_PARSER.add_argument("--size", type=int, choices=(2, 3), default=2)
_PARSER.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
_ARGS = _PARSER.parse_args()
_DTYPE = _ARGS.dtype
_CASE = _ARGS.case
_SIZE = _ARGS.size
if _DTYPE == "int8" and _CASE != "base":
    _PARSER.error("Only the base case supports int8")
if _SIZE != 2 and _CASE != "base":
    _PARSER.error("Only the base case supports size 3")
_TENSOR_TYPE = (
    tflite.TensorType.FLOAT32 if _DTYPE == "float32" else tflite.TensorType.INT8
)
_OP_SUFFIX = "fp32" if _DTYPE == "float32" else "i8"

builder = flatbuffers.Builder(4096)


def int_vector(table: str, field: str, values: list[int]) -> int:
    getattr(tflite, f"{table}Start{field}Vector")(builder, len(values))
    for value in reversed(values):
        builder.PrependInt32(value)
    return builder.EndVector()


def offset_vector(table: str, field: str, values: list[int]) -> int:
    getattr(tflite, f"{table}Start{field}Vector")(builder, len(values))
    for value in reversed(values):
        builder.PrependUOffsetTRelative(value)
    return builder.EndVector()


def tensor(name: str, dimensions: list[int]) -> int:
    shape = int_vector("Tensor", "Shape", dimensions)
    shape_signature = (
        int_vector("Tensor", "ShapeSignature", [-1] * len(dimensions))
        if _CASE == "different"
        else None
    )
    name_offset = builder.CreateString(name)
    tflite.TensorStart(builder)
    tflite.TensorAddShape(builder, shape)
    if shape_signature is not None:
        tflite.TensorAddShapeSignature(builder, shape_signature)
    tflite.TensorAddType(builder, _TENSOR_TYPE)
    tflite.TensorAddBuffer(builder, 0)
    tflite.TensorAddName(builder, name_offset)
    return tflite.TensorEnd(builder)


def opcode(name: str) -> int:
    name_offset = builder.CreateString(name)
    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, tflite.BuiltinOperator.CUSTOM)
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.CUSTOM)
    tflite.OperatorCodeAddCustomCode(builder, name_offset)
    tflite.OperatorCodeAddVersion(builder, 1)
    return tflite.OperatorCodeEnd(builder)


def operator(opcode_index: int, input_indices: list[int], output_index: int) -> int:
    inputs = int_vector("Operator", "Inputs", input_indices)
    outputs = int_vector("Operator", "Outputs", [output_index])
    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, opcode_index)
    tflite.OperatorAddInputs(builder, inputs)
    tflite.OperatorAddOutputs(builder, outputs)
    return tflite.OperatorEnd(builder)


def tensor_map(name: str, index: int) -> int:
    name_offset = builder.CreateString(name)
    tflite.TensorMapStart(builder)
    tflite.TensorMapAddName(builder, name_offset)
    tflite.TensorMapAddTensorIndex(builder, index)
    return tflite.TensorMapEnd(builder)


def subgraph(
    name: str,
    tensor_specs: list[tuple[str, list[int]]],
    operator_specs: list[tuple[int, list[int], int]],
    input_indices: list[int],
    output_indices: list[int],
    prebuilt_tensors: list[int] | None = None,
) -> int:
    tensor_offsets = (
        prebuilt_tensors
        if prebuilt_tensors is not None
        else [tensor(tensor_name, shape) for tensor_name, shape in tensor_specs]
    )
    operator_offsets = [operator(*spec) for spec in operator_specs]
    tensors = offset_vector("SubGraph", "Tensors", tensor_offsets)
    operators = offset_vector("SubGraph", "Operators", operator_offsets)
    inputs = int_vector("SubGraph", "Inputs", input_indices)
    outputs = int_vector("SubGraph", "Outputs", output_indices)
    name_offset = builder.CreateString(name)
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensors)
    tflite.SubGraphAddInputs(builder, inputs)
    tflite.SubGraphAddOutputs(builder, outputs)
    tflite.SubGraphAddOperators(builder, operators)
    tflite.SubGraphAddName(builder, name_offset)
    return tflite.SubGraphEnd(builder)


tflite.BufferStart(builder)
empty_buffer = tflite.BufferEnd(builder)

if _CASE == "base":
    specs = [(name, [_SIZE]) for name in ("lhs", "rhs", "sum", "difference")]
    tensor_offsets = [tensor(name, shape) for name, shape in specs]
    opcode_offsets = [
        opcode(name) for name in (f"add_{_OP_SUFFIX}", f"sub_{_OP_SUFFIX}")
    ]
    primary = subgraph(
        "main",
        specs,
        [(0, [0, 1], 2), (1, [0, 1], 3)],
        [0, 1],
        [2, 3],
        tensor_offsets,
    )
    input_names = ("lhs", "rhs")
    output_names = ("sum", "difference")
elif _CASE == "subgraphs":
    opcode_offsets = [opcode("add_fp32")]
    specs = [(name, [2]) for name in ("lhs", "rhs", "sum")]
    primary = subgraph("main", specs, [(0, [0, 1], 2)], [0, 1], [2])
    second = subgraph("secondary", specs, [(0, [0, 1], 2)], [0, 1], [2])
    input_names = ("lhs", "rhs")
    output_names = ("sum",)
elif _CASE == "shared_constant":
    opcode_offsets = [opcode("add_scalar_fp32")]
    specs = [
        ("lhs_a", [2]),
        ("lhs_b", [3]),
        ("rhs", [1]),
        ("sum_a", [2]),
        ("sum_b", [3]),
    ]
    primary = subgraph(
        "main",
        specs,
        [(0, [0, 2], 3), (0, [1, 2], 4)],
        [0, 1, 2],
        [3, 4],
    )
    input_names = ("lhs_a", "lhs_b", "rhs")
    output_names = ("sum_a", "sum_b")
else:
    opcode_offsets = [opcode("add_fp32")]
    second_shape = [3] if _CASE == "different" else [2]
    specs = [
        ("lhs_a", [2]),
        ("rhs_a", [2]),
        ("lhs_b", second_shape),
        ("rhs_b", second_shape),
        ("sum_a", [2]),
        ("sum_b", second_shape),
    ]
    primary = subgraph(
        "main",
        specs,
        [(0, [0, 1], 4), (0, [2, 3], 5)],
        [0, 1, 2, 3],
        [4, 5],
    )
    input_names = ("lhs_a", "rhs_a", "lhs_b", "rhs_b")
    output_names = ("sum_a", "sum_b")

signature_inputs = offset_vector(
    "SignatureDef",
    "Inputs",
    [tensor_map(name, index) for index, name in enumerate(input_names)],
)
signature_outputs = offset_vector(
    "SignatureDef",
    "Outputs",
    [
        tensor_map(name, index + len(input_names))
        for index, name in enumerate(output_names)
    ],
)
signature_key = builder.CreateString("serving_default")
tflite.SignatureDefStart(builder)
tflite.SignatureDefAddInputs(builder, signature_inputs)
tflite.SignatureDefAddOutputs(builder, signature_outputs)
tflite.SignatureDefAddSignatureKey(builder, signature_key)
tflite.SignatureDefAddSubgraphIndex(builder, 0)
signature = tflite.SignatureDefEnd(builder)

buffers = offset_vector("Model", "Buffers", [empty_buffer])
opcodes = offset_vector("Model", "OperatorCodes", opcode_offsets)
subgraphs = offset_vector(
    "Model", "Subgraphs", [primary, second] if _CASE == "subgraphs" else [primary]
)
signatures = offset_vector("Model", "SignatureDefs", [signature])
description = builder.CreateString("Compute Flow custom add/subtract integration test")
tflite.ModelStart(builder)
tflite.ModelAddVersion(builder, 3)
tflite.ModelAddOperatorCodes(builder, opcodes)
tflite.ModelAddSubgraphs(builder, subgraphs)
tflite.ModelAddBuffers(builder, buffers)
tflite.ModelAddSignatureDefs(builder, signatures)
tflite.ModelAddDescription(builder, description)
model = tflite.ModelEnd(builder)
builder.Finish(model, file_identifier=b"TFL3")

if __name__ == "__main__":
    if _CASE == "base":
        if _SIZE == 3:
            filename = "add_sub_three.tflite"
        else:
            filename = "add_sub.tflite" if _DTYPE == "float32" else "add_sub_i8.tflite"
    else:
        filename = f"add_sub_{_CASE}.tflite"
    _ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    (_ARGS.output_dir / filename).write_bytes(builder.Output())
