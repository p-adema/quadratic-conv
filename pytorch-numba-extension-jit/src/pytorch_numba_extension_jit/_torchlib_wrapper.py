from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable

import torch

from ._as_dtype import AsDType
from ._codegen import (
    InputScalar,
    InputTensor,
    KernelParam,
    OutputTensor,
    UnusedParam,
)


def _determine_torchlib_signature(
    kernel_params: Iterable[KernelParam],
) -> tuple[str, set[str]]:
    input_sig = []
    mutable_names = set()
    output_count = 0
    for param in kernel_params:
        if isinstance(param, InputScalar):
            input_sig.append(
                f"{AsDType(param.dtype).as_torchlib_scalar()} {param.name}"
            )
        elif isinstance(param, InputTensor):
            if param.mutable:
                input_sig.append(f"Tensor(mem_{param.name}!) {param.name}")
                mutable_names.add(param.name)
            else:
                input_sig.append(f"Tensor {param.name}")
        elif isinstance(param, OutputTensor):
            output_count += 1
        elif isinstance(param, UnusedParam):
            pass
        else:
            raise TypeError(f"Unknown kernel parameter {type(param)=}: {param=}")

    if output_count == 0:
        output = "-> ()"
    elif output_count == 1:
        output = "-> Tensor"
    else:
        output = "-> (" + ", ".join("Tensor" for _ in range(output_count)) + ")"

    return "(" + ", ".join(input_sig) + ")" + output, mutable_names


def _make_fake(kernel_params: Iterable[KernelParam]):
    parameters, values = [], []
    tensor_names = set()

    for param in kernel_params:
        if isinstance(param, InputTensor):
            tensor_names.add(param.name)
        if isinstance(param, InputTensor | InputScalar):
            parameters.append(param.name)
        if not isinstance(param, OutputTensor):
            continue

        values.append(
            f"torch.empty({param.sizes('py', tensor_names)},"
            f" dtype={AsDType(param.dtype).dtype}, device='cuda')"
        )
        tensor_names.add(param.name)

    func = f"""
def fake({", ".join(parameters)}):
    return {values[0] if len(values) == 1 else ", ".join(values)}"""

    ev_local = {}
    exec(func, {"torch": torch}, ev_local)

    return ev_local["fake"]


def torchlib_wrapper(
    kernel: Callable,
    name: str,
    kernel_params: tuple[KernelParam, ...],
) -> torch.library.CustomOpDef:
    schema, mutates_args = _determine_torchlib_signature(kernel_params)

    op = torch.library.custom_op(
        f"pnex_jit::op_{name}_{uuid.uuid4().hex}",
        schema=schema,
        mutates_args=mutates_args,
        device_types="cuda",
    )(kernel)
    op.register_fake(_make_fake(kernel_params))

    return op
