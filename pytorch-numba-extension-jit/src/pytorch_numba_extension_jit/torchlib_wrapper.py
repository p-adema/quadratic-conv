from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable

import torch

from .as_dtype import AsDType
from .codegen import (
    InputScalar,
    InputTensor,
    KernelParam,
    OutputTensor,
    UnusedParam,
)

# @forwards_wrapper.register_fake
# def _(img, kernel):
#     batch_size = img.shape[0]
#     out_img_shape = (
#         batch_size,
#         meta.out_cs,
#         meta.out_ys,
#         meta.out_xs,
#     )
#     return (
#         img.new_empty(out_img_shape),
#         kernel.new_empty(
#             (*out_img_shape, 3 if meta.krn_cs > 1 else 2), dtype=prov_t.torch_type()
#         ),
#     )


def _determine_torchlib_signature(
    kernel_params: Iterable[KernelParam],
) -> tuple[str, set[str]]:
    input_sig = []
    mutable_names = set()
    output_count = 0
    for param in kernel_params:
        match param:
            case InputScalar(dtype=dtype, name=name):
                input_sig.append(f"{AsDType(dtype).as_torchlib_scalar()} {name}")
            case InputTensor(name=name, mutable=False):
                input_sig.append(f"Tensor {name}")
            case InputTensor(name=name, mutable=True):
                input_sig.append(f"Tensor(mem_{name}!) {name}")
                mutable_names.add(name)
            case OutputTensor():
                output_count += 1
            case UnusedParam():
                pass
            case _:
                raise TypeError(f"Unknown kernel parameter {type(param)=}: {param=}")

    if output_count == 0:
        output = ""
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
        f"ptex_jit::op_{name}_{uuid.uuid4().hex}",
        schema=schema,
        mutates_args=mutates_args,
        device_types="cuda",
    )(kernel)
    op.register_fake(_make_fake(kernel_params))

    return op
