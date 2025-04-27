from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch

from .as_dtype import AsDType
from .cpp_codegen import (
    InputScalar,
    InputTensor,
    KernelParam,
    OutputTensor,
    UnusedParam,
)

# @torch.library.custom_op(
#     f"semifields::select_fwd_{op_id}", mutates_args={}, device_types="cuda"
# )
# def forwards_wrapper(
#     img: torch.Tensor, kernel: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     img, kernel = img.detach(), kernel.detach()
#     assert img.dtype == torch.float32, f"Wrong {img.dtype=}"
#     assert kernel.dtype == torch.float32, f"Wrong {kernel.dtype=}"
#     if debug:
#         print("Warning: running CUDA kernel in debug mode")
#     batch_size = img.shape[0]
#
#     out_img_shape = (
#         batch_size,
#         meta.out_cs,
#         meta.out_ys,
#         meta.out_xs,
#     )
#     out_img = img.new_empty(out_img_shape)
#     out_prov = img.new_empty(
#         (*out_img_shape, 3 if meta.krn_cs > 1 else 2), dtype=prov_t.torch_type()
#     )
#     n_blocks = math.ceil(out_img.nelement() / thread_block_size)
#     forwards[n_blocks, thread_block_size](img, kernel, out_img, out_prov)
#     return out_img, out_prov
#
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


_op_star_args = tuple[torch.Tensor | int | float, ...]
_kernel_args = list


class _InputConstructors(NamedTuple):
    arg_constructors: list[Callable[[_op_star_args], torch.Tensor | int | float]]
    n_blocks: Callable[[_op_star_args, _kernel_args], int]
    ret: Callable[[_kernel_args], None | torch.Tensor | tuple[torch.Tensor, ...]]


def _numba_input_constructors(
    kernel_params: Iterable[KernelParam], threads_per_block: int, n_threads: str | int
) -> _InputConstructors:
    arg_constructors = []
    iscalars = {}
    itensors = {}
    otensors = {}
    nth_input = 0
    for idx, param in enumerate(kernel_params):
        match param:
            case InputScalar(name=name):

                def get_scalar(args, nth=nth_input):
                    return args[nth]

                arg_constructors.append(get_scalar)
                iscalars[name] = nth_input
                nth_input += 1
            case InputTensor(name=name):

                def get_tensor(args, nth=nth_input):
                    return args[nth].detach()

                arg_constructors.append(get_tensor)
                itensors[name] = nth_input
                nth_input += 1

            case UnusedParam():
                arg_constructors.append(lambda _args: 0)

            case OutputTensor(name=name, dtype=dtype, shape=(str(like), int(ndims))):
                nth_arg = itensors[like]

                def make_like(args, nth=nth_arg):
                    return torch.empty(
                        args[nth].shape[:ndims],
                        dtype=AsDType(dtype).dtype,
                        device="cuda",
                    )

                arg_constructors.append(make_like)
                otensors[name] = idx

            case OutputTensor(name=name, dtype=dtype, shape=shape_desc):
                make_variable = _variable_output_constructor(
                    shape_desc, AsDType(dtype).dtype, iscalars, itensors
                )
                arg_constructors.append(make_variable)
                otensors[name] = idx

            case _:
                raise TypeError(f"Invalid kernel parameter {type(param)=}: {param=}")

    n_blocks = _make_n_blocks(
        iscalars, itensors, otensors, threads_per_block, n_threads
    )
    ret = _make_ret(list(otensors.values()))

    return _InputConstructors(arg_constructors, n_blocks, ret)


def _make_n_blocks(
    iscalars: dict[str, int],
    itensors: dict[str, int],
    otensors: dict[str, int],
    threads_per_block: int,
    n_threads: str | int,
):
    if isinstance(n_threads, int):
        return lambda _args: n_threads
    if n_threads in iscalars:
        nth_arg = iscalars[n_threads]

        def n_blocks(op_args, _kernel_args, nth=nth_arg):
            return math.ceil(op_args[nth] / threads_per_block)
    elif n_threads in itensors:
        nth_arg = itensors[n_threads]

        def n_blocks(op_args, _kernel_args, nth=nth_arg):
            return math.ceil(op_args[nth].numel() / threads_per_block)
    elif n_threads in otensors:
        nth_arg = otensors[n_threads]

        def n_blocks(_op_args, kernel_args, nth=nth_arg):
            return math.ceil(kernel_args[nth].numel() / threads_per_block)
    else:
        msg = (
            f"Non-extension backend only supports n_threads based"
            f" on Tensor.numel() or a scalar input, not {n_threads=}"
            f" (instead: pass the name of a KernelParam)"
        )
        raise ValueError(msg)
    return n_blocks


def _make_ret(output_indices: list[int]):
    if len(output_indices) == 0:
        return lambda _kernel_params: None
    if len(output_indices) == 1:
        single_idx = output_indices[0]
        return lambda kernel_params: kernel_params[single_idx]

    return lambda kernel_params: tuple(kernel_params[idx] for idx in output_indices)


def _variable_output_constructor(
    shape_desc: Iterable[int | tuple[str, int]],
    dtype: torch.dtype,
    iscalars: dict[str, int],
    itensors: dict[str, int],
):
    def make(args):
        shape = []
        for dim in shape_desc:
            match dim:
                case int(const_dim):
                    shape.append(const_dim)
                case (str(scalar_name), -1):
                    shape.append(args[iscalars[scalar_name]])
                case (str(tensor_name), int(nth_dim)):
                    shape.append(args[itensors[tensor_name]].shape[nth_dim])

        return torch.empty(shape, dtype=dtype, device="cuda")

    return make


def wrap_numba_jit(
    kernel: Callable,
    name: str,
    kernel_params: tuple[KernelParam, ...],
    threads_per_block: int,
    n_threads: str | int,
) -> Callable:
    schema, mutates_args = _determine_torchlib_signature(kernel_params)
    cons = _numba_input_constructors(kernel_params, threads_per_block, n_threads)

    @torch.library.custom_op(
        f"ptex_jit::op_{name}_{uuid.uuid4().hex}",
        schema=schema,
        mutates_args=mutates_args,
        device_types="cuda",
    )
    def op(*op_args: _op_star_args):
        kernel_args = [c(op_args) for c in cons.arg_constructors]
        n_blocks = cons.n_blocks(op_args, kernel_args)
        kernel[n_blocks, threads_per_block](*kernel_args)
        return cons.ret(kernel_args)

    return op
