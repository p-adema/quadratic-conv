from __future__ import annotations

import re
import typing
from collections.abc import Callable, Iterable
from typing import Literal, NamedTuple

import numpy as np
import torch

from .as_dtype import AsDType


class InputTensor(NamedTuple):
    name: str
    dtype: torch.dtype | np.dtype | str
    shape: tuple[int | str | None, ...] | str
    mutable: bool = False

    def prepare_args(
        self,
        parameters: list[str],
        asserts: list[str],
        declarations: list[str],
        args: list[str],
        _tensor_names: set[str],
        *,
        lang: Literal["cpp", "py"],
    ):
        if lang == "cpp":
            parameters.append(
                f"{'' if self.mutable else 'const '}at::Tensor &{self.name}"
            )
            asserts.extend(
                (
                    f"{self.name}.dtype() == {AsDType(self.dtype).as_aten()}",
                    f"{self.name}.is_contiguous()",
                    f"{self.name}.sizes().size() == {len(self.shape)}",
                    f"{self.name}.device().type() == at::DeviceType::CUDA",
                )
            )
        else:
            parameters.append(self.name)
            asserts.extend(
                (
                    f"{self.name}.dtype == {AsDType(self.dtype).dtype}",
                    f"{self.name}.is_contiguous()",
                    f"{self.name}.ndim == {len(self.shape)}",
                    f"{self.name}.is_cuda",
                )
            )
        for i, dim in enumerate(self.shape):
            if dim is None:
                continue
            asserts.append(f"{self.name}.size({i}) == {dim}")

        _add_tensor_args(self, args, declarations, lang)


# Replace a.shape[2] with a.size(2)
_cpp_replace_shape = re.compile(r"\w+\.shape\[([^]]+)]")


def _replace_shape_size(match: re.Match[str]):
    name, right = match.group().split(".shape[")

    return f"{name}.size({right[:-1]})"


class OutputTensor(NamedTuple):
    name: str
    dtype: torch.dtype | np.dtype | str
    shape: tuple[int | str, ...] | str

    def prepare_args(
        self,
        _parameters: list[str],
        _asserts: list[str],
        declarations: list[str],
        args: list[str],
        tensor_names: set[str],
        *,
        lang: Literal["cpp", "py"],
    ):
        sizes = self.sizes(lang, tensor_names)

        if lang == "cpp":
            declarations.append(
                f"at::Tensor {self.name} = at::empty({sizes}"
                f",     at::device(at::kCUDA).dtype({AsDType(self.dtype).as_aten()}));"
            )
        else:
            declarations.append(
                f"{self.name} = torch.empty({sizes},"
                f" dtype={AsDType(self.dtype).dtype}, device='cuda')"
            )
        _add_tensor_args(self, args, declarations, lang)

    def sizes(self, lang, tensor_names):
        assert len(self.shape) > 0
        if isinstance(self.shape, str):
            if self.shape not in tensor_names:
                msg = (
                    f"Asked for {self.name} to be like {self.shape},"
                    f" but {self.shape} is not (yet) defined ({tensor_names} are)"
                )
                raise ValueError(msg)

            sizes = f"{self.shape}.sizes()" if lang == "cpp" else f"{self.shape}.shape"
        elif lang == "cpp":
            sizes = "{" + ", ".join(str(dim) for dim in self.shape) + "}"
            sizes = _cpp_replace_shape.sub(_replace_shape_size, sizes)
        elif len(self.shape) == 1:
            sizes = str(self.shape[0])
        else:
            sizes = "(" + ", ".join(str(dim) for dim in self.shape) + ")"
        return sizes


class InputScalar(NamedTuple):
    name: str
    dtype: torch.dtype | np.dtype | str

    def prepare_args(
        self,
        parameters: list[str],
        _asserts: list[str],
        _declarations: list[str],
        args: list[str],
        _tensor_names: set[str],
        *,
        lang: Literal["cpp", "py"],
    ):
        if lang == "cpp":
            parameters.append(f"{AsDType(self.dtype).as_c()} {self.name}")
            args.append(f"&{self.name}")
        else:
            parameters.append(self.name)
            args.append(self.name)


class UnusedParam(NamedTuple):
    name: str

    def prepare_args(
        self,
        _parameters: list[str],
        _asserts: list[str],
        declarations: list[str],
        args: list[str],
        _tensor_names: set[str],
        *,
        lang: Literal["cpp", "py"],
    ):
        if lang == "cpp":
            declarations.append(f"uint32_t {self.name} = 0;")
            args.append(f"&{self.name}")
        else:
            args.append("0")


KernelParam: typing.TypeAlias = InputTensor | OutputTensor | InputScalar | UnusedParam


def _add_tensor_args(
    tensor: InputTensor | OutputTensor,
    args: list[str],
    declarations: list[str],
    lang: Literal["cpp", "py"],
):
    if lang == "py":
        args.append(f"{tensor.name}.detach()")
        return

    nbytes = AsDType(tensor.dtype).byte_width()
    declarations.extend(
        (
            f"uint64_t {tensor.name}_meminfo = 0;",
            f"uint64_t {tensor.name}_parent = 0;",
            f"uint64_t {tensor.name}_nitems = {tensor.name}.numel();",
            f"uint64_t {tensor.name}_itemsize = {nbytes};",
            f"uint64_t {tensor.name}_data = (uint64_t)(void *) "
            f"  {tensor.name}.data_ptr<{AsDType(tensor.dtype).as_c()}>();",
        )
    )
    args.extend(
        (
            f"&{tensor.name}_meminfo",
            f"&{tensor.name}_parent",
            f"&{tensor.name}_nitems",
            f"&{tensor.name}_itemsize",
            f"&{tensor.name}_data",
        )
    )
    for i in range(len(tensor.shape)):
        declarations.extend(
            (
                f"uint64_t {tensor.name}_shape_{i} = {tensor.name}.size({i});",
                f"uint64_t {tensor.name}_stride_{i} "
                f"= {tensor.name}.stride({i}) * {nbytes};",
            )
        )

    args.extend(f"&{tensor.name}_shape_{i}" for i in range(len(tensor.shape)))
    args.extend(f"&{tensor.name}_stride_{i}" for i in range(len(tensor.shape)))


_chk_macro_runtime = r"""
#define CHK(x)                                                      \
  do {                                                              \
    cudaError_t result = x;                                         \
    if (result != cudaSuccess) {                                    \
      const char *msg = cudaGetErrorString(result);                 \
      std::cerr << "\nerror: " #x " failed with error "             \
                << msg << '\n';                                     \
      exit(1);                                                      \
    }                                                               \
  } while(0);
"""
_chk_macro_driver = r"""
#define CHK(x)                                                      \
  do {                                                              \
    CUresult result = x;                                            \
    if (result != CUDA_SUCCESS) {                                   \
      const char *msg;                                              \
      cuGetErrorName(result, &msg);                                 \
      std::cerr << "\nerror: " #x " failed with error "             \
                << msg << '\n';                                     \
      exit(1);                                                      \
    }                                                               \
  } while(0);
"""


def _cpp_return_type(outputs: list[str]):
    if len(outputs) == 0:
        return "void"
    if len(outputs) == 1:
        return "at::Tensor"
    return f"std::tuple<{', '.join(['at::Tensor'] * len(outputs))}>"


def _return_statement(outputs: list[str]):
    if len(outputs) == 0:
        return "return"
    if len(outputs) == 1:
        return f"return {outputs[0]}"
    return f"return {{{', '.join(outputs)}}}"


def _cpp_kernel_invocation(name: str, use_runtime_api: bool):
    return (
        f"""
    static cudaKernel_t ptex_internal_kernel;
    if (!ptex_internal_kernel) {{
        cudaLibrary_t library;
        CHK(cudaLibraryLoadData(&library, ptx, 0,0,0,0,0,0));
        CHK(cudaLibraryGetKernel(&ptex_internal_kernel, library, "{name}"));
    }}
    CHK(cudaLaunchKernel((void*)ptex_internal_kernel, 
                        {{blocks_per_grid, 1, 1}}, {{threads_per_block, 1, 1}},
                         args, 0, NULL));
"""
        if use_runtime_api
        else f"""
    static CUfunction ptex_internal_kernel;
    if (!ptex_internal_kernel) {{
        CUmodule cuModule;
        CHK(cuModuleLoadData(&cuModule, ptx));
        CHK(cuModuleGetFunction(&ptex_internal_kernel, cuModule, "{name}"));
    }}
    CHK(cuLaunchKernel(ptex_internal_kernel, blocks_per_grid, 1, 1, 
                       threads_per_block, 1, 1, 0, 0, args, NULL));
"""
    )


_ptx_name_regex = re.compile(r"\.visible \.entry (\w+)\(")


def kernel_wrapper(
    kernel_inner: str | Callable,
    name: str,
    kernel_params: Iterable[KernelParam],
    *,
    n_threads: str,
    threads_per_block: int = 256,
    use_runtime_api: bool = False,
    lang: Literal["cpp", "py"],
) -> str:
    assert lang in ("cpp", "py")
    parameters, asserts, declarations, args, outputs = [], [], [], [], []

    all_names = set()
    tensor_names = set()

    if lang == "cpp":
        assert isinstance(kernel_inner, str), "CPP must be provided PTX"
        ptx_match = _ptx_name_regex.search(kernel_inner)
        assert ptx_match is not None, "Strange PTX with no function"
        kernel_inner = (
            kernel_inner[: ptx_match.span(1)[0]]
            + name
            + kernel_inner[ptx_match.span(1)[1] :]
        ).replace("\n", "    \\n\\t\\\n")

    else:
        assert isinstance(kernel_inner, Callable), "PY must be provided a Dispatcher"

    for param in kernel_params:
        if param.name in all_names:
            raise ValueError(f"Duplicate name {param.name=}")
        all_names.add(param.name)

        param.prepare_args(
            parameters, asserts, declarations, args, tensor_names, lang=lang
        )

        if isinstance(param, OutputTensor):
            outputs.append(param.name)
        if isinstance(param, OutputTensor | InputTensor):
            tensor_names.add(param.name)
    if n_threads in tensor_names:
        n_threads = f"{n_threads}.numel()"

    newline = "\n    "
    return (
        f"""
#include <{"cuda_runtime.h" if use_runtime_api else "cuda.h"}>
namespace ptex_jit {{
    {_chk_macro_runtime if use_runtime_api else _chk_macro_driver}
    const char *ptx = "{kernel_inner}";
    
    {_cpp_return_type(outputs)} kernel_{name}({", ".join(parameters)}) {{
    
        {newline.join(f"TORCH_CHECK({cond});" for cond in asserts)}
        {newline.join(declarations)}
    
        void *args[] = {{{", ".join(args)}}};
        unsigned int threads_per_block = {threads_per_block};
        unsigned int blocks_per_grid = ( ({n_threads}) 
                                          + threads_per_block - 1) / threads_per_block;
    
        {_cpp_kernel_invocation(name, use_runtime_api)}
        {_return_statement(outputs)};
    }}
}}
"""
        if lang == "cpp"
        else f"""
def kernel_{name}({", ".join(parameters)}):
    {newline.join(f"assert {cond}" for cond in asserts)}
    {newline.join(declarations)}
    blocks_per_grid = ( ({n_threads}) 
                        + {threads_per_block} - 1) // {threads_per_block}
    kernel_inner[blocks_per_grid, {threads_per_block}]({", ".join(args)})
    {_return_statement(outputs)}
"""
    )
