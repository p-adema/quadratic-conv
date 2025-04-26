from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple, Protocol

import numpy as np
import torch

from .as_dtype import AsDType


class KernelParam(Protocol):
    def prepare_args(
        self,
        parameters: list[str],
        asserts: list[str],
        declarations: list[str],
        args: list[str],
    ) -> None: ...


class InputTensor(NamedTuple):
    name: str
    dtype: torch.dtype | np.dtype | NumbaDType | str
    shape: tuple[int | None, ...]
    mutable: bool = False

    def prepare_args(
        self,
        parameters: list[str],
        asserts: list[str],
        declarations: list[str],
        args: list[str],
    ):
        parameters.append(f"{'' if self.mutable else 'const '}at::Tensor &{self.name}")
        asserts.extend(
            (
                f"TORCH_CHECK({self.name}.dtype() == {AsDType(self.dtype).as_aten()});",
                f"TORCH_CHECK({self.name}.is_contiguous());",
                f"TORCH_CHECK({self.name}.sizes().size() == {len(self.shape)});",
                f"TORCH_INTERNAL_ASSERT({self.name}.device().type() == at::DeviceType::CUDA);",
            )
        )
        for i, dim in enumerate(self.shape):
            if dim is None:
                continue
            asserts.append(
                f"TORCH_CHECK({self.name}.size({i}) == {dim});",
            )

        _add_tensor_args(self, args, declarations)


class InputScalar(NamedTuple):
    name: str
    dtype: torch.dtype | np.dtype | NumbaDType | str

    # noinspection PyUnusedLocal
    def prepare_args(
        self,
        parameters: list[str],
        asserts: list[str],
        declarations: list[str],
        args: list[str],
    ):
        parameters.append(f"{AsDType(self.dtype).as_c()} {self.name}")
        args.append(f"&{self.name}")


class OutputTensor(NamedTuple):
    name: str
    dtype: torch.dtype | np.dtype | NumbaDType | str
    shape: tuple[int | tuple[str, int], ...] | tuple[str, int]

    # noinspection PyUnusedLocal
    def prepare_args(
        self,
        parameters: list[str],
        asserts: list[str],
        declarations: list[str],
        args: list[str],
    ):
        assert len(self.shape) > 0

        if isinstance(self.shape[0], str):
            sizes = f"{self.shape[0]}.sizes()"
        else:
            sizes_parts = []
            for dim in self.shape:
                if isinstance(dim, int):
                    sizes_parts.append(str(dim))
                else:
                    name, nth_dim = dim
                    sizes_parts.append(f"{name}.size({nth_dim})")
            sizes = "{" + ", ".join(sizes_parts) + "}"

        declarations.append(
            f"at::Tensor {self.name} = at::empty({sizes}"
            f", at::device(at::kCUDA).dtype({AsDType(self.dtype).as_aten()}));"
        )
        _add_tensor_args(self, args, declarations)


def _add_tensor_args(tensor: InputTensor | OutputTensor, args, declarations):
    nbytes = AsDType(tensor.dtype).byte_width()
    declarations.extend(
        (
            f"uint64_t {tensor.name}_meminfo = 0;",
            f"uint64_t {tensor.name}_parent = 0;",
            f"uint64_t {tensor.name}_nitems = {tensor.name}.numel();",
            f"uint64_t {tensor.name}_itemsize = {nbytes};",
            f"uint64_t {tensor.name}_data = (uint64_t)(void *) {tensor.name}.data_ptr<{AsDType(tensor.dtype).as_c()}>();",
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
                f"uint64_t {tensor.name}_stride_{i} = {tensor.name}.stride({i}) * {nbytes};",
            )
        )

    for i in range(len(tensor.shape)):
        args.append(f"&{tensor.name}_shape_{i}")
    for i in range(len(tensor.shape)):
        args.append(f"&{tensor.name}_stride_{i}")


_chk_macro_runtime = r"""
#define CHK(x) do { \
  cudaError_t result = x; \
  if (result != cudaSuccess) { \
    printf("CUDART error %d at %d: %s\n", (int)result, __LINE__, cudaGetErrorString(result)); \
    exit(1); \
  }  \
  } while(0);
"""
_chk_macro_driver = r"""
#define CHK(x)                                                    \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0);
"""


def _return_type(outputs: list[str]):
    if len(outputs) == 0:
        return "void"
    if len(outputs) == 1:
        return "at::Tensor"
    return f"std::tuple<{', '.join(['at::Tensor'] * len(outputs))}>"


def _return_statement(outputs: list[str]):
    if len(outputs) == 0:
        return "return;"
    if len(outputs) == 1:
        return f"return {outputs[0]};"
    return f"return {{{', '.join(outputs)}}}"


def _kernel_invocation(name: str, use_runtime_api: bool):
    return (
        f"""
    static cudaKernel_t kernel;
    if (!kernel) {{
        cudaLibrary_t library;
        CHK(cudaLibraryLoadData(&library, ptx, 0,0,0,0,0,0));
        CHK(cudaLibraryGetKernel(&kernel, library, "{name}"));
    }}
    CHK(cudaLaunchKernel((void*)kernel, {{blocksPerGrid, 1, 1}}, {{threadsPerBlock, 1, 1}}, args, 0, NULL));
"""
        if use_runtime_api
        else f"""
    static CUfunction kernel;
    if (!kernel) {{
        CUmodule cuModule;
        CHK(cuModuleLoadData(&cuModule, ptx));
        CHK(cuModuleGetFunction(&kernel, cuModule, "{name}"));
    }}
    CHK(cuLaunchKernel(kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, NULL));
"""
    )


_ptx_name_regex = re.compile(r"\.visible \.entry (\w+)\(")


def ptx_to_cpp(
    ptx: str,
    name: str,
    kernel_params: Iterable[KernelParam],
    *,
    n_threads_expr: str,
    threads_per_block: int = 256,
    use_runtime_api: bool = False,
) -> str:
    parameters, asserts, declarations, args = [], [], [], []
    outputs = []
    tensor_names = set()
    ptx_match = _ptx_name_regex.search(ptx)
    assert ptx_match is not None, "Strange PTX with no function"
    ptx_modified = ptx[: ptx_match.span(1)[0]] + name + ptx[ptx_match.span(1)[1] :]
    for param in kernel_params:
        param.prepare_args(parameters, asserts, declarations, args)
        if isinstance(param, OutputTensor):
            outputs.append(param.name)
        if isinstance(param, (OutputTensor, InputTensor)):
            assert param.name not in tensor_names, (
                f"Duplicate tensor name `{param.name}`"
            )
            tensor_names.add(param.name)

    if n_threads_expr in tensor_names:
        n_threads_expr = f"{n_threads_expr}.numel()"

    return f"""
#include <{"cuda_runtime.h" if use_runtime_api else "cuda.h"}>
namespace ptex_jit {{
    {_chk_macro_runtime if use_runtime_api else _chk_macro_driver}
    const char *ptx = "{ptx_modified.replace("\n", "    \\n\\t\\\n")}";
    
    {_return_type(outputs)} {name}({", ".join(parameters)}) {{
    
        {"\n    ".join(asserts)}
        {"\n    ".join(declarations)}
    
        void *args[] = {{{", ".join(args)}}};
        unsigned int threadsPerBlock = {threads_per_block};
        unsigned int blocksPerGrid = (({n_threads_expr}) + threadsPerBlock - 1) / threadsPerBlock;
    
        {_kernel_invocation(name, use_runtime_api)}
        {_return_statement(outputs)}
    }}
}}
"""
