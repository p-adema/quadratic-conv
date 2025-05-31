from __future__ import annotations

import re
import typing
from collections.abc import Callable, Iterable
from typing import Literal, NamedTuple

import numpy as np
import torch

from ._as_dtype import AsDType

# Replace a.shape[2] with a.size(2)
_cpp_replace_shape = re.compile(r"\w+\.shape\[([^]]+)]")


def _replace_shape_size_func(match: re.Match[str]):
    name, right = match.group().split(".shape[")

    return f"{name}.size({right[:-1]})"


def _to_size(s: str | int) -> str:
    if isinstance(s, int):
        return str(s)
    return _cpp_replace_shape.sub(_replace_shape_size_func, s)


def _to_shape_param(s: str | int, i: int, tensor_ndims):
    if s in tensor_ndims:
        return f"{s}.size({i})"
    return _to_size(s)


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
        tensor_ndims: dict[str, int],
        *,
        lang: Literal["cpp", "py"],
    ):
        shape = self.sizes(tensor_ndims)
        tensor_ndims[self.name] = len(shape)
        if lang == "cpp":
            parameters.append(
                f"{'' if self.mutable else 'const '}at::Tensor &{self.name}"
            )
            asserts.extend(
                (
                    f"{self.name}.dtype() == {AsDType(self.dtype).as_tcpp()}",
                    # f"{self.name}.is_contiguous()",
                    f"{self.name}.sizes().size() == {len(shape)}",
                    f"{self.name}.device().type() == at::DeviceType::CUDA",
                )
            )
        else:
            parameters.append(self.name)
            asserts.extend(
                (
                    f"{self.name}.dtype == {AsDType(self.dtype).dtype}",
                    # f"{self.name}.is_contiguous()",
                    f"{self.name}.ndim == {len(shape)}",
                    f"{self.name}.is_cuda",
                )
            )
        for i, dim in enumerate(shape):
            if dim is None:
                continue
            asserts.append(
                f"{self.name}.size({i}) == {_to_shape_param(dim, i, tensor_ndims)}"
            )

        _add_tensor_args(self, args, declarations, len(shape), lang)

    def sizes(self, tensor_ndims) -> tuple[int | str | None, ...]:
        if isinstance(self.shape, str):
            if self.shape not in tensor_ndims:
                msg = (
                    f"Asked for {self.name} to be like {self.shape},"
                    f" but {self.shape} is not (yet) defined ({tensor_ndims} are)"
                )
                raise ValueError(msg)
            return tuple(
                f"{self.shape}.size({i})" for i in range(tensor_ndims[self.shape])
            )

        assert len(self.shape) > 0
        return self.shape


class OutputTensor(NamedTuple):
    name: str
    dtype: torch.dtype | np.dtype | str
    shape: tuple[int | str, ...] | str
    init: int | float | None = None

    def prepare_args(
        self,
        _parameters: list[str],
        _asserts: list[str],
        declarations: list[str],
        args: list[str],
        tensor_ndims: dict[str, int],
        *,
        lang: Literal["cpp", "py"],
    ):
        sizes = self.sizes(lang, tensor_ndims)
        ndim = (
            tensor_ndims[self.shape] if self.shape in tensor_ndims else len(self.shape)
        )
        tensor_ndims[self.name] = ndim

        if lang == "cpp":
            declarations.append(
                f"at::Tensor {self.name} = "
                + (
                    f"at::empty({sizes}"
                    if self.init is None
                    else f"at::full({sizes}, {self.init}"
                )
                + f", at::device(at::kCUDA).dtype({AsDType(self.dtype).as_tcpp()}));"
            )
        else:
            declarations.append(
                f"{self.name} = "
                + (
                    f"torch.empty({sizes}"
                    if self.init is None
                    else f"torch.full({sizes}, {self.init}"
                )
                + f", dtype={AsDType(self.dtype).dtype}, device='cuda')"
            )
        _add_tensor_args(self, args, declarations, ndim, lang)

    def sizes(self, lang, tensor_ndims):
        assert len(self.shape) > 0
        if isinstance(self.shape, str):
            if self.shape not in tensor_ndims:
                msg = (
                    f"Asked for {self.name} to be like {self.shape},"
                    f" but {self.shape} is not (yet) defined ({tensor_ndims} are)"
                )
                raise ValueError(msg)

            sizes = f"{self.shape}.sizes()" if lang == "cpp" else f"{self.shape}.shape"
        elif lang == "cpp":
            sizes = (
                "{"
                + ", ".join(
                    _to_shape_param(dim, i, tensor_ndims)
                    for i, dim in enumerate(self.shape)
                )
                + "}"
            )
            sizes = _to_size(sizes)
        elif len(self.shape) == 1:
            sizes = str(self.shape[0])
        else:
            sizes = (
                "("
                + ", ".join(
                    _to_shape_param(dim, i, tensor_ndims)
                    for i, dim in enumerate(self.shape)
                )
                + ")"
            )
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
        _tensor_ndims: dict[str, int],
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
        _tensor_ndims: dict[str, int],
        *,
        lang: Literal["cpp", "py"],
    ):
        if lang == "cpp":
            declarations.append(f"uint32_t {self.name} = 0;")
            args.append(f"&{self.name}")
        else:
            args.append("0")

    def __class_getitem__(cls, item):
        raise ValueError(f"UnusedParam does not take type arguments (was given {item})")


KernelParam: typing.TypeAlias = InputTensor | OutputTensor | InputScalar | UnusedParam


def _add_tensor_args(
    tensor: InputTensor | OutputTensor,
    args: list[str],
    declarations: list[str],
    ndim: int,
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
    for i in range(ndim):
        declarations.extend(
            (
                f"uint64_t {tensor.name}_shape_{i} = {tensor.name}.size({i});",
                f"uint64_t {tensor.name}_stride_{i} "
                f"= {tensor.name}.stride({i}) * {nbytes};",
            )
        )

    args.extend(f"&{tensor.name}_shape_{i}" for i in range(ndim))
    args.extend(f"&{tensor.name}_stride_{i}" for i in range(ndim))


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


def _return_values(outputs: list[str], lang: Literal["cpp", "py"]):
    if len(outputs) == 0:
        return ""
    if len(outputs) == 1:
        return outputs[0]
    outs = ", ".join(outputs)
    return "{" + outs + "}" if lang == "cpp" else outs


def _cpp_kernel_invocation(name: str, use_runtime_api: bool):
    return (
        f"""
    static cudaKernel_t pnex_internal_kernel;
    if (!pnex_internal_kernel) {{
        cudaLibrary_t library;
        CHK(cudaLibraryLoadData(&library, ptx, 0,0,0,0,0,0));
        CHK(cudaLibraryGetKernel(&pnex_internal_kernel, library, "{name}"));
    }}
    CHK(cudaLaunchKernel((void*)pnex_internal_kernel, 
                        {{bpg_x, bpg_y, bpg_z}}, {{tpb_x, tpb_y, tpb_z}},
                         args, 0, c10::cuda::getCurrentCUDAStream()));
"""
        if use_runtime_api
        else f"""
    static CUfunction pnex_internal_kernel;
    if (!pnex_internal_kernel) {{
        CUmodule cuModule;
        CHK(cuModuleLoadData(&cuModule, ptx));
        CHK(cuModuleGetFunction(&pnex_internal_kernel, cuModule, "{name}"));
    }}
    CHK(cuLaunchKernel(pnex_internal_kernel, bpg_x, bpg_y, bpg_z, 
                       tpb_x, tpb_y, tpb_z,
                       0, c10::cuda::getCurrentCUDAStream(), args, NULL));
"""
    )


def _thread_calculation(tpb: int, bpg: str, suffix: str, lang: Literal["cpp", "py"]):
    return (
        f"""
    unsigned int tpb_{suffix} = {tpb};
    unsigned int bpg_{suffix} = ( ({bpg}) + {tpb} - 1) / {tpb};"""
        if lang == "cpp"
        else f"""
    bpg_{suffix} = ( ({bpg}) + {tpb} - 1) // {tpb}"""
    )


_ptx_name_regex = re.compile(r"\.visible \.entry (\w+)\(")
_ptx_env_regex = re.compile(r"\.common \.global \.align 8 \.u64 (\w+NumbaEnv\w+);")


def kernel_wrapper(
    kernel_inner: str | Callable,
    name: str,
    kernel_params: Iterable[KernelParam],
    *,
    n_threads: tuple[str, str, str],
    threads_per_block: tuple[int, int, int],
    use_runtime_api: bool = False,
    lang: Literal["cpp", "py"],
) -> str:
    assert lang in ("cpp", "py")
    parameters, asserts, declarations, args, outputs = [], [], [], [], []

    all_names = set()
    tensor_ndims = {}

    if lang == "cpp":
        assert isinstance(kernel_inner, str), "CPP must be provided PTX"
        ptx_match = _ptx_name_regex.search(kernel_inner)
        assert ptx_match is not None, "Strange PTX with no function"
        env_stripped = _ptx_env_regex.sub("", kernel_inner)
        kernel_inner = (
            env_stripped.replace(ptx_match.group(1), name)
            .replace("\n", "    \\n\\t\\\n")
            .replace('"', '\\"')
        )

    for param in kernel_params:
        if param.name in all_names:
            raise ValueError(f"Duplicate name {param.name=}")
        all_names.add(param.name)

        param.prepare_args(
            parameters, asserts, declarations, args, tensor_ndims, lang=lang
        )

        if isinstance(param, OutputTensor):
            outputs.append(param.name)

    n_threads = (
        _to_size(n_threads[0]),
        _to_size(n_threads[1]),
        _to_size(n_threads[2]),
    )
    newline = "\n    "
    return (
        f"""
#include <{"cuda_runtime.h" if use_runtime_api else "cuda.h"}>
#include <c10/cuda/CUDAStream.h>

namespace pnex_jit {{
    {_chk_macro_runtime if use_runtime_api else _chk_macro_driver}
    const char *ptx = "{kernel_inner}";
    
    {_cpp_return_type(outputs)} kernel_{name}({", ".join(parameters)}) {{
    
        {newline.join(f"TORCH_CHECK({cond});" for cond in asserts)}
        {newline.join(declarations)}
    
        void *args[] = {{{", ".join(args)}}};
        
        {_thread_calculation(threads_per_block[0], n_threads[0], "x", "cpp")}
        {_thread_calculation(threads_per_block[1], n_threads[1], "y", "cpp")}
        {_thread_calculation(threads_per_block[2], n_threads[2], "z", "cpp")}
    
        {_cpp_kernel_invocation(name, use_runtime_api)}
        return {_return_values(outputs, "cpp")};
    }}
}}
"""
        if lang == "cpp"
        else f"""
def kernel_{name}({", ".join(parameters)}):
    {newline.join(f"assert {cond}, '{cond}'" for cond in asserts)}
    {newline.join(declarations)}
    
    {_thread_calculation(threads_per_block[0], n_threads[0], "x", "py")}
    {_thread_calculation(threads_per_block[1], n_threads[1], "y", "py")}
    {_thread_calculation(threads_per_block[2], n_threads[2], "z", "py")}
    kernel_inner[
        (bpg_x, bpg_y, bpg_z), {threads_per_block},
        cuda.external_stream(torch.cuda.current_stream().cuda_stream)
    ]({", ".join(args)})
    return {_return_values(outputs, "py")}
"""
    )
