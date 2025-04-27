from __future__ import annotations

import math
import os
import sys
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import numba
import torch
from numba import cuda
from torch.utils import cpp_extension

from .as_dtype import AsDType
from .codegen import (
    InputScalar,
    InputTensor,
    KernelParam,
    OutputTensor,
    UnusedParam,
    kernel_wrapper,
)
from .torchlib_wrapper import torchlib_wrapper

warnings.filterwarnings(
    "ignore",
    r"Grid size \d+ will likely result in GPU under-utilization due to low occupancy.",
    numba.NumbaPerformanceWarning,
)

_cuda_major, _cuda_minor = cuda.get_current_device().compute_capability
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_cuda_major}.{_cuda_minor}")


def _find_cudart() -> Path:
    py_lib = Path(sys.exec_prefix) / "lib"
    site_packages = next(py_lib.glob("python3.*")) / "site-packages"
    cuda_lib = site_packages / "nvidia" / "cuda_runtime" / "lib"
    if not cuda_lib.exists():
        raise FileNotFoundError(f"Seem to be missing runtime: looked for {cuda_lib=}")
    cudart = cuda_lib / "libcudart.so"
    if not cudart.exists():
        cudart_versioned = next(cuda_lib.glob("libcudart.so.*"), None)
        if cudart_versioned is None:
            raise FileNotFoundError(f"Missing libcudart.so.* in {cuda_lib=}")
        cudart.symlink_to(cudart_versioned)
    return cuda_lib


def ptx_to_extension(
    ptx: str,
    name: str,
    kernel_params: Iterable[KernelParam],
    *,
    n_threads: str,
    threads_per_block: int = 256,
    use_runtime_api: bool = False,
    verbose: bool = True,
):
    cpp = kernel_wrapper(
        ptx,
        name,
        kernel_params,
        n_threads=n_threads,
        threads_per_block=threads_per_block,
        use_runtime_api=use_runtime_api,
        lang="cpp",
    )
    if verbose:
        print("=" * 10, f"BEGIN CPP {name}", "=" * 10)
        print(cpp)
        print("=" * 10, f"END CPP {name}", "=" * 10)

    mod = cpp_extension.load_inline(
        f"ptex_jit_{name}",
        cpp,
        with_cuda=True,
        verbose=verbose,
        functions=[f"ptex_jit::kernel_{name}"],
        keep_intermediates=True,
        extra_ldflags=[f"-L{_find_cudart()}"] + ([] if use_runtime_api else ["-lcuda"]),
    )
    return getattr(mod, f"ptex_jit::kernel_{name}")


def _determine_numba_signature(kernel_params: tuple[KernelParam, ...]) -> str:
    sig = []
    prev_dims = {}
    for param in kernel_params:
        if isinstance(param, InputScalar):
            sig.append(AsDType(param.dtype).as_numba())
        elif isinstance(param, InputTensor | OutputTensor):
            if isinstance(param.shape, str):
                if param.shape not in prev_dims:
                    msg = (
                        f"Asked for {param.name} to be like {param.shape},"
                        f" but {param.shape} is not (yet) defined"
                        + (
                            f" ({set(prev_dims.keys())} "
                            f"{'are' if len(prev_dims) > 1 else 'is'})"
                            if prev_dims
                            else " (there are no tensors defined yet)"
                        )
                    )
                    raise ValueError(msg)

                ndim = prev_dims[param.shape]
            else:
                ndim = len(param.shape)

            prev_dims[param.name] = ndim

            sig.append(
                f"{AsDType(param.dtype).as_numba()}"
                f"[{', '.join(':' for _ in range(ndim))}]"
            )
        elif isinstance(param, UnusedParam):
            pass
        else:
            raise TypeError(f"Unknown kernel parameter {type(param)=}: {param=}")

    return "void(" + ", ".join(sig) + ")"


def jit(
    kernel_params: Iterable[KernelParam],
    *,
    n_threads: str,
    threads_per_block: int = 256,
    use_runtime_api: bool = False,
    verbose: bool = False,
    compile_extension: bool = True,
    cache_id: str | None = None,
) -> Callable[[Callable], torch.library.CustomOpDef]:
    cache_id = "_" + cache_id if cache_id else ""

    kernel_params = tuple(kernel_params)
    sig = _determine_numba_signature(kernel_params)
    if verbose:
        print(f"SIGNATURE {cache_id if cache_id else ''} : {sig}")

    if compile_extension:

        def decorator(pyfunc: Callable) -> torch.library.CustomOpDef:
            name = pyfunc.__name__ + cache_id
            ptx = cuda.compile_for_current_device(
                pyfunc,
                sig,
                device=False,
                abi="numba",
                lineinfo=False,
                output="ptx",
            )[0]
            kernel = ptx_to_extension(
                ptx,
                name,
                kernel_params,
                n_threads=str(n_threads),
                threads_per_block=threads_per_block,
                use_runtime_api=use_runtime_api,
                verbose=verbose,
            )
            torchlib_op = torchlib_wrapper(
                kernel=kernel,
                name=name,
                kernel_params=kernel_params,
            )
            return torchlib_op
    else:

        def decorator(pyfunc: Callable) -> torch.library.CustomOpDef:
            name = pyfunc.__name__ + cache_id
            numba_kernel = cuda.jit(sig, cache=True)(pyfunc)
            kernel_py_code = kernel_wrapper(
                numba_kernel,
                name,
                kernel_params,
                n_threads=n_threads,
                threads_per_block=threads_per_block,
                lang="py",
            )
            ev_locals = {}
            if verbose:
                print("=" * 10, f"BEGIN PY {name}", "=" * 10)
                print(kernel_py_code)
                print("=" * 10, f"END PY {name}", "=" * 10)

            exec(
                kernel_py_code,
                {"torch": torch, "kernel_inner": numba_kernel},
                ev_locals,
            )
            kernel = ev_locals[f"kernel_{name}"]
            torchlib_op = torchlib_wrapper(
                kernel=kernel,
                name=name,
                kernel_params=kernel_params,
            )
            return torchlib_op

    return decorator
