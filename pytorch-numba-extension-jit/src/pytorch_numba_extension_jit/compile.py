from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from numba import cuda
from torch.utils import cpp_extension

from .as_dtype import AsDType
from .cpp_codegen import (
    InputScalar,
    InputTensor,
    KernelParam,
    OutputTensor,
    UnusedParam,
    ptx_to_cpp,
)
from .torchlib_wrap import wrap_numba_jit

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
    cpp = ptx_to_cpp(
        ptx,
        name,
        kernel_params,
        n_threads=n_threads,
        threads_per_block=threads_per_block,
        use_runtime_api=use_runtime_api,
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
    for param in kernel_params:
        if isinstance(param, InputScalar):
            sig.append(AsDType(param.dtype).as_numba())
        elif isinstance(param, InputTensor | OutputTensor):
            if isinstance(param.shape[0], str):
                ndim = param.shape[1]
            else:
                ndim = len(param.shape)

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
    name: str,
    kernel_params: Iterable[KernelParam],
    *,
    n_threads: str | int,
    threads_per_block: int = 256,
    use_runtime_api: bool = False,
    verbose: bool = False,
    compile_extension: bool = True,
) -> Callable[[Callable], Callable]:
    kernel_params = tuple(kernel_params)
    sig = _determine_numba_signature(kernel_params)
    if verbose:
        print(f"SIGNATURE {name} : {sig}")

    if compile_extension:

        def decorator(pyfunc: Callable) -> Callable:
            ptx = cuda.compile_for_current_device(
                pyfunc,
                sig,
                device=False,
                abi="numba",
                lineinfo=False,
                output="ptx",
            )[0]
            if verbose:
                print("=" * 10, f"BEGIN PTX {name}", "=" * 10)
                print(ptx)
                print("=" * 10, f"END PTX {name}", "=" * 10)
            return ptx_to_extension(
                ptx,
                name,
                kernel_params,
                n_threads=str(n_threads),
                threads_per_block=threads_per_block,
                use_runtime_api=use_runtime_api,
                verbose=verbose,
            )
    else:

        def decorator(pyfunc: Callable) -> Callable:
            numba_kernel = cuda.jit(sig, cache=True)(pyfunc)
            torchlib_op = wrap_numba_jit(
                kernel=numba_kernel,
                name=name,
                kernel_params=kernel_params,
                threads_per_block=threads_per_block,
                n_threads=n_threads,
            )
            return torchlib_op

    return decorator
