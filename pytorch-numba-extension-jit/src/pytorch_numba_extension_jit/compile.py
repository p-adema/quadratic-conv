from __future__ import annotations

from collections.abc import Callable, Iterable

from numba import cuda
from torch.utils import cpp_extension

from .as_dtype import AsDType
from .cpp_codegen import InputScalar, InputTensor, KernelParam, OutputTensor, ptx_to_cpp


def ptx_to_extension(
    ptx: str,
    name: str,
    kernel_params: Iterable[KernelParam],
    *,
    n_threads_expr: str,
    threads_per_block: int = 256,
    use_runtime_api: bool = False,
    verbose: bool = True,
):
    cpp = ptx_to_cpp(
        ptx,
        name,
        kernel_params,
        n_threads_expr=n_threads_expr,
        threads_per_block=threads_per_block,
        use_runtime_api=use_runtime_api,
    )
    if verbose:
        print("=" * 10, "BEGIN CPP", "=" * 10)
        print(cpp)
        print("=" * 10, "END CPP", "=" * 10)
    mod = cpp_extension.load_inline(
        f"ptex_jit_{name}",
        cpp,
        with_cuda=True,
        verbose=verbose,
        functions=[f"ptex_jit::{name}"],
        keep_intermediates=True,
        extra_ldflags=[] if use_runtime_api else ["-lcuda"],
    )
    return getattr(mod, f"ptex_jit::{name}")


def _determine_signature(kernel_params: tuple[KernelParam, ...]) -> str:
    sig = []
    for param in kernel_params:
        if isinstance(param, InputScalar):
            sig.append(AsDType(param.dtype).as_numba())
        elif isinstance(param, (InputTensor, OutputTensor)):
            if isinstance(param.shape[0], str):
                ndim = param.shape[1]
            else:
                ndim = len(param.shape)

            sig.append(
                f"{AsDType(param.dtype).as_numba()}"
                f"[{', '.join(':' for _ in range(ndim))}]"
            )
        else:
            raise TypeError(f"Unknown kernel parameter {type(param)=}: {param=}")

    return "void(" + ", ".join(sig) + ")"


def jit(
    name: str,
    kernel_params: Iterable[KernelParam],
    *,
    n_threads_expr: str,
    threads_per_block: int = 256,
    use_runtime_api: bool = False,
    verbose: bool = True,
) -> Callable[[Callable], Callable]:
    kernel_params = tuple(kernel_params)
    sig = _determine_signature(kernel_params)
    if verbose:
        print(f"{sig=}")

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
            print("=" * 10, "BEGIN PTX", "=" * 10)
            print(ptx)
            print("=" * 10, "END PTX", "=" * 10)
        return ptx_to_extension(
            ptx,
            name,
            kernel_params,
            n_threads_expr=n_threads_expr,
            threads_per_block=threads_per_block,
            use_runtime_api=use_runtime_api,
            verbose=verbose,
        )

    return decorator
