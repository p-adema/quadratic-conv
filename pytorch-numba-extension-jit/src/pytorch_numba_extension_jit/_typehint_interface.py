import inspect
from collections.abc import Callable

import numpy as np
import torch

from ._as_dtype import AsDType
from ._codegen import InputScalar, InputTensor, OutputTensor, UnusedParam
from ._compiler import compile_array_api, compile_extension


class In:
    """
    A type annotation for immutable input tensor parameters in a `jit` function

    An input tensor is part of the argument list in the final operator, meaning it must
    be provided by the caller. This variant is immutable, meaning the kernel must not
    modify the tensor.

    To use this annotation, use the syntax `param: In(dtype, shape)`.

    Parameters
    -------
    dtype : torch.dtype, np.dtype, str
        The data type of the input tensor.

        Some equivalent examples: `torch.float32`, `float`, `"float32"` or `"f32"`
    shape : str, tuple of (int or str or None)
        The shape of the input tensor.

        If `shape` is a string, it must be the name of a previously defined
        tensor parameter, and the shape of this parameter must be equal to
        the shape of the parameter named by `shape`.

        If `shape` is a tuple, every element in the tuple corresponds with one axis in
        the input tensor. For every such element:

        - `int` constrains the axis to be exactly of the given dimension.
        - `str` represents an expression that evaluates to an integer, and constrains
          the axis to be equal to the result of the expression. If the name of a tensor
          parameter is provided, this is equivalent to `param_name.shape[nth_dim]` where
          `nth_dim` is the index of the current axis.
        - `None` does not constrain the size of the axis.
    """

    def __new__(
        cls,
        dtype: torch.dtype | np.dtype | str,
        shape: tuple[int | str | None, ...] | str,
    ):
        return InputTensor("<MISSING>", AsDType(dtype).dtype, shape, mutable=False)

    def __class_getitem__(
        cls,
        item: tuple[torch.dtype | np.dtype | str, tuple[int | str | None, ...] | str],
    ):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("`In` parameter must be of the form In[type, shape]")
        dtype, shape = item
        return cls(dtype, shape)


class InMut:
    """
    A type annotation for **mutable** input tensor parameters in a `jit` function

    An input tensor is part of the argument list in the final operator, meaning it must
    be provided by the caller. This variant **is mutable**, meaning the kernel **may**
    modify the tensor.

    To use this annotation, use the syntax `param: InMut(dtype, shape)`.

    For information on the parameters, see `In`.
    """

    def __new__(
        cls,
        dtype: torch.dtype | np.dtype | str,
        shape: tuple[int | str | None, ...] | str,
    ):
        return InputTensor("<MISSING>", AsDType(dtype).dtype, shape, mutable=True)

    def __class_getitem__(
        cls,
        item: tuple[torch.dtype | np.dtype | str, tuple[int | str | None, ...] | str],
    ):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("`InMut` parameter must be of the form InMut[type, shape]")
        dtype, shape = item
        return cls(dtype, shape)


class Out:
    """
    A type annotation for output tensor parameters in a `jit` function

    An output tensor is not part of the argument list in the final operator, meaning the
    caller *must not* attempt to provide it. Instead, parameters marked as `Out` are
    created by the wrapper code before being passed to the kernel, and are returned to
    the caller afterwards as return values from the final operator. Since parameters
    marked `Out` are returned, they can receive a gradient and can work with the PyTorch
    autograd system.

    To use this annotation, use the syntax `param: Out(dtype, shape[, init=init])`.

    Parameters
    -------
    dtype : torch.dtype, np.dtype, str
        The data type of the output tensor.

        Some equivalent examples: `torch.float32`, `float`, `"float32"` or `"f32"`
    shape : str, tuple of (int or str)
        The shape of the output tensor.

        If `shape` is a string, it must be the name of
        a previously defined tensor parameter, and this tensor will be constructed to
        have the same shape as the parameter named by `shape`

        If `shape` is a tuple, every element in the tuple corresponds with one axis in
        the output tensor. For every such element:

        - `int` sets the size to be exactly the provided value.
        - `str` represents an expression that evaluates to an integer, and sets the size
          of the axis to be equal to the result of the expression. If the name of a
          tensor parameter is provided, this is equivalent to
          `param_name.shape[nth_dim]` where `nth_dim` is the index of the current axis.
    init : float or int, optional
        The initial value used to fill the output tensor with. If not provided, the
        output tensor will contain uninitialised memory (in the style of
        [`torch.empty`](https://docs.pytorch.org/docs/stable/generated/torch.empty.html)).

        Example: gradient tensors for the backward pass should be initialised with 0.
    """

    def __new__(
        cls,
        dtype: torch.dtype | np.dtype | str,
        shape: tuple[int | str, ...] | str,
        init: float = None,
    ):
        return OutputTensor("<MISSING>", AsDType(dtype).dtype, shape, init)

    def __class_getitem__(
        cls,
        item: tuple[torch.dtype | np.dtype | str, tuple[int | str, ...] | str]
        | tuple[torch.dtype | np.dtype | str, tuple[int | str, ...] | str, None],
    ):
        if not isinstance(item, tuple) or len(item) not in (2, 3):
            raise ValueError(
                "`Out` parameter must be of the form Out[type, shape]"
                " or Out[type, shape, init]"
            )
        return cls(*item)


class Scalar:
    """
    A type annotation for scalar input parameters in a `jit` function

    A scalar input is part of the argument list in the final operator, meaning the
    caller must provide it. It is not returned: for scalar outputs,
     use `Out(dtype, (1,))` instead.

    To use this annotation, use the syntax `param: Scalar(dtype)`,
    or the shorthand `param: dtype`.

    Parameters
    -------
    dtype : torch.dtype, np.dtype, str
        The data type of the scalar.

        Some equivalent examples: `torch.float32`, `float`, `"float32"` or `"f32"`
    """

    def __new__(cls, dtype: torch.dtype | np.dtype | str):
        return InputScalar("<MISSING>", AsDType(dtype).dtype)

    def __class_getitem__(
        cls,
        item: torch.dtype | np.dtype | str,
    ):
        return cls(item)


class Unused:
    """
    A type annotation for ignored parameters in a `jit` function

    This is a utility class for marking certain parameters to be skipped during
    compilation. An example of this would be a kernel which can optionally return an
    additional output (such as provenance indices for a maximum operation), allowing
    this output to be skipped programmatically.

    Note that all array accesses of a parameter marked `Unused` must be statically
    determined to be dead code (e.g. `if False`), as compilation will otherwise fail.

    To use this annotation, use e.g. `param: Out(...) if condition else Unused`
    """

    def __new__(cls):
        return cls

    def __class_getitem__(cls, item):
        return cls


def _as_kernel_param(
    name: str,
    typehint: InputTensor
    | InputScalar
    | OutputTensor
    | object
    | torch.dtype
    | np.dtype
    | str,
):
    if typehint is Unused:
        return UnusedParam(name)
    if isinstance(typehint, InputTensor | InputScalar | OutputTensor):
        return typehint._replace(name=name)

    if typehint in (torch.Tensor, np.ndarray):
        msg = (
            "Cannot directly type-hint as Tensor or NDarray, as shape information is"
            "required. Please use pnex.In, pnex.InMut or pnex.Out for Tensor in/outputs"
        )
        raise ValueError(msg)

    return InputScalar(name, AsDType(typehint).dtype)


def _standardise_thread_args(
    n_threads: str | tuple[str, str] | tuple[str, str, str],
    threads_per_block: int | tuple[int, int] | tuple[int, int, int] | None,
) -> tuple[tuple[str, str, str], tuple[int, int, int]]:
    if isinstance(n_threads, str | int):
        n_threads = (str(n_threads), "1", "1")
        threads_dim = 1
    elif len(n_threads) == 2:
        n_threads = (str(n_threads[0]), str(n_threads[1]), "1")
        threads_dim = 2
    else:
        n_threads = (str(n_threads[0]), str(n_threads[1]), str(n_threads[2]))
        threads_dim = len(n_threads)

    if threads_dim == 0 or threads_dim > 3:
        raise ValueError("Kernel launch parameters only supported between 1 and 3 dims")

    if threads_per_block is None:
        threads_per_block = [256, (16, 16), (8, 8, 4)][threads_dim - 1]

    if isinstance(threads_per_block, int):
        threads_per_block = (int(threads_per_block), 1, 1)
        blocks_dim = 1
    elif len(threads_per_block) == 2:
        threads_per_block = (int(threads_per_block[0]), int(threads_per_block[1]), 1)
        blocks_dim = 2
    else:
        threads_per_block = (
            int(threads_per_block[0]),
            int(threads_per_block[1]),
            int(threads_per_block[2]),
        )
        blocks_dim = len(threads_per_block)

    if threads_dim != blocks_dim:
        msg = f"Invalid n_threads / threads_per_block: {threads_dim=} but {blocks_dim=}"
        raise ValueError(msg)

    return n_threads, threads_per_block


def jit(
    *,
    n_threads: str | tuple[str, str] | tuple[str, str, str],
    to_extension: bool = False,
    cache_id: str = None,
    verbose: bool = False,
    threads_per_block: int | tuple[int, int] | tuple[int, int, int] = None,
    max_registers: int = None,
) -> Callable[[Callable[..., None]], torch.library.CustomOpDef]:
    """
    Compile a Python function in the form of a Numba-CUDA kernel to a PyTorch operator

    All parameters must be annotated with one of the argument types exported by this
    module, and the resulting operator will take `In`/`InMut`/`Scalar` parameters
    as arguments, while returning `Out` parameters.

    The keyword-only argument `n_threads` must be specified to indicate with how many
    threads the resulting kernel should be launched. The dimensionality of `n_threads`
    indicates the dimensionality of the launched kernel, while `threads_per_block`
    controls the size of each block.

    With `to_extension=True`, this function will also compile the PTX generated by Numba
    to a PyTorch native C++ extension, thereby reducing the overhead per call. If the
    resulting compilation times (first several seconds, then cached) are not acceptable,
    this additional compilation step can be skipped with `to_extension=False`.

    Parameters
    ----------
    n_threads : str, tuple[str, str], tuple[str, str, str]
        Expression(s) that evaluate to the total number of threads that the kernel
        should be launched with. Thread axes are filled in the order X, Y, Z: as such,
        passing only a single string `n_threads` is equivalent to passing
        ``(n_threads, 1, 1)``, with only the X thread-dimension being non-unit.

        In practice, this number is then divided by `threads_per_block` and rounded up
        to get the number of blocks for a single kernel invocation (blocks per grid).
    to_extension : bool = False
        Whether the function should be compiled to a PyTorch C++ extension or instead
        be left as a wrapped Numba-CUDA kernel. The signature of the returned function
        is identical in both cases, but compiling an extension can take 5+ seconds,
        while not compiling an extension incurs a small runtime overhead on every call.

        For neural networks, it is best to keep `to_extension` as False and use
        CUDA Graphs via `torch.compile(model, mode="reduce-overhead", fullgraph=True)`
        to eliminate the wrapper code.
        If this is not possible (due to highly dynamic code or irregular shapes), then
        the next best option would be to use `to_extension` and minimise call overhead.
    cache_id : str, optional
        The name to save the compiled extension under: clashing `cache_id`s will
        result in recompilations (clashing functions will evict each-other
        from the cache), but not miscompilations (the results will be correct).

        Only used when `to_extension=True`

    Returns
    -------
    decorator : (kernel) -> torch.library.CustomOpDef
        The resulting decorator will transform a Python function
        (if properly annotated, and the function is a valid Numba-CUDA kernel) into a
        `CustomOpDef`, where the signature is such that all parameters
        annotated with `In`, `InMut` or `Scalar` must be provided as arguments, and
        all `Out` parameters are returned.

        All parameters must be annotated with
        one of `In`, `InMut`, `Out`, `Scalar` or `Unused`

    Other Parameters
    -------
    verbose : bool = False
        Whether to print additional information about the compilation process.
        Compilation errors are always printed.
    threads_per_block : int, tuple[int, int], tuple[int, int, int] = None
        The number of threads within a thread block across the various dimensions.

        Depending on the dimensionality of `n_threads`, this defaults to one of:

        - For 1 dimension: 256
        - For 2 dimensions: (16, 16)
        - For 3 dimensions: (8, 8, 4)
    max_registers : int, optional
        Specify the maximum number of registers to be used by the kernel, with excess
        spilling over to local memory.
        Typically, the compiler is quite good at guessing the number of registers it
        should use, but limiting this to hit occupancy targets may help in some cases.
        This option is only available with `to_extension=False`, due to the structure
        of the Numba-CUDA API.

    Examples
    -------
    This is an example implementation of the `mymuladd` function from the PyTorch
    [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
    documentation, where we take 2D inputs instead of flattening.
    A variety of methods for specifying dtype and shape are used in this example, but
    sticking to one convention may be better for readability.

    >>> import pytorch_numba_extension_jit as pnex
    >>> # Can be invoked as mymuladd_2d(A, B, C) to return RESULT
    ... @pnex.jit(n_threads="result.numel()")
    ... def mymuladd_2d(
    ...     a: pnex.In(torch.float32, (None, None)),
    ...     b: pnex.In("f32", ("a.size(0)", "a.size(1)")),
    ...     c: float,  # : pnex.Scalar(float)
    ...     result: pnex.Out("float32", "a"),
    ... ):
    ...     idx = cuda.grid(1)
    ...     y, x = divmod(idx, result.shape[0])
    ...     if y < result.shape[0]:
    ...         result[y, x] = a[y, x] * b[y, x] + c

    Here, we can see an alternate version that uses
    [multidimensional blocks](https://nvidia.github.io/numba-cuda/user/kernels.html#multi-dimensional-blocks-and-grids)
    to achieve the same task, while compiling the result to a C++ operator using
    `to_extension`. Note that the `n_threads` argument is given sizes
    in the X, Y, Z order (consistent with C++ CUDA kernels), and that `numba.cuda.grid`
    also returns indices in this order, even if we might later use indices in e.g.
    `y, x` order.

    >>> @pnex.jit(n_threads=("result.size(1)", "result.size(0)"), to_extension=True)
    ... def mymuladd_grid(
    ...     a: pnex.In("f32", (None, None)),
    ...     b: pnex.In("f32", ("a.size(0)", "a.size(1)")),
    ...     c: float,
    ...     result: pnex.Out("f32", "a"),
    ... ):
    ...     # always use this order for names to be consistent with CUDA terminology:
    ...     x, y = cuda.grid(2)
    ...
    ...     if y < result.shape[0] and x < result.shape[1]:
    ...         result[y, x] = a[y, x] * b[y, x] + c

    Notes
    -------
    This function relies heavily on internals and undocumented behaviour of the
    Numba-CUDA PTX compiler. However, these internals have not changed in over 3 years,
    so it is reasonable to assume they will remain similar in future versions as well.
    Versions 0.9.0 and 0.10.0 of Numba-CUDA have been verified to work as expected.

    Additionally, it should be noted that storing the function to be compiled for
    compilation in a different stack frame may cause issues if some annotations use
    local variables and the module is using `from __future__ import annotations`.
    This is because annotations are not considered part of the function proper, so they
    are not closed over during the construction of a function (no cell is created).
    Using `jit` directly with the decorator syntax ``@pnex.jit(n_threads=...)``
    has no such problems, or one can selectively disable ``annotations`` for the file
    where the function to be compiled is defined.

    See Also
    -------
    numba.cuda.compile_for_current_device :
        used to compile the Python function into PTX: all functions must therefore also
        be [valid `numba.cuda` kernels](https://nvidia.github.io/numba-cuda/user/kernels.html).
    numba.cuda.jit : used instead to allow `to_extension=False`
    torch.utils.cpp_extension.load_inline : used to compile the PyTorch C++ extension
    """
    if to_extension and max_registers is not None:
        msg_regs = (
            "Numba-CUDA does not support passing max_registers "
            "to normal compilation functions, only to cuda.jit"
            "\n(can't pass both to_extension=True and max_registers!=None)"
        )
        raise ValueError(msg_regs)

    n_threads, threads_per_block = _standardise_thread_args(
        n_threads, threads_per_block
    )
    compile_kwargs = {
        "n_threads": n_threads,
        "threads_per_block": threads_per_block,
        "verbose": verbose,
        "cache_id": cache_id,
    }

    def decorator(pyfunc: Callable) -> torch.library.CustomOpDef:
        parameters = inspect.signature(pyfunc).parameters
        annotations = inspect.get_annotations(
            pyfunc,
            eval_str=True,
            # Is this illegal? Yes!
            # However, it's the only way to work around __future__.annotations
            locals=inspect.stack()[1].frame.f_locals,
            globals=inspect.stack()[1].frame.f_globals,
        )
        if "return" in annotations:
            if annotations["return"] is not None:
                msg = (
                    f"{pyfunc.__name__} annotated with return type "
                    f"`{annotations['return']}`, but kernels must return None"
                )
                raise TypeError(msg)

            del annotations["return"]
        if len(annotations) != len(parameters):
            msg = f"{pyfunc.__name__} did not have annotations for all arguments"
            raise TypeError(msg)

        kernel_params = [
            _as_kernel_param(name, annotations[name]) for name in parameters
        ]

        if verbose:
            print("Inferred params:", kernel_params)

        if to_extension:
            return compile_extension(
                pyfunc,
                kernel_params=kernel_params,
                **compile_kwargs,
            )

        return compile_array_api(
            pyfunc,
            kernel_params=kernel_params,
            max_registers=max_registers,
            **compile_kwargs,
        )

    return decorator
