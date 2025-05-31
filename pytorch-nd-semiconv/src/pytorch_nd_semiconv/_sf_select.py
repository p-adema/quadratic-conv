import math
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any, NamedTuple, Self

import torch

from ._compiled_conv import CompiledConv, CompiledConvFixedLazy
from ._sf_select_codegen import (
    CompiledSelectSemifield,
    compile_backwards,
    compile_forwards,
)
from ._utils import ConvMeta


class SelectSemifield(NamedTuple):
    r"""
    A semifield definition where semifield addition selects a single value

    For such semifields, the backwards pass can be done very efficiently by memoizing
    the output provenance (index of the chosen value).
    The resulting module is compiled and works only on CUDA devices.

    Parameters
    -------
    add_select : (float, float) -> bool
        Given two values, return whether we should pick the second value (`True`), or
        instead keep the first (`False`).
        If there is no meaningful difference between the two values, `False` should be
        preferred.
    times : (float, float) -> float
        Given an image and a kernel value, perform scalar semifield multiplication
        \(\otimes\).
    d_times_d_img : (float, float) -> float
        Given the two arguments to `times`, compute the derivative to the first:
        \[\frac{\delta (\textrm{img} \otimes \textrm{kernel}) }{\delta\textrm{img}}\]
    d_times_d_kernel : (float, float) -> float
        Given the two arguments to `times`, compute the derivative to the second:
        \[\frac{\delta (\textrm{img} \otimes \textrm{kernel}) }{\delta\textrm{kernel}}\]
    zero : float
        The semifield zero.
    cache_name : str, optional
        Identifier for this semifield, allows for extension compilations to be cached.

        Instances of `SelectSemifield` that are meaningfully different should not have
        the same `cache_name`, as this may lead to the wrong compilation being used.

    Examples
    -------
    \(T_+\) convolution that will recompile for new inputs:

    >>> dilation = SelectSemifield.tropical_max().dynamic()

    \(T_-\) convolution that will compile only once:

    >>> erosion = SelectSemifield.tropical_min_negated().lazy_fixed()

    For examples of how to construct a `SelectSemifield` manually, see the source code.
    """

    add_select: Callable[[float, float], bool]  # Return True if we should pick right
    times: Callable[[float, float], float]  # (img_val, krn_val) -> multiplied_val
    d_times_d_img: Callable[[float, float], float]
    d_times_d_kernel: Callable[[float, float], float]
    zero: float
    cache_name: str = None  # Cache identifier: distinct for different operators

    @classmethod
    def tropical_max(cls) -> Self:
        r"""
        Construct a \(T_+\) `SelectSemifield`.

        The tropical max semifield / semiring is defined as:
        \[(\mathbb{R}\cup \{-\infty\}, \max, +)\]
        """
        return cls(
            add_select=lambda left, right: left < right,
            times=lambda img_val, kernel_val: img_val + kernel_val,
            d_times_d_img=lambda _i, _k: 1.0,
            d_times_d_kernel=lambda _i, _k: 1.0,
            zero=-math.inf,
            cache_name="_tropical_max",
        )

    @classmethod
    def tropical_min_negated(cls) -> Self:
        r"""
        Construct a `SelectSemifield` similar to \(T_-\), where the kernel is negated.

        The usual tropical min semifield / semiring is defined as:
        \[(\mathbb{R}\cup \{\infty\}, \min, +)\]

        This version is slightly modified:
        while performing erosion using \(T_-\) requires first negating the kernel, this
        modified semifield has \(-\) instead of \(+\) as the semifield multiplication.
        As such, the resulting convolution will work with non-negated kernels as inputs,
        making the interface more similar to the dilation in \(T_+\).
        """
        return cls(
            add_select=lambda left, right: left > right,
            times=lambda img_val, kernel_val: img_val - kernel_val,
            d_times_d_img=lambda _i, _k: 1.0,
            d_times_d_kernel=lambda _i, _k: -1.0,
            zero=math.inf,
            cache_name="_tropical_min",
        )

    # The torch compiler doesn't understand the Numba compiler
    @torch.compiler.disable
    @lru_cache  # noqa: B019
    def _compile(
        self,
        meta: ConvMeta,
        compile_options: Mapping[str, Any],
    ) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        impl = compile_options.get("impl", "glb")
        if impl not in ("glb",):
            raise ValueError(f"Unknown {impl=}")

        cmp_semi = CompiledSelectSemifield.compile(self)

        impls = {
            "glb": compile_forwards,
        }
        forwards = impls[impl](
            semifield=cmp_semi,
            meta=meta,
            thread_block_size=compile_options.get("thread_block_size"),
            debug=compile_options.get("debug", False),
            cache_name="_temporary" if self.cache_name is None else self.cache_name,
            to_extension=compile_options.get("to_extension", False),
        )
        backwards, backwards_setup = compile_backwards(
            semifield=cmp_semi,
            meta=meta,
            thread_block_size=compile_options.get("thread_block_size"),
            debug=compile_options.get("debug", False),
            cache_name="_temporary" if self.cache_name is None else self.cache_name,
            to_extension=compile_options.get("to_extension", False),
            kernel_inflation=compile_options.get("kernel_inflation", 16),
        )
        forwards.register_autograd(backwards, setup_context=backwards_setup)

        return forwards

    def dynamic(
        self,
        thread_block_size: int = None,
        to_extension: bool = False,
        debug: bool = False,
        kernel_inflation: int = 16,
    ) -> torch.nn.Module:
        """
        Create a *recompiling* convolution Module based on this `SelectSemifield`.

        Returns
        -------
        conv : nn.Module
            A convolution module, suitable for use in `GenericConv`.
            Note that the compilation process is not traceable, and recompilations
            **may cause errors when using `torch.compile`** for backends other than
            CUDA Graphs

        Other Parameters
        ----------
        thread_block_size : int = 128
            The number of threads per CUDA block.
        to_extension : bool = False
            Whether the resulting module should compile to a PyTorch extension.
            Doing so increases compilation times, but reduces per-call overhead
            when not using CUDA-Graphs.

            For neural networks, it is best to keep `to_extension` as False and use
            CUDA Graphs via `torch.compile(model, mode="reduce-overhead",
            fullgraph=True)` to eliminate the wrapper code.
            If this is not possible (due to highly dynamic code or irregular shapes),
            then the next best option would be to use `to_extension`
            and minimise call overhead.
        debug : bool = False
            Whether to print additional debugging and compilation information.
        kernel_inflation : int = 16
            The factor to inflate the kernel gradient with, to better distribute
            atomic operations.
            A larger factor can improve performance when the number of output pixels
            per kernel value is high, but only up to a point, and at the cost of memory
            efficiency.
        """
        return CompiledConv(
            self,
            {
                "thread_block_size": thread_block_size,
                "debug": debug,
                "to_extension": to_extension,
                "kernel_inflation": kernel_inflation,
            },
        )

    def lazy_fixed(
        self,
        thread_block_size: int = None,
        to_extension: bool = False,
        debug: bool = False,
        kernel_inflation: int = 16,
    ) -> torch.nn.Module:
        """
        Create a *once-compiling* convolution Module based on this `SelectSemifield`.

        In general, `SelectSemifield.dynamic` should be preferred for testing and also
        for training if the model can be traced by CUDA Graphs.
        If CUDA Graphs cannot capture the model code due to dynamic elements, then using
        `SelectSemifield.lazy_fixed` with `to_extension=True` will minimise overhead.

        Returns
        -------
        conv : nn.Module
            A convolution module, suitable for use in `GenericConv`.
            Note that compilation will be based on the first inputs seen, after which
            the operation will be fixed: **only batch size may be changed afterwards**.
            The module is, however, traceable by e.g. `torch.compile` on all backends.

        Other Parameters
        ----------
        thread_block_size : int = 128
            The number of threads per CUDA block.
        to_extension : bool = False
            Whether the resulting module should compile to a PyTorch extension.
            Doing so increases compilation times, but reduces per-call overhead
            when not using CUDA-Graphs.

            For neural networks, it is best to keep `to_extension` as False and use
            CUDA Graphs via `torch.compile(model, mode="reduce-overhead",
            fullgraph=True)` to eliminate the wrapper code.
            If this is not possible (due to highly dynamic code or irregular shapes),
            then the next best option would be to use `to_extension`
            and minimise call overhead.
        debug : bool = False
            Whether to print additional debugging and compilation information.
        kernel_inflation : int = 16
            The factor to inflate the kernel gradient with, to better distribute
            atomic operations.
            A larger factor can improve performance when the number of output pixels
            per kernel value is high, but only up to a point, and at the cost of memory
            efficiency.
        """
        return CompiledConvFixedLazy(
            self,
            {
                "thread_block_size": thread_block_size,
                "debug": debug,
                "to_extension": to_extension,
                "kernel_inflation": kernel_inflation,
            },
        )

    def __hash__(self):
        if self.cache_name is not None:
            return hash(self.cache_name)

        return hash(
            (
                self.add_select,
                self.times,
                self.d_times_d_img,
                self.d_times_d_kernel,
                self.zero,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, SelectSemifield):
            return False
        if self.cache_name is not None:
            return self.cache_name == other.cache_name

        return self is other

    @staticmethod
    def _get_result(res: tuple[torch.Tensor, torch.Tensor]):
        return res[0]
