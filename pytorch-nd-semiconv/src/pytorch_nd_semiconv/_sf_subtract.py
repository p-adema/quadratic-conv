import math
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any, NamedTuple, Self

import torch

from ._compiled_conv import CompiledConv, CompiledConvFixedLazy
from ._sf_subtract_codegen import (
    CompiledSubtractSemifield,
    compile_backwards,
    compile_forwards,
)
from ._utils import ConvMeta


class SubtractSemifield(NamedTuple):
    r"""
    A semifield definition where semifield addition has an inverse

    For such semifields, the backwards pass can be done by 'subtracting' every value
    from the result to get the arguments for the additive derivative.
    The resulting module is compiled and works only on CUDA devices.

    Note that, while this implementation is more memory-efficient than
    `BroadcastSemifield`, it is typically slower in execution speed.
    If memory usage is not a concern but training speed is, then `BroadastSemifield`
    should therefore be preferred.

    Parameters
    -------
    add : (float, float) -> float
        Given an accumulator and a multiplied value, perform scalar semifield addition
        \(\oplus\).
    times : (float, float) -> float
        Given an image and a kernel value, perform scalar semifield multiplication
        \(\otimes\).
    d_times_d_img : (float, float) -> float
        Given the two arguments to `times`, compute the derivative to the first:
        \[\frac{\delta (\textrm{img} \otimes \textrm{kernel}) }{\delta\textrm{img}}\]
    d_times_d_kernel : (float, float) -> float
        Given the two arguments to `times`, compute the derivative to the second:
        \[\frac{\delta (\textrm{img} \otimes \textrm{kernel}) }{\delta\textrm{kernel}}\]
    subtract : (float, float) -> float
        Given the final accumulator value `res` and a multiplied value `val`,
        use the inverse of `add` and

        return an `acc` such that `add(acc, val) == res`.
        In other words: perform semifield subtraction.
    d_add_d_right : (float, float) -> float
        Given the two arguments to `add`, compute the derivative to the second:
        \[\frac{\delta (\textrm{acc} \oplus \textrm{val}) }{\delta\textrm{val}}\]
    zero : float
        The semifield zero.
    cache_name : str, optional
        Identifier for this semifield, allows for extension compilations to be cached.

        Instances of `SubtractSemifield` that are meaningfully different should not have
        the same `cache_name`, as this may lead to the wrong compilation being used.

    Other Parameters
    -------
    post_sum : (float) -> float, optional
        Some semifield additions are fairly complex and computationally expensive, but
        can be reinterpreted as a repeated simpler operation, followed by a scalar
        transformation of the final accumulator value.
        `post_sum` is then this scalar transformation, taking the final accumulator
        value `res` and transforming it into a value `out`.

        Taking the root semifield \(R_3\) as an example, we can see that if we use

        - `times` as \(a \otimes_3 b = (a \times b)^3 \)
        - `add` as \(+\)
        - `post_sum` as \(\textrm{out} = \sqrt[3]{\textrm{res}} \)

        then we can perform the reduction in terms of simple scalar addition, instead
        of having to take the power and root at every step.

        Using such a transfrom does, however, require defining two other operators,
        namely the inverse and the derivative.
        When these are given, `subtract` and `d_add_d_right` will be given untransformed
        arguments: in the root semifield example, that would mean that the arguments
        to `subtract` and `d_add_d_right` are not yet taken to the `p`'th root.
    undo_post_sum : (float) -> float, optional
        The inverse of `post_sum`, required if `post_sum` is given.
    d_post_d_acc : (float) -> float, optional
        The derivative of `post_sum` to its argument, required if `post_sum` is given:
        \[\frac{\delta \textrm{post_sum}(\textrm{res}) }{\delta\textrm{res}}\]

    Examples
    -------
    Linear convolution that will recompile for new parameters:

    >>> linear = SubtractSemifield.linear().dynamic()

    \(R_3\) convolution that will compile only once:

    >>> root = SubtractSemifield.root(3.0).lazy_fixed()

    For examples of how to construct a `SubtractSemifield`, see the source code.
    """

    add: Callable[[float, float], float]  # (acc, val) -> acc (+) val
    times: Callable[[float, float], float]  # (img_val, krn_val) -> multiplied_val
    d_times_d_img: Callable[[float, float], float]
    d_times_d_kernel: Callable[[float, float], float]
    # (res, val) -> res-val, such that val (+) (res - val) == res
    subtract: Callable[[float, float], float]
    # d(acc (+) val) / dval
    d_add_d_right: Callable[[float, float], float]
    zero: float
    cache_name: str = None  # Cache identifier: distinct for different operators

    post_sum: Callable[[float], float] = None  # (final_acc) -> res
    undo_post_sum: Callable[[float], float] = None  # (res) -> final_acc
    d_post_d_acc: Callable[[float], float] = None  # (final_acc) -> dacc

    @classmethod
    def linear(cls) -> Self:
        r"""
        Construct a linear `SubtractSemifield`

        The linear field is defined as:
        \[(\mathbb{R}, +, \times)\]

        Mainly for comparison purposes: the linear convolutions offered by PyTorch
        use CUDNN, which is far better optimised for CUDA devices.
        """
        return cls(
            add=lambda acc, val: acc + val,
            times=lambda img_val, kernel_val: img_val * kernel_val,
            d_times_d_img=lambda _i, kernel_val: kernel_val,
            d_times_d_kernel=lambda img_val, _k: img_val,
            subtract=lambda res, val: res - val,
            d_add_d_right=lambda _a, _v: 1,
            zero=0,
            cache_name="_linear",
        )

    @classmethod
    def root(cls, p: float) -> Self:
        r"""
        Construct a \(R_p\) `SubtractSemifield`.

        The root semifields are defined as:
        \[(\mathbb{R}_+, \oplus_p, \times) \textrm{ for all } p\ne0 \textrm{ where }
        a\oplus_p b= \sqrt[p]{a^p+b^p} \]
        with the semifield zero being \(0\) and the semifield one being \(1\).

        Parameters
        ----------
        p : int
            The power to use in \(\oplus_p\).
            May not be zero.
        """
        assert p != 0, f"Invalid value: {p=}"
        return cls(
            times=lambda img_val, kernel_val: (img_val * kernel_val) ** p,
            add=lambda acc, val: (acc + val),
            post_sum=lambda acc: acc ** (1 / p),
            zero=0,
            cache_name=f"_root_{cls._number_to_cache(p)}",
            undo_post_sum=lambda res: res**p,
            subtract=lambda acc, val: acc - val,
            d_times_d_img=lambda a, b: ((a * b) ** p) * p / a,
            d_times_d_kernel=lambda a, b: ((a * b) ** p) * p / b,
            d_add_d_right=lambda _a, _b: 1,
            d_post_d_acc=lambda acc: (1 / p) * acc ** (1 / p - 1),
        )

    @classmethod
    def log(cls, mu: float) -> Self:
        r"""
        Construct a \(L_+\mu\) or \(L_-\mu\) `SubtractSemifield`.

        The log semifields are defined as:
        \[(\mathbb{R}\cup \{\pm\infty\}, \oplus_\mu, +) \textrm{ for all } \mu\ne0
        \textrm{ where }
        a\oplus_\mu b= \frac{1}{\mu}\ln(e^{\mu a}+e^{\mu b}) \]
        with the semifield zero being \(-\infty\) for \(\mu>0\) and \(\infty\)
        otherwise, and the semifield one being \(0\).

        Parameters
        ----------
        mu : int
            The base to use in \(\oplus_\mu\).
            May not be zero.
        """
        assert mu != 0, f"Invalid value: {mu=}"
        return cls(
            times=lambda img_val, kernel_val: math.exp((img_val + kernel_val) * mu),
            add=lambda acc, val: (acc + val),
            post_sum=lambda acc: math.log(acc) / mu,
            zero=0,
            cache_name=f"_log_{cls._number_to_cache(mu)}",
            d_times_d_img=lambda a, b: mu * math.exp((a + b) * mu),
            d_times_d_kernel=lambda a, b: mu * math.exp((a + b) * mu),
            undo_post_sum=lambda res: math.exp(res * mu),
            subtract=lambda acc, val: acc - val,
            d_add_d_right=lambda _a, _v: 1,
            d_post_d_acc=lambda acc: 1 / (mu * acc),
        )

    # The torch compiler doesn't understand the Numba compiler
    @torch.compiler.disable
    @lru_cache  # noqa: B019
    def _compile(
        self,
        meta: ConvMeta,
        compile_options: Mapping[str, Any],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        impl = compile_options.get("impl", "glb")
        if impl not in ("glb",):
            raise ValueError(f"Unknown {impl=}")

        cmp_semi = CompiledSubtractSemifield.compile(self)
        impls = {"glb": compile_forwards}

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
        )
        forwards.register_autograd(backwards, setup_context=backwards_setup)

        return forwards

    def dynamic(
        self,
        thread_block_size: int = 256,
        to_extension: bool = False,
        debug: bool = False,
        kernel_inflation: int = 16,
    ) -> torch.nn.Module:
        """
        Create a *recompiling* convolution Module based on this `SubtractSemifield`.

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
        thread_block_size: int = 256,
        to_extension: bool = False,
        debug: bool = False,
        kernel_inflation: int = 16,
    ) -> torch.nn.Module:
        """
        Create a *once-compiling* convolution Module based on this `SubtractSemifield`.

        In general, `SubtractSemifield.dynamic` should be preferred for testing and also
        for training if the model can be traced by CUDA Graphs.
        If CUDA Graphs cannot capture the model code due to dynamic elements, then using
        `SubtractSemifield.lazy_fixed` with `to_extension=True` will minimise overhead.

        Returns
        -------
        conv : nn.Module
            A convolution module, suitable for use in `GenericConv`.
            Note that compilation will be based on the first inputs seen, after which
            the operation will be fixed: **only batch size may be changed afterwards**.
            The module is, however, traceable by e.g. `torch.compile`.

        Other Parameters
        ----------
        thread_block_size : int = 256
            The number of threads per CUDA block
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
                self.add,
                self.times,
                self.d_times_d_img,
                self.d_times_d_kernel,
                self.subtract,
                self.d_add_d_right,
                self.zero,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, SubtractSemifield):
            return False
        if self.cache_name is not None:
            return self.cache_name == other.cache_name

        return self is other

    @staticmethod
    def _get_result(res: torch.Tensor):
        return res

    @staticmethod
    def _number_to_cache(n: float):
        return str(n).replace(".", "_").replace("-", "_minus_")
