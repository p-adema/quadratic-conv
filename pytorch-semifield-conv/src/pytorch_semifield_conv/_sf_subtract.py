import math
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple, Self

import numba
import pytorch_numba_extension_jit as pnex
import torch
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

from ._compiled_conv import CompiledConv, CompiledConvFixedLazy
from ._utils import ConvMeta

warnings.simplefilter("ignore", numba.NumbaPerformanceWarning, 536)


class SubtractSemifield(NamedTuple):
    r"""
    A semifield definition where semifield addition has an inverse

    For such semifields, the backwards pass can be done by 'subtracting' every value
    from the result to get the arguments for the additive derivative.
    The resulting module is compiled and works only on CUDA devices.

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

        return an `acc` such that `add(acc, val) == res`.
        In other words: perform semifield subtraction.
    d_add_d_right : (float, float) -> float
        Given the two arguments to `add`, compute the derivative to the second:
        \[\frac{\delta (\textrm{acc} \oplus \textrm{val}) }{\delta\textrm{val}}\]
    zero : float
        The semifield zero.
    cache_name : str, optional
        Identifier for this semifield, allows for compilations to be cached.

        Instances of `SelectSemifield` that are meaningfully different should not have
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

    For examples of how to construct a `SubtractSemifield` manually, see the source code.
    """

    add: Callable[[float, float], float]  # (acc, val) -> acc (+) val
    times: Callable[[float, float], float]  # (img_val, krn_val) -> multiplied_val
    d_times_d_img: Callable[[float, float], float]
    d_times_d_kernel: Callable[[float, float], float]
    # (res, val) -> res-val, such that val (+) (res - val) == res
    subtract: Callable[[float, float], float]
    # d(acc (+) val) / dval
    d_add_d_right: Callable[[float, float], float]
    neutral: float
    cache_name: str = None  # Cache identifier: distinct for different operators

    post_sum: Callable[[float], float] = None  # (final_acc) -> res
    undo_post_sum: Callable[[float], float] = None  # (res) -> final_acc
    d_post_d_acc: Callable[[float], float] = None  # (final_acc) -> dacc

    @classmethod
    def linear(cls) -> Self:
        """Construct a linear `SubtractSemifield`"""
        return cls(
            add=lambda acc, val: acc + val,
            times=lambda img_val, kernel_val: img_val * kernel_val,
            d_times_d_img=lambda _i, kernel_val: kernel_val,
            d_times_d_kernel=lambda img_val, _k: img_val,
            subtract=lambda res, val: res - val,
            d_add_d_right=lambda _a, _v: 1,
            neutral=0,
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
            neutral=0,
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
            The base to use in \(\oplus_mu\).
            May not be zero.
        """
        assert mu != 0, f"Invalid value: {mu=}"
        return cls(
            times=lambda img_val, kernel_val: math.exp((img_val + kernel_val) * mu),
            add=lambda acc, val: (acc + val),
            post_sum=lambda acc: math.log(acc) / mu,
            neutral=0,
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
        thread_block_size: int = 256,
        debug: bool = False,
        to_extension: bool = True,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        cmp_semi = _CompiledSubtractSemifield.compile(self)

        forwards = _compile_forwards(
            semifield=cmp_semi,
            meta=meta,
            thread_block_size=thread_block_size,
            debug=debug,
            cache_name="_temporary" if self.cache_name is None else self.cache_name,
            to_extension=to_extension,
        )
        backwards, backwards_setup = _compile_backwards(
            semifield=cmp_semi,
            meta=meta,
            thread_block_size=thread_block_size,
            debug=debug,
            cache_name="_temporary" if self.cache_name is None else self.cache_name,
            to_extension=to_extension,
        )
        forwards.register_autograd(backwards, setup_context=backwards_setup)

        return forwards

    def dynamic(
        self,
        thread_block_size: int = 256,
        debug: bool = False,
        to_extension: bool = True,
    ) -> torch.nn.Module:
        """
        Create a *recompiling* convolution Module based on this `SubtractSemifield`.


        Parameters
        ----------
        thread_block_size : int = 256
            The number of threads per CUDA block
        to_extension : bool = True
            Whether the resulting module should compile to a PyTorch extension.
            Doing so increases compilation times, but reduces per-call overhead.
        debug : bool = False
            Whether to print additional debugging and compilation information.

        Returns
        -------
        conv : nn.Module
            A convolution module, suitable for use in `GenericConv2D`.
            Note that the compilation process is not traceable, and recompilations
            **may cause errors when using `torch.compile`**.
        """
        return CompiledConv(self, thread_block_size, debug, to_extension)

    def lazy_fixed(
        self,
        thread_block_size: int = 256,
        debug: bool = False,
        to_extension: bool = True,
    ) -> torch.nn.Module:
        """
        Create a *once-compiling* convolution Module based on this `SubtractSemifield`.

        Parameters
        ----------
        thread_block_size : int = 256
            The number of threads per CUDA block
        to_extension : bool = True
            Whether the resulting module should compile to a PyTorch extension.
            Doing so increases compilation times, but reduces per-call overhead.
        debug : bool = False
            Whether to print additional debugging and compilation information.

        Returns
        -------
        conv : nn.Module
            A convolution module, suitable for use in `GenericConv2D`.
            Note that compilation will be based on the first inputs seen, after which
            the operation will be fixed: **only batch size may be changed afterwards**.
            The module is, however, traceable by e.g. `torch.compile`.
        """
        return CompiledConvFixedLazy(self, thread_block_size, debug, to_extension)

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
                self.neutral,
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


class _CompiledSubtractSemifield(NamedTuple):
    add: CUDADispatcher
    times: CUDADispatcher
    d_times_d_img: CUDADispatcher
    d_times_d_kernel: CUDADispatcher
    subtract: CUDADispatcher
    d_add_d_right: CUDADispatcher
    neutral: float

    # Optional:
    post_sum: CUDADispatcher
    undo_post: CUDADispatcher
    post_sum_bwd: CUDADispatcher

    @classmethod
    def compile(cls, semifield: SubtractSemifield) -> Self:
        if semifield.post_sum is None:
            assert semifield.undo_post_sum is None, "post_sum not specified"
            assert semifield.d_post_d_acc is None, "post_sum not specified"
            post_sum, undo_post, post_bwd = lambda i: i, lambda i: i, lambda _: 1
        else:
            assert semifield.undo_post_sum is not None, "need inverse of post_sum"
            assert semifield.d_post_d_acc is not None, "need derivative of post_sum"
            post_sum = semifield.post_sum
            undo_post = semifield.undo_post_sum
            post_bwd = semifield.d_post_d_acc

        return _CompiledSubtractSemifield(
            cuda.jit(semifield.add, device=True, inline="always", cache=True),
            cuda.jit(semifield.times, device=True, inline="always", cache=True),
            cuda.jit(semifield.d_times_d_img, device=True, inline="always", cache=True),
            cuda.jit(
                semifield.d_times_d_kernel, device=True, inline="always", cache=True
            ),
            cuda.jit(semifield.subtract, device=True, inline="always", cache=True),
            cuda.jit(semifield.d_add_d_right, device=True, inline="always", cache=True),
            semifield.neutral,
            cuda.jit(post_sum, device=True, inline="always", cache=True),
            cuda.jit(undo_post, device=True, inline="always", cache=True),
            cuda.jit(post_bwd, device=True, inline="always", cache=True),
        )


def _compile_forwards(
    semifield: _CompiledSubtractSemifield,
    meta: ConvMeta,
    thread_block_size: int = 256,
    debug: bool = False,
    cache_name: str = "",
    to_extension: bool = True,
):
    # noinspection DuplicatedCode
    @pnex.jit(
        n_threads="out_img.numel()",
        to_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
        cache_id=f"subtract_{cache_name}_{meta.cache_id()}",
    )
    def forwards(
        img: pnex.In("f32", (None, meta.img_cs, meta.img_ys, meta.img_xs)),
        kernel: pnex.In("f32", (meta.krn_os, meta.krn_cs, meta.krn_ys, meta.krn_xs)),
        out_img: pnex.Out("f32", ("img", meta.out_cs, meta.out_ys, meta.out_xs)),
    ):
        rem, o_x = divmod(cuda.grid(1), meta.out_xs)
        rem, o_y = divmod(rem, meta.out_ys)
        b, o_c = divmod(rem, meta.out_cs)
        if b >= img.shape[0]:
            return

        i_top_y = o_y * meta.str_y - meta.pad_y_beg
        i_left_x = o_x * meta.str_x - meta.pad_x_beg

        acc = semifield.neutral

        group_number = o_c // meta.grp_o
        # If we're not broadcasting, then we have a separate kernel
        # for every output channel. If we are broadcasting, we instead loop
        # around the kernels every k_os (which == krn_group_size)
        k_o = o_c if not meta.group_broadcasting else o_c % meta.grp_o

        # For a pooling, we have only one input channel, so group_idx is always 0
        for group_idx in range(meta.krn_cs):
            for y_step, i_y in enumerate(
                range(i_top_y, i_top_y + meta.krn_ys * meta.dil_y, meta.dil_y)
            ):
                for x_step, i_x in enumerate(
                    range(
                        i_left_x,
                        i_left_x + meta.krn_xs * meta.dil_x,
                        meta.dil_x,
                    )
                ):
                    if i_x < 0 or i_x >= meta.img_xs or i_y < 0 or i_y >= meta.img_ys:
                        continue

                    # Need to explicitly use seperate variable, due to compiler error

                    if meta.mirror_kernel:
                        k_x = meta.krn_xs - 1 - x_step
                        k_y = meta.krn_ys - 1 - y_step
                    else:
                        k_x = x_step
                        k_y = y_step

                    i_c = group_number * meta.krn_cs + group_idx
                    img_val = img[b, i_c, i_y, i_x]
                    kernel_val = kernel[k_o, group_idx, k_y, k_x]

                    val = semifield.times(img_val, kernel_val)
                    acc = semifield.add(acc, val)

        out_img[b, o_c, o_y, o_x] = semifield.post_sum(acc)

    return forwards


def _compile_backwards(
    semifield: _CompiledSubtractSemifield,
    meta: ConvMeta,
    thread_block_size: int = 256,
    debug: bool = False,
    cache_name: str = "",
    to_extension: bool = True,
):
    # noinspection PyArgumentList,DuplicatedCode
    @pnex.jit(
        n_threads="gradient.numel()",
        to_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
        cache_id=f"subtract_{cache_name}_{meta.cache_id()}",
    )
    def backwards(
        img: pnex.In("f32", (None, meta.img_cs, meta.img_ys, meta.img_xs)),
        kernel: pnex.In("f32", (meta.krn_os, meta.krn_cs, meta.krn_ys, meta.krn_xs)),
        gradient: pnex.In("f32", ("img", meta.out_cs, meta.out_ys, meta.out_xs)),
        res_img: pnex.In("f32", "gradient"),
        out_img_grad: pnex.Out("f32", "img", init=0),
        out_kernel_grad: pnex.Out("f32", "kernel", init=0),
    ):
        rem, o_x = divmod(cuda.grid(1), meta.out_xs)
        rem, o_y = divmod(rem, meta.out_ys)
        b, o_c = divmod(rem, meta.out_cs)
        if b >= img.shape[0]:
            return

        i_top_y = o_y * meta.str_y - meta.pad_y_beg
        i_left_x = o_x * meta.str_x - meta.pad_x_beg

        res = semifield.undo_post(res_img[b, o_c, o_y, o_x])
        res_grad = gradient[b, o_c, o_y, o_x] * semifield.post_sum_bwd(res)

        group_number = o_c // meta.grp_o
        k_o = o_c if not meta.group_broadcasting else o_c % meta.grp_o
        for group_idx in range(meta.krn_cs):
            for y_step, i_y in enumerate(
                range(i_top_y, i_top_y + meta.krn_ys * meta.dil_y, meta.dil_y)
            ):
                for x_step, i_x in enumerate(
                    range(
                        i_left_x,
                        i_left_x + meta.krn_xs * meta.dil_x,
                        meta.dil_x,
                    )
                ):
                    if i_x < 0 or i_x >= meta.img_xs or i_y < 0 or i_y >= meta.img_ys:
                        continue

                    if meta.mirror_kernel:
                        k_x = meta.krn_xs - 1 - x_step
                        k_y = meta.krn_ys - 1 - y_step
                    else:
                        k_x = x_step
                        k_y = y_step

                    i_c = group_number * meta.krn_cs + group_idx
                    img_val = img[b, i_c, i_y, i_x]
                    kernel_val = kernel[k_o, group_idx, k_y, k_x]

                    val = semifield.times(img_val, kernel_val)
                    acc = semifield.subtract(res, val)
                    val_grad = semifield.d_add_d_right(acc, val) * res_grad

                    cuda.atomic.add(
                        out_img_grad,
                        (b, i_c, i_y, i_x),
                        semifield.d_times_d_img(img_val, kernel_val) * val_grad,
                    )
                    cuda.atomic.add(
                        out_kernel_grad,
                        (k_o, group_idx, k_y, k_x),
                        semifield.d_times_d_kernel(img_val, kernel_val) * val_grad,
                    )

    def backwards_setup(ctx, inputs, output):
        img, kernel = inputs
        ctx.img = img
        ctx.kernel = kernel
        ctx.res_img = output

    def backwards_entry(ctx, grad_output):
        g_img, g_kern = backwards(ctx.img, ctx.kernel, grad_output, ctx.res_img)
        return g_img, g_kern

    return backwards_entry, backwards_setup
