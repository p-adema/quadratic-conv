from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import numba
import pytorch_numba_extension_jit as ptex
import torch
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

from .compiled_conv import CompiledConv, CompiledConvFixedLazy
from .utils import ConvMeta

warnings.simplefilter("ignore", numba.NumbaPerformanceWarning, 536)


class SubtractSemifield(NamedTuple):
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

    @classmethod
    def linear(cls) -> SubtractSemifield:
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

    # The torch compiler doesn't understand the Numba compiler
    @torch.compiler.disable
    @lru_cache  # noqa: B019
    def compile(
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
    ) -> CompiledConv:
        """
        Use a dynamic module that will recompile if it sees a new ConvMeta.
        Cannot be fully traced due to potential compiler calls.
        """
        return CompiledConv(self, thread_block_size, debug, to_extension)

    def lazy_fixed(
        self,
        thread_block_size: int = 256,
        debug: bool = False,
        to_extension: bool = True,
    ) -> CompiledConvFixedLazy:
        """
        Use a lazy module that will compile exactly once, then ignore all auxilliary
        arguments (undefined behaviour if future inputs differ in any way other than
        batch size).
        Can be fully traced.
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
    def get_result(res: torch.Tensor):
        return res


class _CompiledSubtractSemifield(NamedTuple):
    add: CUDADispatcher
    times: CUDADispatcher
    d_times_d_img: CUDADispatcher
    d_times_d_kernel: CUDADispatcher
    subtract: CUDADispatcher
    d_add_d_right: CUDADispatcher
    neutral: float

    @classmethod
    def compile(cls, semifield: SubtractSemifield) -> _CompiledSubtractSemifield:
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
    @ptex.jit(
        [
            ptex.InputTensor(
                "img", "f32", (None, meta.img_cs, meta.img_ys, meta.img_xs)
            ),
            ptex.InputTensor(
                "kernel", "f32", (meta.krn_os, meta.krn_cs, meta.krn_ys, meta.krn_xs)
            ),
            ptex.OutputTensor(
                "out_img",
                "f32",
                ("img.shape[0]", meta.out_cs, meta.out_ys, meta.out_xs),
            ),
        ],
        n_threads="out_img",
        compile_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
        cache_id=f"subtract_{cache_name}_{meta.cache_id()}",
    )
    def forwards(img, kernel, out_img):
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

        out_img[b, o_c, o_y, o_x] = acc

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
    @ptex.jit(
        [
            ptex.InputTensor(
                "img", "f32", (None, meta.img_cs, meta.img_ys, meta.img_xs)
            ),
            ptex.InputTensor(
                "kernel", "f32", (meta.krn_os, meta.krn_cs, meta.krn_ys, meta.krn_xs)
            ),
            ptex.InputTensor(
                "gradient",
                "f32",
                ("img.shape[0]", meta.out_cs, meta.out_ys, meta.out_xs),
            ),
            ptex.InputTensor("res_img", "f32", "gradient"),
            ptex.OutputTensor("out_img_grad", "f32", "img", init=0),
            ptex.OutputTensor("out_kernel_grad", "f32", "kernel", init=0),
        ],
        n_threads="gradient",
        compile_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
        cache_id=f"subtract_{cache_name}_{meta.cache_id()}",
    )
    def backwards(img, kernel, gradient, res_img, out_img_grad, out_kernel_grad):
        rem, o_x = divmod(cuda.grid(1), meta.out_xs)
        rem, o_y = divmod(rem, meta.out_ys)
        b, o_c = divmod(rem, meta.out_cs)
        if b >= img.shape[0]:
            return

        i_top_y = o_y * meta.str_y - meta.pad_y_beg
        i_left_x = o_x * meta.str_x - meta.pad_x_beg

        res = res_img[b, o_c, o_y, o_x]
        res_grad = gradient[b, o_c, o_y, o_x]

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
