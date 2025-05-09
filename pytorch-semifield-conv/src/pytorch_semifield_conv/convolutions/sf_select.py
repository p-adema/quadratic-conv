from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import numba
import numpy as np
import pytorch_numba_extension_jit as ptex
import torch
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

from .compiled_conv import CompiledConv, CompiledConvFixedLazy
from .utils import ConvMeta

warnings.simplefilter("ignore", numba.NumbaPerformanceWarning, 536)


class SelectSemifield(NamedTuple):
    add_select: Callable[[float, float], bool]  # Return True if we should pick right
    times: Callable[[float, float], float]  # (img_val, krn_val) -> multiplied_val
    d_times_d_img: Callable[[float, float], float]
    d_times_d_kernel: Callable[[float, float], float]
    neutral: float
    cache_name: str = None  # Cache identifier: distinct for different operators

    @classmethod
    def tropical_max(cls) -> SelectSemifield:
        return cls(
            add_select=lambda left, right: left < right,
            times=lambda img_val, kernel_val: img_val + kernel_val,
            d_times_d_img=lambda _i, _k: 1.0,
            d_times_d_kernel=lambda _i, _k: 1.0,
            neutral=-float("inf"),
            cache_name="_tropical_max",
        )

    @classmethod
    def tropical_min_negated(cls) -> SelectSemifield:
        return cls(
            add_select=lambda left, right: left > right,
            times=lambda img_val, kernel_val: img_val - kernel_val,
            d_times_d_img=lambda _i, _k: 1.0,
            d_times_d_kernel=lambda _i, _k: -1.0,
            neutral=float("inf"),
            cache_name="_tropical_min",
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
    ) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        prov_t = _ProvType.smallest_required(meta)
        cmp_semi = _CompiledSelectSemifield.compile(self)

        forwards = _compile_forwards(
            semifield=cmp_semi,
            meta=meta,
            prov_t=prov_t,
            thread_block_size=thread_block_size,
            debug=debug,
            cache_name="_temporary" if self.cache_name is None else self.cache_name,
            to_extension=to_extension,
        )
        backwards, backwards_setup = _compile_backwards(
            semifield=cmp_semi,
            meta=meta,
            prov_t=prov_t,
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
        Use a dynamic module that will recompile if it sees a new ConvMeta, up to
        `max_compilations` times.
        Cannot be fully traced
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
        Can be fully traced
        """
        return CompiledConvFixedLazy(self, thread_block_size, debug, to_extension)

    def __hash__(self):
        if self.cache_name is not None:
            return hash(self.cache_name)

        return hash(
            (
                self.add_select,
                self.times,
                self.d_times_d_img,
                self.d_times_d_kernel,
                self.neutral,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, SelectSemifield):
            return False
        if self.cache_name is not None:
            return self.cache_name == other.cache_name

        return self is other

    @staticmethod
    def get_result(res: tuple[torch.Tensor, torch.Tensor]):
        return res[0]


class _CompiledSelectSemifield(NamedTuple):
    add_select: CUDADispatcher
    times: CUDADispatcher
    d_times_d_img: CUDADispatcher
    d_times_d_kernel: CUDADispatcher
    neutral: float

    @classmethod
    def compile(cls, semifield: SelectSemifield) -> _CompiledSelectSemifield:
        return _CompiledSelectSemifield(
            cuda.jit(semifield.add_select, device=True, inline="always", cache=True),
            cuda.jit(semifield.times, device=True, inline="always", cache=True),
            cuda.jit(semifield.d_times_d_img, device=True, inline="always", cache=True),
            cuda.jit(
                semifield.d_times_d_kernel, device=True, inline="always", cache=True
            ),
            semifield.neutral,
        )


class _ProvType(NamedTuple):
    typename: str
    maxval: int

    @classmethod
    def smallest_required(cls, meta: ConvMeta):
        largest = max(meta.krn_xs, meta.krn_ys, meta.krn_cs)
        if largest < np.iinfo(np.uint8).max:
            return cls("uint8", np.iinfo(np.uint8).max)

        assert largest < np.iinfo(np.uint16).max, "That's not going to fit in memory"
        return cls("uint16", np.iinfo(np.uint16).max)

    @property
    def torch_type(self):
        if self.typename == "uint8":
            return torch.uint8
        if self.typename == "uint16":
            return torch.uint16

        raise ValueError


def _compile_forwards(  # noqa: C901
    semifield: _CompiledSelectSemifield,
    meta: ConvMeta,
    prov_t: _ProvType,
    thread_block_size: int = 256,
    debug: bool = False,
    cache_name: str = "",
    to_extension: bool = True,
):
    # noinspection DuplicatedCode
    @ptex.jit(
        n_threads="out_img.numel()",
        to_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
        cache_id=f"select_{cache_name}_{meta.cache_id()}",
    )
    def forwards(
        img: ptex.In("f32", (None, meta.img_cs, meta.img_ys, meta.img_xs)),
        kernel: ptex.In("f32", (meta.krn_os, meta.krn_cs, meta.krn_ys, meta.krn_xs)),
        out_img: ptex.Out("f32", ("img", meta.out_cs, meta.out_ys, meta.out_xs)),
        out_prov: ptex.Out(
            prov_t.torch_type,
            (
                "img.shape[0]",
                meta.out_cs,
                meta.out_ys,
                meta.out_xs,
                3 if meta.krn_cs > 1 else 2,
            ),
        ),
    ):
        rem, o_x = divmod(cuda.grid(1), meta.out_xs)
        rem, o_y = divmod(rem, meta.out_ys)
        b, o_c = divmod(rem, meta.out_cs)
        if b >= img.shape[0]:
            return

        i_top_y = o_y * meta.str_y - meta.pad_y_beg
        i_left_x = o_x * meta.str_x - meta.pad_x_beg

        prov_x = prov_y = prov_group_idx = prov_t.maxval
        selected_val = semifield.neutral

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
                    if semifield.add_select(selected_val, val):
                        selected_val = val
                        prov_y, prov_x = k_y, k_x
                        if meta.krn_cs > 1:
                            prov_group_idx = group_idx

        out_img[b, o_c, o_y, o_x] = selected_val

        out_prov[b, o_c, o_y, o_x, 0] = prov_y
        out_prov[b, o_c, o_y, o_x, 1] = prov_x
        if meta.krn_cs > 1:
            # out_prov is only size 3 if we require an index within the group
            out_prov[b, o_c, o_y, o_x, 2] = prov_group_idx

    return forwards


def _compile_backwards(
    semifield: _CompiledSelectSemifield,
    meta: ConvMeta,
    prov_t: _ProvType,
    thread_block_size: int = 256,
    debug: bool = False,
    cache_name: str = "",
    to_extension: bool = True,
):
    # noinspection PyArgumentList
    @ptex.jit(
        n_threads="gradient.numel()",
        to_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
        cache_id=f"select_{cache_name}_{meta.cache_id()}",
    )
    def backwards(
        img: ptex.In("f32", (None, meta.img_cs, meta.img_ys, meta.img_xs)),
        kernel: ptex.In("f32", (meta.krn_os, meta.krn_cs, meta.krn_ys, meta.krn_xs)),
        gradient: ptex.In(
            "f32",
            ("img.shape[0]", meta.out_cs, meta.out_ys, meta.out_xs),
        ),
        prov: ptex.In(
            prov_t.torch_type,
            (
                "img.shape[0]",
                meta.out_cs,
                meta.out_ys,
                meta.out_xs,
                3 if meta.krn_cs > 1 else 2,
            ),
        ),
        out_img_grad: ptex.Out("f32", "img", init=0),
        out_kernel_grad: ptex.Out("f32", "kernel", init=0),
    ):
        rem, o_x = divmod(cuda.grid(1), meta.out_xs)
        rem, o_y = divmod(rem, meta.out_ys)
        b, o_c = divmod(rem, meta.out_cs)
        if b >= img.shape[0]:
            return

        group_number = o_c // meta.grp_o
        k_o = o_c if not meta.group_broadcasting else o_c % meta.grp_o

        grad_val = gradient[b, o_c, o_y, o_x]
        k_prov_y = prov[b, o_c, o_y, o_x, 0]
        k_prov_x = prov[b, o_c, o_y, o_x, 1]
        # Index within our group, for which of the channels we ended up picking
        # If krn_cs == 1, we can only pick the singular channel we have: always 0
        prov_group_idx = prov[b, o_c, o_y, o_x, 2] if meta.krn_cs > 1 else 0

        if k_prov_y == prov_t.maxval:
            # We kept the original neutral element,
            # so our gradient can't be related to the image
            return

        if meta.mirror_kernel:
            x_steps = meta.krn_xs - 1 - k_prov_x
            y_steps = meta.krn_ys - 1 - k_prov_y
        else:
            x_steps = k_prov_x
            y_steps = k_prov_y

        i_top_y = o_y * meta.str_y - meta.pad_y_beg
        i_left_x = o_x * meta.str_x - meta.pad_x_beg
        i_prov_c = group_number * meta.krn_cs + prov_group_idx
        i_prov_y = i_top_y + meta.dil_y * y_steps
        i_prov_x = i_left_x + meta.dil_x * x_steps

        kernel_val = kernel[k_o, prov_group_idx, k_prov_y, k_prov_x]
        img_val = img[b, i_prov_c, i_prov_y, i_prov_x]

        d_img = semifield.d_times_d_img(img_val, kernel_val) * grad_val
        d_kernel = semifield.d_times_d_kernel(img_val, kernel_val) * grad_val

        cuda.atomic.add(out_img_grad, (b, i_prov_c, i_prov_y, i_prov_x), d_img)
        cuda.atomic.add(
            out_kernel_grad, (k_o, prov_group_idx, k_prov_y, k_prov_x), d_kernel
        )

    def backwards_setup(ctx, inputs, output):
        img, kernel = inputs
        _out_img, prov = output
        ctx.img = img
        ctx.kernel = kernel
        ctx.prov = prov

    def backwards_entry(ctx, grad_output, _grad_prov):
        # return backwards_wrapper(ctx.img, ctx.kernel, grad_output, ctx.prov)
        g_img, g_kern = backwards(ctx.img, ctx.kernel, grad_output, ctx.prov)
        return g_img, g_kern

    return backwards_entry, backwards_setup
