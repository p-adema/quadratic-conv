from __future__ import annotations

import math
import uuid
import warnings
from collections.abc import Callable
from typing import Literal, NamedTuple

import numba
import numpy as np
import torch
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin

from .utils import ConvMeta

warnings.simplefilter("ignore", numba.NumbaPerformanceWarning, 536)


class SelectSemifield(NamedTuple):
    add_select: Callable[[float, float], bool]  # Return True if we should pick right
    times: Callable[[float, float], float]  # (img_val, krn_val) -> multiplied_val
    d_times_d_img: Callable[[float, float], float]
    d_times_d_kernel: Callable[[float, float], float]
    neutral: float
    kind: Literal["conv", "corr"]  # Whether the kernel should be mirrored (conv) or not

    @classmethod
    def tropical_max(cls) -> SelectSemifield:
        return cls(
            add_select=lambda left, right: left < right,
            times=lambda img_val, kernel_val: img_val + kernel_val,
            d_times_d_img=lambda _i, _k: 1.0,
            d_times_d_kernel=lambda _i, _k: 1.0,
            neutral=-float("inf"),
            kind="conv",
        )

    @classmethod
    def tropical_min(cls) -> SelectSemifield:
        return cls(
            add_select=lambda left, right: left > right,
            times=lambda img_val, kernel_val: img_val - kernel_val,
            d_times_d_img=lambda _i, _k: 1.0,
            d_times_d_kernel=lambda _i, _k: -1.0,
            neutral=float("inf"),
            kind="corr",
        )

    # The torch compiler doesn't understand the Numba compiler
    @torch.compiler.disable
    def compile(
        self,
        example_imgs: torch.Tensor,
        example_kernels: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        thread_block_size: int = 256,
        debug: bool = False,
    ) -> tuple[
        ConvMeta,
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ]:
        meta = ConvMeta.infer(
            example_imgs,
            example_kernels,
            stride,
            padding,
            dilation,
            groups,
            group_broadcasting,
            kind=self.kind,
        )
        prov_t = _ProvType.smallest_required(meta)
        op_id = uuid.uuid4().hex
        cmp_semi = _CompiledSelectSemifield.compile(self)

        forwards = _compile_forwards(
            semifield=cmp_semi,
            meta=meta,
            prov_t=prov_t,
            op_id=op_id,
            group_broadcasting=group_broadcasting,
            thread_block_size=thread_block_size,
            debug=debug,
        )
        backwards, backwards_setup = _compile_backwards(
            semifield=cmp_semi,
            meta=meta,
            prov_t=prov_t,
            op_id=op_id,
            group_broadcasting=group_broadcasting,
            thread_block_size=thread_block_size,
            debug=debug,
        )
        forwards.register_autograd(backwards, setup_context=backwards_setup)

        if debug:
            _debug_tests(forwards, example_imgs, example_kernels, meta)
        # Nothing should capture these, but delete anyway for safety's sake
        del example_imgs, example_kernels

        return meta, forwards

    def dynamic(self, max_compilations: int = 3) -> SelectConv:
        """
        Use a dynamic module that will recompile if it sees a new ConvMeta, up to
        `max_compilations` times.
        Cannot be fully traced
        """
        return SelectConv(self, max_compilations)

    def lazy_fixed(self) -> SelectConvFixedLazy:
        """
        Use a lazy module that will compile exactly once, then ignore all auxilliary
        arguments (undefined behaviour if future inputs differ in any way other than
        batch size).
        Can be fully traced
        """
        return SelectConvFixedLazy(self)


class SelectConv(nn.Module):
    def __init__(self, semifield: SelectSemifield, max_compilations: int = 3):
        super().__init__()
        self.semifield = semifield
        self.compilations: list[tuple[ConvMeta, Callable]] = []
        self.max_compilations = max_compilations

    def forward(
        self,
        img: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        thread_block_size: int = 256,
        debug: bool = False,
    ):
        for i, (meta, maybe_fwd) in enumerate(self.compilations):
            if meta.check_matches(
                img, kernel, stride, padding, dilation, groups, group_broadcasting
            ):
                meta_idx = i
                fwd = maybe_fwd
                break
        else:
            if len(self.compilations) >= self.max_compilations:
                raise ValueError(
                    f"Need to recompile, but at {self.max_compilations}\n"
                    f"{self.compilations=}"
                )

            new_meta, fwd = self.semifield.compile(
                img,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                group_broadcasting=group_broadcasting,
                thread_block_size=thread_block_size,
                debug=debug,
            )
            meta_idx = len(self.compilations)
            self.compilations.append((new_meta, fwd))

        if meta_idx:
            # Always place current at the front
            self.compilations[0], self.compilations[meta_idx] = (
                self.compilations[meta_idx],
                self.compilations[0],
            )
        res, _prov = fwd(img, kernel)
        return res


class SelectConvFixed(nn.Module):
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
    meta: ConvMeta | None
    semifield: SelectSemifield

    def forward(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        *_args,
        debug: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        if debug:
            if not self.meta.check_matches(imgs, kernel, *_args, **_kwargs):
                raise ValueError("Failed to match arguments!")
            if self.op is None:
                raise ValueError("Operator not initialised!")

        res, _prov = self.op(imgs, kernel)
        return res


class SelectConvFixedLazy(LazyModuleMixin, SelectConvFixed):
    cls_to_become = SelectConvFixed

    def __init__(self, semifield: SelectSemifield):
        super().__init__()
        self.semifield = semifield
        self.op = None
        self.meta = None
        self.done = False

    def initialize_parameters(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        thread_block_size: int = 256,
        debug: bool = False,
    ):
        assert not self.done
        self.meta, self.op = self.semifield.compile(
            imgs,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            group_broadcasting,
            thread_block_size,
            debug,
        )
        self.done = True

    def has_uninitialized_params(self):
        return self.op is None


class _CompiledSelectSemifield(NamedTuple):
    add_select: CUDADispatcher
    times: CUDADispatcher
    d_times_d_img: CUDADispatcher
    d_times_d_kernel: CUDADispatcher
    neutral: float

    @classmethod
    def compile(cls, semifield: SelectSemifield) -> _CompiledSelectSemifield:
        return _CompiledSelectSemifield(
            cuda.jit(semifield.add_select, device=True, inline="always"),
            cuda.jit(semifield.times, device=True, inline="always"),
            cuda.jit(semifield.d_times_d_img, device=True, inline="always"),
            cuda.jit(semifield.d_times_d_kernel, device=True, inline="always"),
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
    op_id: str,
    group_broadcasting: bool = False,
    thread_block_size: int = 256,
    debug: bool = False,
):
    # === Forwards CUDA-side ===
    @cuda.jit(
        "void(float32[:, :, :, :],"  # img: [Batch, Channel, Img-Y, Img-X]
        " float32[:, :, :, :],"  # kernel: [Out-chan, Group-chan, Kernel-Y, Kernel-X]
        " float32[:, :, :, :],"  # out_img: [Batch, Out-chan, Out-Y, Out-X]
        # out_prov: [B, O-chan, O-Y, O-X, 2 / 3 (y, x[, gi])]
        f" {prov_t.typename}[:, :, :, :, :])",
        debug=debug,
        opt=not debug,
    )
    def forwards(img, kernel, out_img, out_prov):
        rem, o_x = divmod(cuda.grid(1), meta.out_xs)
        rem, o_y = divmod(rem, meta.out_ys)
        b, o_c = divmod(rem, meta.out_cs)
        if b >= img.shape[0]:
            return

        i_top_y = o_y * meta.stride - meta.padding
        i_left_x = o_x * meta.stride - meta.padding

        prov_x = prov_y = prov_group_idx = prov_t.maxval
        selected_val = semifield.neutral

        group_number = o_c // meta.krn_o_group_size
        # If we're not broadcasting, then we have a separate kernel
        # for every output channel. If we are broadcasting, we instead loop
        # around the kernels every k_os (which == krn_group_size)
        k_o = o_c if not group_broadcasting else o_c % meta.krn_o_group_size

        # For a pooling, we have only one input channel, so group_idx is always 0
        for group_idx in range(meta.krn_cs):
            for y_step, i_y in enumerate(
                range(i_top_y, i_top_y + meta.krn_ys * meta.dilation, meta.dilation)
            ):
                for x_step, i_x in enumerate(
                    range(
                        i_left_x,
                        i_left_x + meta.krn_xs * meta.dilation,
                        meta.dilation,
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

    fowards_bindings = _torch_bindings_forwards(
        forwards, meta, prov_t, op_id, thread_block_size, debug
    )

    return fowards_bindings


def _compile_backwards(
    semifield: _CompiledSelectSemifield,
    meta: ConvMeta,
    prov_t: _ProvType,
    op_id: str,
    group_broadcasting: bool = False,
    thread_block_size: int = 256,
    debug: bool = False,
):
    # === Backwards CUDA-side ===
    # noinspection PyArgumentList
    @cuda.jit(
        "void(float32[:, :, :, :],"  # img: [Batch, Channel, Img-Y, Img-X]
        " float32[:, :, :, :],"  # kernel: [Out-chan, Group-chan, Kernel_Y, Kernel_X]
        " float32[:, :, :, :],"  # gradient: [Batch, Out-chan, Out-Y, Out-X]
        # prov: [B, O-chan, O-Y, O-X, 2 / 3 (y, x[, gi])]
        f" {prov_t.typename}[:, :, :, :, :],"
        " float32[:, :, :, :],"  # out_img_grad: [Batch, Channel, Img-Y, Img-X]
        " float32[:, :, :, :])",  # out_kernel_grad: [Out-chan, Group-chan, K-Y, K-X]
        debug=debug,
        opt=not debug,
    )
    def backwards(img, kernel, gradient, prov, out_img_grad, out_kernel_grad):
        rem, o_x = divmod(cuda.grid(1), meta.out_xs)
        rem, o_y = divmod(rem, meta.out_ys)
        b, o_c = divmod(rem, meta.out_cs)
        if b >= img.shape[0]:
            return

        group_number = o_c // meta.krn_o_group_size
        k_o = o_c if not group_broadcasting else o_c % meta.krn_o_group_size

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
        i_top_y = o_y * meta.stride - meta.padding
        i_left_x = o_x * meta.stride - meta.padding
        i_prov_c = group_number * meta.krn_cs + prov_group_idx
        i_prov_y = i_top_y + meta.dilation * y_steps
        i_prov_x = i_left_x + meta.dilation * x_steps
        # if (
        #     i_prov_x < 0
        #     or i_prov_x >= shapes.img_xs
        #     or i_prov_y < 0
        #     or i_prov_y >= shapes.img_ys
        # ):
        #     # Out-of-bounds providence?
        #     return

        kernel_val = kernel[k_o, prov_group_idx, k_prov_y, k_prov_x]
        img_val = img[b, i_prov_c, i_prov_y, i_prov_x]

        d_img = semifield.d_times_d_img(img_val, kernel_val) * grad_val
        d_kernel = semifield.d_times_d_kernel(img_val, kernel_val) * grad_val

        cuda.atomic.add(out_img_grad, (b, i_prov_c, i_prov_y, i_prov_x), d_img)
        cuda.atomic.add(
            out_kernel_grad, (k_o, prov_group_idx, k_prov_y, k_prov_x), d_kernel
        )

    # === Backwards torch-side ===
    @torch.library.custom_op(
        f"semifields::select_bwd_{op_id}", mutates_args={}, device_types="cuda"
    )
    def backwards_wrapper(
        img: torch.Tensor,
        kernel: torch.Tensor,
        gradient: torch.Tensor,
        prov: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img, kernel = img.detach(), kernel.detach()
        out_img_grad = torch.zeros_like(img)
        out_kernel_grad = torch.zeros_like(kernel)
        n_blocks = math.ceil(gradient.nelement() / thread_block_size)
        backwards[n_blocks, thread_block_size](
            img, kernel, gradient, prov, out_img_grad, out_kernel_grad
        )
        return out_img_grad, out_kernel_grad

    @backwards_wrapper.register_fake
    def _(img, kernel, _gradient, _prov):
        return torch.empty_like(img), torch.empty_like(kernel)

    def backwards_setup(ctx, inputs, output):
        img, kernel = inputs
        _out_img, prov = output
        ctx.img = img
        ctx.kernel = kernel
        ctx.prov = prov

    def backwards_entry(ctx, grad_output, _grad_prov):
        return backwards_wrapper(ctx.img, ctx.kernel, grad_output, ctx.prov)

    return backwards_entry, backwards_setup


def _torch_bindings_forwards(
    forwards: Callable,
    meta: ConvMeta,
    prov_t: _ProvType,
    op_id: str,
    thread_block_size: int = 256,
    debug: bool = False,
):
    @torch.library.custom_op(
        f"semifields::select_fwd_{op_id}", mutates_args={}, device_types="cuda"
    )
    def forwards_wrapper(
        img: torch.Tensor, kernel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img, kernel = img.detach(), kernel.detach()
        assert img.dtype == torch.float32, f"Wrong {img.dtype=}"
        assert kernel.dtype == torch.float32, f"Wrong {kernel.dtype=}"
        if debug:
            print("Warning: running CUDA kernel in debug mode")
        batch_size = img.shape[0]

        out_img_shape = (
            batch_size,
            meta.out_cs,
            meta.out_ys,
            meta.out_xs,
        )
        out_img = img.new_empty(out_img_shape)
        out_prov = img.new_empty(
            (*out_img_shape, 3 if meta.krn_cs > 1 else 2), dtype=prov_t.torch_type()
        )
        n_blocks = math.ceil(out_img.nelement() / thread_block_size)
        forwards[n_blocks, thread_block_size](img, kernel, out_img, out_prov)
        return out_img, out_prov

    @forwards_wrapper.register_fake
    def _(img, kernel):
        batch_size = img.shape[0]
        out_img_shape = (
            batch_size,
            meta.out_cs,
            meta.out_ys,
            meta.out_xs,
        )
        return (
            img.new_empty(out_img_shape),
            kernel.new_empty(
                (*out_img_shape, 3 if meta.krn_cs > 1 else 2), dtype=prov_t.torch_type()
            ),
        )

    return forwards_wrapper


def _debug_tests(
    forwards_wrapper: Callable,
    example_imgs: torch.Tensor,
    example_kernels: torch.Tensor,
    meta: ConvMeta,
):
    print(f"Inferred {meta=}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with torch.autograd.detect_anomaly():
            # noinspection PyTypeChecker
            torch.library.opcheck(
                forwards_wrapper,
                (example_imgs, example_kernels),
            )
