from __future__ import annotations

import math
import uuid
import warnings
from collections.abc import Callable
from typing import NamedTuple

import numba
import numpy as np
import torch
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

warnings.simplefilter("ignore", numba.NumbaPerformanceWarning, 536)
UINT8_MAX = np.iinfo(np.uint8).max


class SelectSemifield(NamedTuple):
    select_right: Callable[[float, float], bool]
    times: Callable[[float, float], float]
    d_times_d_img: Callable[[float, float], float]
    d_times_d_kernel: Callable[[float, float], float]
    neutral: float

    @classmethod
    def tropical_max(cls) -> SelectSemifield:
        return cls(
            select_right=lambda left, right: left < right,
            times=lambda img_val, kernel_val: img_val + kernel_val,
            d_times_d_img=lambda _i, _k: 1.0,
            d_times_d_kernel=lambda _i, _k: 1.0,
            neutral=-float("inf"),
        )

    @classmethod
    def tropical_min(cls) -> SelectSemifield:
        return cls(
            select_right=lambda left, right: left > right,
            times=lambda img_val, kernel_val: img_val - kernel_val,
            d_times_d_img=lambda _i, _k: 1.0,
            d_times_d_kernel=lambda _i, _k: -1.0,
            neutral=float("inf"),
        )

    def compile(
        self,
        example_imgs: torch.Tensor,
        example_kernels: torch.Tensor,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        thread_block_size: int = 256,
        debug: bool = False,
    ):
        params = _ConvParams(stride, padding, dilation, groups)
        shapes = _ShapeInfo.infer(
            example_imgs, example_kernels, params, group_broadcasting
        )
        op_id = uuid.uuid4().hex
        cmp_semi = _CompiledSelectSemifield.compile(self)

        forwards = _compile_forwards(
            semifield=cmp_semi,
            shapes=shapes,
            params=params,
            op_id=op_id,
            group_broadcasting=group_broadcasting,
            thread_block_size=thread_block_size,
            debug=debug,
        )
        backwards, backwards_setup = _compile_backwards(
            semifield=cmp_semi,
            shapes=shapes,
            params=params,
            op_id=op_id,
            group_broadcasting=group_broadcasting,
            thread_block_size=thread_block_size,
            debug=debug,
        )
        forwards.register_autograd(backwards, setup_context=backwards_setup)

        if debug:
            _debug_tests(forwards, example_imgs, example_kernels, shapes)
        # Nothing should capture these, but delete anyway for safety's sake
        del example_imgs, example_kernels

        conv = _entrypoint(params, forwards)

        return conv


class _ConvParams(NamedTuple):
    stride: int
    padding: int
    dilation: int
    groups: int

    def output_size(self, input_size: int, kernel_size: int):
        return math.floor(
            (input_size + 2 * self.padding - self.dilation * (kernel_size - 1) - 1)
            / self.stride
            + 1
        )


class _ShapeInfo(NamedTuple):
    img_cs: int  # Image channels
    img_ys: int  # Image y-size
    img_xs: int  # Image x-size
    krn_o_group_size: int  # Size of a convolution group in kernel output channels
    krn_os: int  # Kernel output channels
    krn_cs: int  # Kernel input channels. Therefore, also equal to image group size
    krn_ys: int  # Kernel y-size
    krn_xs: int  # Kernel x-size
    out_cs: int  # Output image channels. Equal to krn_os, except when group broacasting
    out_ys: int  # Output image y-size
    out_xs: int  # Output image x-size

    @classmethod
    def infer(
        cls,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        params: _ConvParams,
        group_broadcasting: bool,
    ) -> _ShapeInfo:
        # === Check imgs
        assert imgs.dtype == torch.float32, f"{imgs.dtype=}"
        assert kernel.dtype == torch.float32, f"{kernel.dtype=}"
        assert len(imgs.shape) == 4, f"{imgs.shape=} needs to be BCHW"
        img_cs, img_ys, img_xs = imgs.shape[1:]
        assert img_cs % params.groups == 0, (
            f"{img_cs=} not a multiple of {params.groups=}"
        )
        img_group_size = img_cs // params.groups
        # === Check kernels
        assert len(kernel.shape) == 4, f"{kernel.shape=} needs to be OIHW"
        krn_os, krn_cs, krn_ys, krn_xs = kernel.shape
        assert krn_cs == img_group_size, f"Groups: {krn_cs=} != {img_group_size}"
        if not group_broadcasting:
            # If we *are* group-broadcasting, then we effectively multiply
            # krn_os by params.groups
            assert krn_os % params.groups == 0, (
                f"{krn_os=} not a multiple of {params.groups=}"
            )
            krn_o_group_size = krn_os // params.groups
        else:
            krn_o_group_size = krn_os

        # We reserve u8::MAX for marking invalid provenances
        assert krn_ys < UINT8_MAX, f"Provenance represented as u8, but {krn_ys=}"
        assert krn_xs < UINT8_MAX, f"Provenances represented as u8, but {krn_xs=}"
        assert krn_cs < UINT8_MAX, f"Provenances represented as u8, but {krn_cs=}"
        out_xs = params.output_size(img_xs, krn_xs)
        out_ys = params.output_size(img_ys, krn_ys)
        assert out_xs > 0, f"Output image collapsed in x-direction: {out_xs=}"
        assert out_ys > 0, f"Output image collapsed in y-direction: {out_ys=}"
        out_cs = krn_os if not group_broadcasting else krn_os * params.groups
        shape = cls(
            img_cs=img_cs,
            img_ys=img_ys,
            img_xs=img_xs,
            krn_o_group_size=krn_o_group_size,
            krn_os=krn_os,
            krn_cs=krn_cs,
            krn_ys=krn_ys,
            krn_xs=krn_xs,
            out_cs=out_cs,
            out_ys=out_ys,
            out_xs=out_xs,
        )
        assert all(s > 0 for s in shape), f"Invalid value in {shape=}"
        return shape

    def assert_matches(self, img: torch.Tensor, kernel: torch.Tensor):
        assert img.shape[1] == self.img_cs, f"Wrong image channels: {img.shape=}"
        assert img.shape[2] == self.img_ys, f"Wrong image ys: {img.shape=}"
        assert img.shape[3] == self.img_xs, f"Wrong image xs: {img.shape=}"
        assert kernel.shape[0] == self.krn_os, f"Wrong kernel outs: {kernel.shape=}"
        assert kernel.shape[1] == self.krn_cs, f"Wrong kernel channels: {kernel.shape=}"
        assert kernel.shape[2] == self.krn_ys, f"Wrong kernel ys: {kernel.shape=}"
        assert kernel.shape[3] == self.krn_xs, f"Wrong kernel xs: {kernel.shape=}"


class _CompiledSelectSemifield(NamedTuple):
    select_right: CUDADispatcher
    times: CUDADispatcher
    d_times_d_img: CUDADispatcher
    d_times_d_kernel: CUDADispatcher
    neutral: float

    @classmethod
    def compile(cls, semifield: SelectSemifield) -> _CompiledSelectSemifield:
        return _CompiledSelectSemifield(
            cuda.jit(semifield.select_right, device=True, inline="always"),
            cuda.jit(semifield.times, device=True, inline="always"),
            cuda.jit(semifield.d_times_d_img, device=True, inline="always"),
            cuda.jit(semifield.d_times_d_kernel, device=True, inline="always"),
            semifield.neutral,
        )


def _entrypoint(params: _ConvParams, conv_op: Callable):
    def conv(
        img: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = params.stride,
        padding: int = params.padding,
        dilation: int = params.dilation,
        groups: int = params.groups,
    ):
        if (stride, padding, dilation, groups) != params:
            raise ValueError("Cannot change params after compilation")

        out_img, _out_prov = conv_op(img, kernel)
        return out_img

    return conv


def _compile_forwards(
    semifield: _CompiledSelectSemifield,
    shapes: _ShapeInfo,
    params: _ConvParams,
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
        " uint8[:, :, :, :, :])",  # out_prov: [B, O-chan, O-Y, O-X, 2 / 3 (y, x[, gi])]
        debug=debug,
        opt=not debug,
    )
    def forwards(img, kernel, out_img, out_prov):
        rem, o_x = divmod(cuda.grid(1), shapes.out_xs)
        rem, o_y = divmod(rem, shapes.out_ys)
        b, o_c = divmod(rem, shapes.img_cs)
        if b >= img.shape[0]:
            return

        i_top_y = o_y * params.stride - params.padding
        i_left_x = o_x * params.stride - params.padding

        prov_x = prov_y = prov_group_idx = UINT8_MAX
        selected_val = semifield.neutral

        group_number = o_c // shapes.krn_o_group_size
        # If we're not broadcasting, then we have a separate kernel
        # for every output channel. If we are broadcasting, we instead loop
        # around the kernels every k_os (which == krn_group_size)
        k_o = o_c if not group_broadcasting else o_c % shapes.krn_o_group_size

        # For a pooling, we have only one input channel, so group_idx is always 0
        for group_idx in range(shapes.krn_cs):
            for k_y, i_y in enumerate(
                range(
                    i_top_y, i_top_y + shapes.krn_ys * params.dilation, params.dilation
                )
            ):
                for k_x, i_x in enumerate(
                    range(
                        i_left_x,
                        i_left_x + shapes.krn_xs * params.dilation,
                        params.dilation,
                    )
                ):
                    if (
                        i_x < 0
                        or i_x >= shapes.img_xs
                        or i_y < 0
                        or i_y >= shapes.img_ys
                    ):
                        continue
                    i_c = group_number * shapes.krn_cs + group_idx
                    img_val = img[b, i_c, i_y, i_x]
                    kernel_val = kernel[k_o, group_idx, k_y, k_x]

                    val = semifield.times(img_val, kernel_val)
                    if semifield.select_right(selected_val, val):
                        selected_val = val
                        prov_y, prov_x = k_y, k_x
                        if shapes.krn_cs > 1:
                            prov_group_idx = group_idx

        out_img[b, o_c, o_y, o_x] = selected_val

        out_prov[b, o_c, o_y, o_x, 0] = prov_y
        out_prov[b, o_c, o_y, o_x, 1] = prov_x
        if shapes.krn_cs > 1:
            # out_prov is only size 3 if we require an index within the group
            out_prov[b, o_c, o_y, o_x, 2] = prov_group_idx

    fowards_bindings = _torch_bindings_forwards(
        forwards, shapes, op_id, thread_block_size, debug
    )

    return fowards_bindings


def _compile_backwards(
    semifield: _CompiledSelectSemifield,
    shapes: _ShapeInfo,
    params: _ConvParams,
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
        " uint8[:, :, :, :, :],"  # prov: [B, O-chan, O-Y, O-X, 2 / 3 (y, x[, gi])]
        " float32[:, :, :, :],"  # out_img_grad: [Batch, Channel, Img-Y, Img-X]
        " float32[:, :, :, :])",  # out_kernel_grad: [Out-chan, Group-chan, K-Y, K-X]
        debug=debug,
        opt=not debug,
    )
    def backwards(img, kernel, gradient, prov, out_img_grad, out_kernel_grad):
        rem, o_x = divmod(cuda.grid(1), shapes.out_xs)
        rem, o_y = divmod(rem, shapes.out_ys)
        b, o_c = divmod(rem, shapes.img_cs)
        if b >= img.shape[0]:
            return

        group_number = o_c // shapes.krn_o_group_size
        k_o = o_c if not group_broadcasting else o_c % shapes.krn_o_group_size

        grad_val = gradient[b, o_c, o_y, o_x]
        k_prov_y = prov[b, o_c, o_y, o_x, 0]
        k_prov_x = prov[b, o_c, o_y, o_x, 1]
        # Index within our group, for which of the channels we ended up picking
        # If krn_cs == 1, we can only pick the singular channel we have: always 0
        prov_group_idx = prov[b, o_c, o_y, o_x, 2] if shapes.krn_cs > 1 else 0

        if k_prov_y == UINT8_MAX:
            # We kept the original neutral element,
            # so our gradient can't be related to the image
            return

        i_top_y = o_y * params.stride - params.padding
        i_left_x = o_x * params.stride - params.padding
        i_prov_c = group_number * shapes.krn_cs + prov_group_idx
        i_prov_y = i_top_y + params.dilation * k_prov_y
        i_prov_x = i_left_x + params.dilation * k_prov_x
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
    shapes: _ShapeInfo,
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
        shapes.assert_matches(img, kernel)
        if debug:
            print("Warning: running CUDA kernel in debug mode")
        batch_size = img.shape[0]

        out_img_shape = (
            batch_size,
            shapes.out_cs,
            shapes.out_ys,
            shapes.out_xs,
        )
        out_img = img.new_empty(out_img_shape)
        out_prov = img.new_empty(
            (*out_img_shape, 3 if shapes.krn_cs > 1 else 2), dtype=torch.uint8
        )
        n_blocks = math.ceil(out_img.nelement() / thread_block_size)
        forwards[n_blocks, thread_block_size](img, kernel, out_img, out_prov)
        return out_img, out_prov

    @forwards_wrapper.register_fake
    def _(img, kernel):
        batch_size = img.shape[0]
        out_img_shape = (
            batch_size,
            shapes.out_cs,
            shapes.out_ys,
            shapes.out_xs,
        )
        return (
            img.new_empty(out_img_shape),
            kernel.new_empty(
                (*out_img_shape, 3 if shapes.krn_cs > 1 else 2), dtype=torch.uint8
            ),
        )

    return forwards_wrapper


def _debug_tests(
    forwards_wrapper: Callable,
    example_imgs: torch.Tensor,
    example_kernels: torch.Tensor,
    shapes: _ShapeInfo,
):
    print("Calculated output shape", shapes.out_cs, shapes.out_ys, shapes.out_xs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with torch.autograd.detect_anomaly():
            # noinspection PyTypeChecker
            torch.library.opcheck(forwards_wrapper, (example_imgs, example_kernels))
