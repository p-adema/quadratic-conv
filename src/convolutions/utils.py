from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
import torch
from torch import nn


class GenericConv2D(nn.Module):
    def __init__(
        self,
        kernel: nn.Module,
        conv: nn.Module,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel = kernel
        self.conv = conv

    def forward(self, x):
        return self.conv(
            x,
            self.kernel(),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )


class CoerceImage4D(nn.Module):
    def __init__(self, img_channels: int):
        super().__init__()
        self.img_channels = img_channels

    def forward(self, x: torch.Tensor):
        x_dims = len(x.shape)
        if x_dims > 4:
            raise ValueError(f"Probably invalid image dims {x.shape=}")
        if x_dims == 4:
            assert x.shape[1] == self.img_channels, f"Invalid channels {x.shape=}"
        elif x_dims == 3:
            if x.shape[0] == self.img_channels:
                # Probably a single image, unbatched
                x = x.unsqueeze(0)
            else:
                # Must be a single-channel image, batched
                assert self.img_channels == 1, f"Strange image {x.shape=}"
                x = x.unsqueeze(1)
        else:
            assert x_dims == 2, "not an image?"
            # Unbatched, single channel image
            assert self.img_channels == 1, f"Missing channels {x.shape=}"
            x = x.unsqueeze(0).unsqueeze(1)

        return x


class LinearConv2D(nn.Module):
    """A convolution in the linear field"""

    forward = staticmethod(nn.functional.conv2d)


class ConvMeta(NamedTuple):
    img_bs: int  # Batch size
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
    stride: int  # Stride in both x and y
    padding: int  # Padding in both x and y
    dilation: int  # Dilation in both x and y
    groups: int  # Number of convolutional groups
    group_broadcasting: int  # Whether kernels should be broadcast along groups
    mirror_kernel: bool  # When true, the kernel is mirrored as in a convolution

    @classmethod
    def infer(
        cls,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        *,
        kind: Literal["conv", "corr"],
    ) -> ConvMeta:
        # === Check params
        assert stride > 0, f"Cannot have zero {stride=}"
        assert dilation > 0, f"Cannot have zero {dilation=}"
        assert groups > 0, f"Cannot have zero {groups=}"
        # === Check imgs
        assert imgs.dtype == torch.float32, f"{imgs.dtype=}"
        assert kernel.dtype == torch.float32, f"{kernel.dtype=}"
        assert len(imgs.shape) == 4, f"{imgs.shape=} needs to be BCHW"
        img_bs, img_cs, img_ys, img_xs = imgs.shape
        assert img_cs % groups == 0, f"{img_cs=} not a multiple of {groups=}"
        img_group_size = img_cs // groups
        # === Check kernels
        assert len(kernel.shape) == 4, f"{kernel.shape=} needs to be OIHW"
        krn_os, krn_cs, krn_ys, krn_xs = kernel.shape
        assert krn_cs == img_group_size, f"Groups: {krn_cs=} != {img_group_size}"
        if not group_broadcasting:
            # If we *are* group-broadcasting, then we effectively multiply
            # krn_os by params.groups
            assert krn_os % groups == 0, f"{krn_os=} not a multiple of {groups=}"
            krn_o_group_size = krn_os // groups
        else:
            krn_o_group_size = krn_os

        out_xs = _output_size(img_xs, krn_xs, stride, padding, dilation)
        out_ys = _output_size(img_ys, krn_ys, stride, padding, dilation)
        assert out_xs > 0, f"Output image collapsed in x-direction: {out_xs=}"
        assert out_ys > 0, f"Output image collapsed in y-direction: {out_ys=}"
        out_cs = krn_os if not group_broadcasting else krn_os * groups
        shape = cls(
            img_bs=img_bs,
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
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            group_broadcasting=group_broadcasting,
            mirror_kernel=kind == "conv",
        )
        return shape

    def assert_matches(
        self,
        img: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
    ):
        assert img.shape[1] == self.img_cs, f"Wrong image channels: {img.shape=}"
        assert img.shape[2] == self.img_ys, f"Wrong image ys: {img.shape=}"
        assert img.shape[3] == self.img_xs, f"Wrong image xs: {img.shape=}"
        assert kernel.shape[0] == self.krn_os, f"Wrong kernel outs: {kernel.shape=}"
        assert kernel.shape[1] == self.krn_cs, f"Wrong kernel channels: {kernel.shape=}"
        assert kernel.shape[2] == self.krn_ys, f"Wrong kernel ys: {kernel.shape=}"
        assert kernel.shape[3] == self.krn_xs, f"Wrong kernel xs: {kernel.shape=}"
        assert stride == self.stride, f"Cannot change {stride=}"
        assert padding == self.padding, f"Cannot change {padding=}"
        assert dilation == self.dilation, f"Cannot change {dilation=}"
        assert groups == self.groups, f"Cannot change {groups=}"
        assert group_broadcasting == self.group_broadcasting, (
            f"Cannot change {group_broadcasting=}"
        )

    def check_matches(
        self,
        img: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
    ):
        return (
            img.shape[1] == self.img_cs
            and img.shape[2] == self.img_ys
            and img.shape[3] == self.img_xs
            and kernel.shape[0] == self.krn_os
            and kernel.shape[1] == self.krn_cs
            and kernel.shape[2] == self.krn_ys
            and kernel.shape[3] == self.krn_xs
            and stride == self.stride
            and padding == self.padding
            and dilation == self.dilation
            and groups == self.groups
            and group_broadcasting == self.group_broadcasting
        )


def _output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    return math.floor(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
