from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
from torch import nn


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
    img_cs: int  # Image channels
    img_ys: int  # Image y-size
    img_xs: int  # Image x-size
    krn_os: int  # Kernel output channels
    krn_cs: int  # Kernel input channels (== grp_i)
    krn_ys: int  # Kernel y-size
    krn_xs: int  # Kernel x-size
    out_cs: int  # Output image channels. Equal to krn_os, except when group broacasting
    out_ys: int  # Output image y-size
    out_xs: int  # Output image x-size
    str_y: int  # Stride in y-direction
    str_x: int  # Stride in x-direction
    pad_y_beg: int  # Padding at the start of y-axis
    pad_y_end: int  # Padding at the end of y-axis
    pad_x_beg: int  # Padding at the start of x-axis
    pad_x_end: int  # Padding at the end of x-axis
    dil_y: int  # Dilation in y-direction
    dil_x: int  # Dilation in x-direction
    groups: int  # Number of convolutional groups
    grp_i: int  # Size of a convolutional group in input channels (== krn_cs)
    grp_o: int  # Size of a convolutional group in kernel output channels
    group_broadcasting: bool  # Whether kernels should be broadcast along groups
    mirror_kernel: bool  # When true, the kernel is mirrored as in a convolution

    @classmethod
    def infer(
        cls,
        img_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ) -> ConvMeta:
        str_y, str_x = _as_tup(stride)
        pad_y, pad_x = _as_tup(padding)
        pad_y_beg, pad_y_end = _as_tup(pad_y)
        pad_x_beg, pad_x_end = _as_tup(pad_x)
        dil_y, dil_x = _as_tup(dilation)

        # === Check params
        assert str_y > 0, f"{str_y=} must be positive"
        assert str_x > 0, f"{str_x=} must be positive"
        assert dil_x > 0, f"{dil_x=} must be positive"
        assert dil_y > 0, f"{dil_y=} must be positive"
        assert groups > 0, f"{groups=} must be positive"
        # Negative padding is strange, but not a logic error.

        # === Check imgs
        assert len(img_shape) == 4, f"{img_shape=} needs to be BCHW"
        assert all(s > 0 for s in img_shape), f"Invalid {img_shape=}"
        img_bs, img_cs, img_ys, img_xs = img_shape
        assert img_cs % groups == 0, f"{img_cs=} not a multiple of {groups=}"
        grp_i = img_cs // groups
        # === Check kernels
        assert len(kernel_shape) == 4, f"{kernel_shape=} needs to be OIHW"
        assert all(s > 0 for s in kernel_shape), f"Invalid {kernel_shape=}"
        krn_os, krn_cs, krn_ys, krn_xs = kernel_shape
        assert krn_cs == grp_i, f"Groups: {krn_cs=} != {grp_i=}"
        if not group_broadcasting:
            # If we *are* group-broadcasting, then we effectively multiply
            # krn_os by params.groups
            assert krn_os % groups == 0, f"{krn_os=} not a multiple of {groups=}"
            grp_o = krn_os // groups
        else:
            grp_o = krn_os

        out_xs = _output_size(img_xs, krn_xs, str_x, pad_x_beg, pad_x_end, dil_x)
        out_ys = _output_size(img_ys, krn_ys, str_y, pad_y_beg, pad_y_end, dil_y)

        out_cs = krn_os if not group_broadcasting else krn_os * groups
        assert out_xs > 0, f"Output image collapsed in x-direction: {out_xs=}"
        assert out_ys > 0, f"Output image collapsed in y-direction: {out_ys=}"
        shape = cls(
            img_cs=int(img_cs),
            img_ys=int(img_ys),
            img_xs=int(img_xs),
            krn_os=int(krn_os),
            krn_cs=int(krn_cs),
            krn_ys=int(krn_ys),
            krn_xs=int(krn_xs),
            out_cs=int(out_cs),
            out_ys=int(out_ys),
            out_xs=int(out_xs),
            str_x=int(str_x),
            str_y=int(str_y),
            pad_y_beg=int(pad_y_beg),
            pad_y_end=int(pad_y_end),
            pad_x_beg=int(pad_x_beg),
            pad_x_end=int(pad_x_end),
            dil_y=int(dil_y),
            dil_x=int(dil_x),
            groups=int(groups),
            grp_i=int(grp_i),
            grp_o=int(grp_o),
            group_broadcasting=bool(group_broadcasting),
            mirror_kernel=bool(kind == "conv"),
        )
        return shape

    def check_matches(
        self,
        img_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ):
        str_y, str_x = _as_tup(stride)
        pad_y, pad_x = _as_tup(padding)
        pad_y_beg, pad_y_end = _as_tup(pad_y)
        pad_x_beg, pad_x_end = _as_tup(pad_x)
        dil_y, dil_x = _as_tup(dilation)
        return (
            img_shape[1] == self.img_cs
            and img_shape[2] == self.img_ys
            and img_shape[3] == self.img_xs
            and kernel_shape[0] == self.krn_os
            and kernel_shape[1] == self.krn_cs
            and kernel_shape[2] == self.krn_ys
            and kernel_shape[3] == self.krn_xs
            and str_x == self.str_x
            and str_y == self.str_y
            and pad_y_beg == self.pad_y_beg
            and pad_y_end == self.pad_y_end
            and pad_x_beg == self.pad_x_beg
            and pad_x_end == self.pad_x_end
            and dil_y == self.dil_y
            and dil_x == self.dil_x
            and groups == self.groups
            and group_broadcasting == self.group_broadcasting
            and (kind == "conv") == self.mirror_kernel
        )

    def cache_id(self) -> str:
        return (
            f"meta"
            f"_{self.img_cs}_{self.img_ys}_{self.img_xs}"
            f"_{self.krn_os}_{self.krn_cs}_{self.krn_ys}_{self.krn_xs}"
            f"_{self.out_cs}_{self.out_ys}_{self.out_xs}"
            f"_{self.str_x}_{self.str_y}"
            f"_{self.pad_y_beg}_{self.pad_y_end}_{self.pad_x_beg}_{self.pad_x_end}"
            f"_{self.dil_x}_{self.dil_y}"
            f"_{self.groups}_{self.grp_i}_{self.grp_o}"
            f"_{int(self.group_broadcasting)}_{int(self.mirror_kernel)}"
        )


def _as_tup(v: int | tuple[int] | tuple[int, int]):
    if isinstance(v, int):
        return v, v
    if len(v) == 1:
        return v[0], v[0]
    if len(v) == 2:
        return v

    raise ValueError(f"Invalid 2-tuple-like object {v=}")


def _output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding_begin: int,
    padding_end: int,
    dilation: int,
):
    return math.floor(
        (input_size + padding_begin + padding_end - dilation * (kernel_size - 1) - 1)
        / stride
        + 1
    )
