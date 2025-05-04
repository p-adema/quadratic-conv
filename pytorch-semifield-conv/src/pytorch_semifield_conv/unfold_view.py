import math
from collections.abc import Sequence
from typing import NamedTuple

import torch


def unfold_view(
    imgs: torch.Tensor,
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
):
    """
    Returns an unfolded view of imgs, shaped:
        [Batch, Channels, Kernel-Y, Kernel-X, Out-Y, Out-X]

    Does not support padding, as it returns a view.

    :param imgs: [B, C, H, W] tensor to be unfolded
    :param kernel_size: int or tuple of (Kernel-Y, Kernel-X)
    :param dilation: int or tuple of (Dilation-Y, Dilation-X)
    :param stride: int or tuple of (Stride-Y, Stride-X)
    :return: [Batch, Channels, Kernel-Y, Kernel-X, Out-Y, Out-X] tensor view

    Examples:

    >>> images = torch.empty((1024, 5, 28, 28))
    >>> unfold_view(images, (3, 3)).shape
    torch.Size([1024, 5, 3, 3, 26, 26])
    >>> unfold_view(images, (7, 6), stride=(1, 2)).shape
    torch.Size([1024, 5, 7, 6, 22, 12])
    """
    meta = _UnfoldMeta.infer(imgs.shape, kernel_size, dilation, stride)

    return imgs.as_strided(
        (imgs.shape[0], imgs.shape[1], meta.krs_y, meta.krs_x, meta.out_y, meta.out_x),
        (
            imgs.stride(0),
            imgs.stride(1),
            imgs.stride(2) * meta.dil_y,
            imgs.stride(3) * meta.dil_x,
            imgs.stride(2) * meta.str_y,
            imgs.stride(3) * meta.str_x,
        ),
    )


def unfold_copy(
    imgs: torch.Tensor,
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
):
    """For comparison: unfold-view, but then via a copy with nn.functional.unfold"""
    meta = _UnfoldMeta.infer(imgs.shape, kernel_size, dilation, stride)

    return torch.nn.functional.unfold(
        imgs, kernel_size=kernel_size, dilation=dilation, stride=stride
    ).view(imgs.shape[0], imgs.shape[1], meta.krs_y, meta.krs_x, meta.out_y, meta.out_x)


def _as_tup(v: int | tuple[int] | tuple[int, int]):
    if isinstance(v, int):
        return v, v
    if len(v) == 1:
        return v[0], v[0]
    if len(v) == 2:
        return v

    raise ValueError(f"Invalid 2-tuple-like object {v=}")


class _UnfoldMeta(NamedTuple):
    krs_y: int
    krs_x: int
    dil_y: int
    dil_x: int
    str_y: int
    str_x: int
    out_y: int
    out_x: int

    @classmethod
    def infer(
        cls,
        imgs_shape: Sequence[int],
        kernel_size: int | tuple[int, int],
        dilation: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
    ):
        if len(imgs_shape) != 4:
            raise ValueError("imgs must be in BCHW")
        krs_y, krs_x = _as_tup(kernel_size)
        dil_y, dil_x = _as_tup(dilation)
        str_y, str_x = _as_tup(stride)

        out_y = math.floor((imgs_shape[2] - dil_y * (krs_y - 1) - 1) / str_y + 1)
        out_x = math.floor((imgs_shape[3] - dil_x * (krs_x - 1) - 1) / str_x + 1)

        if out_y <= 0:
            raise ValueError("Output collapsed in y-dimension")
        if out_x <= 0:
            raise ValueError("Output collapsed in x-dimension")

        return cls(krs_y, krs_x, dil_y, dil_x, str_y, str_x, out_y, out_x)
