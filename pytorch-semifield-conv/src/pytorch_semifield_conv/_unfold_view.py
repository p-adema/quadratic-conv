import math
from typing import Any, NamedTuple

import torch


def unfold_view(
    imgs: torch.Tensor,
    kernel_size: int | tuple[int, ...],
    dilation: int | tuple[int, ...] = 1,
    stride: int | tuple[int, ...] = 1,
):
    """
    Returns an unfolded view of imgs, without copying.

    The view is shaped: [Batch, Channels, Kernel dims..., Output dims...]

    where kernel and output dimensions are in the same order as they are in the image.
    Example: a [B, C, Y, X] image will become [B, C, Ky, Kx, Oy, Ox].

    Does not support padding, as it returns a view.

    Parameters
    ----------
    imgs : Tensor of (B, C, *Spatial)
        Input to be unfolded
    kernel_size : int or tuple of int
        Size of the window, must be the same length as the number of spatial dimensions
        in imgs if provided as a tuple, otherwise repeated that many times.
    dilation : int or tuple of int
    stride : int or tuple of int

    Returns
    -------
    view : Tensor of [Batch, Channels, Kernel dims..., Output dims...]

    Examples
    -------

    >>> images_2d = torch.empty((1024, 5, 28, 28))
    >>> unfold_view(images_2d, 3).shape
    torch.Size([1024, 5, 3, 3, 26, 26])
    >>> unfold_view(images_2d, (7, 6), stride=(1, 2)).shape
    torch.Size([1024, 5, 7, 6, 22, 12])
    """
    meta = _UnfoldMeta.infer(tuple(imgs.shape), kernel_size, dilation, stride)

    return imgs.as_strided(
        (imgs.shape[0], imgs.shape[1], *meta.kernel_size, *meta.output_size),
        (
            imgs.stride(0),
            imgs.stride(1),
            *(imgs.stride(2 + n) * meta.dilation[n] for n in range(meta.ndim)),
            *(imgs.stride(2 + n) * meta.stride[n] for n in range(meta.ndim)),
        ),
    )


def unfold_copy_2d(
    imgs: torch.Tensor,
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
):
    """For comparison: unfold-view, but then via a copy with nn.functional.unfold"""
    assert len(imgs.shape) == 4, "unfold_copy_2d only supports batch+channel+2D images"
    meta = _UnfoldMeta.infer(imgs.shape, kernel_size, dilation, stride)

    return torch.nn.functional.unfold(
        imgs,
        kernel_size=meta.kernel_size,
        dilation=meta.dilation,
        stride=meta.stride,
    ).view(imgs.shape[0], imgs.shape[1], *meta.kernel_size, *meta.output_size)


def _as_tup_n(v: int | tuple[Any] | tuple[Any, ...], n: int):
    if isinstance(v, int):
        return tuple(v for _ in range(n))
    if len(v) == n:
        return v
    if len(v) == 1:
        return tuple(v[0] for _ in range(n))

    raise ValueError(
        f"Invalid {n}-int-tuple-like object {v=}\n(expected dimensionality {n})"
    )


class _UnfoldMeta(NamedTuple):
    ndim: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    dilation: tuple[int, ...]
    output_size: tuple[int, ...]

    @classmethod
    def infer(
        cls,
        imgs_shape: tuple[int, ...],
        kernel_size: int | tuple[int, int],
        dilation: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
    ):
        if len(imgs_shape) <= 2:
            raise ValueError("imgs must have Batch and Channel as leading dimensions")
        ndim = len(imgs_shape) - 2
        kernel_size = _as_tup_n(kernel_size, ndim)
        dilation = _as_tup_n(dilation, ndim)
        stride = _as_tup_n(stride, ndim)

        output_size = tuple(
            math.floor((i - d * (k - 1) - 1) / s + 1)
            for i, d, k, s in zip(
                imgs_shape[2:], dilation, kernel_size, stride, strict=True
            )
        )

        if any(o <= 0 for o in output_size):
            raise ValueError(f"Output collapsed: {output_size}")

        return cls(ndim, kernel_size, stride, dilation, output_size)
