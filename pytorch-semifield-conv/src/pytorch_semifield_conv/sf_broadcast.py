from __future__ import annotations

import typing
import warnings
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from .unfold_view import unfold_copy as unfold_copy_fn
from .unfold_view import unfold_view as unfold_view_fn
from .utils import ConvMeta


class BroadcastSemifield(typing.NamedTuple):
    # (multiplied, dims) -> `multipled` reduced with (+) along every dim in `dims`
    add_reduce: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]
    # (img, krn) -> `img` (x) `krn`
    multiply: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    # forall a, b: `zero` (x) a  (+) b  ==  b
    zero: float
    # Similar to add_reduce, but only used for channel axis (so takes one dimension)
    add_reduce_channels: Callable[[torch.Tensor, int], torch.Tensor] | None = None

    @classmethod
    def tropical_max(cls, channels_add: bool = False, spread_gradient: bool = False):
        return cls(
            add_reduce=(lambda multiplied, dim: torch.amax(multiplied, dim=dim))
            if spread_gradient
            else (
                _repeated_dim(
                    lambda multiplied, dim: torch.max(multiplied, dim=dim).values
                )
            ),
            multiply=lambda img, krn: img + krn,
            zero=-float("inf"),
            add_reduce_channels=(
                (lambda multiplied, dim: torch.sum(multiplied, dim=dim))
                if channels_add
                else None
            ),
        )

    @classmethod
    def tropical_min_negated(
        cls, channels_add: bool = False, spread_gradient: bool = False
    ):
        return cls(
            add_reduce=(lambda multiplied, dim: torch.amin(multiplied, dim=dim))
            if spread_gradient
            else (
                _repeated_dim(
                    lambda multiplied, dim: torch.min(multiplied, dim=dim).values
                )
            ),
            multiply=lambda img, krn: img - krn,
            zero=float("inf"),
            add_reduce_channels=(
                (lambda multiplied, dim: torch.sum(multiplied, dim=dim))
                if channels_add
                else None
            ),
        )

    def dynamic(
        self, unfold_copy: bool = False, warn_changing_meta: bool = True
    ) -> BroadcastConv:
        return BroadcastConv(self, unfold_copy, warn_changing_meta)


class BroadcastConv(nn.Module):
    def __init__(
        self,
        semifield: BroadcastSemifield,
        unfold_copy: bool = False,
        warn_changing_meta: bool = True,
    ):
        super().__init__()
        self.semifield = semifield
        self.last_meta: ConvMeta | None = None
        self.warn_changing_meta = warn_changing_meta
        self.unfold = unfold_copy_fn if unfold_copy else unfold_view_fn

    def forward(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ) -> torch.Tensor:
        meta = self.get_meta(
            imgs,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            group_broadcasting,
        )

        imgs_padded = torch.constant_pad_nd(
            imgs, (padding, padding, padding, padding), self.semifield.zero
        )

        # [b, groups * krn_cs, krn_ys, krn_xs, out_ys, out_xs]
        windows_flat_channels = self.unfold(
            imgs_padded,
            (meta.krn_ys, meta.krn_xs),
            dilation=dilation,
            stride=stride,
        )
        # print(windows_flat.shape)
        windows = windows_flat_channels.view(
            imgs.shape[0],
            groups,
            1,  # Broadcast along krn_o_group_size
            meta.krn_cs,
            meta.krn_ys,
            meta.krn_xs,
            meta.out_ys,
            meta.out_xs,
        )
        if kind == "conv":
            # Very bad, but this is only a reference implementation
            kernel = kernel.flip((2, 3))

        weights = kernel.view(
            1,  # Broadcast along batch dimension
            1 if group_broadcasting else groups,  # Maybe broadcast along groups
            meta.krn_o_group_size,  # Number of kernels per group
            meta.krn_cs,  # 3: Neighbourhood Channels
            meta.krn_ys,  # 4: Neighbourhood Ys
            meta.krn_xs,  # 5: Neighbourhood Xs
            1,  # Broadcast along window Y
            1,  # Broadcast along window X
        )
        multiplied = self.semifield.multiply(windows, weights)
        if self.semifield.add_reduce_channels is None:
            reduced = self.semifield.add_reduce(multiplied, (3, 4, 5))
        else:
            reduced_with_channels = self.semifield.add_reduce(multiplied, (4, 5))
            reduced = self.semifield.add_reduce_channels(reduced_with_channels, 3)

        res = reduced.view(
            imgs.shape[0],
            meta.out_cs,
            meta.out_ys,
            meta.out_xs,
        )
        return res

    def get_meta(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ) -> ConvMeta:
        if self.last_meta is not None and self.last_meta.check_matches(
            imgs, kernel, stride, padding, dilation, groups, group_broadcasting, kind
        ):
            meta = self.last_meta
        else:
            if self.last_meta is not None and self.warn_changing_meta:
                warnings.warn("Convolution parameters changed", stacklevel=5)

            meta = ConvMeta.infer(
                imgs,
                kernel,
                stride,
                padding,
                dilation,
                groups,
                group_broadcasting,
                kind,
            )
            self.last_meta = meta
        return meta

    if typing.TYPE_CHECKING:
        __call__ = forward


def _repeated_dim(single_dim_broadcast: Callable):
    def func(x: torch.Tensor, dims: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dims, int):
            dims = (dims,)

        for dim in sorted(dims, reverse=True):
            x = single_dim_broadcast(x, dim=dim)

        return x

    return func
