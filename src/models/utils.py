from __future__ import annotations

import collections
import struct
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import torch
from pytorch_semifield_conv import (
    BroadcastSemifield,
    GenericClosing2D,
    GenericConv2D,
    QuadraticKernelIso2D,
    QuadraticKernelSpectral2D,
    SelectSemifield,
    TorchMaxPool2D,
)
from torch import nn

from ..load_data import Dataset

if TYPE_CHECKING:
    from .trainer import Trainer


class CheckNan(nn.Module):
    def __init__(self, i: int):
        super().__init__()
        self.nth_module = i

    @torch.compiler.disable
    def forward(self, *args):
        for i, arr in enumerate(args, start=1):
            if torch.isnan(arr).any().item():
                raise ValueError(f"{self.nth_module}: Item {i}/{len(args)} was NaN")

            print(
                f"{self.nth_module}: Item {i}/{len(args)} was OK,"
                f" shape={tuple(arr.shape)} min={arr.min()} max={arr.max()}"
            )
        return args if len(args) > 1 else args[0]


def split_seed(size: int, seed: int, groups: int = 1) -> tuple[tuple[int, ...], ...]:
    return tuple(
        struct.iter_unpack(
            f"@{size}i", np.random.default_rng(seed).bytes(4 * size * groups)
        )
    )


def reports_to_df(reports: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(pl.Series("reports", reports))
    cat_fields = df["reports"].struct.fields[:-3]
    cat_f1s = pl.all().struct.field(cat_fields).struct.field("f1-score")
    return df.select(
        acc=pl.all().struct.field("accuracy"),
        min_f1=pl.min_horizontal(cat_f1s),
        max_f1=pl.max_horizontal(cat_f1s),
        macro_f1=pl.all().struct.field("macro avg").struct.field("f1-score"),
    )


class HistoryCallback:
    def __init__(self, data: Dataset | tuple[torch.Tensor, torch.Tensor]):
        self.losses = []
        self.reports = []
        self.data = data

    def __call__(self, model: Trainer, train_loss: float):
        self.losses.append(train_loss)
        self.reports.append(model.evaluate(self.data))

    def reset(self):
        self.losses.clear()
        self.reports.clear()

    def result(self) -> tuple[pl.DataFrame, list[float]]:
        return reports_to_df(self.reports), self.losses


POOLING_JIT_DEFAULT = True


def make_pooling_function(
    kind: Literal["standard", "iso", "aniso"],
    kernel_size: int,
    stride: int = 2,
    padding: (
        int
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | Literal["valid", "same"]
    ) = "same",
    groups: int | None = None,
    group_size: int | None = None,  # S_i
    group_broadcasting: bool = False,
    jit: bool = None,
    channel_add: bool = False,
    spread_gradient: bool = False,
    is_closing: bool = False,
) -> Callable[[int, dict], torch.Module]:
    jit = _get_jit_status(jit, channel_add, spread_gradient)

    def pooling_fn(channels: int, init: dict[str, float | int]) -> nn.Module:
        grp_size, grps = _calculate_groups(channels, groups, group_size)

        if kind == "standard":
            assert grp_size == 1, "Standard max pool doesn't support group sizes > 1"
            assert not group_broadcasting, "Standard max pool doesn't have parameters"
            return TorchMaxPool2D(kernel_size, stride, padding)

        out_channels = channels if not group_broadcasting else grp_size

        if kind == "iso":
            kernel = QuadraticKernelIso2D(
                grp_size, out_channels, kernel_size=kernel_size, init=init
            )
        elif kind == "aniso":
            kernel = QuadraticKernelSpectral2D(
                grp_size, out_channels, kernel_size=kernel_size, init=init
            )
        else:
            raise ValueError(f"Invalid {kind=}")

        conv_dilation = (
            SelectSemifield.tropical_max().lazy_fixed()
            if jit
            else BroadcastSemifield.tropical_max(
                channels_add=channel_add, spread_gradient=spread_gradient
            ).dynamic()
        )

        if is_closing:
            conv_erosion = (
                SelectSemifield.tropical_min_negated().lazy_fixed()
                if jit
                else BroadcastSemifield.tropical_min_negated(
                    channels_add=channel_add, spread_gradient=spread_gradient
                ).dynamic()
            )
            return GenericClosing2D(
                kernel=kernel,
                conv_dilation=conv_dilation,
                conv_erosion=conv_erosion,
                padding=padding,
                stride=stride,
                groups=grps,
                group_broadcasting=group_broadcasting,
            )

        return GenericConv2D(
            kernel=kernel,
            conv=conv_dilation,
            padding=padding,
            stride=stride,
            groups=grps,
            group_broadcasting=group_broadcasting,
        )

    return pooling_fn


def _calculate_groups(channels: int, groups: int, group_size: int) -> tuple[int, int]:
    if groups is not None and group_size is not None:
        expected_channels = groups * group_size
        assert channels == expected_channels, (
            f"Provided both {groups=} and {group_size=},"
            f" which implies {expected_channels=}, but actual {channels=}"
        )
        return group_size, groups

    if groups is None and group_size is None:
        grps = channels
        grp_size = 1
    elif groups is not None:
        grps = groups
        assert channels % grps == 0, f"{channels=} not evenly divided by {groups=}"
        grp_size = channels // grps
    else:
        grp_size = group_size
        assert channels % group_size == 0, (
            f"{channels=} not evenly divided by {group_size=}"
        )
        grps = channels // grp_size
    return grp_size, grps


def _get_jit_status(jit: bool | None, channel_add: bool, spread_gradient: bool):
    if (channel_add or spread_gradient) and jit is None:
        return False
    if jit is None:
        return POOLING_JIT_DEFAULT

    if channel_add or spread_gradient:
        assert not jit, "JIT doesn't support channel-add or spreading gradient"
    return jit


EXAMPLE_POOLING_FUNCTIONS: dict[str, Callable[[int, dict], nn.Module]] = {
    "standard-1": make_pooling_function("standard", 1),
    "standard-2": make_pooling_function("standard", 2),
    "standard-3": make_pooling_function("standard", 3),
    "standard-5": make_pooling_function("standard", 5),
    "iso-2": make_pooling_function("iso", 2),
    "iso-3": make_pooling_function("iso", 3),
    "iso-5": make_pooling_function("iso", 5),
    "iso-7": make_pooling_function("iso", 7),
    "iso-11": make_pooling_function("iso", 11),
    "aniso-2": make_pooling_function("aniso", 2),
    "aniso-3": make_pooling_function("aniso", 3),
    "aniso-5": make_pooling_function("aniso", 5),
    "aniso-7": make_pooling_function("aniso", 7),
    "aniso-11": make_pooling_function("aniso", 11),
}
