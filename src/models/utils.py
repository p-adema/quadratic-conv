from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import torch
from torch import nn

from src.convolutions import BroadcastSemifield, GenericConv2D, SelectSemifield
from src.kernels import QuadraticKernelIso2D, QuadraticKernelSpectral2D

from ..load_data import Dataset

if TYPE_CHECKING:
    from .trainer import Trainer


class CheckNan(nn.Module):
    def __init__(self, i: int):
        super().__init__()
        self.i = i

    @torch.compiler.disable
    def forward(self, *args):
        for i, arr in enumerate(args, start=1):
            # noinspection PyProtectedMember
            torch._check_value(
                not torch.isnan(arr).any().item(),
                lambda: f"{self.i}: Item {i}/{len(args)} was NaN",  # noqa: B023
            )
            print(
                f"{self.i}: Item {i}/{len(args)} was OK,"
                f" shape={tuple(arr.shape)} min={arr.min()} max={arr.max()}"
            )
        return tuple(args) if len(args) > 1 else args[0]


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


def history_callback(data: Dataset | tuple[torch.Tensor, torch.Tensor]):
    class Hist:
        def __init__(self):
            self.losses = []
            self.reports = []

        def __call__(self, model: Trainer, train_loss: float):
            self.losses.append(train_loss)
            self.reports.append(model.evaluate(data))

        def result(self) -> tuple[pl.DataFrame, list[float]]:
            return reports_to_df(self.reports), self.losses

    return Hist()


POOLING_JIT = True
POOLING_FUNCTIONS: dict[str, Callable[[int, str], nn.Module]] = {
    "standard-2": lambda _, __: nn.MaxPool2d(kernel_size=2, stride=2),
    "standard-3": lambda _, __: nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    "standard-5": lambda _, __: nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
    "standard-7": lambda _, __: nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
    "iso-3": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelIso2D(1, c, kernel_size=3, init=init),
        conv=SelectSemifield.tropical_max().lazy_fixed()
        if POOLING_JIT
        else BroadcastSemifield.tropical_max().module(),
        padding=1,
        stride=2,
        groups=c,
    ),
    "iso-5": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelIso2D(1, c, kernel_size=5, init=init),
        conv=SelectSemifield.tropical_max().lazy_fixed()
        if POOLING_JIT
        else BroadcastSemifield.tropical_max().module(),
        padding=2,
        stride=2,
        groups=c,
    ),
    "iso-7": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelIso2D(1, c, kernel_size=7, init=init),
        conv=SelectSemifield.tropical_max().lazy_fixed()
        if POOLING_JIT
        else BroadcastSemifield.tropical_max().module(),
        padding=3,
        stride=2,
        groups=c,
    ),
    "aniso-3": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelSpectral2D(1, c, kernel_size=3, init=init),
        conv=SelectSemifield.tropical_max().lazy_fixed()
        if POOLING_JIT
        else BroadcastSemifield.tropical_max().module(),
        padding=1,
        stride=2,
        groups=c,
    ),
    "aniso-5": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelSpectral2D(1, c, kernel_size=5, init=init),
        conv=SelectSemifield.tropical_max().lazy_fixed()
        if POOLING_JIT
        else BroadcastSemifield.tropical_max().module(),
        padding=2,
        stride=2,
        groups=c,
    ),
    "aniso-7": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelSpectral2D(1, c, kernel_size=7, init=init),
        conv=SelectSemifield.tropical_max().lazy_fixed()
        if POOLING_JIT
        else BroadcastSemifield.tropical_max().module(),
        padding=3,
        stride=2,
        groups=c,
    ),
}
