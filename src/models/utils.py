from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    import keras


def quiet_model(model: keras.Model) -> keras.Model:
    fit_fn = model.fit

    def quiet_fit(*args, **kwargs):
        quiet_kwargs = kwargs | {"verbose": False}
        return fit_fn(*args, **quiet_kwargs)

    model.fit = quiet_fit
    return model


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
