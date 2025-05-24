from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .trainer import Trainer
from .utils import EXAMPLE_POOLING_FUNCTIONS, CheckNan


class LeNet(Trainer):
    """LeNet-like model, returns logits"""

    def __init__(
        self,
        img_channels: int,
        num_classes: int,
        pool_fn: Callable[[int, dict], nn.Module] | str,
        conv_kernel_size: int = 5,
        linear_units: int = 500,
        conv_channels: tuple[int, int] = (20, 50),
        init: dict[str, str | float] | None = None,
        init_seed: int | None = None,
        conv_dilation: int = 1,
        convs: tuple[nn.Module, nn.Module] = None,
        debug: bool = False,
    ):
        super().__init__()
        if init_seed is not None:
            torch.manual_seed(init_seed)

        if isinstance(pool_fn, str):
            pool_fn = EXAMPLE_POOLING_FUNCTIONS[pool_fn]

        modules = [
            (
                nn.Conv2d(
                    img_channels,
                    conv_channels[0],
                    conv_kernel_size,
                    dilation=conv_dilation,
                )
                if convs is None
                else convs[0]
            ),
            nn.ReLU(),
            pool_fn(conv_channels[0], init),
            (
                nn.Conv2d(
                    conv_channels[0],
                    conv_channels[1],
                    conv_kernel_size,
                    dilation=conv_dilation,
                )
                if convs is None
                else convs[1]
            ),
            nn.ReLU(),
            pool_fn(conv_channels[1], init),
            nn.Flatten(),
            nn.LazyLinear(linear_units),
            nn.ReLU(),
            nn.Linear(linear_units, num_classes),
        ]
        if debug:
            print(f"Debug on, {init=}")
            for i in reversed(range(len(modules))):
                modules.insert(i, CheckNan(i))
            modules.append(CheckNan(len(modules)))
        self.net = nn.Sequential(*modules)
        self.debug = debug

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.net(imgs)
