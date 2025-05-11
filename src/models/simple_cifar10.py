from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .trainer import Trainer
from .utils import EXAMPLE_POOLING_FUNCTIONS, CheckNan


# Based on:  https://www.kaggle.com/code/ektasharma/simple-cifar10-cnn-keras-code-with-88-accuracy
class CIFAR10CNN(Trainer):
    """CNN model for CIFAR10, returns logits"""

    def __init__(
        self,
        img_channels: int,
        num_classes: int,
        pool_fn: Callable[[int, dict], nn.Module] | str,
        conv_kernel_size: int = 3,
        linear_units: int = 128,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        init: dict[str, str | float] | None = None,
        init_seed: int | None = None,
        debug: bool = False,
    ):
        super().__init__()
        if init_seed is not None:
            torch.manual_seed(init_seed)

        if isinstance(pool_fn, str):
            pool_fn = EXAMPLE_POOLING_FUNCTIONS[pool_fn]

        def conv_block(in_chan: int, out_chan: int, dropout: float):
            return (
                nn.Conv2d(in_chan, out_chan, conv_kernel_size, padding="same"),
                nn.ReLU(),
                nn.LazyBatchNorm2d(),
                nn.Conv2d(out_chan, out_chan, conv_kernel_size, padding="same"),
                nn.ReLU(),
                nn.LazyBatchNorm2d(),
                pool_fn(out_chan, init),
                nn.Dropout2d(p=dropout),
            )

        modules = [
            *conv_block(img_channels, conv_channels[0], dropout=0.3),
            *conv_block(conv_channels[0], conv_channels[1], dropout=0.5),
            *conv_block(conv_channels[1], conv_channels[2], dropout=0.5),
            nn.Flatten(),
            nn.LazyLinear(linear_units),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(p=0.5),
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
