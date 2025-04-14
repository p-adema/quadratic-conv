from __future__ import annotations

import sys
import typing
from collections.abc import Callable

# import keras
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import multiprocessing, nn
from tqdm.auto import tqdm, trange

sys.path.extend(".")

from ..load_data import Dataset
from . import POOLING_FUNCTIONS
from .trainer import Trainer
from .utils import CheckNan, split_seed


class LeNet(Trainer):
    """LeNet-like model, returns logits"""

    def __init__(
        self,
        img_channels: int,
        num_classes: int,
        pool_fn: Callable[[int, str], nn.Module] | str,
        conv_kernel_size: int = 5,
        linear_units: int = 500,
        init: str | float = 3.0,
        debug: bool = False,
    ):
        super().__init__()

        if isinstance(pool_fn, str):
            pool_fn = POOLING_FUNCTIONS[pool_fn]

        modules = [
            nn.Conv2d(img_channels, 20, conv_kernel_size),
            nn.ReLU(),
            pool_fn(20, init),
            nn.Conv2d(20, 50, conv_kernel_size),
            nn.ReLU(),
            pool_fn(50, init),
            nn.Flatten(),
            nn.Linear(800, 500),
            # nn.LazyLinear(linear_units),
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
