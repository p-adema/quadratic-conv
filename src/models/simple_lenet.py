from __future__ import annotations

import typing
from collections.abc import Callable

# import keras
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from tqdm.auto import trange

from ..convolutions import BroadcastSemifield, GenericConv2D, SelectSemifield
from ..kernels import QuadraticKernelIso2D, QuadraticKernelSpectral2D
from ..load_data import Dataset
from .utils import CheckNan, quiet_model


class LeNet(nn.Module):
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
            pool_fn = LENET_POOLING_FUNCTIONS[pool_fn]

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
            modules.append(CheckNan(999))
        self.net = nn.Sequential(*modules)
        self.debug = debug

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.net(imgs)

    if typing.TYPE_CHECKING:
        __call__ = forward

    def _epoch(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor,
        opt: torch.optim.Optimizer,
        batch_size: int,
    ):
        for batch_start in range(0, imgs.shape[0], batch_size):
            y = labels[batch_start : batch_start + batch_size]
            # print(y)
            res = self(imgs[batch_start : batch_start + batch_size])
            # print(torch.log_softmax(res, dim=1))
            loss = nn.functional.cross_entropy(res, y)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # input()

    def fit(
        self,
        data: Dataset,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.001,
        epoch_callback: Callable[[LeNet], None] | None = None,
        shuffle_rng: torch.random.Generator | int = 0,
        verbose: bool = True,
        shuffle: bool = True,
    ) -> LeNet:
        self.train().to("cuda")

        if isinstance(shuffle_rng, int):
            shuffle_rng = torch.Generator(device="cuda").manual_seed(shuffle_rng)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        imgs = torch.as_tensor(data.x_train, device="cuda")
        labels = torch.as_tensor(data.y_train, device="cuda")
        for _ in trange(
            epochs, desc="Training", unit="epoch", smoothing=0, disable=not verbose
        ):
            if shuffle:
                idxs = torch.randperm(
                    imgs.shape[0], generator=shuffle_rng, device="cuda"
                )
            else:
                idxs = torch.arange(imgs.shape[0])
            self._epoch(imgs[idxs], labels[idxs], opt, batch_size)
            if epoch_callback is not None:
                epoch_callback(self)

        return self

    @torch.no_grad()
    def evaluate(self, data: Dataset, batch_size: int = 10_000) -> dict:
        self.eval().to("cuda")
        preds = []
        imgs = torch.as_tensor(data.x_test, device="cuda")
        for batch_start in range(0, imgs.shape[0], batch_size):
            preds.append(
                self(imgs[batch_start : batch_start + batch_size])
                .argmax(1)
                .numpy(force=True)
            )
        preds = np.concat(preds)
        return classification_report(
            data.y_test, preds, output_dict=True, zero_division=0
        )

    @classmethod
    def fit_many(
        cls,
        data: Dataset,
        epochs: int = 5,
        pool_fn: str = "standard-2",
        init: str | float = 3.0,
        seed: int = 0,
        debug: bool = False,
        count: int = 20,
        epoch_callback: Callable[[LeNet], None] | None = None,
    ) -> list[dict]:
        run_scores = []
        shuffle_rng = torch.Generator("cuda").manual_seed(seed + 42)
        torch.manual_seed(seed)
        bar = trange(count, unit="run", desc=pool_fn)
        for _ in bar:
            run_scores.append(
                cls(
                    img_channels=data.img_channels,
                    num_classes=data.num_classes,
                    pool_fn=pool_fn,
                    debug=debug,
                    init=init,
                )
                .fit(
                    data,
                    epochs=epochs,
                    shuffle_rng=shuffle_rng,
                    verbose=False,
                    epoch_callback=epoch_callback,
                )
                .evaluate(data)
            )
            bar.set_postfix(last_acc=run_scores[-1]["accuracy"])
        return run_scores


LENET_POOLING_FUNCTIONS: dict[str, Callable[[int, str], nn.Module]] = {
    "standard-2": lambda _, __: nn.MaxPool2d(kernel_size=2, stride=2),
    "standard-3": lambda _, __: nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    "standard-5": lambda _, __: nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
    "standard-7": lambda _, __: nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
    "iso-3": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelIso2D(1, c, kernel_size=3, init=init),
        # conv=SelectSemifield.tropical_max().lazy_fixed(),
        conv=BroadcastSemifield.tropical_max().module(),
        padding=1,
        stride=2,
        groups=c,
    ),
    "iso-5": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelIso2D(1, c, kernel_size=5, init=init),
        # conv=SelectSemifield.tropical_max().lazy_fixed(),
        conv=BroadcastSemifield.tropical_max().module(),
        padding=2,
        stride=2,
        groups=c,
    ),
    "iso-7": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelIso2D(1, c, kernel_size=7, init=init),
        # conv=SelectSemifield.tropical_max().lazy_fixed(),
        conv=BroadcastSemifield.tropical_max().module(),
        padding=3,
        stride=2,
        groups=c,
    ),
    "aniso-3": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelSpectral2D(1, c, kernel_size=3, init=init),
        # conv=SelectSemifield.tropical_max().lazy_fixed(),
        conv=BroadcastSemifield.tropical_max().module(),
        padding=1,
        stride=2,
        groups=c,
    ),
    "aniso-5": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelSpectral2D(1, c, kernel_size=5, init=init),
        # conv=SelectSemifield.tropical_max().lazy_fixed(),
        conv=BroadcastSemifield.tropical_max().module(),
        padding=2,
        stride=2,
        groups=c,
    ),
    "aniso-7": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelSpectral2D(1, c, kernel_size=7, init=init),
        # conv=SelectSemifield.tropical_max().lazy_fixed(),
        conv=BroadcastSemifield.tropical_max().module(),
        padding=3,
        stride=2,
        groups=c,
    ),
}
