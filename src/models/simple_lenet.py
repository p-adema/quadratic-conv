import typing
from collections.abc import Callable

import keras
import numpy as np
import torch
from torch import nn

from ..convolutions import GenericConv2D, SelectSemifield
from ..kernels import QuadraticKernelCholesky2D, QuadraticKernelIso2D
from ..load_data import Dataset


class LeNet(nn.Module):
    """LeNet-like model, returns logits"""

    def __init__(
        self,
        img_channels: int,
        num_classes: int,
        pool_fn: Callable[[int, str], nn.Module],
        conv_kernel_size: int = 5,
        linear_units: int = 500,
        init_kind: str = "zero",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 20, conv_kernel_size),
            nn.ReLU(),
            pool_fn(20, init_kind),
            nn.Conv2d(20, 50, conv_kernel_size),
            nn.ReLU(),
            pool_fn(50, init_kind),
            nn.Flatten(),
            nn.LazyLinear(linear_units),
            nn.ReLU(),
            nn.Linear(linear_units, num_classes),
        )

    @torch.compile(fullgraph=True)
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.net(imgs)

    if typing.TYPE_CHECKING:
        __call__ = forward

    def to_keras(
        self, example: torch.Tensor | np.ndarray | Dataset, batch_size: int = 256
    ) -> keras.Model:
        with torch.no_grad():
            if isinstance(example, tuple):
                example = example[0]
            example = torch.as_tensor(example[:batch_size], device="cuda")
            self(example)
            model: keras.Model = keras.Sequential(
                (keras.layers.TorchModuleWrapper(self, name="LeNet"),)
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[
                    "accuracy",
                    keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
                ],
            )
            model(example)
        return model


LENET_POOLING_FUNCTIONS: dict[str, Callable[[int, str], nn.Module]] = {
    "standard": lambda _, __: nn.MaxPool2d(kernel_size=2, stride=2),
    "iso": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelIso2D(c, c, kernel_size=3, init=init),
        conv=SelectSemifield.tropical_max().lazy_fixed(),
        padding=1,
        stride=2,
    ),
    "aniso": lambda c, init: GenericConv2D(
        kernel=QuadraticKernelCholesky2D(c, c, kernel_size=3, init=init),
        conv=SelectSemifield.tropical_max().lazy_fixed(),
        padding=1,
        stride=2,
    ),
}
