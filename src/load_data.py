from __future__ import annotations

from typing import NamedTuple

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

# When True, normalise to zero-mean, unit variance.
# When False, normalise to [0, 1]
STD_NORM = False


class Dataset(NamedTuple):
    x_train: torch.Tensor
    x_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    img_channels: int
    num_classes: int
    label_names: list[str]

    def __repr__(self):
        return (
            f"Dataset(x_train={self.x_train.shape}, x_test={self.x_test.shape},"
            f" y_train={self.y_train.shape}, y_test={self.y_test.shape})"
        )

    def train_loader(self, *, batch_size: int, shuffle_seed: int = 0) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.as_tensor(self.x_train), torch.as_tensor(self.y_train)),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            generator=torch.Generator().manual_seed(shuffle_seed),
        )

    def test_loader(self, *, batch_size: int) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.as_tensor(self.x_test), torch.as_tensor(self.y_test)),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def as_cuda(
        self, except_y_test: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x_train.cuda(non_blocking=True),
            self.y_train.cuda(non_blocking=True),
            self.x_test.cuda(non_blocking=True),
            self.y_test if except_y_test else self.y_test.cuda(non_blocking=True),
        )

    def tuning_split(self, val_prop: float = 0.3, seed: int = 0) -> Dataset:
        gen = torch.Generator("cpu").manual_seed(seed)
        train_size = self.x_train.shape[0]
        shuf_idxs = torch.randperm(train_size, generator=gen)
        val_size = int(train_size * val_prop)
        val_idxs, train_idxs = torch.split(shuf_idxs, [val_size, train_size - val_size])
        assert val_size
        assert train_size - val_size
        return Dataset(
            self.x_train[train_idxs],
            self.x_train[val_idxs],
            self.y_train[train_idxs],
            self.y_train[val_idxs],
            self.img_channels,
            self.num_classes,
            self.label_names,
        )

    def upsample(self, factor: int) -> Dataset:
        assert factor > 1
        return Dataset(
            self.x_train.repeat_interleave(factor, dim=2).repeat_interleave(
                factor, dim=3
            ),
            self.x_test.repeat_interleave(factor, dim=2).repeat_interleave(
                factor, dim=3
            ),
            self.y_train,
            self.y_test,
            self.img_channels,
            self.num_classes,
            self.label_names,
        )


def _mnist_like_normalisation(
    train: torchvision.datasets.VisionDataset,
    test: torchvision.datasets.VisionDataset,
    has_channels: bool,
    img_channels: int,
    num_classes: int,
) -> Dataset:
    x_train_unnorm = torch.as_tensor(train.data, dtype=torch.float32)
    if STD_NORM:
        stds, means = torch.std_mean(
            torch.as_tensor(x_train_unnorm), (0, 1, 2), keepdim=True
        )
        x_train = (x_train_unnorm - means) / stds
        x_test = (torch.as_tensor(test.data, dtype=torch.float32) - means) / stds
    else:
        aminmax = torch.aminmax(torch.as_tensor(x_train_unnorm))
        low, high = aminmax.min, aminmax.max
        x_train = (x_train_unnorm - low) / (high - low)
        x_test = (torch.as_tensor(test.data, dtype=torch.float32) - low) / (high - low)
    y_train = torch.asarray(train.targets, dtype=torch.int64)

    if has_channels:
        x_train = x_train.movedim(3, 1)
        x_test = x_test.movedim(3, 1)
    else:
        x_train.unsqueeze_(1)
        x_test.unsqueeze_(1)
    assert x_train.shape[1] == img_channels, f"Wrong channels in {x_train.shape=}"
    assert x_test.shape[1] == img_channels, f"Wrong channels in {x_test.shape=}"
    assert num_classes == len(train.classes), "Classes seem wrong"
    y_test = torch.as_tensor(test.targets, dtype=torch.int64)
    if torch.cuda.is_available():
        return Dataset(
            x_train.contiguous().cpu().pin_memory(),
            x_test.contiguous().cpu().pin_memory(),
            y_train.contiguous().cpu().pin_memory(),
            y_test.contiguous().cpu().pin_memory(),
            img_channels,
            num_classes,
            train.classes,
        )
    return Dataset(
        x_train.contiguous(),
        x_test.contiguous(),
        y_train.contiguous(),
        y_test.contiguous(),
        img_channels,
        num_classes,
        train.classes,
    )


def _mnist_like(
    module, has_channels: bool = False, img_channels: int = 1, num_classes: int = 10
):
    def get():
        train = module(root="./.data/", train=True, download=True)
        test = module(root="./.data/", train=False, download=True)
        return _mnist_like_normalisation(
            train,
            test,
            has_channels=has_channels,
            img_channels=img_channels,
            num_classes=num_classes,
        )

    return get


cifar10 = _mnist_like(torchvision.datasets.CIFAR10, has_channels=True, img_channels=3)
cifar100 = _mnist_like(
    torchvision.datasets.CIFAR100, has_channels=True, img_channels=3, num_classes=100
)
mnist = _mnist_like(torchvision.datasets.MNIST)
fashion_mnist = _mnist_like(torchvision.datasets.FashionMNIST)
k_mnist = _mnist_like(torchvision.datasets.KMNIST)
