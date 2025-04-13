from typing import NamedTuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

STD_NORM = False


class Dataset(NamedTuple):
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    img_channels: int
    num_classes: int

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
            drop_last=True,
        )

    def test_loader(self, *, batch_size: int) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.as_tensor(self.x_test), torch.as_tensor(self.y_test)),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
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
        print("normalisation:", means, stds)
        x_train = (x_train_unnorm - means) / stds
        x_test = (torch.as_tensor(test.data, dtype=torch.float32) - means) / stds
    else:
        aminmax = torch.aminmax(torch.as_tensor(x_train_unnorm))
        low, high = aminmax.min, aminmax.max
        x_train = (x_train_unnorm - low) / (high - low)
        x_test = (torch.as_tensor(test.data, dtype=torch.float32) - low) / (high - low)
    y_train = np.asarray(train.targets)

    if has_channels:
        x_train = x_train.movedim(3, 1)
        x_test = x_test.movedim(3, 1)
    else:
        x_train.unsqueeze_(1)
        x_test.unsqueeze_(1)
    assert x_train.shape[1] == img_channels, f"Wrong channels in {x_train.shape=}"
    assert x_test.shape[1] == img_channels, f"Wrong channels in {x_test.shape=}"
    y_test = np.asarray(test.targets)
    return Dataset(
        np.asarray(x_train),
        np.asarray(x_test),
        y_train,
        y_test,
        img_channels,
        num_classes,
    )


def _mnist_like(
    module, has_channels: bool = False, img_channels: int = 1, num_classes: int = 10
):
    def get():
        train = module(root="./data", train=True, download=True)
        test = module(root="./data", train=False, download=True)
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
