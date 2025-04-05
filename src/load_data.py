from typing import NamedTuple

import numpy as np
import torch
import torchvision


class Dataset(NamedTuple):
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    def __repr__(self):
        return (
            f"Dataset(x_train={self.x_train.shape}, x_test={self.x_test.shape},"
            f" y_train={self.y_train.shape}, y_test={self.y_test.shape})"
        )


def _mnist_like_normalisation(
    train: torchvision.datasets.VisionDataset,
    test: torchvision.datasets.VisionDataset,
    has_channels: bool,
) -> Dataset:
    x_train_unnorm = torch.as_tensor(train.data, dtype=torch.float32)
    stds, means = torch.std_mean(
        torch.as_tensor(x_train_unnorm), (0, 1, 2), keepdim=True
    )
    print("normalisation:", means, stds)
    x_train = (x_train_unnorm - means) / stds
    y_train = np.asarray(train.targets)

    x_test = (torch.as_tensor(test.data, dtype=torch.float32) - means) / stds
    if has_channels:
        x_train = x_train.movedim(3, 1)
        x_test = x_test.movedim(3, 1)
    y_test = np.asarray(test.targets)
    return Dataset(np.asarray(x_train), np.asarray(x_test), y_train, y_test)


def _mnist_like(module, has_channels: bool = False):
    def get():
        train = module(root="./data", train=True, download=True)
        test = module(root="./data", train=False, download=True)
        return _mnist_like_normalisation(train, test, has_channels=has_channels)

    return get


cifar10 = _mnist_like(torchvision.datasets.CIFAR10, has_channels=True)
cifar100 = _mnist_like(torchvision.datasets.CIFAR100, has_channels=True)
mnist = _mnist_like(torchvision.datasets.MNIST)
fashion_mnist = _mnist_like(torchvision.datasets.FashionMNIST)
k_mnist = _mnist_like(torchvision.datasets.KMNIST)
