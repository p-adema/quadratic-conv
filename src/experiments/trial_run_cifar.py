import sys

import torch

sys.path.extend(".")

from src import load_data
from src.models import CIFAR10CNN
from src.models.configurations.simple_lenet import (
    grad_configs,
    group_configs,
    standard_configs,
)

torch.set_float32_matmul_precision("high")
cifar10 = load_data.cifar10()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 2,
    "lr": 0.004,
    "count": 3,
    "progress_bar": False,
}

for desc, config_kwargs in standard_configs(name="Trial run basics (cifar10)"):
    CIFAR10CNN.fit_many(
        data=cifar10,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    )


for desc, config_kwargs in group_configs(name="Trial run groups (cifar10)"):
    CIFAR10CNN.fit_many(
        data=cifar10,
        description=desc,
        conv_channels=(30, 60, 120),
        **base_kwargs,
        **config_kwargs,
    )

for desc, config_kwargs in grad_configs(name="Trial run grad (cifar10)"):
    CIFAR10CNN.fit_many(
        data=cifar10,
        description=desc,
        conv_channels=(30, 60, 120),
        **base_kwargs,
        **config_kwargs,
    )
