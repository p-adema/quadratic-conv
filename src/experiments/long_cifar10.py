import sys

import polars as pl
import torch

sys.path.extend(".")

from src import load_data
from src.models import CIFAR10CNN
from src.models.configurations.simple_lenet import standard_configs

torch.set_float32_matmul_precision("high")
cifar10 = load_data.cifar10()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 150,
    "lr": 0.004,
    "count": 40,
    "progress_bar": True,
}
result = {}

for desc, config_kwargs in standard_configs(name="Basics (cifar10)"):
    result[desc] = CIFAR10CNN.fit_many(
        data=cifar10,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores

pl.DataFrame(result).write_parquet("./.data/long_cifar10.pq")
result.clear()
