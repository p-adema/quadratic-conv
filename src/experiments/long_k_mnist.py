import sys
from pathlib import Path

import polars as pl
import torch

sys.path.extend(".")

from src import load_data
from src.models import LeNet
from src.models.configurations.simple_lenet import (
    grad_configs,
    group_configs,
    standard_configs,
)

torch.set_float32_matmul_precision("high")
k_mnist = load_data.k_mnist()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 30,
    "lr": 0.004,
    "count": 100,
    "progress_bar": True,
}

assert not Path("./.data/long_k_mnist.pq").exists(), "Move or delete old data"

result = {}
for desc, config_kwargs in standard_configs(name="Basics (k_mnist)"):
    result[desc] = LeNet.fit_many(
        data=k_mnist,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores
pl.DataFrame(result).write_parquet("./.data/long_k_mnist.pq")

result = {}
for desc, config_kwargs in group_configs(name="Groups (k_mnist)"):
    result[desc] = LeNet.fit_many(
        data=k_mnist,
        description=desc,
        conv_channels=(24, 60),
        **base_kwargs,
        **config_kwargs,
    ).scores
pl.DataFrame(result).write_parquet("./.data/groups_k_mnist.pq")

result = {}
for desc, config_kwargs in grad_configs(name="Grad (k_mnist)"):
    result[desc] = LeNet.fit_many(
        data=k_mnist,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores
pl.DataFrame(result).write_parquet("./.data/grad_k_mnist.pq")
