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
fashion = load_data.fashion_mnist()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 30,
    "lr": 0.004,
    "count": 100,
    "progress_bar": True,
}

assert not Path("./.data/long_k_mnist.pq").exists(), "Move or delete old data"

for data_name, data in {"k_mnist": k_mnist, "fashion": fashion}.items():
    result = {}
    for desc, config_kwargs in standard_configs(name=f"Basics ({data_name})"):
        result[desc] = LeNet.fit_many(
            data=data,
            description=desc,
            **base_kwargs,
            **config_kwargs,
        ).scores
    pl.DataFrame(result).write_parquet(f"./.data/long_{data_name}.pq")

    result = {}
    for desc, config_kwargs in group_configs(name=f"Groups ({data_name})"):
        result[desc] = LeNet.fit_many(
            data=data,
            description=desc,
            conv_channels=(24, 60),
            **base_kwargs,
            **config_kwargs,
        ).scores
    pl.DataFrame(result).write_parquet(f"./.data/groups_{data_name}.pq")

    result = {}
    for desc, config_kwargs in grad_configs(name=f"Grad ({data_name})"):
        result[desc] = LeNet.fit_many(
            data=data,
            description=desc,
            **base_kwargs,
            **config_kwargs,
        ).scores
    pl.DataFrame(result).write_parquet(f"./.data/grad_{data_name}.pq")
