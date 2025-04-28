import sys

import polars as pl
import torch

sys.path.extend(".")

from src import load_data
from src.models import LeNet
from src.models.configurations.simple_lenet import group_configs, standard_configs

torch.set_float32_matmul_precision("high")
k_mnist = load_data.k_mnist()
fashion = load_data.fashion_mnist()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 5,
    "lr": 0.004,
    "count": 5,
    "progress_bar": False,
}

for data_name, data in {"k_mnist": k_mnist, "fashion": fashion}.items():
    for desc, config_kwargs in standard_configs(name=f"Test basics ({data_name})"):
        res = LeNet.fit_many(
            data=data,
            description=desc,
            **base_kwargs,
            **config_kwargs,
        ).scores
        median_acc = res.select(pl.median("acc")).item()
        assert median_acc > 0.7, f"Suspicously low performance for {desc}:\n{res}"

    for desc, config_kwargs in group_configs(name=f"Test groups ({data_name})"):
        res = LeNet.fit_many(
            data=data,
            description=desc,
            conv_channels=(24, 60),
            **base_kwargs,
            **config_kwargs,
        ).scores
        median_acc = res.select(pl.median("acc")).item()
        assert median_acc > 0.4, f"Suspicously low performance for {desc}:\n{res}"
