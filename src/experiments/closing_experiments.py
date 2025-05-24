import sys
from pathlib import Path

import polars as pl
import torch

sys.path.extend(".")

from src import load_data
from src.models import CIFAR10CNN, LeNet
from src.models.configurations.simple_lenet import extra_configs

k_mnist = load_data.k_mnist()
cifar10 = load_data.cifar10()

km_res = Path("./.data/closing_k_mnist.pq")
cf_res = Path("./.data/closing_cifar10.pq")

assert km_res.parent.exists(), f"Results directory missing for {km_res}"
assert not km_res.exists(), f"Move or delete old data at {km_res}"
assert not cf_res.exists(), f"Move or delete old data at {cf_res}"

base_kwargs = {
    "batch_size": 1024,
    "epochs": 30,
    "lr": 0.004,
    "count": 100,
    "progress_bar": True,
}


result = {}
for desc, config_kwargs in extra_configs(7, name="Extras (k_mnist)"):
    result[desc] = LeNet.fit_many(
        data=k_mnist,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores
pl.DataFrame(result).write_parquet(km_res)

base_kwargs = {
    "batch_size": 1024,
    "epochs": 150,
    "lr": 0.004,
    "count": 40,
    "progress_bar": True,
}

result = {}
for desc, config_kwargs in extra_configs(5, name="Extras (cifar10)"):
    result[desc] = CIFAR10CNN.fit_many(
        data=cifar10,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores
pl.DataFrame(result).write_parquet(cf_res)
