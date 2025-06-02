import sys

import polars as pl
import torch

sys.path.extend(".")

from src import load_data
from src.models import CIFAR10CNN
from src.models.configurations.simple_lenet import lenet_configs, standard_configs

svhn = load_data.svhn()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 250,
    "lr": 0.003,
    "count": 40,
    "progress_bar": True,
}
result = {}

for desc, config_kwargs in standard_configs(name="Basics (svhn)"):
    result[desc] = CIFAR10CNN.fit_many(
        data=svhn,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores

pl.DataFrame(result).write_parquet("./.data/long_svhn.pq")
