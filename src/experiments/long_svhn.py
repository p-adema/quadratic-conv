import sys

import polars as pl
import torch

sys.path.extend(".")

from src import load_data
from src.models import CIFAR10CNN
from src.models.configurations.simple_lenet import lenet_configs, standard_configs

torch.set_float32_matmul_precision("high")
svhn = load_data.svhn()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 250,
    "lr": 0.003,
    "count": 40,
    "progress_bar": True,
}
result = {}

# for desc, config_kwargs in standard_configs(name="Basics (svhn)"):
#     result[desc] = CIFAR10CNN.fit_many(
#         data=svhn,
#         description=desc,
#         **base_kwargs,
#         **config_kwargs,
#     ).scores
#
# pl.DataFrame(result).write_parquet("./.data/long_svhn.pq")


for desc, config_kwargs in lenet_configs(
    progress_bar="Temp extra (svhn)", standard_sizes=(7, 11), kernel_sizes=(13,)
):
    result[desc] = CIFAR10CNN.fit_many(
        data=svhn,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores

pl.DataFrame(result).write_parquet("./.data/extra_svhn.pq")
