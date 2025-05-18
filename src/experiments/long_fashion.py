import sys
from pathlib import Path

import polars as pl
import torch

sys.path.extend(".")

from src import load_data
from src.models import LeNet
from src.models.configurations.simple_lenet import (
    group_configs,
    standard_configs,
)

torch.set_float32_matmul_precision("high")
fashion = load_data.fashion_mnist()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 80,
    "lr": 0.003,
    "count": 100,
    "progress_bar": True,
}

assert not Path("./.data/long_fashion.pq").exists(), "Move or delete old data"

result = {}
for desc, config_kwargs in standard_configs(name="Basics (fashion)"):
    result[desc] = LeNet.fit_many(
        data=fashion,
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores
pl.DataFrame(result).write_parquet("./.data/long_fashion.pq")


# result = {}
# for desc, config_kwargs in group_configs(name="Groups (fashion)"):
#     result[desc] = LeNet.fit_many(
#         data=fashion,
#         description=desc,
#         conv_channels=(24, 60),
#         **base_kwargs,
#         **config_kwargs,
#     ).scores
# pl.DataFrame(result).write_parquet("./.data/groups_fashion.pq")
