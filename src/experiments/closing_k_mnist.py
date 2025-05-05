import sys
from pathlib import Path

import polars as pl
import torch

sys.path.extend(".")

from src import load_data
from src.models import LeNet
from src.models.utils import make_pooling_function

torch.set_float32_matmul_precision("high")
k_mnist = load_data.k_mnist()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 30,
    "lr": 0.004,
    "count": 100,
    "progress_bar": True,
}

assert not Path("./.data/closing_k_mnist.pq").exists(), "Move or delete old data"

result = {}
for closing in (False, True):
    extra_desc = "-closing" if closing else ""
    result[f"aniso-7{extra_desc}"] = LeNet.fit_many(
        data=k_mnist,
        description=f"aniso-7{extra_desc}",
        pool_fn=make_pooling_function("aniso", 7, closing=closing),
        init={"var": "ss-iso", "theta": "spin"},
        **base_kwargs,
    ).scores
    result[f"iso-7{extra_desc}"] = LeNet.fit_many(
        data=k_mnist,
        description=f"iso-7{extra_desc}",
        pool_fn=make_pooling_function("iso", 7, closing=closing),
        init={"var": "ss"},
        **base_kwargs,
    ).scores

pl.DataFrame(result).write_parquet("./.data/closing_k_mnist.pq")
