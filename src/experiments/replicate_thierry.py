import sys

import polars as pl
import torch
import tqdm.auto as tqdm

sys.path.extend(".")

from src import load_data
from src.models import EXAMPLE_POOLING_FUNCTIONS, LeNet

k_mnist = load_data.k_mnist()

res = {
    pool_fn: LeNet.fit_many(
        k_mnist,
        pool_fn=pool_fn,
        batch_size=32,
        epochs=5,
        lr=0.001,
        count=100,
    ).scores
    for pool_fn in tqdm.tqdm(
        sorted(EXAMPLE_POOLING_FUNCTIONS), desc="Pooling types", unit="trial"
    )
}

pl.DataFrame(res).write_parquet(".data/thierry_long.pq")

res = {}


for size in tqdm.tqdm((3, 5, 7), desc="Aniso sizes", unit="trial"):
    res[f"aniso-{size}-spin"] = LeNet.fit_many(
        k_mnist,
        pool_fn=f"aniso-{size}",
        init={"var": "skewed", "theta": "spin"},
        batch_size=32,
        epochs=5,
        lr=0.001,
        count=100,
    ).scores
    res[f"aniso-{size}-uiso"] = LeNet.fit_many(
        k_mnist,
        pool_fn=f"aniso-{size}",
        init={"var": "uniform-iso", "theta": "uniform"},
        batch_size=32,
        epochs=5,
        lr=0.001,
        count=100,
    ).scores


pl.DataFrame(res).write_parquet(".data/thierry_extend.pq")
