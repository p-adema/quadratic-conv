import sys
from collections.abc import Callable
from itertools import product

import polars as pl
import torch
import tqdm.auto as tqdm

sys.path.extend(".")

from src import load_data
from src.models import LeNet, make_pooling_function

torch.set_float32_matmul_precision("high")
k_mnist = load_data.k_mnist()

trials: dict[str, tuple[Callable, dict | None]] = {}

kernel_sizes = (2, 3, 5, 7, 9)
for size in kernel_sizes:
    trials[f"standard-{size}"] = make_pooling_function("standard", size), None

for size, init in product(kernel_sizes, ("uniform", "ss")):
    trials[f"iso-{size}-{init}"] = make_pooling_function("iso", size), {"var": init}

for size, (v_init, t_init) in product(
    kernel_sizes,
    (
        ("uniform", "uniform"),
        ("uniform-iso", "uniform"),
        ("uniform-iso", "spin"),
        ("ss-iso", "spin"),
        ("skewed", "spin"),
    ),
):
    trials[f"aniso-{size}-{v_init}-{t_init}"] = (
        make_pooling_function("aniso", size),
        {"var": v_init, "theta": t_init},
    )

results = {
    desc: LeNet.fit_many(
        k_mnist,
        pool_fn=pool_fn,
        init=init,
        description=desc,
        batch_size=1024,
        epochs=30,
        lr=0.004,
        count=100,
    )
    for desc, (pool_fn, init) in tqdm.tqdm(
        trials.items(), desc="K-MNIST Long", unit="trial"
    )
}


pl.DataFrame(results).write_parquet("data/long_kmnist.pq")
