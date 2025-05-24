import sys
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from pytorch_semifield_conv import BroadcastSemifield, GenericConv2D, LearnedKernel
from tqdm import tqdm

sys.path.extend(".")

from src import load_data
from src.models import LeNet
from src.models.utils import make_pooling_function

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
k_mnist = load_data.k_mnist()

base_kwargs = {
    "batch_size": 1024,
    "epochs": 30,
    "lr": 0.004,
    "count": 20,
    "progress_bar": True,
}

# assert not Path("./.data/nonlinear_k_mnist.pq").exists(), "Move or delete old data"
assert not Path("./.data/nonlinear_k_mnist_zoom.pq").exists(), "Move or delete old data"


def make_convs(kind: str, p_or_mu: float):
    semifield = (
        BroadcastSemifield.log(p_or_mu)
        if kind == "log"
        else BroadcastSemifield.root(p_or_mu)
    )
    return (
        GenericConv2D(
            kernel=LearnedKernel(1, 20, 5),
            conv=semifield.dynamic(),
        ),
        GenericConv2D(
            kernel=LearnedKernel(20, 50, 5),
            conv=semifield.dynamic(),
        ),
    )


# result = {}

# for param in tqdm(
#     (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2), desc="Param values"
# ):
#     for field_kind in ("log", "root"):
#         result[f"aniso-7-{field_kind}-{param}"] = LeNet.fit_many(
#             data=k_mnist,
#             description=f"aniso-7-{field_kind}-{param}",
#             pool_fn="aniso-7",
#             init={"var": "ss-iso", "theta": "spin"},
#             convs=make_convs(field_kind, param),
#             **base_kwargs,
#         ).scores
#         result[f"standard-3-{field_kind}-{param}"] = LeNet.fit_many(
#             data=k_mnist,
#             description=f"standard-3-{field_kind}-{param}",
#             pool_fn="standard-3",
#             convs=make_convs(field_kind, param),
#             **base_kwargs,
#         ).scores
#
# pl.DataFrame(result).write_parquet("./.data/nonlinear_k_mnist.pq")

base_kwargs = {
    "batch_size": 1024,
    "epochs": 30,
    "lr": 0.004,
    "count": 5,
    "progress_bar": True,
}
result = {}
tiny_delta = 0.0000000000000001
for param in tqdm(
    np.linspace(1 - tiny_delta, 1 + tiny_delta, num=5).tolist(),
    desc="Param values",
):
    for field_kind in ("root", "log"):
        result[f"aniso-7-{field_kind}-{param}"] = LeNet.fit_many(
            data=k_mnist,
            description=f"aniso-7-{field_kind}-{param}",
            pool_fn="aniso-7",
            init={"var": "ss-iso", "theta": "spin"},
            convs=make_convs(field_kind, param),
            **base_kwargs,
        ).scores
        result[f"standard-3-{field_kind}-{param}"] = LeNet.fit_many(
            data=k_mnist,
            description=f"standard-3-{field_kind}-{param}",
            pool_fn="standard-3",
            convs=make_convs(field_kind, param),
            **base_kwargs,
        ).scores

pl.DataFrame(result).write_parquet("./.data/nonlinear_k_mnist_zoom.pq")
