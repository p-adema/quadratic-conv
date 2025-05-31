import sys
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from pytorch_nd_semiconv import (
    BroadcastSemifield,
    GenericConv,
    LearnedKernel2D,
    QuadraticKernelSpectral2D,
)
from tqdm import tqdm

sys.path.extend(".")

from src import load_data
from src.models import LeNet

base_kwargs = {
    "data": load_data.k_mnist(),
    "batch_size": 1024,
    "epochs": 30,
    "lr": 0.004,
    "count": 20,
    "progress_bar": True,
    "pool_fn": "aniso-7",
    "init": {"var": "ss-iso", "theta": "spin"},
}

assert not Path("./.data/nonlinear_k_mnist.pq").exists(), "Move or delete old data"


class ClipRootConv(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.conv = BroadcastSemifield.root(p).dynamic()
        self.bn = torch.nn.LazyBatchNorm2d()

    def forward(self, img: torch.Tensor, kernel, *args, **kwargs):
        img = img.clip(0.001)
        kernel = kernel.clip(0.001)
        out = self.conv(img, kernel, *args, **kwargs)
        return self.bn(out)


result = {}
param_space = np.logspace(-2, 2, 9).tolist()
print("Parameters:", param_space)

for param in tqdm(param_space, desc="Param values"):
    # For log semifields, we use a quadratic kernel (like with dilations)
    result[f"aniso-7-log-{param}"] = LeNet.fit_many(
        description=f"aniso-7-log-{param}",
        convs=(
            GenericConv(
                kernel=QuadraticKernelSpectral2D(1, 20, 5),
                conv=BroadcastSemifield.log(param).dynamic(),
            ),
            GenericConv(
                kernel=QuadraticKernelSpectral2D(20, 50, 5),
                conv=BroadcastSemifield.log(param).dynamic(),
            ),
        ),
        **base_kwargs,
    ).scores
    # For root semifields, we use a modified convolution that clips and batch-norms,
    # but keep the kernel entirely learned.
    result[f"aniso-7-root-{param}"] = LeNet.fit_many(
        description=f"aniso-7-root-{param}",
        convs=(
            GenericConv(
                kernel=LearnedKernel2D(1, 20, 5),
                conv=ClipRootConv(param),
            ),
            GenericConv(
                kernel=LearnedKernel2D(20, 50, 5),
                conv=ClipRootConv(param),
            ),
        ),
        **base_kwargs,
    ).scores

pl.DataFrame(result).write_parquet("./.data/nonlinear_k_mnist.pq")
