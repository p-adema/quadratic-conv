import sys
import warnings
from pathlib import Path

import polars as pl

sys.path.extend(".")

from src import load_data
from src.models import CIFAR10CNN
from src.models.configurations.simple_lenet import standard_configs

imagenette = load_data.imagenette()

base_kwargs = {
    "data": imagenette,
    "batch_size": 64,
    "epochs": 15,
    "lr": 0.001,
    "count": 40,
    "progress_bar": True,
    "conv_channels": (32, 64, 128, 128, 128),
}
result = {}
save_path = Path("./.data/long_imagenette.pq")
assert save_path.parent.exists(), "Data directory missing"
assert not save_path.exists(), f"Move or delete old data at {save_path=}"

# Ignore the "no fast path for CUDAGraphs" warning.
# This warning is valid but not actionable: with the current build of PyTorch, adding
# mark_step_begin causes the model to hang, while no_grad causes recompilations.
warnings.simplefilter("ignore", UserWarning, 2442)

for desc, config_kwargs in standard_configs(name="Basics (imagenette)"):
    result[desc] = CIFAR10CNN.fit_many(
        description=desc,
        **base_kwargs,
        **config_kwargs,
    ).scores

pl.DataFrame(result).write_parquet(save_path)
