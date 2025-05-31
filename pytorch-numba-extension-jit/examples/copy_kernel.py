import pytorch_numba_extension_jit as pnex
import torch
from numba import cuda


@pnex.jit(n_threads="a.numel()")
def copy(
    a: pnex.In(dtype="f32", shape=(None,)),
    result: pnex.Out(dtype="f32", shape="a"),
):
    x = cuda.grid(1)
    if x < a.shape[0]:
        result[x] = a[x]


A = torch.arange(5, dtype=torch.float32, device="cuda")

RESULT = copy(A)

print("copy:", RESULT, sep="\n")
