import pytorch_numba_extension_jit as pnex
import torch
from numba import cuda


def mymuladd_2d(
    a: pnex.In("f32", (None, None)),
    # Within the generated code, .size can be called to retrieve the shape.
    # Indexing .shape may also work, but this is not guaranteed.
    b: pnex.In("f32", ("a.size(0)", "a.size(1)")),
    c: float,
    result: pnex.Out("f32", "a"),
):
    idx = cuda.grid(1)
    # Within the function body, .shape can be indexed to retrieve the shape.
    y, x = divmod(idx, result.shape[0])
    if y < result.shape[0]:
        result[y, x] = a[y, x] * b[y, x] + c


A = torch.arange(10, dtype=torch.float32, device="cuda").view(5, 2)
B = torch.full((5, 2), 2.0, device="cuda")
C = 10.0

RESULT = mymuladd_2d(A, B, C)

print("mymuladd_2d:", RESULT, sep="\n")


# CUDA always orders threads in X, Y, Z: this means that if you have an array which
# you would like to use e.g. Y, X indexing for, then you have to flip the order
# to make the names align:
@pnex.jit(n_threads=("result.size(1)", "result.size(0)"), to_extension=True)
def mymuladd_grid(
    a: pnex.In("f32", (None, None)),
    b: pnex.In("f32", ("a.size(0)", "a.size(1)")),
    c: float,
    result: pnex.Out("f32", "a"),
):
    # always use this order for names to be consistent with CUDA terminology:
    x, y = cuda.grid(2)

    if y < result.shape[0] and x < result.shape[1]:
        result[y, x] = a[y, x] * b[y, x] + c


RESULT_GRID = mymuladd_grid(A, B, C)

print("\nmymuladd_grid:", RESULT_GRID, sep="\n")
