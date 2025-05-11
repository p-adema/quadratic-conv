import pytorch_numba_extension_jit as pnex
from numba import cuda


@pnex.jit(n_threads="result.numel()")
def mymuladd_2d(
    a: pnex.In("f32", (None, None)),
    b: pnex.In("f32", "a"),
    c: float,
    result: pnex.Out("f32", ("a", "a.shape[1]")),
):
    idx = cuda.grid(1)
    y, x = divmod(idx, result.shape[0])
    if y < result.shape[1]:
        result[y, x] = a[y, x] * b[y, x] + c
