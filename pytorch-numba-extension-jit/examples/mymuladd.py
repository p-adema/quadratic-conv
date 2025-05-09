import pytorch_numba_extension_jit as ptex
from numba import cuda


@ptex.jit(n_threads="result")
def mymuladd_2d(
    a: ptex.In("f32", (None, None)),
    b: ptex.In("f32", "a"),
    c: float,
    result: ptex.Out("f32", ("a", "a.shape[1]")),
):
    idx = cuda.grid(1)
    y, x = divmod(idx, result.shape[0])
    if y < result.shape[1]:
        result[y, x] = a[y, x] * b[y, x] + c
