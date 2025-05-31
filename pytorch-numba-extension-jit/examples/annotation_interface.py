from __future__ import annotations

import pytorch_numba_extension_jit as pnex
import torch
from numba import cuda


# We can also place the inputs and outputs mixed
# (though it's not necessarily a good idea)
@pnex.jit(n_threads="inp.numel()", to_extension=True, verbose=True)
def foo(
    inp: pnex.In("f32", (None, 3)),
    out: pnex.Out("f32", ("inp.size(0)", "inp.size(1)", 2)),
    scalar: float,
):
    idx = cuda.grid(1)
    y, x = divmod(idx, out.shape[0])
    if x < out.shape[0] and y < out.shape[1]:
        out[x, y, 0] = inp[x, y]
        out[x, y, 1] = inp[x, y] * 4 + scalar


print("\nOperator:", foo, "of type", type(foo))

INP = torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3)
SCALAR = 3

# Note that the Out argument is skipped in the argument list:
OUT = foo(INP, SCALAR)

print("\nInput:")
print(INP)
print("\nOutput shape:", OUT.shape)
print("\nOutput:")
print(OUT)

torch.library.opcheck(foo, (INP, SCALAR))

compiled_foo = torch.compile(foo, fullgraph=True, mode="reduce-overhead")
compiled_foo(INP, SCALAR)
print("\nOutput after compilation:")
print(compiled_foo(INP, SCALAR))
