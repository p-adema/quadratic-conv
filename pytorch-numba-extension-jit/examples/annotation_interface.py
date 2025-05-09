from __future__ import annotations

import pytorch_numba_extension_jit as ptex
import torch
from numba import cuda


@ptex.jit(n_threads="rawr", to_extension=True, verbose=True)
def hoo(
    ree: ptex.In("f32", (None, 3)),
    rawr: ptex.Out("f32", ("ree", "ree.shape[1]", 2)),
    roo: float,
):
    idx = cuda.grid(1)
    y, x = divmod(idx, rawr.shape[0])
    if x < rawr.shape[0] and y < rawr.shape[1]:
        rawr[x, y, 0] = ree[x, y]
        rawr[x, y, 1] = ree[x, y] * 4 + roo


print(hoo)

a = torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3)
out1 = hoo(a, 3)
print(a)
print(out1)
print(type(hoo))
torch.library.opcheck(hoo, (a, 3))
comp_a = torch.compile(hoo, fullgraph=True)
print("Compiled:")
print(comp_a(a, 3).shape)
print(comp_a(a, 3))
