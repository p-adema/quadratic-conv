from __future__ import annotations

import pytorch_numba_extension_jit as ptex_jit
import torch
from numba import cuda


@ptex_jit.jit(
    "hoooo",
    [
        ptex_jit.InputTensor("ree", "f32", (None, 3)),
        ptex_jit.OutputTensor("rawr", "f32", (("ree", 0), ("ree", 1), 2)),
    ],
    n_threads="rawr",
    compile_extension=False,
)
def hoo(ree, rawr):
    idx = cuda.grid(1)
    y, x = divmod(idx, rawr.shape[0])
    if x < rawr.shape[0] and y < rawr.shape[1]:
        rawr[x, y, 0] = ree[x, y]
        rawr[x, y, 1] = ree[x, y] * 4


print(hoo)

a = torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3)
out1 = hoo(a)
print(a)
print(out1)
print(type(hoo))
#
#
# @ptex_jit.jit(
#     "hoooo",
#     [
#         ptex_jit.InputTensor("ree", "f32", (None, 3)),
#         ptex_jit.OutputTensor("rawr", "f32", (("ree", 0), ("ree", 1), 2)),
#     ],
#     n_threads_expr="rawr",
# )
# def hoo2(ree, rawr):
#     idx = cuda.grid(1)
#     y, x = divmod(idx, rawr.shape[0])
#     if x < rawr.shape[0] and y < rawr.shape[1]:
#         rawr[x, y, 0] = ree[x, y]
#         rawr[x, y, 1] = ree[x, y] * 20
#
#
# a = torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3)
# out2 = hoo2(a)
# print(a)
# print(out2)
#
# a = torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3)
# out3 = hoo(a)
# print(a)
# print(out3)
