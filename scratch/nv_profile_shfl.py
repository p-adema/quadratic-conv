import math

import numba
import numpy as np
import torch
from numba import cuda

STRIDE = 2
WINDOW_SIZE = 12


# noinspection PyArgumentList
@cuda.jit("void(float32[:], float32[:])")
def pool_basic(vals, out):
    idx = cuda.grid(1)
    if idx > out.size:
        return

    ox = idx

    begin_x = ox * STRIDE
    acc = numba.float32(-100.0)
    for x in range(begin_x, begin_x + WINDOW_SIZE):
        if x >= vals.size:
            continue
        val = vals[x]
        val_2 = val + numba.float32(1.0)
        if val_2 > acc:
            acc = val_2

    out[ox] = acc


# def _get_deltas(window_size):
#     res = []
#     acc = 1
#     while acc < window_size:
#         res.append(acc)
#         acc *= 2
#     return np.array(res[::-1], dtype=np.uint64)
#
#
# DELTAS = _get_deltas(WINDOW_SIZE)
# print(f"{DELTAS=}")


# noinspection PyArgumentList
# @cuda.jit("void(float32[:], float32[:])")
# def pool_strided(vals, out):
#     idx = cuda.grid(1)
#     num_warps, warp_pos = divmod(idx, 32)
#
#     x = num_warps * (32 // WINDOW_SIZE * WINDOW_SIZE) + warp_pos
#
#     ox = x // STRIDE
#     window_pos = warp_pos % WINDOW_SIZE
#
#     does_output = (
#         (window_pos == 0)
#         and (warp_pos < 32 // WINDOW_SIZE * WINDOW_SIZE)
#         and ox < out.shape[-1]
#     )
#     x_valid = x < vals.shape[-1]
#
#     arr_val = cuda.selp(x_valid, vals[x], numba.float32(-1000))
#     val = arr_val + numba.float32(1.0)
#
#     # send_val = cuda.selp(window_pos >= 4, val, numba.float32(-1000))
#     # other_val = cuda.shfl_down_sync(FULL_MASK, send_val, 4)
#     # if other_val > val:
#     #     val = other_val
#
#     # send_val = cuda.selp(window_pos >= 2, val, numba.float32(-1000))
#     # other_val = cuda.shfl_down_sync(FULL_MASK, send_val, 2)
#     # if other_val > val:
#     #     val = other_val
#
#     send_val = cuda.selp(window_pos >= 1, val, numba.float32(-1000))
#     other_val = cuda.shfl_down_sync(FULL_MASK, send_val, 1)
#     if other_val > val:
#         val = other_val
#
#     if does_output:
#         out[ox] = val


# noinspection PyArgumentList
@cuda.jit("void(float32[:], float32[:])")
def pool_strided(vals, out):
    idx = cuda.grid(1)
    if idx > out.size * STRIDE:
        return

    ox, stride_step = divmod(idx, STRIDE)
    stride_leader = stride_step == 0
    begin_x = ox * STRIDE
    acc = numba.float32(-100.0)
    for i in range(math.ceil(WINDOW_SIZE / STRIDE)):
        step = stride_step + i * STRIDE
        window_valid = step < WINDOW_SIZE
        x = begin_x + step
        x_valid = x < vals.shape[-1]
        val = cuda.selp(window_valid and x_valid, vals[x], numba.float32(-1000))
        val_2 = val + numba.float32(1.0)
        other_val_2 = cuda.shfl_down_sync(numba.uint32(0xFFFF), val_2, 1)

        if other_val_2 > val_2:
            val_2 = other_val_2

        if val_2 > acc:
            acc = val_2

    if stride_leader:
        out[ox] = acc


IN_SIZE = 16_000_000
OUT_SIZE = (IN_SIZE - 1 - (WINDOW_SIZE - 1)) // STRIDE + 1
BLOCK_SIZE = 256

torch.manual_seed(0)
lg_vals_1d = torch.rand(IN_SIZE, device="cuda")

check_out = (
    torch.max_pool1d(lg_vals_1d.unsqueeze(0), WINDOW_SIZE, STRIDE).add(1).squeeze()
)

lg_out_1 = torch.empty(OUT_SIZE, device="cuda")
n_blocks = (OUT_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
pool_basic[n_blocks, BLOCK_SIZE](lg_vals_1d, lg_out_1)

n_blocks = (IN_SIZE + 0 + BLOCK_SIZE - 1) // (
    (BLOCK_SIZE // 32) * (32 // WINDOW_SIZE * WINDOW_SIZE)
)
lg_out_2 = torch.empty(OUT_SIZE, device="cuda")
pool_strided[n_blocks, BLOCK_SIZE](lg_vals_1d, lg_out_2)

# print("Part close:")
# torch.testing.assert_close(check_out[::2], lg_out_1[::2])
# torch.testing.assert_close(check_out[::2], lg_out_2[::2])
print("Full close:")
torch.testing.assert_close(check_out, lg_out_1)
torch.testing.assert_close(check_out, lg_out_2)
