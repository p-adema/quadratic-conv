import math

import numba
import torch
from numba import cuda

STRIDE = 2
WINDOW_SIZE = 7
BLOCK_SIZE = 128
CACHE_SIZE = (BLOCK_SIZE - 1) * STRIDE + WINDOW_SIZE
FILL_STEPS = math.ceil(CACHE_SIZE / BLOCK_SIZE)


# noinspection PyArgumentList,PyTypeChecker
@cuda.jit("void(float32[:], float32[:])")
def pool_basic_select(vals, out):
    ox = cuda.grid(1)
    if ox > out.size:
        return

    acc = numba.float32(-100.0)

    for ts in range(WINDOW_SIZE):
        begin_x = (
            cuda.blockIdx.x * BLOCK_SIZE
            + ((cuda.threadIdx.x + ts) % BLOCK_SIZE) * STRIDE * STRIDE
        )
        for x in range(begin_x, begin_x + WINDOW_SIZE):
            val = cuda.selp(x < vals.size, vals[x], numba.float32(-1000))
            val_2 = val + numba.float32(1.0)
            acc = cuda.selp(val_2 > acc, val_2, acc)

    out[ox] = acc


# noinspection PyArgumentList,PyTypeChecker
@cuda.jit("void(float32[:], float32[:])")
def pool_shmm(inp, out):
    ox = cuda.grid(1)

    inp_cache = cuda.shared.array(CACHE_SIZE, numba.float32)
    acc = numba.float32(-100.0)

    cache_pos = cuda.threadIdx.x
    block_begin_x = cuda.blockIdx.x * BLOCK_SIZE * STRIDE
    for _ in range(FILL_STEPS - 1):
        x = block_begin_x + cache_pos
        inp_cache[cache_pos] = cuda.selp(x < inp.size, inp[x], numba.float32(-200))

        cache_pos += BLOCK_SIZE

    if cache_pos < CACHE_SIZE:
        x = block_begin_x + cache_pos
        inp_cache[cache_pos] = cuda.selp(x < inp.size, inp[x], numba.float32(-200))

    cuda.syncthreads()

    if ox >= out.shape[-1]:
        return

    for ts in range(WINDOW_SIZE):
        cache_begin_x = ((cuda.threadIdx.x + ts) % BLOCK_SIZE) * STRIDE
        step = 0
        for _ in range(WINDOW_SIZE):
            val = inp_cache[cache_begin_x + step]
            val_2 = val + numba.float32(1.0)
            if val_2 > acc:
                acc = val_2
            # acc = cuda.selp(val_2 > acc, val_2, acc)
            step += 1

    out[ox] = acc


IN_SIZE = 16_000_000
OUT_SIZE = (IN_SIZE - 1 - (WINDOW_SIZE - 1)) // STRIDE + 1

block_size = BLOCK_SIZE
n_blocks = (OUT_SIZE + block_size - 1) // block_size

torch.manual_seed(0)
lg_vals_1d = torch.rand(IN_SIZE, device="cuda")
check_out = (
    torch.max_pool1d(lg_vals_1d.unsqueeze(0), WINDOW_SIZE, STRIDE).add(1).squeeze()
)

for pool in (pool_basic_select, pool_shmm):
    lg_out = torch.empty(OUT_SIZE, device="cuda")
    pool[n_blocks, block_size](lg_vals_1d, lg_out)
    # torch.testing.assert_close(check_out, lg_out)
