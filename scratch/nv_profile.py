import torch
from pytorch_semifield_conv import SelectSemifield

torch.set_float32_matmul_precision("high")
torch.manual_seed(0)
ex_data = torch.rand((1024, 6, 28, 28), device="cuda")
ex_kernel = torch.rand((6, 1, 11, 11), device="cuda")

op = SelectSemifield.tropical_max().lazy_fixed(thread_block_size=128)

g_inp = ex_data.clone().requires_grad_(True)
g_krn = ex_kernel.clone().requires_grad_(True)
g_tangent = torch.randn_like(op(ex_data, ex_kernel, groups=6, padding="same", stride=2))

print(torch.max_pool2d(g_inp, 11, 2, 5).backward(g_tangent))


def run_one():
    res = op(g_inp, g_krn)
    res.backward(g_tangent)
    torch.cuda.synchronize()


run_one()
