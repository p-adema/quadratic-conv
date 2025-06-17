import torch
from pytorch_nd_semiconv import SelectSemifield

torch.set_float32_matmul_precision("high")
torch.manual_seed(0)
ex_data = torch.rand((1024, 6, 28, 28), device="cuda")
ex_kernel = torch.rand((6, 1, 11, 11), device="cuda")


print("GLB IMPL")
op = SelectSemifield.tropical_max().lazy_fixed(to_extension=False)

g_inp = ex_data.clone().requires_grad_(True)
g_krn = ex_kernel.clone().requires_grad_(True)
g_tangent = torch.randn_like(op(ex_data, ex_kernel, groups=6, padding="same", stride=2))

print(torch.max_pool2d(g_inp, 11, 2, 5).backward(g_tangent))


res1 = op(g_inp, g_krn, groups=6, padding="same", stride=2)
res1.backward(g_tangent)
torch.cuda.synchronize()

# torch.testing.assert_close(res1, res2)
