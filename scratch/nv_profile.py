import torch
from pytorch_semifield_conv import SelectSemifield

torch.set_float32_matmul_precision("high")
torch.manual_seed(0)
ex_data = torch.rand((1024, 6, 28, 28), device="cuda")
ex_kernel = torch.rand((6, 1, 11, 11), device="cuda")


print("GLB IMPL")
op = SelectSemifield.tropical_max().lazy_fixed(impl="glb", to_extension=False)

g_inp = ex_data.clone().requires_grad_(True)
g_krn = ex_kernel.clone().requires_grad_(True)
g_tangent = torch.randn_like(op(ex_data, ex_kernel, groups=6, padding="same", stride=2))

print(torch.max_pool2d(g_inp, 11, 2, 5).backward(g_tangent))


res1 = op(g_inp, g_krn, groups=6, padding="same", stride=2)
res1.backward(g_tangent)
torch.cuda.synchronize()


print("EXTENSION IMPL")
op2 = SelectSemifield.tropical_max().lazy_fixed(impl="glb", to_extension=True)

torch.max_pool2d(g_inp, 11, 2, 5)
op(g_inp, g_krn, groups=6, padding="same", stride=2)
res2 = op2(g_inp, g_krn, groups=6, padding="same", stride=2)

res2.backward(g_tangent)
torch.cuda.synchronize()

torch.testing.assert_close(res1, res2)
