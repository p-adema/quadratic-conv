"""
This package is aimed at simplifying the usage of
[Numba-CUDA](https://github.com/NVIDIA/numba-cuda) kernels within projects
using the [PyTorch](https://pytorch.org) deep learning framework.

By annotating a function written in the style of a
[Numba-CUDA kernel](https://nvidia.github.io/numba-cuda/user/kernels.html)
with type hints from this package, `jit` can generate
[PyTorch Custom Operator](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
bindings that allow the kernel to be used within a traced (e.g. `torch.compile`)
environment.
Furthermore, by setting `to_extension=True`, the kernel can also be transformed into
PTX, and C++ code can be generated to invoke the kernel with minimal overhead.

As a toy example, consider the task of creating a copy of a 1D array:

>>> import pytorch_numba_extension_jit as pnex
>>> @pnex.jit(n_threads="a.numel()")
... def copy(
...     a: pnex.In(dtype="f32", shape=(None,)),
...     result: pnex.Out(dtype="f32", shape="a"),
... ):
...     x = cuda.grid(1)
...     if x < a.shape[0]:
...         result[x] = a[x]
>>> A = torch.arange(5, dtype=torch.float32, device="cuda")
>>> copy(A)
tensor([0., 1., 2., 3., 4.], device='cuda:0')

For more examples of usage, see `jit` and the examples directory of the project.
"""

from ._typehint_interface import In, InMut, Out, Scalar, Unused, jit

__all__ = [  # noqa: RUF022
    "jit",
    "In",
    "InMut",
    "Out",
    "Scalar",
    "Unused",
]
