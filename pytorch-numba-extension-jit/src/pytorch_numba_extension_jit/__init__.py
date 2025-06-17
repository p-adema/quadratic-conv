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

.. note:: Correct CUDA toolkit versions
    When this package is installed via Pip, a version of nvidia-cuda-nvcc and
    nvidia-cuda-runtime will likely be installed.
    However, depending on the weather outside, this version may not be correct.
    As such, if you experience issues during compilation (especially if you see the
    error `cuModuleLoadData(&cuModule, ptx) failed with error
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), then it may be worth verifying your
    installation.
    This can be done by running `nvidia-smi` to find your CUDA version, and then
    `pip list` to find the currently installed versions of the
    relevant NVIDIA libraries.
    These libraries should begin with your CUDA version, e.g. for CUDA 12.8 the expected
    output might look like:

        $ pip list | grep nvidia-cuda-
        nvidia-cuda-cupti-cu12    12.8.57
        nvidia-cuda-nvcc-cu12     12.8.61
        nvidia-cuda-nvrtc-cu12    12.8.61
        nvidia-cuda-runtime-cu12  12.8.57
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
