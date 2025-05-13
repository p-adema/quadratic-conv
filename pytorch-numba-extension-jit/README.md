# Pytorch-Numba Extension JIT

[Documentation](https://p-adema.github.io/quadratic-conv/pytorch-numba-extension-jit/html/pytorch_numba_extension_jit/index.html)

Writing custom CUDA operators in C and CPP can make certain operations significantly
more efficient, but requires setting up a full C++ project and involves a great deal of
boilerplate.
Writing CUDA kernels using `numba-cuda` is significantly easier, but incurs overhead on
every call, and still requires some boilerplate to integrate with the tracing systems
that underlie `torch.compile`.

However, many of the CUDA kernels that would be used for deep learning are relatively
similar (read from a set of input arrays, write to output arrays).
As such, most of the boilerplate and binding code for C++ extensions
could be generated automatically.

This project aims to do exactly that: take a Python function in the form of a Numba CUDA
kernel, along with some type annotations, and compile a user-friendly and
highly-performant PyTorch C++ extension.

For an example usage of this package, see my other
package [pytorch-semifield-conv](https://p-adema.github.io/quadratic-conv/pytorch-semifield-conv/html/pytorch_semifield_conv/index.html)