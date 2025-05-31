# Pytorch N-Dimensional Semifield Convolutions

[Documentation](https://p-adema.github.io/quadratic-conv/pytorch-nd-semiconv/docs.html)

PyTorch provides efficient implementations of linear convolution operators, as well
as max-pooling operators. Both of these operators can be considered a kind of
semifield convolution, where the semifield defines what 'addition' and 'multiplication'
mean.

However, there are other semifields that we may wish to use than the linear.
As such, this package aims to simplify the process of implementing new semifield
convolutions, as well as providing definitions for standard semifields.

These new semifields can be defined using PyTorch broadcasting operators using
[
BroadcastSemifield](https://p-adema.github.io/quadratic-conv/pytorch-nd-semiconv/docs.html#pytorch_nd_semiconv.BroadcastSemifield),
or using [
SelectSemifield](https://p-adema.github.io/quadratic-conv/pytorch-nd-semiconv/docs.html#pytorch_nd_semiconv.SelectSemifield)
/ [
SubtractSemifield](https://p-adema.github.io/quadratic-conv/pytorch-nd-semiconv/docs.html#pytorch_nd_semiconv.SubtractSemifield)
in the cases where no appropriate PyTorch operator exists.

The implementations, while not as optimised as the base PyTorch versions,
have decent performance. [
BroadcastSemifield](https://p-adema.github.io/quadratic-conv/pytorch-nd-semiconv/docs.html#pytorch_nd_semiconv.BroadcastSemifield)
relies on chaining optimised
PyTorch operators but suffers from higher memory usage, while [
SelectSemifield](https://p-adema.github.io/quadratic-conv/pytorch-nd-semiconv/docs.html#pytorch_nd_semiconv.SelectSemifield)
/ [
SubtractSemifield](https://p-adema.github.io/quadratic-conv/pytorch-nd-semiconv/docs.html#pytorch_nd_semiconv.SubtractSemifield)
are custom CUDA operators, optionally JIT compiled into PyTorch
extensions using my other library [
pytorch-numba-extension-jit](https://p-adema.github.io/quadratic-conv/pytorch-numba-extension-jit/docs.html).

Finally, all three implementations work in arbitrary dimensionality: they support 1D,
2D or any dimensionality of inputs and kernels (though the example kernels provided
are only 2D).