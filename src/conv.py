import math

import torch
from torch import nn


class LinearConv2D(nn.Module):
    """A convolution in the linear field"""

    forward = staticmethod(nn.functional.conv2d)


class TropicalConv2D(nn.Module):
    """A convolution in the tropical-max or tropical-min subfield, also known as dilation or erosion"""

    def __init__(self, *, is_max: bool, softmax_temp: float | None = None):
        super().__init__()
        self.is_max = is_max
        self.softmax_temp = softmax_temp

    @staticmethod
    def _output_size(
        input_size: int, kernel_size: int, dilation: int, padding: int, stride: int
    ):
        return math.floor(
            (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )

    def forward(
        self,
        x: torch.Tensor,
        kernel: torch.Tensor,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ):
        assert len(kernel.shape) == 4, f"Kernel shape seems off: {kernel.shape=}"
        out_channels, in_channels, kernel_size, _ks = kernel.shape
        assert kernel_size == _ks
        xdims = len(x.shape)
        if xdims == 3:
            x = x.unsqueeze(0)
        elif xdims < 3 or xdims > 5:
            raise ValueError(f"Input shape seems off: {x.shape=}")
        inp_batch, _in_ch, inp_x, inp_y = x.shape
        assert in_channels == _in_ch
        new_x = self._output_size(inp_x, kernel_size, dilation, padding, stride)
        new_y = self._output_size(inp_y, kernel_size, dilation, padding, stride)
        assert new_x > 0 and new_y > 0

        stencils = nn.functional.unfold(
            x, (kernel_size, kernel_size), dilation, padding, stride
        ).unsqueeze(1)
        weights = kernel.view(
            1, out_channels, in_channels * kernel_size * kernel_size, 1
        )
        assert weights.shape[2] == stencils.shape[2]

        if self.is_max:
            # [batch, o, i*k*k, y'*x']
            vals = stencils - weights
            if self.softmax_temp is None:
                reduced = torch.max(vals, dim=2).values
            else:
                # I looked at using logsumexp here, but the limits in temperature of logsumexp are
                # t=0 gives val=inf, t=inf gives val=max
                # while the limits in temperature of softmax * x are
                # t=0 gives val=mean, t=inf gives val=max

                # dot product
                reduced = torch.einsum(
                    "boKs,boKs->bos",
                    torch.softmax(vals / self.softmax_temp, dim=2),
                    vals,
                )
        else:
            vals = stencils + weights
            if self.softmax_temp is None:
                reduced = torch.min(vals, dim=2).values
            else:
                reduced = torch.einsum(
                    "boKs,boKs->bos",
                    torch.softmin(vals / self.softmax_temp, dim=2),
                    vals,
                )

        assert reduced.shape == (x.shape[0], out_channels, new_y * new_x), (
            f"{reduced.shape=} != {(x.shape[0], out_channels, new_y * new_x)=}"
        )
        res = reduced.reshape(-1, out_channels, new_y, new_x)
        if xdims == 3:
            res = res.squeeze(0)
        return res
