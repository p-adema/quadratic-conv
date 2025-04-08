import math
import typing

import torch
from torch import nn


class TropicalConv2D(nn.Module):
    """A convolution in the tropical-max or tropical-min subfield,
    also known as dilation or erosion"""

    def __init__(self, *, is_max: bool, softmax_temp: float | None = None):
        super().__init__()
        self.is_max = is_max
        self.softmax_temp = softmax_temp

    @staticmethod
    def _output_size(
        input_size: int, kernel_size: int, stride: int, padding: int, dilation: int
    ):
        return math.floor(
            (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )

    def forward(
        self,
        x: torch.Tensor,
        kernel: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        assert len(kernel.shape) == 4, f"Kernel shape seems off: {kernel.shape=}"
        out_channels, kernel_in_ch, kernel_size, _ks = kernel.shape
        assert kernel_size == _ks
        assert kernel_in_ch == 1, "Max/min kernel should take one channel input"
        xdims = len(x.shape)
        if xdims == 3:
            x = x.unsqueeze(0)
        elif xdims < 3 or xdims > 5:
            raise ValueError(f"Input shape seems off: {x.shape=}")
        inp_batch, img_channels, inp_x, inp_y = x.shape
        assert img_channels == out_channels, f"{out_channels=} != {img_channels=}"
        new_x = self._output_size(inp_x, kernel_size, stride, padding, dilation)
        new_y = self._output_size(inp_y, kernel_size, stride, padding, dilation)
        assert new_x > 0, f"{new_x=}"
        assert new_y > 0, f"{new_y=}"
        neutral = -torch.inf if self.is_max else torch.inf
        x_pad = torch.constant_pad_nd(x, (padding, padding, padding, padding), neutral)

        stencils = nn.functional.unfold(
            x_pad, (kernel_size, kernel_size), dilation, 0, stride
        ).reshape(inp_batch, out_channels, kernel_size * kernel_size, -1)
        weights = kernel.view(1, out_channels, kernel_size * kernel_size, 1)
        assert weights.shape[2] == stencils.shape[2]

        if self.is_max:
            # [batch, o, k*k, y'*x']
            vals = stencils + weights
            # Pad with a neutral at the start, so that if all elements are
            # neutral, then the gradient flows into the void instead of into
            # the top-left element
            vals = torch.constant_pad_nd(vals, (0, 0, 1, 0), neutral)
            if self.softmax_temp is None:
                reduced = torch.max(vals, dim=2).values
            else:
                # I looked at using logsumexp here, but the limits of logsumexp are
                # t=0 gives val=inf, t=inf gives val=max
                # while the limits in temperature of softmax * x are
                # t->0 gives val=max, t->inf gives val=mean

                # dot product
                reduced = torch.einsum(
                    "boKs,boKs->bos",
                    torch.softmax(vals / self.softmax_temp, dim=2),
                    vals,
                )
        else:
            vals = stencils - weights
            # Pad with a neutral at the start, so that if all elements are
            # neutral, then the gradient flows into the void instead of into
            # the top-left element
            vals = torch.constant_pad_nd(vals, (0, 0, 1, 0), neutral)
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

    if typing.TYPE_CHECKING:
        __call__ = forward
