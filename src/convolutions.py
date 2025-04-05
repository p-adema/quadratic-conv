import math
import typing

import torch
from torch import nn


class LinearConv2D(nn.Module):
    """A convolution in the linear field"""

    forward = staticmethod(nn.functional.conv2d)


class TropicalConv2D(nn.Module):
    """A convolution in the tropical-max or tropical-min subfield,
    also known as dilation or erosion"""

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
        new_x = self._output_size(inp_x, kernel_size, dilation, padding, stride)
        new_y = self._output_size(inp_y, kernel_size, dilation, padding, stride)
        assert new_x > 0, f"{new_x=}"
        assert new_y > 0, f"{new_y=}"

        stencils = nn.functional.unfold(
            x, (kernel_size, kernel_size), dilation, padding, stride
        ).reshape(inp_batch, out_channels, kernel_size * kernel_size, -1)
        weights = kernel.view(1, out_channels, kernel_size * kernel_size, 1)
        assert weights.shape[2] == stencils.shape[2]

        if self.is_max:
            # [batch, o, k*k, y'*x']
            vals = stencils + weights
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


class GenericConv2D(nn.Module):
    def __init__(
        self,
        kernel: nn.Module,
        conv: nn.Module,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel = kernel
        self.conv = conv

    def forward(self, x):
        return self.conv(
            x,
            self.kernel(),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )


class CoerceImage4D(nn.Module):
    def __init__(self, img_channels: int):
        super().__init__()
        self.img_channels = img_channels

    def forward(self, x: torch.Tensor):
        x_dims = len(x.shape)
        if x_dims > 4:
            raise ValueError(f"Probably invalid image dims {x.shape=}")
        if x_dims == 4:
            assert x.shape[1] == self.img_channels, f"Invalid channels {x.shape=}"
        elif x_dims == 3:
            if x.shape[0] == self.img_channels:
                # Probably a single image, unbatched
                x = x.unsqueeze(0)
            else:
                # Must be a single-channel image, batched
                assert self.img_channels == 1, f"Strange image {x.shape=}"
                x = x.unsqueeze(1)
        else:
            assert x_dims == 2, "not an image?"
            # Unbatched, single channel image
            assert self.img_channels == 1, f"Missing channels {x.shape=}"
            x = x.unsqueeze(0).unsqueeze(1)

        return x
