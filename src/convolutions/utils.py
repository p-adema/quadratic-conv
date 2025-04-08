import torch
from torch import nn


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


class LinearConv2D(nn.Module):
    """A convolution in the linear field"""

    forward = staticmethod(nn.functional.conv2d)
