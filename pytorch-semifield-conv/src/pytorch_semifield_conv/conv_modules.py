from __future__ import annotations

from typing import Literal

from torch import nn


class GenericConv2D(nn.Module):
    def __init__(
        self,
        kernel: nn.Module,
        conv: nn.Module,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel = kernel
        self.conv = conv
        self.groups = groups
        self.group_broadcasting = group_broadcasting
        self.kind = kind

        # Since these are custom arguments, we only want to pass them if they differ
        # from the default values (otherwise, they may be unexpected)
        self.kwargs = {}
        if self.group_broadcasting:
            self.kwargs["group_broadcasting"] = True
        if self.kind == "corr":
            self.kwargs["kind"] = "corr"

    def forward(self, x):
        return self.conv(
            x,
            self.kernel(),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
            groups=self.groups,
            **self.kwargs,
        )

    def extra_repr(self) -> str:
        res = []
        if self.padding:
            res.append(f"padding={self.padding}")
        if self.stride != 1:
            res.append(f"stride={self.stride}")
        if self.dilation != 1:
            res.append(f"dilation={self.dilation}")
        if self.groups != 1:
            res.append(f"groups={self.groups}")
        if self.group_broadcasting:
            res.append("group_broadcasting=True")
        if self.kind == "corr":
            res.append("kind=corr")

        return ", ".join(res)


class Closing2D(nn.Module):
    def __init__(
        self,
        kernel: nn.Module,
        conv_dilation: nn.Module,
        conv_erosion: nn.Module,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel = kernel
        self.conv_dilation = conv_dilation
        self.conv_erosion = conv_erosion
        self.groups = groups
        self.group_broadcasting = group_broadcasting
        self.kind = kind

        # Since these are custom arguments, we only want to pass them if they differ
        # from the default values (otherwise, they may be unexpected)
        self.kwargs = {}
        if self.group_broadcasting:
            self.kwargs["group_broadcasting"] = True
        if self.kind == "corr":
            self.kwargs["kind"] = "corr"

    def forward(self, x):
        kernel = self.kernel()
        dilated = self.conv_dilation(
            x,
            kernel,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
            groups=self.groups,
            **self.kwargs,
        )
        closed = self.conv_erosion(
            dilated,
            kernel,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
            groups=self.groups,
            **self.kwargs,
        )
        return closed

    extra_repr = GenericConv2D.extra_repr
