import math
from typing import Any, Literal, NamedTuple, Self

import torch
from torch import nn


class TorchLinearConv2D(nn.Module):
    """
    A utility that provides PyTorch Conv2D in a form compatible with `GenericConv2D`.
    """

    @staticmethod
    def forward(
        img: torch.Tensor,
        kernel: torch.Tensor,
        stride: int | tuple[int, int] = 1,
        padding: (
            int
            | tuple[int, int]
            | tuple[tuple[int, int], tuple[int, int]]
            | Literal["valid", "same"]
        ) = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ):
        if group_broadcasting:
            if kernel.shape[0] != 1:
                raise ValueError("Torch conv2d cannot broadcast groups with grp_o > 1")

            kernel = kernel.broadcast_to(
                (groups, kernel.shape[1], kernel.shape[2], kernel.shape[3])
            )
        if kind == "conv":
            kernel = kernel.flip((2, 3))

        dil_y, dil_x = as_tup2(dilation)
        (pad_y_beg, pad_y_end), (pad_x_beg, pad_x_end) = get_padding(
            padding, dil_x, dil_y, kernel.shape[2], kernel.shape[3]
        )

        if pad_y_beg != pad_y_end or pad_x_beg != pad_x_end:
            padded = torch.constant_pad_nd(
                img,
                # Yes, the padding really is in this order.
                (pad_x_beg, pad_x_end, pad_y_beg, pad_y_end),
            )
            return torch.nn.functional.conv2d(
                padded, kernel, stride=stride, dilation=dilation, groups=groups
            )

        return torch.nn.functional.conv2d(
            img,
            kernel,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=(pad_y_beg, pad_x_beg),
        )


class TorchMaxPool2D(nn.Module):
    """
    A utility that provides torch.nn.MaxPool2d with padding like `GenericConv2D`.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = None,
        padding: (
            int
            | tuple[int, int]
            | tuple[tuple[int, int], tuple[int, int]]
            | Literal["valid", "same"]
        ) = 0,
        dilation: int | tuple[int, int] = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation

    def forward(
        self,
        img: torch.Tensor,
    ):
        dil_y, dil_x = as_tup2(self.dilation)
        krn_y, krn_x = as_tup2(self.kernel_size)
        (pad_y_beg, pad_y_end), (pad_x_beg, pad_x_end) = get_padding(
            self.padding, dil_x, dil_y, krn_y, krn_x
        )

        if pad_y_beg == pad_y_end and pad_x_beg == pad_x_end:
            use_padding = (pad_y_beg, pad_x_beg)
        else:
            img = torch.constant_pad_nd(
                img,
                # Yes, the padding really is in this order.
                (pad_x_beg, pad_x_end, pad_y_beg, pad_y_end),
            )
            use_padding = 0

        return torch.nn.functional.max_pool2d(
            input=img,
            kernel_size=(krn_y, krn_x),
            stride=self.stride,
            padding=use_padding,
            dilation=(dil_y, dil_x),
            ceil_mode=False,
            return_indices=False,
        )


class ConvMeta(NamedTuple):
    img_cs: int  # Image channels
    img_ys: int  # Image y-size
    img_xs: int  # Image x-size
    krn_os: int  # Kernel output channels
    krn_cs: int  # Kernel input channels (== grp_i)
    krn_ys: int  # Kernel y-size
    krn_xs: int  # Kernel x-size
    out_cs: int  # Output image channels. Equal to krn_os, except when group broacasting
    out_ys: int  # Output image y-size
    out_xs: int  # Output image x-size
    str_y: int  # Stride in y-direction
    str_x: int  # Stride in x-direction
    pad_y_beg: int  # Padding at the start of y-axis
    pad_y_end: int  # Padding at the end of y-axis
    pad_x_beg: int  # Padding at the start of x-axis
    pad_x_end: int  # Padding at the end of x-axis
    dil_y: int  # Dilation in y-direction
    dil_x: int  # Dilation in x-direction
    groups: int  # Number of convolutional groups
    grp_i: int  # Size of a convolutional group in input channels (== krn_cs)
    grp_o: int  # Size of a convolutional group in kernel output channels
    group_broadcasting: bool  # Whether kernels should be broadcast along groups
    mirror_kernel: bool  # When true, the kernel is mirrored as in a convolution

    @classmethod
    def infer(
        cls,
        img_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        stride: int | tuple[int, int] = 1,
        padding: (
            int
            | tuple[int, int]
            | tuple[tuple[int, int], tuple[int, int]]
            | Literal["valid", "same"]
        ) = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ) -> Self:
        str_y, str_x = as_tup2(stride)
        dil_y, dil_x = as_tup2(dilation)

        # === Check params
        assert str_y > 0, f"{str_y=} must be positive"
        assert str_x > 0, f"{str_x=} must be positive"
        assert dil_x > 0, f"{dil_x=} must be positive"
        assert dil_y > 0, f"{dil_y=} must be positive"
        assert groups > 0, f"{groups=} must be positive"
        assert kind in ("conv", "corr"), f"Invalid {kind=}"
        # Negative padding is strange, but not a logic error.

        # === Check imgs
        assert len(img_shape) == 4, f"{img_shape=} needs to be BCHW"
        assert all(s > 0 for s in img_shape), f"Invalid {img_shape=}"
        img_bs, img_cs, img_ys, img_xs = img_shape
        assert img_cs % groups == 0, f"{img_cs=} not a multiple of {groups=}"
        grp_i = img_cs // groups
        # === Check kernels
        assert len(kernel_shape) == 4, f"{kernel_shape=} needs to be OIHW"
        assert all(s > 0 for s in kernel_shape), f"Invalid {kernel_shape=}"
        krn_os, krn_cs, krn_ys, krn_xs = kernel_shape
        assert krn_cs == grp_i, f"Groups: {krn_cs=} != {grp_i=}"
        if not group_broadcasting:
            # If we *are* group-broadcasting, then we effectively multiply
            # krn_os by params.groups
            assert krn_os % groups == 0, f"{krn_os=} not a multiple of {groups=}"
            grp_o = krn_os // groups
        else:
            grp_o = krn_os

        (pad_y_beg, pad_y_end), (pad_x_beg, pad_x_end) = get_padding(
            padding, dil_x, dil_y, krn_ys, krn_xs
        )

        out_xs = output_size(img_xs, krn_xs, str_x, pad_x_beg, pad_x_end, dil_x)
        out_ys = output_size(img_ys, krn_ys, str_y, pad_y_beg, pad_y_end, dil_y)

        out_cs = krn_os if not group_broadcasting else krn_os * groups
        assert out_xs > 0, f"Output image collapsed in x-direction: {out_xs=}"
        assert out_ys > 0, f"Output image collapsed in y-direction: {out_ys=}"
        shape = cls(
            img_cs=int(img_cs),
            img_ys=int(img_ys),
            img_xs=int(img_xs),
            krn_os=int(krn_os),
            krn_cs=int(krn_cs),
            krn_ys=int(krn_ys),
            krn_xs=int(krn_xs),
            out_cs=int(out_cs),
            out_ys=int(out_ys),
            out_xs=int(out_xs),
            str_x=int(str_x),
            str_y=int(str_y),
            pad_y_beg=int(pad_y_beg),
            pad_y_end=int(pad_y_end),
            pad_x_beg=int(pad_x_beg),
            pad_x_end=int(pad_x_end),
            dil_y=int(dil_y),
            dil_x=int(dil_x),
            groups=int(groups),
            grp_i=int(grp_i),
            grp_o=int(grp_o),
            group_broadcasting=bool(group_broadcasting),
            mirror_kernel=bool(kind == "conv"),
        )
        return shape

    def check_matches(
        self,
        img_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        stride: int | tuple[int, int] = 1,
        padding: (
            int
            | tuple[int, int]
            | tuple[tuple[int, int], tuple[int, int]]
            | Literal["valid", "same"]
        ) = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ):
        assert len(img_shape) == 4, "Image shape is not BCHW?"
        assert len(kernel_shape) == 4, "Kernel shape is not OIHW?"
        assert kind in ("conv", "corr"), f"Invalid {kind=}"

        str_y, str_x = as_tup2(stride)
        dil_y, dil_x = as_tup2(dilation)

        (pad_y_beg, pad_y_end), (pad_x_beg, pad_x_end) = get_padding(
            padding,
            dil_x,
            dil_y,
            kernel_shape[2],
            kernel_shape[3],
        )
        return (
            img_shape[1] == self.img_cs
            and img_shape[2] == self.img_ys
            and img_shape[3] == self.img_xs
            and kernel_shape[0] == self.krn_os
            and kernel_shape[1] == self.krn_cs
            and kernel_shape[2] == self.krn_ys
            and kernel_shape[3] == self.krn_xs
            and str_x == self.str_x
            and str_y == self.str_y
            and pad_y_beg == self.pad_y_beg
            and pad_y_end == self.pad_y_end
            and pad_x_beg == self.pad_x_beg
            and pad_x_end == self.pad_x_end
            and dil_y == self.dil_y
            and dil_x == self.dil_x
            and groups == self.groups
            and group_broadcasting == self.group_broadcasting
            and (kind == "conv") == self.mirror_kernel
        )

    def cache_id(self) -> str:
        return (
            f"meta"
            f"_{self.img_cs}_{self.img_ys}_{self.img_xs}"
            f"_{self.krn_os}_{self.krn_cs}_{self.krn_ys}_{self.krn_xs}"
            f"_{self.out_cs}_{self.out_ys}_{self.out_xs}"
            f"_{self.str_x}_{self.str_y}"
            f"_{self.pad_y_beg}_{self.pad_y_end}_{self.pad_x_beg}_{self.pad_x_end}"
            f"_{self.dil_x}_{self.dil_y}"
            f"_{self.groups}_{self.grp_i}_{self.grp_o}"
            f"_{int(self.group_broadcasting)}_{int(self.mirror_kernel)}"
        )


def as_tup2(v: int | tuple[Any] | tuple[Any, Any]):
    if isinstance(v, int):
        return v, v
    if len(v) == 1:
        return v[0], v[0]
    if len(v) == 2:
        return v

    raise ValueError(f"Invalid 2-tuple-like object {v=}")


def output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding_begin: int,
    padding_end: int,
    dilation: int,
):
    return math.floor(
        (input_size + padding_begin + padding_end - dilation * (kernel_size - 1) - 1)
        / stride
        + 1
    )


def get_padding(
    padding: int
    | tuple[int, int]
    | tuple[tuple[int, int], tuple[int, int]]
    | Literal["valid", "same"],
    dil_x: int,
    dil_y: int,
    krn_ys: int,
    krn_xs: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(padding, str):
        if padding == "valid":
            return (0, 0), (0, 0)
        if padding == "same":
            return (
                calculate_same(krn_ys, dil_y),
                calculate_same(krn_xs, dil_x),
            )

        raise ValueError(f"Invalid {padding=}")

    pad_y, pad_x = as_tup2(padding)
    return as_tup2(pad_y), as_tup2(pad_x)


def calculate_same(kernel_size: int, dilation: int) -> tuple[int, int]:
    zero_out = output_size(0, kernel_size, 1, 0, 0, dilation)
    padding_total = -zero_out
    assert padding_total % dilation == 0
    # We calculate padding in terms of dilated steps, to ensure that the output is
    # centred on the input for even kernel sizes.
    # i.e. for calculate_same(4, 2) we return (2, 4) not (3, 3)
    padding = padding_total // dilation
    # If the required padding is odd, we place the extra padding at the end, such that
    # the kernel centre is offset to the top-left.
    pad_beg = (padding // 2) * dilation
    pad_end = (padding // 2 + (padding % 2)) * dilation
    same_out = output_size(0, kernel_size, 1, pad_beg, pad_end, dilation)
    assert same_out == 0, f"calculate_same failed! {same_out=}"
    return pad_beg, pad_end


class LearnedKernel(nn.Module):
    """
    A utility that provides a fully learnable kernel compatible with `GenericConv2D`

    Parameters
    -------
    in_channels : int
        The number of input channels: the `I` in `OIHW`.
    out_channels : int
        The number of output channels: the `O` in `OIHW`.
    kernel_size : int
        The height `H` and width `W` of the kernel (rectangular kernels are not supported).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.normal_(self.kernel)

    def forward(self):
        return self.kernel


def plot_kernels(
    kernels: torch.Tensor, cut_zero=True, high_cut: float = 0.95, at_most: int = 4
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    dev_name = "(CPU)" if kernels.get_device() == -1 else "(CUDA)"
    kernels = kernels.detach().cpu()[:at_most, :at_most]
    high = torch.quantile(kernels.view(-1), torch.tensor([high_cut])).cpu().item()
    low = (
        torch.quantile(kernels.view(-1), torch.tensor([1 - high_cut])).cpu().item()
        if not cut_zero
        else 0
    )
    out_channels, in_channels, kernel_size, _ks = kernels.shape
    fig, axss = plt.subplots(
        out_channels, in_channels, sharex=True, sharey=True, layout="compressed"
    )
    if out_channels == 1:
        axss = (axss,)
    for o, axs in enumerate(axss):
        if in_channels == 1:
            axs = (axs,)
        for i, ax in enumerate(axs):
            ax: plt.Axes
            sns.heatmap(
                kernels[o, i],
                vmin=low,
                vmax=high,
                square=True,
                ax=ax,
                cbar=False,
            )
            ax.set_axis_off()
            ax.set_title(f"Sum {kernels[o, i].sum():.2f}", fontsize=6)
    plt.suptitle(f"Convolution kernels: {dev_name}\n (out-channels x in-channels)")
    plt.show()


def make_pos_grid(kernel_size: int, grid_at_end: bool = False) -> torch.Tensor:
    positions = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
    )
    return (
        (
            torch.cartesian_prod(positions, positions)
            .unsqueeze(1)  # Broadcast along out_channels
            .unsqueeze(2)  # Broadcast along in_channels
        )
        .movedim(0, -1 if grid_at_end else 0)
        .float()
    )
