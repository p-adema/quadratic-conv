import math
from typing import Literal, NamedTuple, Self

import torch
from torch import nn

from ._unfold_view import _as_tup_n


class TorchLinearConv2D(nn.Module):
    """
    A utility that provides PyTorch Conv2D in a form compatible with `GenericConv`.
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

        dilation = _as_tup_n(dilation, 2)
        (pad_y_beg, pad_y_end), (pad_x_beg, pad_x_end) = get_padding(
            padding, 2, dilation, kernel.shape[2:]
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
    A utility that provides torch.nn.MaxPool2d with padding like `GenericConv`.
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
        dilation = _as_tup_n(self.dilation, 2)
        krn_spatial = _as_tup_n(self.kernel_size, 2)
        (pad_y_beg, pad_y_end), (pad_x_beg, pad_x_end) = get_padding(
            self.padding, 2, dilation, krn_spatial
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
            kernel_size=krn_spatial,
            stride=self.stride,
            padding=use_padding,
            dilation=dilation,
            ceil_mode=False,
            return_indices=False,
        )


class ConvMeta(NamedTuple):
    """TODO: short docs"""

    ndim: int  # Number of spatial dimensions
    img_cs: int  # Image channels
    img_spatial: tuple[int, ...]  # Image ...Z/Y/X
    krn_os: int  # Kernel output channels
    krn_cs: int  # Kernel input channels (== grp_i)
    krn_spatial: tuple[int, ...]  # Kernel ...Z/Y/X
    out_cs: int  # Output image channels. Equal to krn_os, except when group broacasting
    out_spatial: tuple[int, ...]  # Output ...Z/Y/X
    stride: tuple[int, ...]  # Stride ...Z/Y/X
    pad_begs: tuple[int, ...]  # Pad beginning ...Z/Y/X
    pad_ends: tuple[int, ...]  # Pad beginning ...Z/Y/X
    dilation: tuple[int, ...]  # Dilation ...Z/Y/X
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
        stride: int | tuple[int, ...] = 1,
        padding: (
            int
            | tuple[int, ...]
            | tuple[tuple[int, int], ...]
            | Literal["valid", "same"]
        ) = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ) -> Self:
        assert len(img_shape) == len(kernel_shape), (
            f"{img_shape=} but {kernel_shape=}: For an ND convolution,"
            f" the image and kernel must have the same number of axes.\n"
            f"The first two axes of img_shape must be Batch and Channels, while"
            f"the first two axes of kernel_shape must be Out and In.\n"
            f"After these initial two axes come the spatial axes in order ...Z, Y, X."
        )
        assert len(img_shape) > 2, (
            "An image in an ND-convolution must have Batch and Channel axes, as well as"
            " at least one spatial axis (so at least three axes in total)."
        )
        ndim = len(img_shape) - 2
        stride = _as_tup_n(stride, ndim)
        dilation = _as_tup_n(dilation, ndim)

        # === Check params
        assert all(s > 0 for s in stride), f"{stride=} must be positive"
        assert all(d > 0 for d in dilation), f"{dilation=} must be positive"
        assert groups > 0, f"{groups=} must be positive"
        assert kind in ("conv", "corr"), f"Invalid {kind=}"
        # Negative padding is strange, but not necessarily a logic error.

        # === Check imgs
        assert all(s > 0 for s in img_shape), f"Nonpositive? {img_shape=}"
        img_bs, img_cs, *img_spatial = img_shape
        assert img_cs % groups == 0, f"{img_cs=} not a multiple of {groups=}"
        grp_i = img_cs // groups
        # === Check kernels
        assert all(s > 0 for s in kernel_shape), f"Nonpositive? {kernel_shape=}"
        krn_os, krn_cs, *krn_spatial = kernel_shape
        assert krn_cs == grp_i, f"Groups: {krn_cs=} != {grp_i=}"
        if not group_broadcasting:
            # If we *are* group-broadcasting, then we effectively multiply
            # krn_os by params.groups
            assert krn_os % groups == 0, f"{krn_os=} not a multiple of {groups=}"
            grp_o = krn_os // groups
        else:
            grp_o = krn_os

        padding = get_padding(padding, ndim, dilation, krn_spatial)
        out_spatial = tuple(
            output_size(i, k, s, pb, pe, d)
            for i, k, s, (pb, pe), d in zip(
                img_spatial, krn_spatial, stride, padding, dilation, strict=True
            )
        )
        assert all(o > 0 for o in out_spatial), f"Output image collapsed: {out_spatial}"

        out_cs = krn_os if not group_broadcasting else krn_os * groups

        # We need to do explicit conversions, because otherwise the FakeTensor Symints
        # will bleed through and crash the Numba compiler
        return cls(
            ndim=int(ndim),
            img_cs=int(img_cs),
            img_spatial=tuple(map(int, img_spatial)),
            krn_os=int(krn_os),
            krn_cs=int(krn_cs),
            krn_spatial=tuple(map(int, krn_spatial)),
            out_cs=int(out_cs),
            out_spatial=tuple(map(int, out_spatial)),
            stride=tuple(map(int, stride)),
            pad_begs=tuple(int(b) for b, _ in padding),
            pad_ends=tuple(int(e) for _, e in padding),
            dilation=tuple(map(int, dilation)),
            groups=int(groups),
            grp_i=int(grp_i),
            grp_o=int(grp_o),
            group_broadcasting=bool(group_broadcasting),
            mirror_kernel=bool(kind == "conv"),
        )

    def check_matches(
        self,
        img_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: (
            int
            | tuple[int, ...]
            | tuple[tuple[int, int], ...]
            | Literal["valid", "same"]
        ) = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ):
        assert len(img_shape) == len(kernel_shape), (
            f"{img_shape=} but {kernel_shape=}: For an ND convolution,"
            f" the image and kernel must have the same number of axes.\n"
            f"The first two axes of img_shape must be Batch and Channels, while"
            f"the first two axes of kernel_shape must be Out and In.\n"
            f"After these initial two axes come the spatial axes in order ...Z, Y, X."
        )
        assert len(img_shape) > 2, (
            "An image in an ND-convolution must have Batch and Channel axes, as well as"
            " at least one spatial axis (so at least three axes in total)."
        )
        ndim = len(img_shape) - 2
        assert kind in ("conv", "corr"), f"Invalid {kind=}"

        stride = _as_tup_n(stride, ndim)
        dilation = _as_tup_n(dilation, ndim)

        padding = get_padding(padding, ndim, dilation, kernel_shape[2:])

        return (
            ndim == self.ndim
            and img_shape[1] == self.img_cs
            and img_shape[2:] == self.img_spatial
            and kernel_shape[0] == self.krn_os
            and kernel_shape[1] == self.krn_cs
            and kernel_shape[2:] == self.krn_spatial
            and stride == self.stride
            and padding == tuple(zip(self.pad_begs, self.pad_ends, strict=True))
            and dilation == self.dilation
            and groups == self.groups
            and group_broadcasting == self.group_broadcasting
            and (kind == "conv") == self.mirror_kernel
        )

    def cache_id(self) -> str:
        def fmt(tup: tuple[int, ...]):
            return "_".join(str(i) for i in tup)

        return (
            f"meta"
            f"_{self.img_cs}_{fmt(self.img_spatial)}"
            f"_{self.krn_os}_{self.krn_cs}_{fmt(self.krn_spatial)}"
            f"_{self.out_cs}_{fmt(self.out_spatial)}"
            f"_{fmt(self.stride)}"
            f"_{fmt(self.pad_begs)}_{fmt(self.pad_ends)}"
            f"_{fmt(self.dilation)}"
            f"_{self.groups}_{self.grp_i}_{self.grp_o}"
            f"_{int(self.group_broadcasting)}_{int(self.mirror_kernel)}"
        )


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
    | tuple[int, ...]
    | tuple[tuple[int, int], ...]
    | Literal["valid", "same"],
    ndim: int,
    dilation: tuple[int, ...],
    krn_spatial: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    if isinstance(padding, str):
        if padding == "valid":
            return (0, 0), (0, 0)
        if padding == "same":
            assert len(krn_spatial) == ndim, "Strange kernel spatial dimensions"
            return tuple(
                calculate_same(k, d) for k, d in zip(krn_spatial, dilation, strict=True)
            )

        raise ValueError(f"Invalid {padding=}")

    padding = _as_tup_n(padding, ndim)
    # noinspection PyTypeChecker
    return tuple(_as_tup_n(p, 2) for p in padding)


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


class LearnedKernel2D(nn.Module):
    """
    A utility that provides a fully learnable kernel compatible with `GenericConv`

    Parameters
    -------
    in_channels : int
        The number of input channels: the `I` in `OIHW`.
    out_channels : int
        The number of output channels: the `O` in `OIHW`.
    kernel_size : int
        The height `H` and width `W` of the kernel (rectangular kernels not supported).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.normal_(self.kernel)

    def forward(self):
        return self.kernel


def plot_kernels(kernels: torch.Tensor, at_most: int = 5) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    dev_name = "(CPU)" if kernels.get_device() == -1 else "(CUDA)"
    kernels = kernels.detach().cpu()[:at_most, :at_most]
    # Assuming dilation kernels for [0, 1] data.
    low, high = -1, 0

    out_channels, in_channels, kernel_size, _ks = kernels.shape
    fig, axss = plt.subplots(
        out_channels,
        in_channels,
        sharex=True,
        sharey=True,
        layout="compressed",
        squeeze=False,
    )
    for o, axs in enumerate(axss):
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
