from __future__ import annotations

from typing import Literal

from torch import nn


class GenericConv2D(nn.Module):
    """
    A generic convolution Module using a kernel and a convolution Module

    Parameters
    -------
    kernel : nn.Module
        A module that produces a convolutional kernel from its `forward` method.
        Must not take arguments.

        See e.g. `QuadraticKernelSpectral2D` or `LearnedKernel`.
    conv : nn.Module
        A module that can take `image, kernel` as positional arguments, as well as
        `dilation`, `padding`, `stride` and `groups` as keyword arguments, optionally
        supporting `group_broadcasting` and `kind`.

        See e.g. `BroadcastSemifield.dynamic` or `SelectSemifield.lazy_fixed`.
    stride : int, (int, int) = 1
        The stride passed to `conv`, either for both spatial dimensions or `(Y, X)`
        seperately.
    padding : int, (int, int), ((int, int), (int, int)), "valid", "same" = 0
        The padding passed to `conv`.
        Depending on the type of `padding`:

        - `P` indicates padding at the start and end of all spatial axes with `P`.
        - `(Y, X)` indicates padding at start and end of the Y-axis with `Y`,
          and padding at the start and end of the X-axis with `X`.
        - `((BY, EY), (BX, EX))` indicates padding the start of the Y-axis with `BY` and
          the end of the Y-axis with `EY`, similarly for X.
        - `"valid"` indicates to only perform the convolution with valid values of the
          image, i.e. no padding.
        - `"same"` indicates to pad the input such that a stride-1 convolution would
          produce an output of the same spatial size.
          Convolutions with higher stride will use the same padding scheme, but result
          in outputs of reduced size.
    dilation : int, (int, int) = 1
        The dilation passed to `conv`, either for both spatial dimensions or `(Y, X)`
        seperately.
    groups : int = 1
        The number of convolutional groups for this convolution.
    group_broadcasting : bool = False
        Whether to take the input kernels as a single output group, and broadcast
        across all input groups.
        `group_broadcasting` has no effect for `groups=1`
    kind : literal "conv" or "corr"
        Represents whether the kernel should be mirrored during the convolution `"conv"`
        or not `"corr"`.

    Examples
    -------
    >>> import pytorch_semifield_conv as semiconv
    >>> dilation = semiconv.GenericConv2D(
    ...     semiconv.QuadraticKernelSpectral2D(5, 5, 3),
    ...     semiconv.SelectSemifield.tropical_max().lazy_fixed(),
    ...     padding="same",
    ...     stride=2,
    ...     groups=5,
    ... )
    >>> root = semiconv.GenericConv2D(
    ...     semiconv.QuadraticKernelIso2D(5, 10, 3),
    ...     semiconv.BroadcastSemifield.root(3.0).dynamic(),
    ...     padding="same",
    ... )
    """

    def __init__(
        self,
        kernel: nn.Module,
        conv: nn.Module,
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

    def forward(self, img):
        """
        Run a forward step with this convolution.

        Parameters
        ----------
        img : Tensor (B, C, H, W)
            The input images as a 4D tensor, of shape (Batch, Channels, Height, Width)

        Returns
        -------
        out_img : Tensor (B, C', H', W')
            The output images as a 4D tensor, with the same batch shape but possibly
            adjusted other dimensions.
        """
        return self.conv(
            img,
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


class GenericClosing2D(nn.Module):
    """
    A generic Module for implement a morphological closing with a dilation and erosion.

    `kind` is fixed to `"conv"` for dilation and `"corr"` for erosion, to simplify the
    implementation of the common morphological closing.

    Parameters
    -------
    kernel : nn.Module
        A module that produces a convolutional kernel from its `forward` method.
    conv_dilation : nn.Module
        A module representing the adjoint dilation that can take `image, kernel`
         as positional arguments, as well as
        `dilation`, `padding`, `stride`, `groups` **and `kind`** as keyword arguments,
        optionally supporting `group_broadcasting` and `kind`.
    conv_erosion : nn.Module
        A module representing the adjoint erosion that can take `image, kernel`
         as positional arguments, as well as
        `dilation`, `padding`, `stride`, `groups` **and `kind`** as keyword arguments,
        optionally supporting `group_broadcasting`.
    stride : int, (int, int) = 1
        The stride passed to `conv`, either for both spatial dimensions or `(Y, X)`
        seperately.
    padding : int, (int, int), ((int, int), (int, int)), "valid", "same" = 0
        The padding passed to `conv`.
        Depending on the type of `padding`:

        - `P` indicates padding at the start and end of all spatial axes with `P`.
        - `(Y, X)` indicates padding at start and end of the Y-axis with `Y`,
          and padding at the start and end of the X-axis with `X`.
        - `((BY, EY), (BX, EX))` indicates padding the start of the Y-axis with `BY` and
          the end of the Y-axis with `EY`, similarly for X.
        - `"valid"` indicates to only perform the convolution with valid values of the
          image, i.e. no padding.
        - `"same"` indicates to pad the input such that a stride-1 convolution would
          produce an output of the same spatial size.
          Convolutions with higher stride will use the same padding scheme, but result
          in outputs of reduced size.
    dilation : int, (int, int) = 1
        The dilation passed to `conv`, either for both spatial dimensions or `(Y, X)`
        seperately.
    groups : int = 1
        The number of convolutional groups for this convolution.
    group_broadcasting : bool = False
        Whether to take the input kernels as a single output group, and broadcast
        across all input groups.
        `group_broadcasting` has no effect for `groups=1`

    Examples
    -------

    >>> import pytorch_semifield_conv as semiconv
    >>> common_closing = semiconv.GenericClosing2D(
    ...     semiconv.QuadraticKernelCholesky2D(5, 5, 3),
    ...     semiconv.SelectSemifield.tropical_max().lazy_fixed(),
    ...     semiconv.SelectSemifield.tropical_min_negated().lazy_fixed(),
    ...     padding="same",
    ...     groups=5,
    ... )
    """

    def __init__(
        self,
        kernel: nn.Module,
        conv_dilation: nn.Module,
        conv_erosion: nn.Module,
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
        self.kind = "closing"

        # Since these are custom arguments, we only want to pass them if they differ
        # from the default values (otherwise, they may be unexpected)
        self.kwargs = {}
        if self.group_broadcasting:
            self.kwargs["group_broadcasting"] = True

    def forward(self, x):
        kernel = self.kernel()
        dilated = self.conv_dilation(
            x,
            kernel,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
            groups=self.groups,
            kind="conv",
            **self.kwargs,
        )
        closed = self.conv_erosion(
            dilated,
            kernel,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
            groups=self.groups,
            kind="corr",
            **self.kwargs,
        )
        return closed

    extra_repr = GenericConv2D.extra_repr
