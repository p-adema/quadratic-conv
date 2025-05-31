import itertools
import typing
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from ._unfold_view import unfold_copy_2d as unfold_copy_fn
from ._unfold_view import unfold_view as unfold_view_fn
from ._utils import ConvMeta


class BroadcastSemifield(typing.NamedTuple):
    r"""
    A semifield definition using PyTorch broadcasting operators

    Using a technique similar to nn.Unfold, we can create a view of the input array and
    apply broadcasting functions along kernel axes to perform a semifield convolution.
    All functions must take PyTorch Tensors, and should have a backwards implementation.

    This function does not use JIT components, and therefore has no compilation time
    (and can be run on non-CUDA devices as well).

    Parameters
    -------
    add_reduce : (Tensor, tuple of ints) -> Tensor
        To characterise semifield summation \(\bigoplus\), this function takes a single
        tensor with several axes, and performs reduction with \(\oplus\) along the axes
        indicated in the second argument.

        Example: ``lambda arr, dims: torch.sum(arr, dim=dims)``
    multiply : (Tensor, Tensor) -> Tensor
        To characterise semifield multiplication \(\otimes\), this function takes two
        tensors and performs a broadcasting, element-wise version of \(\otimes\).

        Example: ``lambda img, krn: img * krn``
    zero : float
        The absorbing semifield zero.

    Other Parameters
    -------
    add_reduce_channels : (Tensor, int) -> Tensor, optional
        An alternate reduction function (similar to `add_reduce`) that is applied along
        specifically the channel dimension.
        This alternate function could be e.g. addition, in a modified version of \(T_+\)
        (see `channels_add` parameter of `BroadcastSemifield.tropical_max`).

    Examples
    -------
    \(T_+\) convolution:

    >>> dilation = BroadcastSemifield.tropical_max().dynamic()

    \(L_{-3}\) convolution:

    >>> log = BroadcastSemifield.log(-3.0).dynamic()

    For examples of how to construct a `BroadcastSemifield` manually, see the source.
    """

    # (multiplied, dims) -> `multipled` reduced with (+) along every dim in `dims`
    add_reduce: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]
    # (img, krn) -> `img` (x) `krn`
    multiply: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    # forall a, b: `zero` (x) a  (+) b  ==  b
    zero: float
    # Similar to add_reduce, but only used for channel axis (so takes one dimension)
    add_reduce_channels: Callable[[torch.Tensor, int], torch.Tensor] = None

    @classmethod
    def tropical_max(cls, channels_add: bool = False, spread_gradient: bool = False):
        r"""
        Construct a \(T_+\) `BroadcastSemifield`.

        The tropical max semifield / semiring is defined as:
        \[(\mathbb{R}\cup \{-\infty\}, \max, +)\]

        Parameters
        ----------
        channels_add : bool = False
            Whether to use standard addition \(+\) instead of the semifield addition
            \(\max\) along specifically the channel axis.
        spread_gradient : bool = False
            Whether to, in cases of multiple equal maxima, spread the gradient equally
            amongst all maxima.
        """
        return cls(
            add_reduce=(lambda multiplied, dim: torch.amax(multiplied, dim=dim))
            if spread_gradient
            else (
                _repeated_dim(
                    lambda multiplied, dim: torch.max(multiplied, dim=dim).values
                )
            ),
            multiply=lambda img, krn: img + krn,
            zero=-torch.inf,
            add_reduce_channels=(
                (lambda multiplied, dim: torch.sum(multiplied, dim=dim))
                if channels_add
                else None
            ),
        )

    @classmethod
    def tropical_min_negated(
        cls, channels_add: bool = False, spread_gradient: bool = False
    ):
        r"""
        Construct a `BroadcastSemifield` similar to \(T_-\), where the kernel is negated

        The usual tropical min semifield / semiring is defined as:
        \[(\mathbb{R}\cup \{\infty\}, \min, +)\]

        This version is slightly modified:
        while performing erosion using \(T_-\) requires first negating the kernel, this
        modified semifield has \(-\) instead of \(+\) as the semifield multiplication.
        As such, the resulting convolution will work with non-negated kernels as inputs,
        making the interface more similar to the dilation in \(T_+\).

        Parameters
        ----------
        channels_add : bool = False
            Whether to use standard addition \(+\) instead of the semifield addition
            \(\min\) along specifically the channel axis.
        spread_gradient : bool = False
            Whether to, in cases of multiple equal minima, spread the gradient equally
            amongst all minima.
        """
        return cls(
            add_reduce=(lambda multiplied, dim: torch.amin(multiplied, dim=dim))
            if spread_gradient
            else (
                _repeated_dim(
                    lambda multiplied, dim: torch.min(multiplied, dim=dim).values
                )
            ),
            multiply=lambda img, krn: img - krn,
            zero=torch.inf,
            add_reduce_channels=(
                (lambda multiplied, dim: torch.sum(multiplied, dim=dim))
                if channels_add
                else None
            ),
        )

    @classmethod
    def linear(cls):
        r"""
        Construct a linear `BroadcastSemifield`.

        The linear field is defined as:
        \[(\mathbb{R}, +, \times)\]

        Mainly for comparison purposes: the linear convolutions offered by PyTorch
        use CUDNN, which is far better optimised for CUDA devices.
        """
        return cls(
            add_reduce=(lambda multiplied, dim: torch.sum(multiplied, dim=dim)),
            multiply=lambda img, krn: img * krn,
            zero=0,
        )

    @classmethod
    def root(cls, p: float):
        r"""
        Construct a \(R_p\) `BroadcastSemifield`.

        The root semifields are defined as:
        \[(\mathbb{R}_+, \oplus_p, \times) \textrm{ for all } p\ne0 \textrm{ where }
        a\oplus_p b= \sqrt[p]{a^p+b^p} \]
        with the semifield zero being \(0\) and the semifield one being \(1\).

        Parameters
        ----------
        p : int
            The power to use in \(\oplus_p\).
            May not be zero.
        """
        assert p != 0, f"Invalid value: {p=}"
        return cls(
            add_reduce=(
                lambda multiplied, dim: multiplied.pow(p).sum(dim=dim).pow(1 / p)
            ),
            multiply=lambda img, krn: img * krn,
            zero=float(torch.finfo(torch.float32).eps),
        )

    @classmethod
    def log(cls, mu: float):
        r"""
        Construct a \(L_+\mu\) or \(L_-\mu\) `BroadcastSemifield`.

        The log semifields are defined as:
        \[(\mathbb{R}\cup \{\pm\infty\}, \oplus_\mu, +) \textrm{ for all } \mu\ne0
        \textrm{ where }
        a\oplus_\mu b= \frac{1}{\mu}\ln(e^{\mu a}+e^{\mu b}) \]
        with the semifield zero being \(-\infty\) for \(\mu>0\) and \(\infty\)
        otherwise, and the semifield one being \(0\).

        Parameters
        ----------
        mu : int
            The base to use in \(\oplus_mu\).
            May not be zero.
        """
        assert mu != 0, f"Invalid value: {mu=}"
        return cls(
            add_reduce=(
                lambda multiplied, dim: torch.logsumexp(multiplied * mu, dim=dim) / mu
            ),
            multiply=lambda img, krn: img + krn,
            zero=-torch.inf if mu > 0 else torch.inf,
        )

    def dynamic(self, unfold_copy: bool = False) -> torch.nn.Module:
        """
        Create a convolution Module based on this `BroadcastSemifield`.

        This method is named `dynamic`, because the Module it creates will dynamically
        adjust itself based on new input types, unlike e.g. `SelectSemifield.lazy_fixed`

        Parameters
        ----------
        unfold_copy : bool = False
            Whether to use `nn.functional.unfold` during computation, which results in
            a copy of the data.
            This is only supported for 2D convolutions 1D or 3+D convolutions cannot use
            `nn.functional.unfold`.

            Mainly for comparison purposes: in tests, it always results in slowdown.

        Returns
        -------
        conv : nn.Module
            A convolution module, suitable for use in `GenericConv`
        """
        return BroadcastConv(self, unfold_copy)


class BroadcastConv(nn.Module):
    """A convolution module, suitable for use in `GenericConv`"""

    def __init__(
        self,
        semifield: BroadcastSemifield,
        unfold_copy: bool = False,
    ):
        super().__init__()
        self.semifield = semifield
        self.last_meta: ConvMeta | None = None
        self.unfold = unfold_copy_fn if unfold_copy else unfold_view_fn

    def forward(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
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
    ) -> torch.Tensor:
        meta = self.get_meta(
            imgs,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            group_broadcasting,
        )
        batch_size = imgs.shape[0]

        imgs_padded = torch.constant_pad_nd(
            imgs,
            tuple(
                itertools.chain.from_iterable(
                    (meta.pad_begs[s], meta.pad_ends[s])
                    for s in reversed(range(meta.ndim))
                )
            ),
            self.semifield.zero,
        )
        # [b, groups * krn_cs, krn_ys, krn_xs, out_ys, out_xs]
        windows_flat_channels = self.unfold(
            imgs_padded,
            meta.krn_spatial,
            dilation=meta.dilation,
            stride=meta.stride,
        )
        # print(windows_flat_channels.shape)
        windows = windows_flat_channels.view(
            batch_size,
            meta.groups,
            1,  # Broadcast along grp_o
            meta.krn_cs,
            *meta.krn_spatial,
            *meta.out_spatial,
        )
        if kind == "conv":
            # Very bad, but this is only a reference implementation
            kernel = kernel.flip(tuple(range(2, 2 + meta.ndim)))

        weights = kernel.view(
            1,  # Broadcast along batch dimension
            1 if group_broadcasting else groups,  # Maybe broadcast along groups
            meta.grp_o,  # Number of kernels per group
            meta.krn_cs,  # 3: Neighbourhood Channels
            *meta.krn_spatial,  # (4, 5) for 2D img kernel
            *(1 for _ in range(meta.ndim)),  # Broadcast along windows
        )
        multiplied = self.semifield.multiply(windows, weights)
        if self.semifield.add_reduce_channels is None:
            reduced = self.semifield.add_reduce(
                multiplied, (3, *range(4, 4 + meta.ndim))
            )
        else:
            reduced_with_channels = self.semifield.add_reduce(
                multiplied, tuple(range(4, 4 + meta.ndim))
            )
            reduced = self.semifield.add_reduce_channels(reduced_with_channels, 3)

        res = reduced.view(
            batch_size,
            meta.out_cs,
            *meta.out_spatial,
        )
        return res

    def get_meta(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
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
    ) -> ConvMeta:
        if self.last_meta is not None and self.last_meta.check_matches(
            tuple(imgs.shape),
            tuple(kernel.shape),
            stride,
            padding,
            dilation,
            groups,
            group_broadcasting,
            kind,
        ):
            meta = self.last_meta
        else:
            meta = ConvMeta.infer(
                tuple(imgs.shape),
                tuple(kernel.shape),
                stride,
                padding,
                dilation,
                groups,
                group_broadcasting,
                kind,
            )
            self.last_meta = meta
        return meta

    if typing.TYPE_CHECKING:
        __call__ = forward


def _repeated_dim(single_dim_broadcast: Callable):
    def func(x: torch.Tensor, dims: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(dims, int):
            dims = (dims,)

        for dim in sorted(dims, reverse=True):
            x = single_dim_broadcast(x, dim=dim)

        return x

    return func
