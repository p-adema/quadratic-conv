import math

import torch
from torch import nn

from ._learned_pos_def import CovCholesky2D, CovSpectral2D
from ._utils import make_pos_grid, plot_kernels


class QuadraticKernelSpectral2D(nn.Module):
    r"""
    A kernel that evaluates \(x^T S^{-1} x\), with skew parameterised as an angle \(\theta\)

    This module takes no arguments in `forward` and produces a
    `Tensor` of `OIHW`, making this Module suitable as a kernel for `GenericConv`.

    Parameters
    -------
    in_channels : int
        The number of input channels: the `I` in `OIHW`.
    out_channels : int
        The number of output channels: the `O` in `OIHW`.
    kernel_size : int
        The height `H` and width `W` of the kernel (rectangular kernels are not supported).
    init : dict, optional
        The initialisation stratergy for the underlying covariance matrices.
        If provided, the dictionary must have keys:

        `"var"` for the variances, which can take values:

        - `float` to indicate a constant initialisation
        - `"normal"` to indicate values normally distributed around 2.0
        - `"uniform"` to indicate uniform-random initialisation
        - `"uniform-iso"` to indicate isotropic uniform-random initialisation
        - `"ss-iso"` to indicate scale-space isotropic initialisation
        - `"skewed"` to indicate uniform-random initialisation with the second primary
          axis having a significantly higher variance (**default**)

        and `"theta"` for the rotations, which can take values:

        - `"uniform"` to indicate uniform-random initialisation
        - `"spin"` to indicate shuffled but evenly spaced angles (**default**)

    Examples
    -------

    >>> kernel = QuadraticKernelSpectral2D(5, 6, 3, {"var": 3.0, "theta": "spin"})
    >>> tuple(kernel().shape)
    (6, 5, 3, 3)
    """

    _pos_grid: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        init: dict[str, str | float] | None = None,
    ):
        super().__init__()
        self.covs = CovSpectral2D(in_channels, out_channels, init, kernel_size)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer(
            "_pos_grid",
            make_pos_grid(kernel_size).reshape(kernel_size * kernel_size, 2),
        )

    def forward(self):
        dists = torch.einsum(
            "kx,oixX,kX->oik", self._pos_grid, self.covs.inverse_cov(), self._pos_grid
        ).view(
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        return -dists

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    def inspect_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inspect the parameters of the underlying covariance matrices.

        Returns
        -------
        log_std : Tensor of (O, I, 2)
            The logathirms of the standard deviations in both axes for all kernels
        theta : Tensor of (O, I)
            The counter-clockwise angles between the first axis and the X-axis for all kernels
        """
        return self.covs.log_std, self.covs.theta

    @torch.no_grad()
    def plot(self):
        """Provide a simple visualisation of some kernels. Requires `seaborn`."""
        plot_kernels(self.forward())


class QuadraticKernelCholesky2D(nn.Module):
    r"""
    A kernel that evaluates \(x^T S^{-1} x\), with skew parameterised as Pearson's R

    This module takes no arguments in `forward` and produces a
    `Tensor` of `OIHW`, making this Module suitable as a kernel for `GenericConv`.

    Parameters
    -------
    in_channels : int
        The number of input channels: the `I` in `OIHW`.
    out_channels : int
        The number of output channels: the `O` in `OIHW`.
    kernel_size : int
        The height `H` and width `W` of the kernel (rectangular kernels are not supported).
    init : dict, optional
        The initialisation stratergy for the underlying covariance matrices.
        If provided, the dictionary must have the key `"var"`, which can take values:

        - `float` to indicate a constant initialisation
        - `"normal"` to indicate values normally distributed around 2.0
        - `"uniform"` to indicate uniform-random initialisation
        - `"uniform-iso"` to indicate isotropic uniform-random initialisation
        - `"ss-iso"` to indicate scale-space isotropic initialisation
        - `"skewed"` to indicate uniform-random initialisation with the second primary
          axis having a significantly higher variance (**default**)

        The skew parameter is always initialised using a clipped normal distribution
        centred around 0.

    Examples
    -------

    >>> kernel = QuadraticKernelCholesky2D(5, 6, 3, {"var": 3.0})
    >>> tuple(kernel().shape)
    (6, 5, 3, 3)
    """

    _pos_grid: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        init: dict[str, str | float] | None = None,
    ):
        super().__init__()
        self.covs = CovCholesky2D(in_channels, out_channels, init, kernel_size)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer("_pos_grid", make_pos_grid(kernel_size, grid_at_end=True))

    def forward(self):
        # [o, i, 2, k*k]
        bs = torch.linalg.solve_triangular(
            self.covs.cholesky(), self._pos_grid, upper=False
        )
        dists = (
            bs.pow(2)
            .sum(-2)
            .view(
                (
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        )
        return -dists

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    def inspect_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inspect the parameters of the underlying covariance matrices.

        Returns
        -------
        log_std : Tensor of (O, I, 2)
            The logathirms of the standard deviations in both axes for all kernels
        corr : Tensor of (O, I)
            The skews, as values for Person's R, for all kernels
        """
        return self.covs.log_std.moveaxis(0, 2), self.covs.corr

    @torch.no_grad()
    def plot(self):
        """Provide a simple visualisation of some kernels. Requires `seaborn`."""
        plot_kernels(self.forward())


class QuadraticKernelIso2D(nn.Module):
    r"""
    A kernel that evaluates \(x^T sI x\), representing an isotropic quadratic

    This module takes no arguments in `forward` and produces a
    `Tensor` of `OIHW`, making this Module suitable as a kernel for `GenericConv`.

    Parameters
    -------
    in_channels : int
        The number of input channels: the `I` in `OIHW`.
    out_channels : int
        The number of output channels: the `O` in `OIHW`.
    kernel_size : int
        The height `H` and width `W` of the kernel (rectangular kernels are not supported).
    init : dict, optional
        The initialisation stratergy for the variances / scale parameters.
        If provided, the dictionary must have the key `"var"`, which can take values:

        - `float` to indicate a constant initialisation
        - `"normal"` to indicate values normally distributed around 2.0
        - `"uniform"` to indicate uniform-random initialisation
        - `"ss"` to indicate scale-space initialisation (**default**)

    Attributes
    -------
    log_std : Tensor of (O, I)
        The logathirms of the standard deviations for all kernels

    Examples
    -------

    >>> kernel = QuadraticKernelIso2D(5, 6, 3, {"var": 3.0})
    >>> tuple(kernel().shape)
    (6, 5, 3, 3)
    """

    log_std: torch.Tensor
    _pos_grid: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        init: dict = None,
    ):
        super().__init__()
        init: dict[str, str | float] = init or {"var": "ss"}

        variances = torch.empty((out_channels, in_channels))
        if isinstance(init["var"], float):
            nn.init.constant_(variances, init["var"])
        elif init["var"] == "uniform":
            nn.init.uniform_(variances, 1, 16)
        elif init["var"] == "ss":
            spaced_vars = torch.linspace(
                1,
                2 * (kernel_size // 2) ** 2,
                steps=out_channels * in_channels,
            )
            permutation = torch.randperm(spaced_vars.shape[0])
            variances[:] = spaced_vars[permutation].reshape(out_channels, in_channels)
        elif init["var"] == "log-ss":
            spaced_vars = torch.logspace(
                math.log10(1),
                math.log10(2 * (kernel_size // 2) ** 2),
                steps=out_channels * in_channels,
            )
            permutation = torch.randperm(spaced_vars.shape[0])
            variances[:] = spaced_vars[permutation].reshape(out_channels, in_channels)
        elif init["var"] == "normal":
            nn.init.trunc_normal_(variances, mean=8.0, a=1.0, b=16.0)
        else:
            raise ValueError(f"Invalid {init['var']=}")

        self.log_std = nn.Parameter(variances.log().mul(0.5))

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer("_pos_grid", make_pos_grid(kernel_size, grid_at_end=True))

    def forward(self):
        dists = (
            self._pos_grid.pow(2).sum(-2) / self.log_std.mul(2).exp().unsqueeze(2)
        ).reshape(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        return -dists

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        """Provide a simple visualisation of some kernels. Requires `seaborn`."""
        plot_kernels(self.forward())
