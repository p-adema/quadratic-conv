import torch
from torch import nn

from .utils import make_pos_grid, plot_kernels

from .learned_pos_def import LearnedCholesky2D, LearnedSpectral2D


class QuadraticKernelSpectral2D(nn.Module):
    """A kernel that evaluates x^T S^-1 x, with skew parameterised as an angle theta"""

    pos_grid: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        init: dict[str, str | float] | None = None,
    ):
        super().__init__()
        self.covs = LearnedSpectral2D(in_channels, out_channels, init)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer(
            "pos_grid",
            make_pos_grid(kernel_size).reshape(kernel_size * kernel_size, 2),
        )

    def forward(self):
        dists = torch.einsum(
            "kx,oixX,kX->oik", self.pos_grid, self.covs.inverse_cov(), self.pos_grid
        ).view(
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        return dists.mul(-0.25)

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        plot_kernels(self.forward())


class QuadraticKernelCholesky2D(nn.Module):
    """A kernel that evaluates x^T S^-1 x, with skew parameterised as Pearson's R"""

    pos_grid: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        init: dict[str, str | float] | None = None,
    ):
        super().__init__()
        self.covs = LearnedCholesky2D(in_channels, out_channels, init)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer("pos_grid", make_pos_grid(kernel_size, grid_at_end=True))

    def forward(self):
        # [o, i, 2, k*k]
        bs = torch.linalg.solve_triangular(
            self.covs.cholesky(), self.pos_grid, upper=False
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
        return dists.mul(-0.25)

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        plot_kernels(self.forward())


class QuadraticKernelIso2D(nn.Module):
    """A kernel that evaluates x^T sI x, representing an isotropic quadratic"""

    pos_grid: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        init: dict = None,
    ):
        super().__init__()
        init: dict[str, str | float] = init or {"var": "uniform"}

        variances = torch.empty((out_channels, in_channels))
        if isinstance(init["var"], float):
            nn.init.constant_(variances, init["var"])
        elif init["var"] == "uniform":
            nn.init.uniform_(variances, 0.1, 4.0)
        elif init["var"] == "ss":
            spaced_vars = torch.linspace(0.1, 4.0, steps=out_channels * in_channels)
            permutation = torch.randperm(spaced_vars.shape[0])
            variances[:] = spaced_vars[permutation].reshape(out_channels, in_channels)
        elif init["var"] == "normal":
            nn.init.trunc_normal_(variances, mean=2.0, a=0.1, b=4.0)
        else:
            raise ValueError(f"Invalid {init['var']=}")

        self.log_std = nn.Parameter(variances.log().mul(0.5))

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer("pos_grid", make_pos_grid(kernel_size, grid_at_end=True))

    def forward(self):
        dists = (
            self.pos_grid.pow(2).sum(-2) / (-4 * self.log_std.mul(2).exp().unsqueeze(2))
        ).reshape(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        return dists

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        plot_kernels(self.forward())
