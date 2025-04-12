import torch
from torch import nn

from src.kernels import utils


class QuadraticKernelSpectral2D(nn.Module):
    """A kernel that evaluates x^T S^-1 x, with skew parameterised as an angle theta"""

    pos_grid: torch.Tensor

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, init: str = "zero"
    ):
        super().__init__()
        self.covs = utils.LearnedSpectral2D(in_channels, out_channels, init)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer(
            "pos_grid",
            utils.make_pos_grid(kernel_size).reshape(kernel_size * kernel_size, 2),
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
        return dists

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        utils.plot_kernels(self.forward())


class QuadraticKernelCholesky2D(nn.Module):
    """A kernel that evaluates x^T S^-1 x, with skew parameterised as Pearson's R"""

    pos_grid: torch.Tensor

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, init: str = "zero"
    ):
        super().__init__()
        self.covs = utils.LearnedCholesky2D(in_channels, out_channels, init)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer(
            "pos_grid", utils.make_pos_grid(kernel_size, grid_at_end=True)
        )

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
        return dists

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        utils.plot_kernels(self.forward())


class QuadraticKernelIso2D(nn.Module):
    """A kernel that evaluates x^T sI x, representing an isotropic quadratic"""

    pos_grid: torch.Tensor

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, init: str = "zero"
    ):
        super().__init__()
        self.std_param = nn.Parameter(torch.empty((out_channels, in_channels)))
        if init == "zero":
            nn.init.zeros_(self.std_param)
        elif init == "normal":
            nn.init.normal_(self.std_param)
        else:
            raise ValueError(f"Invalid {init=}")
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer(
            "pos_grid", utils.make_pos_grid(kernel_size, grid_at_end=True)
        )

    def forward(self):
        dists = (self.pos_grid.pow(2).sum(-2) / self.std_param.unsqueeze(2)).reshape(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        return dists

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        utils.plot_kernels(self.forward())
