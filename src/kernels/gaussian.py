import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from src.kernels import utils


class GaussKernelIso2D(nn.Module):
    """A kernel that evaluates the isotropic N(x; 0, sI), for use in a linear convolution"""

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
        self.register_buffer("pos_grid", utils.make_pos_grid(kernel_size))

    def forward(self):
        std = self.std_param.exp().unsqueeze(2).repeat((1, 1, 2))
        dist = MultivariateNormal(
            torch.zeros((2,), device=std.device), scale_tril=torch.diag_embed(std)
        )
        grid_probs = dist.log_prob(self.pos_grid).exp()
        out_channels, in_channels = self.std_param.shape
        assert grid_probs.shape == (
            self.kernel_size * self.kernel_size,
            out_channels,
            in_channels,
        ), f"Incorect {grid_probs.shape=}"
        kernel = grid_probs.movedim(0, 2).reshape(
            (out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        return kernel

    def extra_repr(self):
        out_channels, in_channels = self.std_param.shape
        kernel_size = self.kernel_size
        return f"{in_channels}, {out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        utils.plot_kernels(self.forward())


class GaussKernelMulti2D(nn.Module):
    """A kernel that evaluates the multivariate N(x; 0, S), for use in a linear convolution"""

    pos_grid: torch.Tensor
    means: torch.Tensor

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, init: str = "zero"
    ):
        super().__init__()
        self.covs = utils.LearnedCholesky2D(in_channels, out_channels, init)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.register_buffer("pos_grid", utils.make_pos_grid(kernel_size))
        self.register_buffer("means", torch.zeros((2,)))

    def forward(self):
        tril = self.covs.cholesky()
        dist = MultivariateNormal(
            loc=self.means,
            scale_tril=tril,
        )
        grid_probs = dist.log_prob(self.pos_grid).exp()
        assert grid_probs.shape == (
            self.kernel_size * self.kernel_size,
            self.out_channels,
            self.in_channels,
        ), f"Incorect {grid_probs.shape=}"
        kernel = grid_probs.movedim(0, 2).reshape(
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        return kernel

    def extra_repr(self):
        kernel_size = self.kernel_size
        return f"{self.in_channels}, {self.out_channels}, {kernel_size=}"

    @torch.no_grad()
    def plot(self):
        utils.plot_kernels(self.forward())
