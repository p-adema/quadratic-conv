import torch
from torch import nn

from src.kernels import utils


class QuadraticKernelMulti2D(nn.Module):
    """A kernel that evaluates xT S^-1 x, for use in a tropical convolution"""

    pos_grid: torch.Tensor

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, init: str = "zero"
    ):
        super().__init__()
        self.covs = utils.LearnedCovs2D(in_channels, out_channels, init)
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
            .reshape(
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
