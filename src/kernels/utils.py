import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn


class LearnedSpectral2D(nn.Module):
    """A utility class that parameterises diagonally decomposed 2D covariance matrices
    using parameters for standard deviations and the rotation of the principal axes."""

    def __init__(self, in_channels: int, out_channels: int, init: str = "zero"):
        super().__init__()
        self.log_stds = nn.Parameter(torch.empty((out_channels, in_channels, 2)))
        self.thetas = nn.Parameter(torch.empty((out_channels, in_channels)))
        if init == "zero":
            nn.init.zeros_(self.log_stds)
            nn.init.zeros_(self.thetas)
        elif init == "normal":
            nn.init.normal_(self.log_stds)
            nn.init.normal_(self.thetas)
        else:
            raise ValueError(f"Invalid {init=}")

    def inverse_cov(self):
        rot = torch.stack(
            [
                torch.stack([torch.cos(self.thetas), -torch.sin(self.thetas)], dim=-1),
                torch.stack([torch.sin(self.thetas), torch.cos(self.thetas)], dim=-1),
            ],
            dim=-2,
        )
        inv_diag = torch.diag_embed(self.log_stds.neg().exp())
        return torch.einsum("oivd,oidD,oiVD->oivV", rot, inv_diag, rot)

    def cov(self):
        return torch.linalg.inv(self.inverse_cov())

    def extra_repr(self):
        out_channels, in_channels = self.thetas.shape
        return f"{in_channels}, {out_channels}"


class LearnedCholesky2D(nn.Module):
    """A utility class that parameterises Cholesky-decomposed 2D covariance matrices
    using parameters for standard deviations and for Pearson's R (`corr`)."""

    def __init__(self, in_channels: int, out_channels: int, init: str = "zero"):
        super().__init__()
        self.log_stds = nn.Parameter(torch.empty((2, out_channels, in_channels)))
        self.corr_param = nn.Parameter(torch.empty((out_channels, in_channels)))
        if init == "zero":
            nn.init.zeros_(self.log_stds)
            nn.init.zeros_(self.corr_param)
        elif init == "normal":
            nn.init.normal_(self.log_stds)
            nn.init.normal_(self.corr_param)
        else:
            raise ValueError(f"Invalid {init=}")

    def cholesky(self):
        out_channels, in_channels = self.corr_param.shape

        std = self.log_stds.exp()
        corr = self.corr_param.tanh()
        l_cross = corr * std[1]

        scale_tril = torch.zeros((out_channels, in_channels, 2, 2), device=std.device)
        scale_tril[:, :, 0, 0] = std[0]
        scale_tril[:, :, 1, 0] = l_cross
        scale_tril[:, :, 1, 1] = (std[1].square() - l_cross.square()).sqrt()
        return scale_tril

    def cov(self):
        tril = self.cholesky()
        return torch.einsum("oivL,oiVL->oivV", tril, tril)

    def extra_repr(self):
        out_channels, in_channels = self.corr_param.shape
        return f"{in_channels}, {out_channels}"


class LearnedKernel(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, init: str = "zero"
    ):
        super().__init__()
        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if init == "zero":
            nn.init.zeros_(self.kernel)
        elif init == "normal":
            nn.init.normal_(self.kernel)
        else:
            raise ValueError(f"Invalid {init=}")

    def forward(self):
        return self.kernel


def plot_kernels(
    kernels: torch.Tensor, cut_zero=True, high_cut: float = 0.95, at_most: int = 4
) -> None:
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
