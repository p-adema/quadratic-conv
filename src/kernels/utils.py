import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn


class LearnedSpectral2D(nn.Module):
    """A utility class that parameterises diagonally decomposed 2D covariance matrices
    using parameters for standard deviations and the rotation of the principal axes."""

    def __init__(self, in_channels: int, out_channels: int, init: str | float = 0.0):
        super().__init__()
        variances = torch.empty((out_channels, in_channels, 2))
        thetas = torch.empty((out_channels, in_channels))
        if isinstance(init, float):
            nn.init.constant_(variances, init)
            nn.init.zeros_(thetas)
        elif init == "uniform":
            nn.init.uniform_(variances, 0.1, 4.0)
            nn.init.uniform_(thetas, 0, 2 * torch.pi)
        elif init == "uniform-iso":
            nn.init.uniform_(variances[..., 0], 0.1, 4.0)
            variances[..., 1] = variances[..., 0]
            nn.init.uniform_(thetas, 0, 2 * torch.pi)
        elif init == "normal":
            nn.init.trunc_normal_(variances, mean=2.0, a=0.1, b=4.0)
            nn.init.uniform_(thetas, 0, 2 * torch.pi)
        elif init == "spin":
            nn.init.constant_(variances[..., 0], 4)
            nn.init.constant_(variances[..., 1], 0.5)
            spaced_thetas = torch.linspace(
                0, 2 * torch.pi, steps=out_channels * in_channels
            )
            thetas[:] = spaced_thetas[torch.randperm(spaced_thetas.shape[0])].reshape(
                out_channels, in_channels
            )
        else:
            raise ValueError(f"Invalid {init=}")
        self.theta = nn.Parameter(thetas)
        self.log_std = nn.Parameter(variances.log().mul(0.5))

    def inverse_cov(self):
        rot = torch.stack(
            [
                torch.stack([torch.cos(self.theta), -torch.sin(self.theta)], dim=-1),
                torch.stack([torch.sin(self.theta), torch.cos(self.theta)], dim=-1),
            ],
            dim=-2,
        )
        # Along the diagonal, we want 1/ std^2
        inv_diag = torch.diag_embed(self.log_std.mul(-2).exp())
        return torch.einsum("oivd,oidD,oiVD->oivV", rot, inv_diag, rot)

    def cov(self):
        return torch.linalg.inv(self.inverse_cov())

    forward = inverse_cov

    def extra_repr(self):
        out_channels, in_channels = self.theta.shape
        return f"{in_channels}, {out_channels}"


class LearnedCholesky2D(nn.Module):
    """A utility class that parameterises Cholesky-decomposed 2D covariance matrices
    using parameters for standard deviations and for Pearson's R (`corr`)."""

    def __init__(self, in_channels: int, out_channels: int, init: str | float = 0.0):
        super().__init__()
        variances = torch.empty((2, out_channels, in_channels))
        corr = torch.empty((out_channels, in_channels))
        if isinstance(init, float):
            nn.init.constant_(variances, init)
            nn.init.zeros_(corr)
        elif init == "normal":
            nn.init.normal_(variances)
            nn.init.normal_(corr)
        else:
            raise ValueError(f"Invalid {init=}")
        self.corr = nn.Parameter(corr)
        self.log_std = nn.Parameter(variances.log().mul(0.5))

    def cholesky(self):
        out_channels, in_channels = self.corr.shape

        std = self.log_std.exp()
        corr = self.corr.tanh()
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
        out_channels, in_channels = self.corr.shape
        return f"{in_channels}, {out_channels}"

    forward = cholesky


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
