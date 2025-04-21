import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn


class LearnedKernel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.normal_(self.kernel)

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
