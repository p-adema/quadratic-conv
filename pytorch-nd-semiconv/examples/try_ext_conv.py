from __future__ import annotations

import typing
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from pytorch_nd_semiconv import SelectSemifield

plt.set_cmap("Spectral_r")

resolution = 1001
kernel_size = 1001


with torch.device("cuda"):
    space = torch.linspace(0, 1, steps=resolution)
    space_2d = torch.cartesian_prod(space, space)
    test_loc = torch.tensor(
        [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]], dtype=torch.float32
    ).unsqueeze(1)
    test_cov = 1e-2 * torch.eye(2).reshape(1, 2, 2).broadcast_to(len(test_loc), 2, 2)
    data_peaked = (
        torch.einsum(
            "cbx,cxX,cbX->cb",
            space_2d - test_loc,
            torch.inverse(test_cov),
            space_2d - test_loc,
        )
        .reshape(1, -1, resolution, resolution)
        .mul(-2)
        .exp()
        .mul(torch.tensor([0.5, 1, 0.8, 0.2], dtype=torch.float32).view(1, -1, 1, 1))
        .sum(1, keepdim=True)
    )
    # data_peaked += torch.randn_like(data_peaked) * 0.01


def plot_2d(
    data: torch.Tensor,
    batch: int = 0,
    channel: int = 0,
    ax=None,
    title: str = "",
    vmin=0,
    vmax=1,
    imshow: bool = False,
):
    assert len(data.shape) == 4
    data = data[batch, channel].numpy(force=True)
    if ax is None:
        _, ax = plt.subplots(layout="compressed")
    if imshow:
        ax.imshow(data, vmin=vmin, vmax=vmax)
        ax.axis("off")
    else:
        ax.matshow(data, vmin=vmin, vmax=vmax)
    ax.set_title(title)


data_slashed = data_peaked[:, :, 150:450].clone()
data_slashed[0, 0, 145:155, 250:350] = 1
data_slashed[0, 0, :, 695:705] = 0


class Adjunction(typing.NamedTuple):
    dilation: Callable
    erosion: Callable
    name: str


def plot_kernels(kernel: torch.Tensor, vmin=-1, vmax=0):
    assert len(kernel.shape) == 4
    kernel = kernel.numpy(force=True)[:, 0]
    _, axs = plt.subplots(ncols=len(kernel), layout="compressed", figsize=(10, 10))
    for k, ax in zip(kernel, axs, strict=True):
        ax.matshow(k, vmin=vmin, vmax=vmax)


with torch.device("cuda"):
    assert kernel_size % 2
    space = torch.linspace(0, 1, steps=kernel_size)
    space_2d = torch.cartesian_prod(space, space)
    space_2d -= space_2d[kernel_size * kernel_size // 2].clone()
    test_cov = 0.4 * torch.tensor(
        [
            [[0.3, 0], [0, 0.3]],
            [[0.3, 0], [0, 0.03]],
            [[0.03, 0], [0, 0.3]],
            [[0.2, 0.1], [0.1, 0.2]],
            [[0.2, -0.18], [-0.18, 0.2]],
            # [[0.5, 0], [0, 0.05]],
        ],
        dtype=torch.float32,
    )
    rounded_kernels = (
        torch.einsum(
            "bx,cxX,bX->cb",
            space_2d,
            torch.inverse(test_cov),
            space_2d,
        )
        .reshape(-1, 1, kernel_size, kernel_size)
        .mul(-2)
        .exp()
    )

    rounded_kernels -= (
        rounded_kernels.view(-1, kernel_size * kernel_size)
        .max(1)
        .values.view(-1, 1, 1, 1)
    )
    # rounded_kernels[rounded_kernels <= -0.97] = -1
    # rounded_kernels[rounded_kernels > -0.05] = 0


ss_adj = Adjunction(
    SelectSemifield.tropical_max().dynamic(20),
    SelectSemifield.tropical_min().dynamic(20),
    "Select",
)


def plot_adjunction(
    data: torch.Tensor,
    adj: Adjunction,
    kernel: torch.Tensor,
    suptitle: str = "",
    surface: bool = True,
    permissible_error: float = 0.00005,
    figsize=(10, 10),
    **conv_kwargs,
):
    plot_fn = plot_2d
    fig, axss = plt.subplots(
        nrows=2,
        ncols=3,
        subplot_kw={"projection": "3d"} if surface else {},
        layout="compressed",
        figsize=figsize,
    )
    plot_fn(data, ax=axss[0, 0], title="Original: $F$", imshow=True)
    plot_fn(
        kernel.movedim(0, 1),
        ax=axss[1, 0],
        title="Kernel: $G$",
        vmin=-1,
        vmax=0,
        imshow=True,
    )

    eroded = adj.erosion(data, kernel, **conv_kwargs)
    opened = adj.dilation(eroded, kernel, **conv_kwargs)

    dilated = adj.dilation(data, kernel, **conv_kwargs)
    closed = adj.erosion(dilated, kernel, **conv_kwargs)

    plot_fn(eroded, ax=axss[0, 1], title=r"Eroded: $F \boxminus G$", imshow=True)
    plot_fn(
        opened,
        ax=axss[0, 2],
        title=r"Opened: $(F \boxminus G) \boxplus G$",
        imshow=True,
    )

    plot_fn(dilated, ax=axss[1, 1], title=r"Dilated: $F \boxplus G$", imshow=True)
    plot_fn(
        closed,
        ax=axss[1, 2],
        title=r"Closed: $(F \boxplus G) \boxminus G$",
        imshow=True,
    )

    dilation_err = data - dilated
    erosion_err = eroded - data
    opened_err = opened - data
    closed_err = data - closed

    for errs, name in zip(
        (dilation_err, erosion_err, opened_err, closed_err),
        ("Dilation", "Erosion", "Opening", "Closing"),
        strict=True,
    ):
        if (errs > permissible_error).any():
            plot_fn(
                errs, title=f"{suptitle + ': ' if suptitle else ''}errors in {name}"
            )

    fig.suptitle(suptitle)
    fig.show()

    assert (dilation_err < permissible_error).all(), (
        f"{suptitle}: Dilation should always be >="
    )
    assert (erosion_err < permissible_error).all(), (
        f"{suptitle}: Erosion should always be <="
    )
    assert (closed_err < permissible_error).all(), (
        f"{suptitle}: Closing should always be >=, {closed_err.max()=}"
    )
    assert (opened_err < permissible_error).all(), (
        f"{suptitle}: Opening should always be <=, {opened_err.max()=}"
    )


with torch.autograd.detect_anomaly():
    dg = (
        (data_peaked + torch.randn_like(data_peaked) * 1e-5)
        .clone()
        .requires_grad_(True)
    )
    test_res = ss_adj.dilation(dg, rounded_kernels[3:4], padding=kernel_size // 2)
    test_res.sum().backward()
    torch.cuda.synchronize()

plot_adjunction(
    data_slashed,
    ss_adj,
    rounded_kernels[4:5, :, ::2, ::2],
    padding=kernel_size // 4,
    surface=False,
    permissible_error=0.0005,
    suptitle="Illustration of effects of morphological operators on gaussians "
    "with minor artefacts ('spikes' and 'holes')",
    figsize=(10, 4),
)
input("Ready. (press Enter to close graph)")
