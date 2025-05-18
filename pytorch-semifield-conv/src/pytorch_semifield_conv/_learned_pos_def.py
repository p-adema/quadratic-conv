import math

import torch
from torch import nn


class CovSpectral2D(nn.Module):
    """A utility class that parameterises diagonally decomposed 2D covariance matrices
    using parameters for standard deviations and the rotation of the principal axes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init: dict = None,
        kernel_size: int = None,
    ):
        super().__init__()
        init: dict[str, str | float] = init or {"var": "skewed", "theta": "spin"}
        variances = torch.empty((out_channels, in_channels, 2))
        thetas = torch.empty((out_channels, in_channels))

        if isinstance(init["var"], float):
            nn.init.constant_(variances, init["var"])
        elif init["var"] == "uniform":
            nn.init.uniform_(variances, 1, 16)
        elif init["var"] == "ss-iso":
            spaced_vars = torch.linspace(
                1.0,
                2.0 * (kernel_size // 2) ** 2,
                steps=out_channels * in_channels,
            )
            permutation = torch.randperm(spaced_vars.shape[0])
            variances[:] = spaced_vars[permutation].reshape(
                out_channels, in_channels, 1
            )
        elif init["var"] == "log-ss-iso":
            spaced_vars = torch.logspace(
                math.log10(1.0),
                math.log10(2.0 * (kernel_size // 2) ** 2),
                steps=out_channels * in_channels,
            )
            permutation = torch.randperm(spaced_vars.shape[0])
            variances[:] = spaced_vars[permutation].reshape(
                out_channels, in_channels, 1
            )
        elif init["var"] == "uniform-iso":
            nn.init.uniform_(variances[..., 0], 1, 16)
            variances[..., 1] = variances[..., 0]
        elif init["var"] == "normal":
            nn.init.trunc_normal_(variances, mean=8.0, a=1.0, b=16.0, std=4.0)
        elif init["var"] == "skewed":
            nn.init.uniform_(variances[..., 0], 1, 8.0)
            nn.init.uniform_(variances[..., 1], 20.0, 24.0)
        else:
            raise ValueError(f"Invalid {init['var']=}")

        if init["theta"] == "uniform":
            nn.init.uniform_(thetas, 0, torch.pi)
        elif init["theta"] == "spin":
            spaced_thetas = torch.linspace(
                0, torch.pi, steps=out_channels * in_channels + 1
            )[:-1]
            permutation = torch.randperm(spaced_thetas.shape[0])
            thetas[:] = spaced_thetas[permutation].reshape(out_channels, in_channels)
        else:
            raise ValueError(f"Invalid {init['theta']=}")

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
        return torch.einsum("oivd,oidD,oiVD->oivV", rot, inv_diag, rot).contiguous()

    def cov(self):
        return torch.linalg.inv(self.inverse_cov())

    def extra_repr(self):
        out_channels, in_channels = self.theta.shape
        return f"{in_channels}, {out_channels}"

    forward = inverse_cov


class CovCholesky2D(nn.Module):
    """A utility class that parameterises Cholesky-decomposed 2D covariance matrices
    using parameters for standard deviations and for Pearson's R (`corr`)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init: dict = 0.0,
        kernel_size: int = None,
    ):
        super().__init__()
        init: dict[str, str | float] = init or {"var": "uniform"}

        variances = torch.empty((2, out_channels, in_channels))
        corr = torch.empty((out_channels, in_channels))
        if isinstance(init["var"], float):
            nn.init.constant_(variances, init["var"])
        elif init["var"] == "uniform":
            nn.init.uniform_(variances, 1, 16)
        elif init["var"] == "ss-iso":
            spaced_vars = torch.linspace(
                1.0,
                2.0 * (kernel_size // 2) ** 2,
                steps=out_channels * in_channels,
            )
            permutation = torch.randperm(spaced_vars.shape[0])
            variances[:] = spaced_vars[permutation].reshape(
                1, out_channels, in_channels
            )
        elif init["var"] == "log-ss-iso":
            spaced_vars = torch.logspace(
                math.log10(1.0),
                math.log10(2.0 * (kernel_size // 2) ** 2),
                steps=out_channels * in_channels,
            )
            permutation = torch.randperm(spaced_vars.shape[0])
            variances[:] = spaced_vars[permutation].reshape(
                1, out_channels, in_channels
            )
        elif init["var"] == "uniform-iso":
            nn.init.uniform_(variances[0], 1, 16)
            variances[1] = variances[0]
        elif init["var"] == "normal":
            nn.init.trunc_normal_(variances, mean=8.0, a=1.0, b=16.0, std=4.0)
        elif init["var"] == "skewed":
            nn.init.uniform_(variances[0], 1, 8.0)
            nn.init.uniform_(variances[1], 20.0, 24.0)
        else:
            raise ValueError(f"Invalid {init['var']=}")

        nn.init.trunc_normal_(corr, mean=0, std=0.5, a=-1, b=1)

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
