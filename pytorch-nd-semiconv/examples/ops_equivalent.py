import contextlib
import itertools
import warnings
from typing import Literal

import numpy as np
import torch
from pytorch_nd_semiconv import (
    BroadcastSemifield,
    SelectSemifield,
    SubtractSemifield,
    TorchLinearConv2D,
)
from tqdm.auto import tqdm

SEED = 0
NUM_TESTS = 100
TEST_BATCH_SIZE = 16

ROOT_P = 3.0
LOG_MU = 3.0

broadcast_max = BroadcastSemifield.tropical_max().dynamic(unfold_copy=False)
ext_max = SelectSemifield.tropical_max().dynamic(to_extension=True)
numba_max = SelectSemifield.tropical_max().dynamic(to_extension=False)

broadcast_min = BroadcastSemifield.tropical_min_negated().dynamic(unfold_copy=False)
ext_min = SelectSemifield.tropical_min_negated().dynamic(to_extension=True)
numba_min = SelectSemifield.tropical_min_negated().dynamic(to_extension=False)

broadcast_lin = BroadcastSemifield.linear().dynamic(unfold_copy=False)
ext_lin = SubtractSemifield.linear().dynamic(to_extension=True)
numba_lin = SubtractSemifield.linear().dynamic(to_extension=False)
torch_lin = TorchLinearConv2D()

broadcast_root = BroadcastSemifield.root(ROOT_P).dynamic(unfold_copy=False)
ext_root = SubtractSemifield.root(ROOT_P).dynamic(to_extension=True)
numba_root = SubtractSemifield.root(ROOT_P).dynamic(to_extension=False)

broadcast_log = BroadcastSemifield.log(LOG_MU).dynamic(unfold_copy=False)
ext_log = SubtractSemifield.log(LOG_MU).dynamic(to_extension=True)
numba_log = SubtractSemifield.log(LOG_MU).dynamic(to_extension=False)


def test_same_conv(
    name: str,
    conv1,
    conv2,
    imgs: torch.Tensor,
    kernels: torch.Tensor,
    stride,
    padding,
    dilation,
    groups: int,
    kind: Literal["corr", "conv"] = "corr",
):
    try:
        with (
            warnings.catch_warnings(action="ignore", category=UserWarning),
            torch.autograd.detect_anomaly(),
        ):
            imgs1 = imgs.clone().requires_grad_(True)
            kernels1 = kernels.clone().requires_grad_(True)
            imgs2 = imgs.clone().requires_grad_(True)
            kernels2 = kernels.clone().requires_grad_(True)

            if "root" in name:
                # Ensure we have values in R+
                imgs1.requires_grad_(False).abs_().add_(0.1).requires_grad_(True)
                imgs2.requires_grad_(False).abs_().add_(0.1).requires_grad_(True)
                kernels1.requires_grad_(False).abs_().add_(0.1).requires_grad_(True)
                kernels2.requires_grad_(False).abs_().add_(0.1).requires_grad_(True)

            res1 = conv1(
                imgs1,
                kernels1,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                kind=kind,
            )
            if torch.isnan(res1).any():
                msg = (
                    f"nan's in left result of {name}"
                    f" ({torch.isnan(res1).float().mean():.1%} nan)"
                )
                raise ValueError(msg)
            res2 = conv2(
                imgs2,
                kernels2,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                kind=kind,
            )
            if torch.isnan(res2).any():
                msg = (
                    f"nan's values in right result of {name}"
                    f" ({torch.isnan(res2).float().mean():.1%} nan)"
                )
                raise ValueError(msg)
            torch.testing.assert_close(
                res1, res2, msg=lambda m: f"Results {name}:\n{m}"
            )

            torch.manual_seed(0)
            tangent = torch.randn_like(res1)
            res1.backward(tangent)
            res2.backward(tangent)

            torch.testing.assert_close(
                imgs1.grad, imgs2.grad, msg=lambda m: f"Img grad {name}:\n{m}"
            )

            torch.testing.assert_close(
                kernels1.grad,
                kernels2.grad,
                msg=lambda m: f"Kernel grad {name}:\n{m}",
                atol=0.01,
                rtol=0.01,
            )
    except Exception as e:
        with contextlib.suppress(AttributeError):
            # python >= 3.11
            e.add_note(f"During {name}")
        raise


# noinspection PyTypeChecker
def check_params(
    image_size: tuple[int, ...],
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    dilation: tuple[int, ...],
    groups: int,
    in_gs: int,
    out_gs: int,
    kind: Literal["corr", "conv"],
):
    in_channels = groups * in_gs
    out_channels = groups * out_gs

    torch.manual_seed(0)
    imgs = torch.randn((TEST_BATCH_SIZE, in_channels, *image_size), device="cuda")
    kernels = -torch.rand((out_channels, in_gs, *kernel_size), device="cuda")
    test_kwargs = {
        "imgs": imgs,
        "kernels": kernels,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "kind": kind,
    }
    for name, op_b, op_e, op_n in (
        ("max", broadcast_max, ext_max, numba_max),
        ("min", broadcast_min, ext_min, numba_min),
    ):
        test_same_conv(f"{name}-B-E", op_b, op_e, **test_kwargs)
        test_same_conv(f"{name}-E-N", op_e, op_n, **test_kwargs)

        torch.library.opcheck(op_e.op, (imgs.clone().requires_grad_(True), kernels))
        torch.library.opcheck(op_n.op, (imgs, kernels.clone().requires_grad_(True)))

    for name, op_b, op_e, op_n in (
        ("lin", broadcast_lin, ext_lin, numba_lin),
        ("root", broadcast_root, ext_root, numba_root),
        ("log", broadcast_log, ext_log, numba_log),
    ):
        test_same_conv(f"{name}-B-E", op_b, op_e, **test_kwargs)
        test_same_conv(f"{name}-E-N", op_e, op_n, **test_kwargs)

        if "root" in name:
            args1 = (
                imgs.abs().add(0.1),
                kernels.abs().add(0.1).clone().requires_grad_(True),
            )
            args2 = (
                imgs.abs().add(0.1).clone().requires_grad_(True),
                kernels.abs().add(0.1),
            )
        else:
            args1 = (imgs, kernels.clone().requires_grad_(True))
            args2 = (imgs.clone().requires_grad_(True), kernels)

        for args in (args1, args2):
            torch.library.opcheck(op_n.op, args, atol=0.01, rtol=0.01)
            torch.library.opcheck(op_e.op, args, atol=0.01, rtol=0.01)


def check_param_space():
    seeds = np.random.SeedSequence(entropy=SEED).generate_state(NUM_TESTS)
    configs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ndim = rng.integers(1, 5).item()
        if ndim < 3:
            image_size = rng.integers(28, 70, ndim)
            kernel_size = rng.integers(1, 6, ndim)
        else:
            image_size = rng.integers(12, 20, ndim)
            kernel_size = rng.integers(1, 4, ndim)
        stride = rng.integers(1, 4, ndim)

        padding = rng.integers(0, 4, (ndim, 2))
        # We can't have padding larger than kernel_size, or the broadcasting impl
        # will behave poorly when the window only sees neutral elements.
        padding[padding.sum(1) >= kernel_size] = 0

        dilation = rng.integers(1, 4, ndim)
        groups, in_gs, out_gs = rng.integers(1, 5, 3).tolist()
        kind = rng.choice(["corr", "conv"]).item()
        configs.append(
            {
                "image_size": image_size.tolist(),
                "kernel_size": kernel_size.tolist(),
                "stride": stride.tolist(),
                "padding": padding.tolist(),
                "dilation": dilation.tolist(),
                "groups": groups,
                "in_gs": in_gs,
                "out_gs": out_gs,
                "kind": kind,
            }
        )

    config = None
    try:
        for config in tqdm(configs, desc="Verifying operators"):
            print(config)
            check_params(**config)

        for config in tqdm(configs, desc="Cached second pass"):
            check_params(**config)

    except Exception as e:
        try:
            e.add_note(f"Failed on {config=}")
        except AttributeError:
            print(f"Failed on {config=}")
        raise


if __name__ == "__main__":
    check_param_space()
    print("All checks OK!")
