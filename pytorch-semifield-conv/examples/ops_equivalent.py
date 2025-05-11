import contextlib
import itertools
import random
import warnings

import torch
from pytorch_semifield_conv import (
    BroadcastSemifield,
    SelectSemifield,
    SubtractSemifield,
    TorchLinearConv2D,
)
from tqdm.auto import tqdm

broadcast_max = BroadcastSemifield.tropical_max().dynamic(unfold_copy=False)
copy_max = BroadcastSemifield.tropical_max().dynamic(unfold_copy=True)
ext_max = SelectSemifield.tropical_max().dynamic(to_extension=True)
numba_max = SelectSemifield.tropical_max().dynamic(to_extension=False)

broadcast_min = BroadcastSemifield.tropical_min_negated().dynamic(unfold_copy=False)
copy_min = BroadcastSemifield.tropical_min_negated().dynamic(unfold_copy=True)
ext_min = SelectSemifield.tropical_min_negated().dynamic(to_extension=True)
numba_min = SelectSemifield.tropical_min_negated().dynamic(to_extension=False)

broadcast_lin = BroadcastSemifield.linear().dynamic(unfold_copy=False)
copy_lin = BroadcastSemifield.linear().dynamic(unfold_copy=True)
ext_lin = SubtractSemifield.linear().dynamic(to_extension=True)
numba_lin = SubtractSemifield.linear().dynamic(to_extension=False)
torch_lin = TorchLinearConv2D()

ROOT_P = 3.0
broadcast_root = BroadcastSemifield.root(ROOT_P).dynamic(unfold_copy=False)
copy_root = BroadcastSemifield.root(ROOT_P).dynamic(unfold_copy=True)
ext_root = SubtractSemifield.root(ROOT_P).dynamic(to_extension=True)
numba_root = SubtractSemifield.root(ROOT_P).dynamic(to_extension=False)

LOG_MU = 3.0
broadcast_log = BroadcastSemifield.log(LOG_MU).dynamic(unfold_copy=False)
copy_log = BroadcastSemifield.log(LOG_MU).dynamic(unfold_copy=True)
ext_log = SubtractSemifield.log(LOG_MU).dynamic(to_extension=True)
numba_log = SubtractSemifield.log(LOG_MU).dynamic(to_extension=False)


IN_CHANNELS = 6
OUT_CHANNELS = 9
GROUPS = 3

NUM_TESTS = 100

torch.manual_seed(0)
test_imgs = torch.randn((64, IN_CHANNELS, 50, 50), device="cuda")
test_kernels = -torch.rand((OUT_CHANNELS, IN_CHANNELS // GROUPS, 5, 5), device="cuda")


def test_same_conv(name: str, conv1, conv2, stride=1, padding=1, dilation=1):
    torch.manual_seed(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with torch.autograd.detect_anomaly():
                imgs1 = test_imgs.clone().requires_grad_(True)
                kernels1 = test_kernels.clone().requires_grad_(True)
                imgs2 = test_imgs.clone().requires_grad_(True)
                kernels2 = test_kernels.clone().requires_grad_(True)

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
                    groups=GROUPS,
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
                    groups=GROUPS,
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
def check_params(stride, padding, dilation):
    for name, op_c, op_b, op_e, op_n in (
        ("max", copy_max, broadcast_max, ext_max, numba_max),
        ("min", copy_min, broadcast_min, ext_min, numba_min),
    ):
        test_same_conv(f"{name}-C-B", op_c, op_b, stride, padding, dilation)
        test_same_conv(f"{name}-B-E", op_b, op_e, stride, padding, dilation)
        test_same_conv(f"{name}-E-N", op_e, op_n, stride, padding, dilation)

        torch.library.opcheck(
            op_e.op, (test_imgs.clone().requires_grad_(True), test_kernels)
        )
        torch.library.opcheck(
            op_n.op, (test_imgs, test_kernels.clone().requires_grad_(True))
        )

    for name, op_t, op_c, op_b, op_e, op_n in (
        ("lin", torch_lin, copy_lin, broadcast_lin, ext_lin, numba_lin),
        ("root", None, copy_root, broadcast_root, ext_root, numba_root),
        ("log", None, copy_log, broadcast_log, ext_log, numba_log),
    ):
        if op_t is not None:
            test_same_conv(f"{name}-T-B", op_t, op_b, stride, padding, dilation)

        test_same_conv(f"{name}-C-B", op_c, op_b, stride, padding, dilation)
        test_same_conv(f"{name}-B-E", op_b, op_e, stride, padding, dilation)
        test_same_conv(f"{name}-E-N", op_e, op_n, stride, padding, dilation)

        if "root" in name:
            args1 = (
                test_imgs.abs().add(0.1),
                test_kernels.abs().add(0.1).clone().requires_grad_(True),
            )
            args2 = (
                test_imgs.abs().add(0.1).clone().requires_grad_(True),
                test_kernels.abs().add(0.1),
            )
        else:
            args1 = (test_imgs, test_kernels.clone().requires_grad_(True))
            args2 = (test_imgs.clone().requires_grad_(True), test_kernels)

        for args in (args1, args2):
            torch.library.opcheck(op_e.op, args, atol=0.01, rtol=0.01)


def check_param_space(
    max_sy: int = 3,
    max_sx: int = 3,
    max_pyb: int = 2,
    max_pye: int = 2,
    max_pxb: int = 2,
    max_pxe: int = 2,
    max_dy: int = 3,
    max_dx: int = 3,
):
    param_space = itertools.product(
        range(1, max_sy + 1),
        range(1, max_sx + 1),
        range(max_pyb + 1),
        range(max_pye + 1),
        range(max_pxb + 1),
        range(max_pxe + 1),
        range(1, max_dx + 1),
        range(1, max_dy + 1),
    )
    random.seed(0)
    samples = random.choices(list(param_space), k=NUM_TESTS)

    bar = tqdm(samples, desc="Verifying operators")
    for sy, sx, pyb, pye, pxb, pxe, dx, dy in bar:
        bar.set_postfix(
            stride=(sy, sx),
            padding=((pyb, pye), (pxb, pxe)),
            dilation=(dy, dx),
            refresh=True,
        )
        check_params((sy, sx), ((pyb, pye), (pxb, pxe)), (dy, dx))

    for sy, sx, pyb, pye, pxb, pxe, dx, dy in tqdm(samples, desc="Cached second pass"):
        check_params((sy, sx), ((pyb, pye), (pxb, pxe)), (dy, dx))


if __name__ == "__main__":
    check_param_space()
    print("All checks OK!")
