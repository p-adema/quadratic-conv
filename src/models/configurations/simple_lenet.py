from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

from ..utils import make_pooling_function
from .utils import handle_common


def lenet_configs(
    standard_sizes: tuple[int, ...] = (1, 2, 3, 4, 5),
    kernel_sizes: tuple[int, ...] = (2, 3, 5, 7, 11),
    group_sizes: tuple[int, ...] = (1,),
    g_broadcasting_options: tuple[bool, ...] = (False,),
    channel_add_options: tuple[bool, ...] = (False,),
    spread_gradient_options: tuple[bool, ...] = (False,),
    do_standard: bool = True,
    do_iso: bool = True,
    do_aniso: bool = True,
    iso_inits: tuple[str, ...] = (
        # "uniform",
        "ss",
        # "log-ss",
    ),
    aniso_inits: tuple[tuple[str, str], ...] = (
        # ("uniform", "uniform"),
        # ("uniform-iso", "uniform"),
        # ("uniform-iso", "spin"),
        ("ss-iso", "spin"),
        ("skewed", "spin"),
    ),
    keep_tag_if: Callable[[str], bool] | None = None,
    progress_bar: str | None = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    trials = {}

    for size in standard_sizes if do_standard else ():
        trials[f"standard-{size}"] = {
            "pool_fn": make_pooling_function("standard", size),
            "init": None,
        }

    for size, init, group_size, g_broadcast, channel_add, spread_grad in (
        product(
            kernel_sizes,
            iso_inits,
            group_sizes,
            g_broadcasting_options,
            channel_add_options,
            spread_gradient_options,
        )
        if do_iso
        else ()
    ):
        trials[
            (
                f"iso-{size}-{init}"
                f"{'-gsize' + str(group_size) if len(group_sizes) > 1 else ''}"
                f"{'-broadcast1' if g_broadcast else ''}"
                f"{'-channeladd' if channel_add else ''}"
                f"{'-spreadgrad' if spread_grad else ''}"
            )
        ] = {
            "pool_fn": make_pooling_function(
                "iso",
                size,
                group_size=group_size,
                group_broadcasting=g_broadcast,
                channel_add=channel_add,
                spread_gradient=spread_grad,
            ),
            "init": {"var": init},
        }

    for size, (
        v_init,
        t_init,
    ), group_size, g_broadcast, channel_add, spread_grad in (
        product(
            kernel_sizes,
            aniso_inits,
            group_sizes,
            g_broadcasting_options,
            channel_add_options,
            spread_gradient_options,
        )
        if do_aniso
        else ()
    ):
        trials[
            f"aniso-{size}-{v_init}-{t_init}"
            f"{'-gsize' + str(group_size) if len(group_sizes) > 1 else ''}"
            f"{'-broadcast1' if g_broadcast else ''}"
            f"{'-channeladd' if channel_add else ''}"
            f"{'-spreadgrad' if spread_grad else ''}"
        ] = {
            "pool_fn": make_pooling_function(
                "aniso",
                size,
                group_size=group_size,
                group_broadcasting=g_broadcast,
                channel_add=channel_add,
                spread_gradient=spread_grad,
            ),
            "init": {"var": v_init, "theta": t_init},
        }

    return handle_common(trials, keep_tag_if, progress_bar)


def standard_configs(name: str | None = None):
    return lenet_configs(progress_bar=name)


def group_configs(name: str | None = None):
    return lenet_configs(
        g_broadcasting_options=(False, True),
        channel_add_options=(False, True),
        group_sizes=(1, 2),
        kernel_sizes=(7,),
        do_standard=False,
        progress_bar=name,
    )
