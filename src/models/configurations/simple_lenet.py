from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

from ..utils import make_pooling_function
from .utils import handle_common


def lenet_configs(
    kernel_sizes: tuple[int, ...] = (2, 3, 5, 7, 11),
    group_sizes: tuple[int, ...] = (1,),
    g_broadcasting_options: tuple[bool, ...] = (False,),
    do_standard: bool = True,
    do_iso: bool = True,
    do_aniso: bool = True,
    iso_inits: tuple[str, ...] = ("uniform", "ss"),
    aniso_inits: tuple[tuple[str, str], ...] = (
        ("uniform", "uniform"),
        ("uniform-iso", "uniform"),
        ("uniform-iso", "spin"),
        ("ss-iso", "spin"),
        ("skewed", "spin"),
    ),
    max_standard_size: int = 7,
    keep_tag_if: Callable[[str], bool] | None = None,
    progress_bar: str | None = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    trials = {}

    if do_standard:
        for size in (s for s in kernel_sizes if s <= max_standard_size):
            trials[f"standard-{size}"] = {
                "pool_fn": make_pooling_function("standard", size),
                "init": None,
            }

    if do_iso:
        for size, init, group_size, g_broadcast in product(
            kernel_sizes, iso_inits, group_sizes, g_broadcasting_options
        ):
            trials[
                (
                    f"iso-{size}-{init}"
                    f"{'-gsize' + str(group_size) if len(group_sizes) > 1 else ''}"
                    f"{'-broadcast1' if g_broadcast else ''}"
                )
            ] = {
                "pool_fn": make_pooling_function(
                    "iso",
                    size,
                    group_size=group_size,
                    group_broadcasting=g_broadcast,
                ),
                "init": {"var": init},
            }

    if do_aniso:
        for size, (v_init, t_init), group_size, g_broadcast in product(
            kernel_sizes, aniso_inits, group_sizes, g_broadcasting_options
        ):
            trials[
                f"aniso-{size}-{v_init}-{t_init}"
                f"{'-gsize' + str(group_size) if len(group_sizes) > 1 else ''}"
                f"{'-broadcast1' if g_broadcast else ''}"
            ] = {
                "pool_fn": make_pooling_function(
                    "aniso",
                    size,
                    group_size=group_size,
                    group_broadcasting=g_broadcast,
                ),
                "init": {"var": v_init, "theta": t_init},
            }

    return handle_common(trials, keep_tag_if, progress_bar)


def standard_configs(name: str | None = None):
    return lenet_configs(progress_bar=name)


def group_configs(name: str | None = None):
    return lenet_configs(
        g_broadcasting_options=(False, True),
        group_sizes=(1, 2, 3),
        kernel_sizes=(9,),
        iso_inits=("uniform", "ss"),
        aniso_inits=(
            ("uniform-iso", "uniform"),
            ("skewed", "spin"),
        ),
        do_standard=False,
        progress_bar=name,
    )
