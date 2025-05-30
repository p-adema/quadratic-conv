from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numba
import numpy as np
import pytorch_numba_extension_jit as pnex
import torch
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

from ._utils import ConvMeta

if TYPE_CHECKING:
    from ._sf_select import SelectSemifield

warnings.simplefilter("ignore", numba.NumbaPerformanceWarning, 536)


class CompiledSelectSemifield(NamedTuple):
    add_select: CUDADispatcher
    times: CUDADispatcher
    d_times_d_img: CUDADispatcher
    d_times_d_kernel: CUDADispatcher
    zero: float

    @classmethod
    def compile(cls, semifield: SelectSemifield) -> Self:
        return CompiledSelectSemifield(
            cuda.jit(semifield.add_select, device=True, inline="always", cache=True),
            cuda.jit(semifield.times, device=True, inline="always", cache=True),
            cuda.jit(semifield.d_times_d_img, device=True, inline="always", cache=True),
            cuda.jit(
                semifield.d_times_d_kernel, device=True, inline="always", cache=True
            ),
            float(semifield.zero),
        )


class _ProvenanceType(NamedTuple):
    bits: int
    maxval: int

    @classmethod
    def smallest_required(cls, meta: ConvMeta):
        largest = max(max(meta.krn_spatial), meta.krn_cs)
        if largest < np.iinfo(np.uint8).max:
            return cls(8, np.iinfo(np.uint8).max)

        assert largest < np.iinfo(np.uint16).max, "That's not going to fit in memory"
        return cls(16, np.iinfo(np.uint16).max)

    @property
    def torch_type(self):
        if self.bits == 8:
            return torch.uint8
        if self.bits == 16:
            return torch.uint16

        raise ValueError


def compile_forwards(
    semifield: CompiledSelectSemifield,
    meta: ConvMeta,
    thread_block_size: int = None,
    debug: bool = False,
    cache_name: str = "",
    to_extension: bool = True,
):
    prov_t = _ProvenanceType.smallest_required(meta)
    if thread_block_size is None:
        thread_block_size = 128

    code = f"""
def forwards(
    img: pnex.In("f32", (None, {meta.img_cs}, {
        ", ".join(str(i) for i in meta.img_spatial)
    })),
    kernel: pnex.In("f32", ({meta.krn_os}, {meta.krn_cs}, {
        ", ".join(str(i) for i in meta.krn_spatial)
    })),
    out_img: pnex.Out("f32", ("img.size(0)", {meta.out_cs}, {
        ", ".join(str(i) for i in meta.out_spatial)
    })),
    out_prov: pnex.Out("u{prov_t.bits}", ("img.size(0)", {meta.out_cs}, {
        ", ".join(str(i) for i in meta.out_spatial)
    }, {meta.ndim + (meta.krn_cs > 1)})),
):
    rem = numba.cuda.grid(1)
    {
        newline().join(
            f"rem, o_{i} = divmod(rem, {meta.out_spatial[i]})"
            for i in reversed(range(meta.ndim))
        )
    }
    b, o_c = divmod(rem, {meta.out_cs})
    if b >= img.shape[0]:
        return

    {
        newline().join(
            f"i_top_{i} = o_{i} * {meta.stride[i]} - {meta.pad_begs[i]}"
            for i in range(meta.ndim)
        )
    }

    {" = ".join(f"prov_{i}" for i in range(meta.ndim))} = prov_group_idx = {
        prov_t.maxval
    }
    selected_val = numba.float32(semifield.zero)

    group_number = o_c // {meta.grp_o}
    k_o = {"o_c" if not meta.group_broadcasting else f"o_c % {meta.grp_o}"}

    for group_idx in range({meta.krn_cs}):
        {
        newline(2).join(
            f"{'    ' * i}for step_{i}, i_{i} in enumerate(range("
            f"i_top_{i},"
            f" i_top_{i} + {meta.krn_spatial[i] * meta.dilation[i]},"
            f" {meta.dilation[i]})):"
            for i in range(meta.ndim)
        )
    }
        {indent(meta.ndim)}if {
        " or ".join(
            f"i_{i} < 0 or i_{i} >= {meta.img_spatial[i]}" for i in range(meta.ndim)
        )
    }:
        {indent(meta.ndim)}    continue

        {indent(meta.ndim)}{
        f"{', '.join(f'k_{i}' for i in range(meta.ndim))} = "
        + ", ".join(f"{meta.krn_spatial[i]} - 1 - step_{i}" for i in range(meta.ndim))
        if meta.mirror_kernel
        else (
            f"{', '.join(f'k_{i}' for i in range(meta.ndim))}"
            f" = {', '.join(f'step_{i}' for i in range(meta.ndim))}"
        )
    }

        {indent(meta.ndim)}i_c = group_number * {meta.krn_cs} + group_idx
        {indent(meta.ndim)}img_val = img[b, i_c, {
        ", ".join(f"i_{i}" for i in range(meta.ndim))
    }]
        {indent(meta.ndim)}kernel_val = kernel[k_o, group_idx, {
        ", ".join(f"k_{i}" for i in range(meta.ndim))
    }]

        {indent(meta.ndim)}val = semifield.times(img_val, kernel_val)
        {indent(meta.ndim)}if semifield.add_select(selected_val, val):
        {indent(meta.ndim)}    selected_val = val
        {indent(meta.ndim)}    {", ".join(f"prov_{i}" for i in range(meta.ndim))} = {
        ", ".join(f"k_{i}" for i in range(meta.ndim))
    }
        {indent(meta.ndim)}    {
        "prov_group_idx = group_idx"
        if meta.krn_cs > 1
        else "# Only one channel, no group index needed"
    }

    out_img[b, o_c, {", ".join(f"o_{i}" for i in range(meta.ndim))}] = selected_val

    {
        newline().join(
            f"out_prov[b, o_c, "
            f"{', '.join(f'o_{i}' for i in range(meta.ndim))}, {prov_i}"
            f"] = prov_{prov_i}"
            for prov_i in range(meta.ndim)
        )
    }
    {
        (
            f"out_prov[b, o_c, "
            f"{', '.join(f'o_{i}' for i in range(meta.ndim))}, {meta.ndim}"
            f"] = prov_group_idx"
        )
        if meta.krn_cs > 1
        else "# Only one channel, no group prov needed"
    }
    """
    if debug:
        print("=" * 10 + "BEGIN SEL FWD KERNEL" + "=" * 10)
        print(code)
        print("=" * 10 + "END SEL FWD KERNEL" + "=" * 10)

    glob: dict[str, Any] = {"numba": numba, "semifield": semifield, "pnex": pnex}
    exec(code, glob)

    forwards = pnex.jit(
        n_threads="out_img.numel()",
        to_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
        cache_id=f"select_{cache_name}_{meta.cache_id()}",
    )(glob["forwards"])

    return forwards


def compile_backwards(
    semifield: CompiledSelectSemifield,
    meta: ConvMeta,
    thread_block_size: int = None,
    debug: bool = False,
    cache_name: str = "",
    to_extension: bool = True,
    kernel_inflation: int = 16,
):
    prov_t = _ProvenanceType.smallest_required(meta)
    if thread_block_size is None:
        thread_block_size = 128

    code = f"""
def backwards(
    img: pnex.In("f32", (None, {meta.img_cs}, {
        ", ".join(str(i) for i in meta.img_spatial)
    })),
    kernel: pnex.In("f32", ({meta.krn_os}, {meta.krn_cs}, {
        ", ".join(str(i) for i in meta.krn_spatial)
    })),
    gradient: pnex.In("f32", ("img.size(0)", {meta.out_cs}, {
        ", ".join(str(i) for i in meta.out_spatial)
    })),
    prov: pnex.In("u{prov_t.bits}", ("img.size(0)", {meta.out_cs}, {
        ", ".join(str(i) for i in meta.out_spatial)
    }, {meta.ndim + (meta.krn_cs > 1)})),
    out_img_grad: pnex.Out("f32", "img", init=0),
    out_kernel_grad: pnex.Out("f32", ({meta.krn_os}, {meta.krn_cs}, {
        ", ".join(str(i) for i in meta.krn_spatial)
    }, {kernel_inflation}), init=0),
):
    rem = numba.cuda.grid(1)
    {
        newline().join(
            f"rem, o_{i} = divmod(rem, {meta.out_spatial[i]})"
            for i in reversed(range(meta.ndim))
        )
    }
    b, o_c = divmod(rem, {meta.out_cs})
    if b >= img.shape[0]:
        return

    group_number = o_c // {meta.grp_o}
    k_o = {"o_c" if not meta.group_broadcasting else f"o_c % {meta.grp_o}"}

    grad_val = gradient[b, o_c, {", ".join(f"o_{i}" for i in range(meta.ndim))}]
    {
        newline().join(
            f"k_prov_{prov_i} = prov[b, o_c,"
            f" {', '.join(f'o_{i}' for i in range(meta.ndim))}, {prov_i}]"
            for prov_i in range(meta.ndim)
        )
    }
    # Index within our group, for which of the channels we ended up picking
    # If krn_cs == 1, we can only pick the singular channel we have: always 0
    prov_group_idx = {
        f"prov[b, o_c, {', '.join(f'o_{i}' for i in range(meta.ndim))}, {meta.ndim}]"
        if meta.krn_cs > 1
        else "0"
    }

    if k_prov_0 == {prov_t.maxval}:
        # We kept the original neutral element,
        # so our gradient can't be related to the image
        return

    {
        f"{', '.join(f'step_{i}' for i in range(meta.ndim))} = "
        + ", ".join(f"{meta.krn_spatial[i]} - 1 - k_prov_{i}" for i in range(meta.ndim))
        if meta.mirror_kernel
        else (
            f"{', '.join(f'step_{i}' for i in range(meta.ndim))} ="
            f" {', '.join(f'k_prov_{i}' for i in range(meta.ndim))}"
        )
    }

    {
        newline().join(
            f"i_top_{i} = o_{i} * {meta.stride[i]} - {meta.pad_begs[i]}"
            for i in range(meta.ndim)
        )
    }

    i_prov_c = group_number * {meta.krn_cs} + prov_group_idx
    {
        newline().join(
            f"i_prov_{i} = i_top_{i} + {meta.dilation[i]} * step_{i}"
            for i in range(meta.ndim)
        )
    }

    kernel_val = kernel[k_o, prov_group_idx, {
        ", ".join(f"k_prov_{i}" for i in range(meta.ndim))
    }]
    img_val = img[b, i_prov_c, {", ".join(f"i_prov_{i}" for i in range(meta.ndim))}]

    d_kernel = semifield.d_times_d_kernel(img_val, kernel_val) * grad_val
    inflate_pos = numba.cuda.threadIdx.x % {kernel_inflation}
    numba.cuda.atomic.add(
        out_kernel_grad,
        (k_o, prov_group_idx, {
        ", ".join(f"k_prov_{i}" for i in range(meta.ndim))
    }, inflate_pos),
        d_kernel,
    )

    d_img = semifield.d_times_d_img(img_val, kernel_val) * grad_val
    numba.cuda.atomic.add(out_img_grad, (b, i_prov_c, {
        ", ".join(f"i_prov_{i}" for i in range(meta.ndim))
    }), d_img)
"""
    if debug:
        print("=" * 10 + "BEGIN SEL BWD KERNEL" + "=" * 10)
        print(code)
        print("=" * 10 + "END SEL BWD KERNEL" + "=" * 10)

    glob: dict[str, Any] = {"numba": numba, "semifield": semifield, "pnex": pnex}
    exec(code, glob)

    backwards = pnex.jit(
        n_threads="gradient.numel()",
        cache_id=f"select_{cache_name}_{meta.cache_id()}",
        to_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
    )(glob["backwards"])

    def backwards_setup(ctx, inputs, output):
        img, kernel = inputs
        _out_img, prov = output
        ctx.img = img
        ctx.kernel = kernel
        ctx.prov = prov

    def backwards_entry(ctx, grad_output, _grad_prov):
        g_img, g_kern = backwards(ctx.img, ctx.kernel, grad_output, ctx.prov)
        return g_img, g_kern.sum(-1)

    return backwards_entry, backwards_setup


def indent(indents: int = 1):
    return "    " * indents


def newline(indents: int = 1):
    return "\n" + "    " * indents
