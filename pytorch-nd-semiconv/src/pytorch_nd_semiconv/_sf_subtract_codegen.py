from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numba
import pytorch_numba_extension_jit as pnex
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

from ._utils import ConvMeta

if TYPE_CHECKING:
    from ._sf_subtract import SubtractSemifield

warnings.simplefilter("ignore", numba.NumbaPerformanceWarning, 536)


class CompiledSubtractSemifield(NamedTuple):
    add: CUDADispatcher
    times: CUDADispatcher
    d_times_d_img: CUDADispatcher
    d_times_d_kernel: CUDADispatcher
    subtract: CUDADispatcher
    d_add_d_right: CUDADispatcher
    zero: float

    # Optional:
    post_sum: CUDADispatcher
    undo_post: CUDADispatcher
    post_sum_bwd: CUDADispatcher

    @classmethod
    def compile(cls, semifield: SubtractSemifield) -> Self:
        if semifield.post_sum is None:
            assert semifield.undo_post_sum is None, "post_sum not specified"
            assert semifield.d_post_d_acc is None, "post_sum not specified"
            post_sum, undo_post, post_bwd = lambda i: i, lambda i: i, lambda _: 1
        else:
            assert semifield.undo_post_sum is not None, "need inverse of post_sum"
            assert semifield.d_post_d_acc is not None, "need derivative of post_sum"
            post_sum = semifield.post_sum
            undo_post = semifield.undo_post_sum
            post_bwd = semifield.d_post_d_acc

        return CompiledSubtractSemifield(
            cuda.jit(semifield.add, device=True, inline="always", cache=True),
            cuda.jit(semifield.times, device=True, inline="always", cache=True),
            cuda.jit(semifield.d_times_d_img, device=True, inline="always", cache=True),
            cuda.jit(
                semifield.d_times_d_kernel, device=True, inline="always", cache=True
            ),
            cuda.jit(semifield.subtract, device=True, inline="always", cache=True),
            cuda.jit(semifield.d_add_d_right, device=True, inline="always", cache=True),
            float(semifield.zero),
            cuda.jit(post_sum, device=True, inline="always", cache=True),
            cuda.jit(undo_post, device=True, inline="always", cache=True),
            cuda.jit(post_bwd, device=True, inline="always", cache=True),
        )


def compile_forwards(
    semifield: CompiledSubtractSemifield,
    meta: ConvMeta,
    thread_block_size: int = 128,
    debug: bool = False,
    cache_name: str = "",
    to_extension: bool = True,
):
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

    acc = numba.float32(semifield.zero)

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
            f"{', '.join(f'k_{i}' for i in range(meta.ndim))} = "
            + ", ".join(f"step_{i}" for i in range(meta.ndim))
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
        {indent(meta.ndim)}acc = semifield.add(acc, val)

    out_img[b, o_c, {
        ", ".join(f"o_{i}" for i in range(meta.ndim))
    }] = semifield.post_sum(acc)
"""
    if debug:
        print("=" * 10 + "BEGIN SUB FWD KERNEL" + "=" * 10)
        print(code)
        print("=" * 10 + "END SUB FWD KERNEL" + "=" * 10)

    glob: dict[str, Any] = {"numba": numba, "semifield": semifield, "pnex": pnex}
    exec(code, glob)

    forwards = pnex.jit(
        n_threads="out_img.numel()",
        cache_id=f"subtract_{cache_name}_{meta.cache_id()}",
        to_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
    )(glob["forwards"])

    return forwards


def compile_backwards(
    semifield: CompiledSubtractSemifield,
    meta: ConvMeta,
    thread_block_size: int = 128,
    debug: bool = False,
    cache_name: str = "",
    to_extension: bool = True,
    kernel_inflation: int = 16,
):
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
    res_img: pnex.In("f32", "gradient"),
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

    res = semifield.undo_post(res_img[b, o_c, {
        ", ".join(f"o_{i}" for i in range(meta.ndim))
    }])
    res_grad = gradient[b, o_c, {
        ", ".join(f"o_{i}" for i in range(meta.ndim))
    }] * semifield.post_sum_bwd(res)
    inflate_pos = numba.cuda.threadIdx.x % {kernel_inflation}
    
    {
        newline().join(
            f"i_top_{i} = o_{i} * {meta.stride[i]} - {meta.pad_begs[i]}"
            for i in range(meta.ndim)
        )
    }

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
            f"{', '.join(f'k_{i}' for i in range(meta.ndim))} = "
            f"{', '.join(f'step_{i}' for i in range(meta.ndim))}"
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
        {indent(meta.ndim)}acc = semifield.subtract(res, val)
        {indent(meta.ndim)}val_grad = semifield.d_add_d_right(acc, val) * res_grad

        {indent(meta.ndim)}numba.cuda.atomic.add(
        {indent(meta.ndim)}    out_kernel_grad,
        {indent(meta.ndim)}    (k_o, group_idx, {
        ", ".join(f"k_{i}" for i in range(meta.ndim))
    }, inflate_pos),
        {
        indent(meta.ndim)
    }    semifield.d_times_d_kernel(img_val, kernel_val) * val_grad,
        {indent(meta.ndim)})
        {indent(meta.ndim)}numba.cuda.atomic.add(
        {indent(meta.ndim)}    out_img_grad,
        {indent(meta.ndim)}    (b, i_c, {
        ", ".join(f"i_{i}" for i in range(meta.ndim))
    }),
        {indent(meta.ndim)}    semifield.d_times_d_img(img_val, kernel_val) * val_grad,
        {indent(meta.ndim)})
"""
    if debug:
        print("=" * 10 + "BEGIN SUB BWD KERNEL" + "=" * 10)
        print(code)
        print("=" * 10 + "END SUB BWD KERNEL" + "=" * 10)

    glob: dict[str, Any] = {"numba": numba, "semifield": semifield, "pnex": pnex}
    exec(code, glob)

    backwards = pnex.jit(
        n_threads="gradient.numel()",
        to_extension=to_extension,
        verbose=debug,
        threads_per_block=thread_block_size,
        cache_id=f"subtract_{cache_name}_{meta.cache_id()}",
    )(glob["backwards"])

    def backwards_setup(ctx, inputs, output):
        img, kernel = inputs
        ctx.img = img
        ctx.kernel = kernel
        ctx.res_img = output

    def backwards_entry(ctx, grad_output):
        g_img, g_kern = backwards(ctx.img, ctx.kernel, grad_output, ctx.res_img)
        return g_img, g_kern.sum(-1)

    return backwards_entry, backwards_setup


def indent(indents: int = 1):
    return "    " * indents


def newline(indents: int = 1):
    return "\n" + "    " * indents
