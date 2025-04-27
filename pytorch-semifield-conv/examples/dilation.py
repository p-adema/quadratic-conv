import pytorch_numba_extension_jit as ptex
import torch
from numba import cuda
from pytorch_semifield_conv.utils import ConvMeta

test_imgs = torch.rand((2048, 6, 28, 28), device="cuda")
test_kernels = torch.zeros((6, 1, 5, 5), device="cuda")
meta = ConvMeta.infer(
    test_imgs, test_kernels, stride=2, padding=1, groups=6, kind="conv"
)
INF = float("inf")


@ptex.jit(
    [
        ptex.InputTensor("img", "f32", test_imgs.shape),
        ptex.InputTensor("kernel", "f32", test_kernels.shape),
        ptex.OutputTensor(
            "out_img", "f32", ("img.shape[0]", meta.out_cs, meta.out_ys, meta.out_xs)
        ),
        ptex.OutputTensor(
            "out_prov",
            "f32",
            (
                "img.shape[0]",
                meta.out_cs,
                meta.out_ys,
                meta.out_xs,
                3 if meta.krn_cs > 1 else 2,
            ),
        ),
    ],
    n_threads="out_img",
    compile_extension=True,
    verbose=False,
)
def test_dilation(img, kernel, out_img, out_prov):
    rem, o_x = divmod(cuda.grid(1), meta.out_xs)
    rem, o_y = divmod(rem, meta.out_ys)
    b, o_c = divmod(rem, meta.out_cs)
    if b >= img.shape[0]:
        return

    i_top_y = o_y * meta.stride - meta.padding
    i_left_x = o_x * meta.stride - meta.padding

    prov_x = prov_y = prov_group_idx = 255
    selected_val = -INF

    group_number = o_c // meta.krn_o_group_size
    # If we're not broadcasting, then we have a separate kernel
    # for every output channel. If we are broadcasting, we instead loop
    # around the kernels every k_os (which == krn_group_size)
    k_o = o_c if not meta.group_broadcasting else o_c % meta.krn_o_group_size

    # For a pooling, we have only one input channel, so group_idx is always 0
    for group_idx in range(meta.krn_cs):
        for y_step, i_y in enumerate(
            range(i_top_y, i_top_y + meta.krn_ys * meta.dilation, meta.dilation)
        ):
            for x_step, i_x in enumerate(
                range(
                    i_left_x,
                    i_left_x + meta.krn_xs * meta.dilation,
                    meta.dilation,
                )
            ):
                if i_x < 0 or i_x >= meta.img_xs or i_y < 0 or i_y >= meta.img_ys:
                    continue

                # Need to explicitly use seperate variable, due to compiler error
                k_x = meta.krn_xs - 1 - x_step if meta.mirror_kernel else x_step
                k_y = meta.krn_ys - 1 - y_step if meta.mirror_kernel else y_step

                i_c = group_number * meta.krn_cs + group_idx
                img_val = img[b, i_c, i_y, i_x]
                kernel_val = kernel[k_o, group_idx, k_y, k_x]

                val = img_val + kernel_val
                if selected_val < val:
                    selected_val = val
                    prov_y, prov_x = k_y, k_x
                    if meta.krn_cs > 1:
                        prov_group_idx = group_idx

    out_img[b, o_c, o_y, o_x] = selected_val

    out_prov[b, o_c, o_y, o_x, 0] = prov_y
    out_prov[b, o_c, o_y, o_x, 1] = prov_x
    if meta.krn_cs > 1:
        # out_prov is only size 3 if we require an index within the group
        out_prov[b, o_c, o_y, o_x, 2] = prov_group_idx


oi, op = test_dilation(test_imgs, test_kernels)
print(f"{oi.shape=} {op.shape=}")

torch.library.opcheck(test_dilation, (test_imgs, test_kernels))
