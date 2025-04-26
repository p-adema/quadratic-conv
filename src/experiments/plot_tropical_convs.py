import sys

import matplotlib.pyplot as plt
import torch

sys.path.extend(".")

from pytorch_semifield_conv import TropicalConv2D

from src import load_data


def try_tropical_conv(kernels: torch.Tensor | list, num_img: int = 0):
    kernels = torch.as_tensor(kernels, dtype=torch.float32)
    img = torch.as_tensor(load_data.mnist().x_train[num_img])
    # img = nn.functional.normalize(img.float())
    out_channels = kernels.shape[0]
    _, ((ax_original, *ax_maxs), (ax_unused, *ax_mins)) = plt.subplots(
        2, 1 + out_channels, layout="compressed"
    )
    ax_original.set_axis_off()
    ax_unused.set_axis_off()
    ax_original.imshow(img)
    ax_original.set_title("Original")
    repeat_img = img.unsqueeze(0).repeat((1, out_channels, 1, 1))
    conv_max = TropicalConv2D(is_max=True)(repeat_img, kernels)[0]
    for i, (ax, img_channel) in enumerate(zip(ax_maxs, conv_max, strict=True), 1):
        ax.set_axis_off()
        ax.imshow(img_channel)
        ax.set_title(f"Max #{i}")
    conv_min = TropicalConv2D(is_max=False)(repeat_img, kernels)[0]
    for i, (ax, img_channel) in enumerate(zip(ax_mins, conv_min, strict=True), 1):
        ax.set_axis_off()
        ax.imshow(img_channel)
        ax.set_title(f"Min #{i}")
    plt.show()


if __name__ == "__main__":
    INF = float("inf")
    test_kernels = -torch.asarray(
        [
            # 1: Does nothing
            [[[INF, INF, INF], [INF, 0, INF], [INF, INF, INF]]],
            # 2: Vertical max
            [[[INF, 0, INF], [INF, 0, INF], [INF, 0, INF]]],
            # 3: Horizontal max
            [[[INF, INF, INF], [0, 0, 0], [INF, INF, INF]]],
            # 4: 3x3 max
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            # 5: small quadratic max, isotropic
            [[[0.4, 0.1, 0.4], [0.1, 0, 0.1], [0.4, 0.1, 0.4]]],
            # 6: small quadratic max, wide horizontally
            [[[0.5, 0.2, 0.5], [0.05, 0, 0.05], [0.5, 0.2, 0.5]]],
        ]
    )
    for num in range(3):
        try_tropical_conv(
            test_kernels,
            num_img=num,
        )
