import os
from collections.abc import Callable

os.environ["KERAS_BACKEND"] = "torch"

import keras
import keras_tuner
from keras import layers

from src import convolutions, kernels
from src.models.utils import quiet_model


def _conv_layer(
    hp: keras_tuner.HyperParameters,
    layer_num: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    init: str,
    smoothing: bool = False,
):
    if smoothing:
        kernel_options = {
            "gauss-iso": kernels.GaussKernelIso2D,
            "gauss-multi": kernels.GaussKernelMulti2D,
        }
        kernel = kernel_options[
            hp.Choice(
                f"conv-{layer_num}-kernel",
                list(kernel_options.keys()),
                parent_name="do_smoothing",
                parent_values=(True,),
            )
        ]
    else:
        kernel = kernels.LearnedKernel

    return layers.TorchModuleWrapper(
        convolutions.GenericConv2D(
            kernel=kernel(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                init=init,
            ),
            conv=convolutions.LinearConv2D(),
        ),
        name=f"conv-{layer_num}",
    )


def _pool_layer(
    hp: keras_tuner.HyperParameters, layer_num: int, channels: int, init: str
):
    if not hp.Boolean("quadratic-pooling"):
        pool_size = hp.Int(
            "classic-pool-size",
            2,
            2,  # for larger images might be able to increase this
            parent_name="quadratic-pooling",
            parent_values=(False,),
        )
        return layers.MaxPool2D(
            data_format="channels_first", pool_size=(pool_size, pool_size)
        )

    kernel_options = {
        "quadratic-iso": kernels.GaussKernelIso2D,
        "quadratic-multi": kernels.QuadraticKernelMulti2D,
    }
    kernel_kind: str = hp.Choice(
        "quadratic-pool-kernel",
        list(kernel_options),
        parent_name="quadratic-pooling",
        parent_values=(True,),
    )
    kernel = kernel_options[kernel_kind]
    if hp.Boolean(
        "quadratic-pool-softpool",
        parent_name="quadratic-pooling",
        parent_values=(True,),
    ):
        soft_temp = hp.Float(
            "quadratic-pool-softpool-temp",
            1,
            1e5,
            sampling="log",
            step=10,
            parent_name="quadratic-pool-softpool",
            parent_values=(True,),
        )
    else:
        soft_temp = None

    pool_size = hp.Int(
        "quadratic-pool-size",
        3,
        4,
        parent_name="quadratic-pooling",
        parent_values=(True,),
    )
    return layers.TorchModuleWrapper(
        convolutions.GenericConv2D(
            kernel=kernel(
                in_channels=1, out_channels=channels, kernel_size=pool_size, init=init
            ),
            conv=convolutions.TropicalConv2D(is_max=True, softmax_temp=soft_temp),
        ),
        name=f"quadratic-{kernel_kind.removeprefix('quadratic-')}-pool-{layer_num}",
    )


def lenet_like(
    img_channels: int = 1, num_classes: int = 10
) -> Callable[[keras_tuner.HyperParameters], keras.Model]:
    def build_model(hp: keras_tuner.HyperParameters = None) -> keras.Model:
        if hp is None:
            hp = keras_tuner.HyperParameters()

        init_kind = hp.Choice("init_kind", ["normal"])

        model = keras.Sequential()
        model.add(
            layers.TorchModuleWrapper(
                convolutions.CoerceImage4D(img_channels), name="reshape"
            )
        )
        if hp.Boolean("do_smoothing"):
            model.add(
                _conv_layer(
                    hp,
                    0,
                    in_channels=img_channels,
                    out_channels=10,
                    kernel_size=7,
                    init=init_kind,
                    smoothing=True,
                )
            )
            first_channels = 10
        else:
            first_channels = img_channels

        model.add(
            _conv_layer(
                hp,
                1,
                in_channels=first_channels,
                out_channels=20,
                kernel_size=5,
                init=init_kind,
            )
        )
        model.add(layers.ReLU())
        model.add(_pool_layer(hp, 1, channels=20, init=init_kind))

        model.add(
            _conv_layer(
                hp, 2, in_channels=20, out_channels=50, kernel_size=5, init=init_kind
            )
        )
        model.add(layers.ReLU())
        model.add(_pool_layer(hp, 2, channels=50, init=init_kind))

        model.add(layers.Flatten())
        if hp.Boolean("logit-hidden"):
            model.add(
                layers.Dense(
                    hp.Choice(
                        "logit-hidden-size",
                        (
                            # 50,
                            # 100,
                            500,
                        ),
                        parent_name="logit-hidden",
                        parent_values=(True,),
                    ),
                    name="logit-hidden",
                )
            )
            model.add(layers.ReLU())

        model.add(layers.Dense(num_classes, name="logit-predictor"))

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                "accuracy",
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            ],
        )
        return quiet_model(model)

    return build_model
