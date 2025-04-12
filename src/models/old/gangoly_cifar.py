import os
from collections.abc import Callable

os.environ["KERAS_BACKEND"] = "torch"

import keras
import keras_tuner
from keras import layers

from src import convolutions, kernels
from src.models.utils import quiet_model

_sentinel = object()


def _conv_block(channels: int, kernel_size: int = 3):
    block = keras.Sequential(name=f"convs-{channels}")
    block.add(
        layers.Conv2D(
            channels, kernel_size, data_format="channels_first", padding="same"
        )
    )
    block.add(layers.BatchNormalization(axis=1))
    block.add(layers.ReLU())
    block.add(
        layers.Conv2D(
            channels, kernel_size, data_format="channels_first", padding="same"
        )
    )
    block.add(layers.ReLU())
    return block


def _fc_block(output_classes: int, h1: int = 1024, h2: int = 512, p_drop: float = 0.1):
    return keras.Sequential(
        [
            layers.Flatten(),
            layers.Dropout(p_drop),
            layers.Dense(h1, activation="relu"),
            layers.Dense(h2, activation="relu"),
            layers.Dropout(p_drop),
            layers.Dense(output_classes),
        ],
        name="fc_block",
    )


def _pool_layer(
    hp: keras_tuner.HyperParameters,
    layer_num: int,
    channels: int,
    init: str,
    stride: int = 1,
    force_temp=_sentinel,
):
    if not hp.Boolean("quadratic-pooling", default=True):
        return layers.MaxPool2D(
            data_format="channels_first",
            pool_size=(2, 2),
            strides=(stride, stride),
        )

    kernel_options = {
        "quadratic-iso": kernels.GaussKernelIso2D,
        "quadratic-multi": kernels.QuadraticKernelCholesky2D,
    }
    kernel_kind: str = hp.Choice(
        "quadratic-pool-kernel",
        list(kernel_options),
        parent_name="quadratic-pooling",
        parent_values=(True,),
    )
    kernel = kernel_options[kernel_kind]
    if force_temp is _sentinel:
        if hp.Boolean(
            "quadratic-pool-softpool",
            # parent_name="quadratic-pooling",
            # parent_values=(True,),
        ):
            soft_temp = hp.Float(
                "quadratic-pool-softpool-temp",
                1,
                16,
                sampling="log",
                step=4,
                # parent_name="quadratic-pool-softpool",
                # parent_values=(True,),
            )
        else:
            soft_temp = None
    else:
        soft_temp = force_temp

    pool_size = hp.Int(
        "quadratic-pool-size",
        3,
        5,  # can be higher
        step=2,
        # default=5,
        # parent_name="quadratic-pooling",
        # parent_values=(True,),
    )
    return layers.TorchModuleWrapper(
        convolutions.GenericConv2D(
            kernel=kernel(
                in_channels=1, out_channels=channels, kernel_size=pool_size, init=init
            ),
            conv=convolutions.TropicalConv2D(is_max=True, softmax_temp=soft_temp),
            # stride=1,  # NOTE: quadratic pools are currently not striding
        ),
        name=f"quadratic-{kernel_kind.removeprefix('quadratic-')}-pool-{layer_num}",
    )


def gangoly_cifar(
    img_channels: int = 1, num_classes: int = 10, force_temp=_sentinel
) -> Callable[[keras_tuner.HyperParameters], keras.Model]:
    """https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844"""

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

        model.add(_conv_block(32))
        model.add(
            _pool_layer(
                hp, 1, channels=32, init=init_kind, stride=2, force_temp=force_temp
            )
        )

        model.add(_conv_block(128))
        model.add(
            _pool_layer(
                hp, 2, channels=128, init=init_kind, stride=2, force_temp=force_temp
            )
        )

        model.add(layers.Dropout(0.05))

        model.add(_conv_block(256))
        model.add(
            _pool_layer(hp, 3, channels=256, init=init_kind, force_temp=force_temp)
        )

        model.add(_fc_block(output_classes=num_classes))

        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                "accuracy",
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            ],
        )
        return model  # quiet_model

    return build_model
