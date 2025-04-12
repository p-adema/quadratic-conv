import os
import sys

os.environ["KERAS_BACKEND"] = "torch"

import keras_tuner

sys.path.extend(".")

from src import load_data
from src.models.old import lenet_like

fixed_hp = keras_tuner.HyperParameters()

tuner = keras_tuner.GridSearch(
    hypermodel=lenet_like.lenet_like(img_channels=1, num_classes=10),
    objective="val_accuracy",
    seed=0,
    executions_per_trial=3,
    overwrite=False,
    directory="checkpoints",
    project_name="basic_kmnist",
    hyperparameters=fixed_hp,
)
tuner.search_space_summary()

cifar10 = load_data.k_mnist()

tuner.search(
    x=cifar10.x_train,
    y=cifar10.y_train,
    batch_size=2**9,
    epochs=40,
    validation_split=0.3,
)

tuner.results_summary()
