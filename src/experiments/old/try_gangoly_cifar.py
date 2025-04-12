import os
import sys

os.environ["KERAS_BACKEND"] = "torch"

import keras_tuner

sys.path.extend(".")

from src import load_data
from src.models.old import gangoly_cifar

tuner = keras_tuner.GridSearch(
    hypermodel=gangoly_cifar.gangoly_cifar(img_channels=3, num_classes=10),
    objective="val_accuracy",
    seed=0,
    executions_per_trial=3,
    overwrite=False,
    directory="checkpoints",
    project_name="gangoly_cifar",
)
tuner.search_space_summary()

cifar10 = load_data.cifar10()

tuner.search(
    x=cifar10.x_train,
    y=cifar10.y_train,
    batch_size=2**8,
    epochs=30,
    validation_split=0.3,
)

tuner.results_summary()
