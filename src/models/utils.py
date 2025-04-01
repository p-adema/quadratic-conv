from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import keras


def quiet_model(model: keras.Model) -> keras.Model:
    fit_fn = model.fit

    def quiet_fit(*args, **kwargs):
        quiet_kwargs = kwargs | {"verbose": False}
        return fit_fn(*args, **quiet_kwargs)

    model.fit = quiet_fit
    return model
