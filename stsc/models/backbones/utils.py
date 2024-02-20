import typing as tp

import keras


class _default:
    pass


DEFAULT = _default()


def complex_decay_rate_activation(x):
    real, imag = keras.ops.split(x, 2, axis=-1)
    real = keras.ops.softplus(real)
    return real, imag


def get_decay_rate_activation(complex_conv: bool) -> tp.Callable:
    return complex_decay_rate_activation if complex_conv else keras.ops.softplus
