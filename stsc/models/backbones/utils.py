import typing as tp

import keras


class _default:
    pass


DEFAULT = _default()


def complex_decay_rate_activation(x):
    real, imag = keras.ops.split(x, 2, axis=0)
    real = keras.ops.softplus(real)
    return real, imag


def get_decay_rate_activation(complex_conv: bool) -> tp.Callable:
    return complex_decay_rate_activation if complex_conv else keras.ops.softplus


def get_bias_initializer(
    activation: str | tp.Callable,
) -> keras.initializers.Initializer:
    activation_name = (
        "None"
        if activation is None
        else activation.__name__
        if callable(activation)
        else activation
    )
    return (
        keras.initializers.Constant(-1.0) if "heaviside" in activation_name else "zeros"
    )
