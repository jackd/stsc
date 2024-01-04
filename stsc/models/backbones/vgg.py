import typing as tp

import keras
import numpy as np

from ... import components

# from jk_utils.layers.masked_batch_norm import MaskedBatchNormalization


def complex_decay_rate_activation(x):
    real, imag = keras.ops.split(x, 2, axis=-1)
    real = keras.ops.tanh(real)
    real_abs = keras.ops.abs(real)
    rate_real = -keras.ops.log(real_abs)
    rate_imag = keras.ops.where(
        real < 0, imag + keras.ops.convert_to_tensor(np.pi, x.dtype), imag
    )
    return rate_real, rate_imag


def real_decay_rate_activation(x):
    x = keras.ops.tanh(x)
    x = keras.ops.abs(x)
    return -keras.ops.log(x)


def real_decay_rate_activation_v2(x, epsilon=1e-2):
    x = keras.ops.abs(keras.ops.convert_to_tensor(x)) + epsilon
    return 1 / x


def conv_block(
    x: components.StreamNode,
    filters: int,
    kernel_size: int = 3,
    stride: int | None = None,
    sample_rate: int | None = None,
    activation="relu",
    min_dt: float = 0.0,
    normalize: bool = True,
    complex_conv: bool = False,
    **kwargs,
) -> components.StreamNode:
    if normalize:
        x = x.masked_batch_norm(momentum=0.9)

    activation_name = activation.__name__ if callable(activation) else activation
    bias_initializer = (
        keras.initializers.Constant(-1.0) if "heaviside" in activation_name else "zeros"
    )
    kwargs.update(
        activation=activation,
        bias_initializer=bias_initializer,
    )
    if complex_conv:
        kwargs.update(
            decay_rate_activation=complex_decay_rate_activation,
            decay_rate_initializer=keras.initializers.RandomNormal(stddev=1.0),
        )
    else:
        kwargs.update(
            # decay_rate_activation="softplus",
            # decay_rate_initializer="zeros",
            decay_rate_activation=real_decay_rate_activation_v2,
            decay_rate_initializer=keras.initializers.RandomNormal(stddev=1.0),
        )
    if stride is not None:
        assert stride == kernel_size, (stride, kernel_size)
        if sample_rate is None:
            sample_rate = stride**2

    if sample_rate:
        x = x.exclusive_conv(
            filters=filters,
            kernel_shape=kernel_size,
            sample_rate=sample_rate,
            min_dt=min_dt,
            **kwargs,
        )
    else:
        x = x.stationary_conv(filters=filters, kernel_shape=kernel_size, **kwargs)
    return x


def vgg_cnn_backbone(
    node: components.StreamNode,
    *,
    filters0: int = 16,
    activation="relu",
    min_dt: float = 0.0,
    complex_conv: bool = False,
) -> tp.Sequence[components.StreamNode]:
    """conv_next classifier."""
    assert node.stream.grid_shape == (128, 128), node.stream.grid_shape

    kwargs = {"activation": activation, "complex_conv": complex_conv}

    streams = []
    x = node

    x = conv_block(
        x, filters0 * 1, 4, 4, min_dt=min_dt, normalize=False, **kwargs
    )  # (32, 32)
    streams.append(x)
    x = conv_block(x, filters0 * 2, **kwargs)
    streams.append(x)
    x = conv_block(x, filters0 * 4, 2, 2, min_dt=min_dt, **kwargs)  # (16, 16)
    for _ in range(2):
        x = conv_block(x, filters0 * 4, **kwargs)
    streams.append(x)
    x = conv_block(x, filters0 * 8, 2, 2, min_dt=min_dt, **kwargs)  # (8, 8)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **kwargs)
    streams.append(x)
    x = conv_block(x, filters0 * 8, 2, 2, min_dt=min_dt, **kwargs)  # (4, 4)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **kwargs)
    streams.append(x)
    x = conv_block(x, filters0 * 8, 2, 2, min_dt=min_dt, **kwargs)  # (2, 2)
    streams.append(x)
    return streams


def vgg_pool_backbone(
    node: components.StreamNode,
    *,
    filters0: int = 16,
    activation="relu",
    min_dt: float = 0.0,
    simple_pooling: bool = False,
    reduction="mean",
    complex_conv: bool = False,
    blocks_per_level: int = 2,
    num_levels: int = 4,
) -> tp.Sequence[components.StreamNode]:
    """conv_next classifier."""
    assert node.stream.grid_shape == (128, 128), node.stream.grid_shape

    kwargs = {"activation": activation, "complex_conv": complex_conv}
    pool_kwargs = {"simple": simple_pooling, "reduction": reduction, "min_dt": min_dt}
    streams = []
    x = node

    x = conv_block(
        x, filters0, 4, 4, min_dt=min_dt, normalize=False, **kwargs
    )  # (32, 32)
    # for _ in range(num_levels - 1):
    #     for _ in range(blocks_per_level):
    #         x = conv_block(x, filters, **kwargs)
    #     streams.append(x)
    #     x = x.exclusive_pool(2, **pool_kwargs)
    #     filters *= 2
    # for _ in range(blocks_per_level):
    #     x = conv_block(x, filters, **kwargs)

    x = conv_block(x, filters0 * 2, **kwargs)
    streams.append(x)
    x = x.exclusive_pool(2, **pool_kwargs)  # (16, 16)
    for _ in range(2):
        x = conv_block(x, filters0 * 4, **kwargs)
    streams.append(x)
    x = x.exclusive_pool(2, **pool_kwargs)  # (8, 8)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **kwargs)
    streams.append(x)
    x = x.exclusive_pool(2, **pool_kwargs)  # (4, 4)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **kwargs)
    streams.append(x)
    return streams
