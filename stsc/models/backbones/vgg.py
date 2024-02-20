import typing as tp

import keras

from ... import components
from .utils import DEFAULT, get_decay_rate_activation


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
        x = x.masked_batch_norm()

    activation_name = (
        "None"
        if activation is None
        else activation.__name__
        if callable(activation)
        else activation
    )
    bias_initializer = (
        keras.initializers.Constant(-1.0) if "heaviside" in activation_name else "zeros"
    )
    kwargs.update(
        activation=activation,
        bias_initializer=bias_initializer,
    )
    kwargs.update(decay_rate_activation=get_decay_rate_activation(complex_conv))
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
    initial_stride: int = 4,
    initial_sample_rate: int | None = None,
    initial_activation=DEFAULT,
) -> tp.Sequence[components.StreamNode]:
    """fully convolutional vgg classifier."""

    conv_kwargs = {"activation": activation, "complex_conv": complex_conv}
    strided_kwargs = dict(conv_kwargs)
    strided_kwargs.update(kernel_size=2, stride=2, min_dt=min_dt)
    streams = []
    x = node

    x = conv_block(
        x,
        filters0,
        # filters0 * 2,  # HACK
        initial_stride,
        initial_stride,
        min_dt=min_dt,
        normalize=False,
        sample_rate=initial_sample_rate,
        activation=activation if initial_activation is DEFAULT else initial_activation,
        complex_conv=complex_conv,
        # channel_multiplier=filters0 // 2,
        channel_multiplier=1,
    )  # (32, 32)

    streams.append(x)
    x = conv_block(x, filters0 * 4, **strided_kwargs)  # (16, 16)
    x = conv_block(x, filters0 * 4, **conv_kwargs)
    streams.append(x)
    x = conv_block(x, filters0 * 8, **strided_kwargs)  # (8, 8)
    x = conv_block(x, filters0 * 8, **conv_kwargs)
    streams.append(x)
    x = conv_block(x, filters0 * 8, **strided_kwargs)  # (4, 4)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **conv_kwargs)
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
    initial_sample_rate: int | None = None,
    initial_stride: int = 4,
    initial_activation=DEFAULT,
) -> tp.Sequence[components.StreamNode]:
    """vgg classifier with pooling."""
    conv_kwargs = {"activation": activation, "complex_conv": complex_conv}
    pool_kwargs = {"simple": simple_pooling, "reduction": reduction, "min_dt": min_dt}
    streams = []
    x = node

    x = conv_block(
        x,
        filters0,
        # filters0 * 2,  # HACK
        initial_stride,
        initial_stride,
        min_dt=min_dt,
        normalize=False,
        sample_rate=initial_sample_rate,
        activation=activation if initial_activation is DEFAULT else initial_activation,
        complex_conv=complex_conv,
        # channel_multiplier=filters0 // 2,
        channel_multiplier=1,
    )  # (32, 32)

    x = conv_block(x, filters0 * 2, **conv_kwargs)
    streams.append(x)
    x = x.exclusive_pool(2, **pool_kwargs)  # (16, 16)
    for _ in range(2):
        x = conv_block(x, filters0 * 4, **conv_kwargs)
    streams.append(x)
    x = x.exclusive_pool(2, **pool_kwargs)  # (8, 8)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **conv_kwargs)
    streams.append(x)
    x = x.exclusive_pool(2, **pool_kwargs)  # (4, 4)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **conv_kwargs)
    streams.append(x)
    return streams
