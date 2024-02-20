import typing as tp

import keras
from jk_utils.layers.layer_scale import LayerScale

from ... import components
from .utils import get_decay_rate_activation


def conv_next_block(
    x: components.StreamNode,
    dropout_rate: float = 0.0,
    activation="gelu",
    kernel_shape: int = 5,
    complex_conv: bool = False,
):
    filters = x.num_channels
    z = x
    z = z.stationary_conv(
        filters=None,
        kernel_shape=kernel_shape,
        decay_rate_activation=get_decay_rate_activation(complex_conv),
    )
    z = z.map_features(keras.layers.LayerNormalization())
    z = z.map_features(keras.layers.Dense(filters * 4, activation=activation))
    z = z.map_features(keras.layers.Dense(filters))
    z = z.map_features(LayerScale(scale_initializer=1e-1))
    if dropout_rate:
        # layer dropout
        z = z.map_features(keras.layers.Dropout(dropout_rate, noise_shape=(None, 1)))
    return x + z


def stem(
    x: components.StreamNode,
    filters: int = 32,
    strides: int = 2,
    sample_rate: tp.Optional[int] = None,
    min_dt: float = 0.0,
    complex_conv: bool = False,
) -> components.StreamNode:
    x = x.exclusive_conv(
        filters,
        strides,
        sample_rate=sample_rate,
        min_dt=min_dt,
        decay_rate_activation=get_decay_rate_activation(complex_conv),
    )
    return x.map_features(keras.layers.LayerNormalization())


def downsample(
    x: components.StreamNode,
    strides: int = 2,
    min_dt: float = 0.0,
    complex_conv: bool = False,
) -> components.StreamNode:
    x = x.map_features(keras.layers.LayerNormalization())
    x = x.exclusive_conv(
        x.num_channels * 2,
        strides,
        min_dt=min_dt,
        decay_rate_activation=get_decay_rate_activation(complex_conv),
    )
    return x


def conv_next_backbone(
    stream: components.StreamNode,
    *,
    filters0: int = 32,
    num_levels: int = 4,
    blocks_per_level: int = 1,
    dropout_rate: float = 0.0,
    activation: str = "gelu",
    kernel_shape: int = 5,
    initial_stride: int = 4,
    initial_sample_rate: tp.Optional[int] = None,
    min_dt: float = 0.0,
    complex_conv: bool = False,
) -> tp.List[components.StreamNode]:
    if complex_conv:
        raise NotImplementedError("TODO")
    x = stream
    streams = []

    x = stem(
        x,
        filters0,
        strides=initial_stride,
        sample_rate=initial_sample_rate,
        min_dt=min_dt,
        complex_conv=complex_conv,
    )
    for _ in range(num_levels - 1):
        for _ in range(blocks_per_level):
            x = conv_next_block(
                x, dropout_rate, activation, kernel_shape, complex_conv=complex_conv
            )

        streams.append(x)
        # down sample
        x = downsample(x, min_dt=min_dt, complex_conv=complex_conv)
    for _ in range(blocks_per_level):
        x = conv_next_block(
            x, dropout_rate, activation, kernel_shape, complex_conv=complex_conv
        )
    streams.append(x)
    return streams
