import typing as tp

from jk_utils.layers.layer_scale import LayerScale
from keras import KerasTensor, layers

from .utils import EmaMode
from .utils import apply_mask as _apply_mask
from .utils import (
    exclusive_conv,
    get_bias_initializer,
    get_decay_rate_activation,
    stationary_conv,
)


def conv_next_block(
    x: KerasTensor,
    dropout_rate: float = 0.0,
    activation="gelu",
    kernel_shape: int = 5,
    complex_conv: bool = False,
    normalize: bool = True,
    ema: bool = True,
):
    filters = x.shape[-1]
    z = x
    z = stationary_conv(
        x,
        filters=None,
        kernel_shape=kernel_shape,
        decay_rate_activation=get_decay_rate_activation(complex_conv),
        normalize=normalize,
        ema=ema,
    )
    z = layers.LayerNormalization()(z)
    z = layers.Dense(
        filters * 4,
        activation=activation,
        bias_initializer=get_bias_initializer(activation),
    )(z)
    z = layers.Dense(filters)(z)
    z = LayerScale(scale_initializer=1e-1)(z)
    if dropout_rate:
        # layer dropout
        z = layers.Dropout(dropout_rate, noise_shape=(None, 1))(z)
    return x + z


def stem(
    x: KerasTensor,
    filters: int = 32,
    strides: int = 2,
    complex_conv: bool = False,
    normalize: bool = False,
    ema: bool = True,
) -> KerasTensor:
    x = exclusive_conv(
        x,
        filters,
        strides,
        decay_rate_activation=get_decay_rate_activation(complex_conv),
        normalize=normalize,
        ema=ema,
    )
    return layers.LayerNormalization()(x)


def downsample(
    x: KerasTensor,
    strides: int = 2,
    complex_conv: bool = False,
    normalize: bool = True,
    ema: bool = True,
) -> KerasTensor:
    x = layers.LayerNormalization()(x)
    x = exclusive_conv(
        x,
        x.shape[-1] * 2,
        strides,
        decay_rate_activation=get_decay_rate_activation(complex_conv),
        normalize=normalize,
        ema=ema,
    )
    return x


def conv_next_backbone(
    stream: KerasTensor,
    *,
    filters0: int = 32,
    num_levels: int = 4,
    blocks_per_level: int = 1,
    dropout_rate: float = 0.0,
    activation: str = "gelu",
    kernel_shape: int = 5,
    initial_stride: int = 4,
    complex_conv: bool = False,
    normalize: bool = True,
    apply_mask: bool = False,
    ema_mode: EmaMode = EmaMode.ALL,
) -> tp.List[KerasTensor]:
    if apply_mask:
        stream = _apply_mask(stream)
    if complex_conv:
        raise NotImplementedError("TODO")
    x = stream
    conv_kwargs = {
        "complex_conv": complex_conv,
        "normalize": normalize,
        "ema": ema_mode == EmaMode.ALL,
    }
    streams = []

    x = stem(
        x,
        filters0,
        strides=initial_stride,
        complex_conv=complex_conv,
        normalize=False,
        ema=ema_mode in {EmaMode.ALL, EmaMode.FIRST},
    )
    for _ in range(num_levels - 1):
        for _ in range(blocks_per_level):
            x = conv_next_block(
                x, dropout_rate, activation, kernel_shape, **conv_kwargs
            )

        streams.append(x)
        # down sample
        x = downsample(x, **conv_kwargs)
    for _ in range(blocks_per_level):
        x = conv_next_block(x, dropout_rate, activation, kernel_shape, **conv_kwargs)
    streams.append(x)
    return streams
