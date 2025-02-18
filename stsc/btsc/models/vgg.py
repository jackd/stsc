import typing as tp

from keras import KerasTensor, layers

from .utils import DEFAULT, EmaMode
from .utils import apply_mask as _apply_mask
from .utils import (
    exclusive_conv,
    exclusive_pool,
    get_bias_initializer,
    get_decay_rate_activation,
    stationary_conv,
)


def conv_block(
    x: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
    filters: int,
    kernel_size: int = 3,
    stride: int | None = None,
    activation="relu",
    initial_batch_norm: bool = True,
    complex_conv: bool = False,
    **kwargs,
) -> KerasTensor:
    if initial_batch_norm:
        x = layers.BatchNormalization()(x)

    bias_initializer = get_bias_initializer(activation)
    kwargs.update(
        activation=activation,
        bias_initializer=bias_initializer,
    )
    kwargs.update(decay_rate_activation=get_decay_rate_activation(complex_conv))
    if stride is not None:
        assert stride == kernel_size, (stride, kernel_size)

    if stride:
        x = exclusive_conv(
            x,
            filters=filters,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs,
        )
    else:
        x = stationary_conv(
            x,
            filters=filters,
            kernel_size=kernel_size,
            **kwargs,
        )
    return x


def vgg_cnn_backbone(
    x: KerasTensor,
    *,
    filters0: int = 16,
    activation="relu",
    complex_conv: bool = False,
    initial_stride: int = 4,
    initial_activation=DEFAULT,
    normalize: bool = True,
    apply_mask: bool = False,
    ema_mode: EmaMode = EmaMode.ALL,
) -> tp.Sequence[KerasTensor]:
    """fully convolutional vgg classifier."""
    if apply_mask:
        x = _apply_mask(x)
    conv_kwargs = {
        "activation": activation,
        "complex_conv": complex_conv,
        "normalize": normalize,
        "ema": ema_mode == EmaMode.ALL,
    }
    strided_kwargs = dict(conv_kwargs)
    strided_kwargs.update(kernel_size=2, stride=2)
    streams = []

    x = conv_block(
        x,
        filters0,
        # filters0 * 2,  # HACK
        initial_stride,
        initial_stride,
        initial_batch_norm=False,
        activation=activation if initial_activation is DEFAULT else initial_activation,
        complex_conv=complex_conv,
        normalize=False,  # don't normalize first layer
        ema=ema_mode in {EmaMode.ALL, EmaMode.FIRST},
        # channel_multiplier=filters0 // 2,
        # channel_multiplier=1,
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
    x: KerasTensor,
    *,
    filters0: int = 16,
    activation="relu",
    reduction="mean",
    complex_conv: bool = False,
    initial_stride: int = 4,
    normalize: bool = True,
    initial_activation=DEFAULT,
    apply_mask: bool = False,
    ema_mode: EmaMode = EmaMode.ALL,
) -> tp.Sequence[KerasTensor]:
    """vgg classifier with pooling."""
    if apply_mask:
        x = _apply_mask(x)
    conv_kwargs = {
        "activation": activation,
        "complex_conv": complex_conv,
        "normalize": normalize,
        "ema": ema_mode == EmaMode.ALL,
    }
    pool_kwargs = {"reduction": reduction}
    streams = []

    x = conv_block(
        x,
        filters0,
        # filters0 * 2,  # HACK
        initial_stride,
        initial_stride,
        initial_batch_norm=False,
        activation=activation if initial_activation is DEFAULT else initial_activation,
        complex_conv=complex_conv,
        # channel_multiplier=filters0 // 2,
        normalize=False,  # don't normalize first layer
        ema=ema_mode in {EmaMode.ALL, EmaMode.FIRST},
        # channel_multiplier=1,
    )  # (32, 32)

    x = conv_block(x, filters0 * 2, **conv_kwargs)
    streams.append(x)
    x = exclusive_pool(x, 2, **pool_kwargs)  # (16, 16)
    for _ in range(2):
        x = conv_block(x, filters0 * 4, **conv_kwargs)
    streams.append(x)
    x = exclusive_pool(x, 2, **pool_kwargs)  # (8, 8)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **conv_kwargs)
    streams.append(x)
    x = exclusive_pool(x, 2, **pool_kwargs)  # (4, 4)
    for _ in range(2):
        x = conv_block(x, filters0 * 8, **conv_kwargs)
    streams.append(x)
    return streams
