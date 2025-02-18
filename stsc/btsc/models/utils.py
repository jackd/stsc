import enum
import typing as tp

from jk_utils.layers.masked_conv import MaskedConv2D
from jk_utils.layers.masked_pooling import MaskedAveragePooling2D, MaskedMaxPooling2D
from keras import KerasTensor, initializers, layers, ops

from ..layers import Ema, EmaConv2D, EventMask


class EmaMode(enum.Enum):
    NONE = "none"
    ALL = "all"
    FIRST = "first"


# class As1D(layers.Layer):
#     def call(self, x, mask=None):
#         return ops.reshape(x, (-1, x.shape[1] * x.shape[2], *x.shape[3:]))

#     def compute_mask(self, x, previous_mask):
#         if previous_mask is None:
#             return None
#         return ops.reshape(
#             previous_mask,
#             (
#                 -1,
#                 previous_mask.shape[1] * previous_mask.shape[2],
#                 *previous_mask.shape[3:],
#             ),
#         )


# class As2D(layers.Layer):
#     def __init__(self, dim1: int, **kwargs):
#         self.dim1 = dim1
#         super().__init__(**kwargs)

#     def get_config(self):
#         config = super().get_config()
#         config["dim1"] = self.dim1
#         return config

#     def call(self, x, mask=None):
#         assert x.shape[1] % self.dim1 == 0, (x.shape[1], self.dim1)
#         return ops.reshape(x, (-1, self.dim1, x.shape[1] // self.dim1, *x.shape[2:]))

#     def compute_mask(self, x, previous_mask):
#         assert x.shape[1] % self.dim1 == 0, (x.shape[1], self.dim1)
#         return ops.reshape(
#             previous_mask, (-1, self.dim1, x.shape[1] // self.dim1, *x.shape[2:])
#         )


# def apply_mask(x: KerasTensor) -> KerasTensor:
#     dim1 = x.shape[1]
#     x = As1D()(x)
#     x = layers.Masking(mask_value=0.0)(x)
#     x = As2D(dim1)(x)
#     return x


def apply_mask(x: KerasTensor) -> KerasTensor:
    return EventMask()(x)


class FlattenLeadingDims(layers.Layer):
    def call(self, x, mask=None):
        return ops.reshape(x, (-1, *x.shape[2:]))

    def compute_mask(self, x, previous_mask):
        if previous_mask is None:
            return None
        return ops.reshape(previous_mask, (-1, *previous_mask.shape[2:]))


class ReshapeLeadingDim(layers.Layer):
    def __init__(self, missing_dim: int, **kwargs):
        self._missing_dim = missing_dim
        super().__init__(**kwargs)

    @property
    def missing_dim(self) -> int:
        return self._missing_dim

    def get_config(self):
        config = super().get_config()
        config["missing_dim"] = self._missing_dim

    def call(self, x):
        return ops.reshape(x, (-1, self._missing_dim, *x.shape[1:]))

    def compute_mask(self, x, previous_mask):
        if previous_mask is None:
            return None
        return ops.reshape(
            previous_mask, (-1, self._missing_dim, *previous_mask.shape[1:])
        )


def as_2d(f: tp.Callable, x):
    missing_dim = x.shape[1]
    x = FlattenLeadingDims()(x)
    x = f(x)
    x = ReshapeLeadingDim(missing_dim)(x)
    return x


def exclusive_conv(
    x,
    filters: int,
    stride: int,
    decay_rate_activation="softplus",
    normalize: bool = False,
    ema: bool = True,
    **conv_kwargs,
):
    ema_layer = (
        Ema(decay_rate_activation=decay_rate_activation, normalize=normalize)
        if ema
        else None
    )
    conv = MaskedConv2D(filters, strides=stride, padding="same", **conv_kwargs)
    x = EmaConv2D(ema_layer, conv)(x)
    return x


def exclusive_pool(x, stride, reduction: str = "mean"):
    if reduction == "mean":
        layer = MaskedAveragePooling2D(stride)
    else:
        assert reduction == "max", reduction
        layer = MaskedMaxPooling2D(stride)
    return as_2d(layer, x)


def stationary_conv(
    x,
    filters: int,
    kernel_size: int,
    decay_rate_activation="softplus",
    normalize: bool = False,
    ema: bool = True,
    **conv_kwargs,
):
    ema_layer = (
        Ema(decay_rate_activation=decay_rate_activation, normalize=normalize)
        if ema
        else None
    )
    conv = MaskedConv2D(filters, kernel_size, padding="same", **conv_kwargs)
    x = EmaConv2D(ema_layer, conv)(x)
    return x


class _default:
    pass


DEFAULT = _default()


def complex_decay_rate_activation(x):
    real, imag = ops.split(x, 2, axis=0)
    real = ops.softplus(real)
    return real, imag


def get_decay_rate_activation(complex_conv: bool) -> tp.Callable:
    return complex_decay_rate_activation if complex_conv else ops.softplus


def get_bias_initializer(
    activation: str | tp.Callable,
) -> initializers.Initializer:
    activation_name = (
        "None"
        if activation is None
        else activation.__name__
        if callable(activation)
        else activation
    )
    return initializers.Constant(-1.0) if "heaviside" in activation_name else "zeros"
