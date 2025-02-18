import typing as tp

from jk_utils.backend import complex as complex_ops
from keras import activations, initializers, layers, ops

from ..ops import ema as ema_ops


class EventMask(layers.Layer):
    def compute_mask(self, x, previous_mask):
        assert previous_mask is None
        return ops.any(ops.greater(x, 0), axis=-1)

    def call(self, x, mask=None):
        return x


class MaskedGlobalAveragePooling(layers.Layer):
    def __init__(self, axis: tp.Sequence[int], **kwargs):
        super().__init__(**kwargs)
        self.axis = tuple(axis)

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        return config

    def compute_mask(self, x, previous_mask=None):
        if previous_mask is None:
            return None
        return ops.any(previous_mask, axis=self.axis)

    def call(self, x, mask=None):
        if mask is None:
            return ops.mean(x, axis=self.axis)
        mask = ops.expand_dims(mask, axis=-1)
        return ops.sum(
            ops.where(mask, x, ops.zeros_like(x)), axis=self.axis
        ) / ops.maximum(ops.sum(ops.cast(mask, x.dtype), axis=self.axis), 1e-3)


class Ema(layers.Layer):
    def __init__(
        self,
        decay_rate_initializer="zeros",
        decay_rate_activation="softplus",
        ema_axis: int = 1,
        channel_axis: int = -1,
        normalize: bool = False,
        time_scale: float = 16.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ema_axis = ema_axis
        self.channel_axis = channel_axis
        self.decay_rate_activation = activations.get(decay_rate_activation)
        self.decay_rate_initializer = initializers.get(decay_rate_initializer)
        self.time_scale = time_scale
        self.normalize = normalize

    def compute_mask(self, inputs, previous_mask):
        return previous_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            ema_axis=self.ema_axis,
            channel_axis=self.channel_axis,
            decay_rate_initializer=activations.serialize(self.decay_rate_initializer),
            decay_rate_activation=initializers.serialize(self.decay_rate_activation),
            normalize=self.normalize,
            time_scale=self.time_scale,
        )
        return config

    def build(self, input_shape):
        if self.built:
            return
        C = input_shape[self.channel_axis]
        self.decay_rate = self.add_weight(
            shape=(C,),
            initializer=self.decay_rate_initializer,
            trainable=True,
            name="decay_rate",
        )
        super().build(input_shape)

    def _broadcast(self, x, input_shape):
        shape = [1] * len(input_shape)
        shape[self.channel_axis] = -1
        x = ops.reshape(x, shape)
        x = ops.broadcast_to(x, input_shape)
        return x

    def call(self, input, mask=None):
        decay_rate = self.decay_rate_activation(self.decay_rate)
        is_complex = not ops.is_tensor(decay_rate)
        if is_complex:
            assert len(decay_rate) == 2, decay_rate
            decay_rate = complex_ops.complex(*decay_rate)
            input = complex_ops.complex(*ops.split(input, 2, axis=self.channel_axis))
        num_frames = input.shape[1]
        # HACK 16 is rescaled_duration that worked well in stsc experiments
        decay_rate = decay_rate * self.time_scale / num_frames
        decay_factor = ops.exp(-decay_rate)
        if mask is not None:
            input = ops.where(
                ops.expand_dims(mask, axis=-1), input, ops.zeros_like(input)
            )
        x = ema_ops.ema(
            input, self._broadcast(decay_factor, input.shape), axis=self.ema_axis
        )

        if self.normalize:
            if is_complex:
                raise NotImplementedError("TODO")
            if mask is None:
                weight = ema_ops.ema(ops.ones_like(decay_factor), decay_factor, axis=0)
                weight = self._broadcast(weight, input.shape)
            else:
                weight = ops.tile(
                    ops.expand_dims(ops.cast(mask, x.dtype), axis=-1),
                    (1,) * len(mask.shape) + tuple(decay_factor.shape),
                )
                weight = ema_ops.ema(
                    weight, ops.broadcast_to(decay_factor, weight.shape), axis=1
                )
            x = x / ops.maximum(weight, 1e-3)

        if is_complex:
            x = ops.concatenate((ops.real(x), ops.imag(x)), axis=self.channel_axis)
        return x


def _as2d(x):
    return ops.reshape(x, (-1, *x.shape[2:])), x.shape[1]


def _as3d(x, dim1: int):
    return ops.reshape(x, (-1, dim1, *x.shape[1:]))


class EmaConv2D(layers.Layer):
    def __init__(
        self, ema_layer: tp.Optional[Ema], conv_layer: layers.Conv2D, **kwargs
    ):
        super().__init__(**kwargs)
        self.ema_layer = ema_layer
        self.conv_layer = conv_layer

    def build(self, x_shape):
        if self.built:
            return
        if self.ema_layer is not None:
            self.ema_layer.build(x_shape)
        self.conv_layer.build((None, *x_shape[2:]))
        super().build(x_shape)

    def compute_mask(self, x, previous_mask):
        if previous_mask is None:
            return None
        x = ops.reshape(x, (-1, *x.shape[2:]))
        previous_mask, dim1 = _as2d(previous_mask)
        mask = self.conv_layer.compute_mask(x, previous_mask)
        return _as3d(mask, dim1)

    def call(self, x, mask=None):
        if self.ema_layer is not None:
            x = self.ema_layer(x, mask=mask)
        elif mask is not None:
            x = ops.where(ops.expand_dims(mask, axis=-1), x, ops.zeros_like(x))
        x, dim1 = _as2d(x)
        x = self.conv_layer(x)  # No mask intentionally
        x = _as3d(x, dim1)
        return x
