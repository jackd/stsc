from keras import activations, initializers, layers

from ..ops import pool as pool_ops
from ..register import register_stsc_serializable


class _PoolingBase(layers.Layer):
    def __init__(
        self,
        reduction: str = "mean",
        decay_rate_initializer="zeros",
        decay_rate_activation="softplus",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert reduction in ("mean", "sum", "max")
        self.reduction = reduction
        self.decay_rate_activation = activations.get(decay_rate_activation)
        self.decay_rate_initializer = initializers.get(decay_rate_initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            reduction=self.reduction,
            decay_rate_initializer=activations.serialize(self.decay_rate_initializer),
            decay_rate_activation=initializers.serialize(self.decay_rate_activation),
        )
        return config

    def _build_decay_rate(self, filters: int):
        shape = (
            (filters,)
            if not hasattr(self, "channel_multiplier")
            else (filters, self.channel_multiplier)
        )
        self.decay_rate = self.add_weight(
            shape,
            initializer=self.decay_rate_initializer,
            name="decay_rate",
        )

    def _pool(self, decay_rate, *args):
        raise NotImplementedError("Abstract method")

    def call(self, *args):
        decay_rate = self.decay_rate_activation(self.decay_rate)
        return self._pool(decay_rate, *args)


@register_stsc_serializable
class Pooling(_PoolingBase):
    def build(
        self,
        features_shape,
        times_in_shape,
        times_out_shape,
        segment_ids_shape,
        predecessor_ids_shape,
    ):
        if self.built:
            return
        self.built = True
        self._build_decay_rate(features_shape[-1])

    def _pool(
        self,
        decay_rate,
        features,
        times_in,
        times_out,
        segment_ids,
        predecessor_ids,
    ):
        return pool_ops.pooling(
            features=features,
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            segment_ids=segment_ids,
            predecessor_ids=predecessor_ids,
            reduction=self.reduction,
        )

    def call(
        self,
        features,
        times_in,
        times_out,
        segment_ids,
        predecessor_ids,
    ):
        decay_rate = self.decay_rate_activation(self.decay_rate)
        return self._pool(
            decay_rate,
            features,
            times_in,
            times_out,
            segment_ids,
            predecessor_ids,
        )


@register_stsc_serializable
class OneHotPooling(_PoolingBase):
    def build(
        self,
        times_in_shape,
        times_out_shape,
        segment_filter_ids_shape,
        one_hot_predecessor_ids_shape,
    ):
        if self.built:
            return
        super().build(
            times_in_shape,
            times_out_shape,
            segment_filter_ids_shape,
            one_hot_predecessor_ids_shape,
        )
        self._build_decay_rate(one_hot_predecessor_ids_shape[-1])

    def _pool(
        self,
        decay_rate,
        times_in,
        times_out,
        segment_filter_ids,
        one_hot_predecessor_ids,
    ):
        return pool_ops.one_hot_pooling(
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            segment_filter_ids=segment_filter_ids,
            one_hot_predecessor_ids=one_hot_predecessor_ids,
            reduction=self.reduction,
        )

    def call(
        self,
        times_in,
        times_out,
        segment_filter_ids,
        one_hot_predecessor_ids,
    ):
        decay_rate = self.decay_rate_activation(self.decay_rate)
        return self._pool(
            decay_rate,
            times_in,
            times_out,
            segment_filter_ids,
            one_hot_predecessor_ids,
        )


@register_stsc_serializable
class ExclusivePooling(_PoolingBase):
    def __init__(self, stride: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = stride

    def get_config(self):
        config = super().get_config()
        config.update(stride=self.stride)
        return config

    def build(
        self,
        features_shape,
        dt_shape,
        times_out_shape,
        successor_kernel_ids_shape,
        segment_ids_out_shape,
    ):
        if self.built:
            return
        self.built = True
        self._build_decay_rate(features_shape[-1])

    def _pool(
        self,
        decay_rate,
        features,
        dt,
        times_out,
        successor_kernel_ids,
        segment_ids_out,
        indices_are_sorted: bool = False,
    ):
        return pool_ops.exclusive_pooling(
            features=features,
            dt=dt,
            times_out=times_out,
            decay_rate=decay_rate,
            successor_kernel_ids=successor_kernel_ids,
            segment_ids_out=segment_ids_out,
            stride=self.stride,
            indices_are_sorted=indices_are_sorted,
            reduction=self.reduction,
        )

    def call(
        self,
        features,
        dt,
        times_out,
        successor_kernel_ids,
        segment_ids_out,
        indices_are_sorted: bool = False,
    ):
        decay_rate = self.decay_rate_activation(self.decay_rate)
        return self._pool(
            decay_rate,
            features,
            dt,
            times_out,
            successor_kernel_ids,
            segment_ids_out,
            indices_are_sorted=indices_are_sorted,
        )


@register_stsc_serializable
class OneHotExclusivePooling(_PoolingBase):
    def __init__(
        self,
        filters: int,
        stride: int,
        channel_multiplier: int = 1,
        reduction: str = "mean",
        *args,
        **kwargs,
    ):
        super().__init__(reduction, *args, **kwargs)
        self.filters = filters
        self.stride = stride
        self.channel_multiplier = channel_multiplier

    def build(
        self,
        dt_shape,
        times_out_shape,
        successor_kernel_channel_ids_shape,
        segment_ids_out_shape,
    ):
        if self.built:
            return
        self.built = True
        self._build_decay_rate(self.filters)

    def get_config(self):
        config = super().get_config()
        config.update(
            filters=self.filters,
            stride=self.stride,
            channel_multiplier=self.channel_multiplier,
        )
        return config

    def _pool(
        self,
        decay_rate,
        dt,
        times_out,
        successor_kernel_channel_ids,
        segment_ids_out,
        indices_are_sorted: bool = False,
    ):
        return pool_ops.one_hot_exclusive_pooling(
            dt=dt,
            times_out=times_out,
            decay_rate=decay_rate,
            successor_kernel_channel_ids=successor_kernel_channel_ids,
            segment_ids_out=segment_ids_out,
            stride=self.stride,
            indices_are_sorted=indices_are_sorted,
        )

    def call(
        self,
        dt,
        times_out,
        successor_kernel_channel_ids,
        segment_ids_out,
        indices_are_sorted: bool = False,
    ):
        decay_rate = self.decay_rate_activation(self.decay_rate)
        return self._pool(
            decay_rate,
            dt,
            times_out,
            successor_kernel_channel_ids,
            segment_ids_out,
            indices_are_sorted=indices_are_sorted,
        )
