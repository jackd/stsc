from keras import activations, initializers, layers

from ..ops import conv as conv_ops
from ..register import register_stsc_serializable


class _ConvBase(layers.Layer):
    def __init__(
        self,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        decay_rate_initializer="zeros",
        decay_rate_activation="softplus",
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.decay_rate_activation = activations.get(decay_rate_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.decay_rate_initializer = initializers.get(decay_rate_initializer)
        self.use_bias = use_bias

    def get_config(self):
        config = super().get_config()
        config.update(
            activation=activations.serialize(self.activation),
            decay_rate_initializer=activations.serialize(self.decay_rate_initializer),
            kernel_initializer=initializers.serialize(self.kernel_initializer),
            bias_initializer=initializers.serialize(self.bias_initializer),
            decay_rate_activation=initializers.serialize(self.decay_rate_activation),
            use_bias=self.use_bias,
        )
        return config

    def _build_base_conv_parameters(self, decay_rate_shape, kernel_shape, bias_shape):
        self.decay_rate = self.add_weight(
            decay_rate_shape,
            initializer=self.decay_rate_initializer,
            name="decay_rate",
        )
        self.kernel = self.add_weight(
            kernel_shape, initializer=self.kernel_initializer, name="kernel"
        )
        if self.use_bias:
            self.bias = self.add_weight(
                bias_shape, initializer=self.bias_initializer, name="bias"
            )

    def _conv(self, decay_rate, kernel, *args, **kwargs):
        """
        Perform convolution with activated `decay_rate` and `kernel`.

        Does not include possibly adding bias or output activation.
        """
        raise NotImplementedError("Abstract method")

    def _finalize(self, x):
        if self.use_bias:
            x = x + self.bias
        return self.activation(x)


@register_stsc_serializable
class Conv(_ConvBase):
    def __init__(
        self,
        filters: int,
        normalize: bool = True,
        **kwargs,
    ):
        self.filters = filters
        self.normalize = normalize
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(filters=self.filters, normalize=self.normalize)
        return config

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
        kernel_size = predecessor_ids_shape[1]
        filters_in = features_shape[1]
        decay_rate_shape = (filters_in,)
        kernel_shape = (kernel_size, filters_in, self.filters)
        bias_shape = (self.filters,)
        self._build_base_conv_parameters(decay_rate_shape, kernel_shape, bias_shape)

    def _conv(
        self,
        decay_rate,
        kernel,
        features,
        times_in,
        times_out,
        segment_ids,
        predecessor_ids,
    ):
        return conv_ops.conv(
            features=features,
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            segment_ids=segment_ids,
            predecessor_ids=predecessor_ids,
            normalize=self.normalize,
        )

    def call(
        self,
        features,
        times_in,
        times_out,
        segment_ids,
        predecessor_ids,
    ):
        return self._finalize(
            self._conv(
                self.decay_rate_activation(self.decay_rate),
                self.kernel,
                features,
                times_in,
                times_out,
                segment_ids,
                predecessor_ids,
            )
        )


@register_stsc_serializable
class DepthwiseConv(_ConvBase):
    def __init__(self, normalize: bool = True, **kwargs):
        self.normalize = normalize
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["normalize"] = self.normalize
        return config

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
        kernel_size = predecessor_ids_shape[1]
        filters = features_shape[1]
        decay_rate_shape = (filters,)
        kernel_shape = (kernel_size, filters)
        bias_shape = (filters,)
        self._build_base_conv_parameters(decay_rate_shape, kernel_shape, bias_shape)

    def _conv(
        self,
        decay_rate,
        kernel,
        features,
        times_in,
        times_out,
        segment_ids,
        predecessor_ids,
    ):
        return conv_ops.depthwise_conv(
            features=features,
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            segment_ids=segment_ids,
            predecessor_ids=predecessor_ids,
            normalize=self.normalize,
        )

    def call(
        self,
        features,
        times_in,
        times_out,
        segment_ids,
        predecessor_ids,
    ):
        return self._finalize(
            self._conv(
                self.decay_rate_activation(self.decay_rate),
                self.kernel,
                features,
                times_in,
                times_out,
                segment_ids,
                predecessor_ids,
            )
        )


@register_stsc_serializable
class OneHotConv(_ConvBase):
    def __init__(
        self,
        filters: int,
        **kwargs,
    ):
        self.filters = filters
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            filters_in=self.filters,
        )
        return config

    def build(
        self,
        times_in_shape,
        times_out_shape,
        segment_filter_ids_shape,
        one_hot_predecessor_ids_shape,
    ):
        if self.built:
            return
        self.built = True
        kernel_size, filters_in = one_hot_predecessor_ids_shape[1:]
        decay_rate_shape = (filters_in,)
        kernel_shape = (kernel_size, filters_in, self.filters)
        bias_shape = (self.filters,)
        self._build_base_conv_parameters(decay_rate_shape, kernel_shape, bias_shape)

    def _conv(
        self,
        decay_rate,
        kernel,
        times_in,
        times_out,
        segment_filter_ids,
        one_hot_predecessor_ids,
    ):
        return conv_ops.one_hot_conv(
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            segment_filter_ids=segment_filter_ids,
            one_hot_predecessor_ids=one_hot_predecessor_ids,
        )

    def call(
        self,
        times_in,
        times_out,
        segment_filter_ids,
        one_hot_predecessor_ids,
    ):
        return self._finalize(
            self._conv(
                self.decay_rate_activation(self.decay_rate),
                self.kernel,
                times_in,
                times_out,
                segment_filter_ids,
                one_hot_predecessor_ids,
            )
        )


@register_stsc_serializable
class OneHotDepthwiseConv(_ConvBase):
    def build(
        self,
        times_in_shape,
        times_out_shape,
        segment_filter_ids_shape,
        one_hot_predecessor_ids_shape,
    ):
        if self.built:
            return
        self.built = True
        kernel_size, filters = one_hot_predecessor_ids_shape[1:]
        decay_rate_shape = (filters,)
        kernel_shape = (kernel_size, filters)
        bias_shape = (filters,)
        self._build_base_conv_parameters(decay_rate_shape, kernel_shape, bias_shape)

    def _conv(
        self,
        decay_rate,
        kernel,
        times_in,
        times_out,
        segment_filter_ids,
        one_hot_predecessor_ids,
    ):
        return conv_ops.one_hot_depthwise_conv(
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            segment_filter_ids=segment_filter_ids,
            one_hot_predecessor_ids=one_hot_predecessor_ids,
        )

    def call(
        self,
        times_in,
        times_out,
        segment_filter_ids,
        one_hot_predecessor_ids,
    ):
        return self._finalize(
            self._conv(
                self.decay_rate_activation(self.decay_rate),
                self.kernel,
                times_in,
                times_out,
                segment_filter_ids,
                one_hot_predecessor_ids,
            )
        )


@register_stsc_serializable
class ExclusiveConv(_ConvBase):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        normalize: bool = True,
        **kwargs,
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.normalize = normalize
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalize=self.normalize,
        )
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
        filters_in = features_shape[1]
        decay_rate_shape = (filters_in,)
        kernel_shape = (self.kernel_size, filters_in, self.filters)
        bias_shape = (self.filters,)
        self._build_base_conv_parameters(decay_rate_shape, kernel_shape, bias_shape)

    def _conv(
        self,
        decay_rate,
        kernel,
        features,
        dt,
        times_out,
        successor_kernel_ids,
        segment_ids_out,
        indices_are_sorted: bool = False,
    ):
        return conv_ops.exclusive_conv(
            features=features,
            dt=dt,
            times_out=times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            successor_kernel_ids=successor_kernel_ids,
            segment_ids_out=segment_ids_out,
            indices_are_sorted=indices_are_sorted,
            normalize=self.normalize,
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
        return self._finalize(
            self._conv(
                self.decay_rate_activation(self.decay_rate),
                self.kernel,
                features,
                dt,
                times_out,
                successor_kernel_ids,
                segment_ids_out,
                indices_are_sorted=indices_are_sorted,
            )
        )


@register_stsc_serializable
class ExclusiveDepthwiseConv(_ConvBase):
    def __init__(
        self,
        kernel_size: int,
        *,
        indices_are_sorted: bool = False,
        normalize: bool = True,
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.indices_are_sorted = indices_are_sorted
        self.normalize = normalize
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            kernel_size=self.kernel_size,
            indices_are_sorted=self.indices_are_sorted,
            normalize=self.normalize,
        )
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
        filters = features_shape[1]
        decay_rate_shape = (filters,)
        kernel_shape = (self.kernel_size, filters)
        bias_shape = (filters,)
        self._build_base_conv_parameters(decay_rate_shape, kernel_shape, bias_shape)

    def _conv(
        self,
        decay_rate,
        kernel,
        features,
        dt,
        times_out,
        successor_kernel_ids,
        segment_ids_out,
        indices_are_sorted: bool,
    ):
        return conv_ops.exclusive_depthwise_conv(
            features=features,
            dt=dt,
            times_out=times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            successor_kernel_ids=successor_kernel_ids,
            segment_ids_out=segment_ids_out,
            indices_are_sorted=indices_are_sorted,
            normalize=self.normalize,
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
        return self._finalize(
            self._conv(
                self.decay_rate_activation(self.decay_rate),
                self.kernel,
                features,
                dt,
                times_out,
                successor_kernel_ids,
                segment_ids_out,
                indices_are_sorted=indices_are_sorted,
            )
        )


@register_stsc_serializable
class OneHotExclusiveConv(_ConvBase):
    def __init__(
        self,
        filters_in: int,
        filters_out: int,
        kernel_size: int,
        channel_multiplier: int = 1,
        **kwargs,
    ):
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.channel_multiplier = channel_multiplier
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            filters_in=self.filters_in,
            filters_out=self.filters_out,
            kernel_size=self.kernel_size,
            channel_multiplier=self.channel_multiplier,
        )
        return config

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
        decay_rate_shape = (self.filters_in, self.channel_multiplier)
        kernel_shape = (
            self.kernel_size,
            self.filters_in * self.channel_multiplier,
            self.filters_out,
        )
        bias_shape = (self.filters_out,)
        self._build_base_conv_parameters(decay_rate_shape, kernel_shape, bias_shape)

    def _conv(
        self,
        decay_rate,
        kernel,
        dt,
        times_out,
        successor_kernel_channel_ids,
        segment_ids_out,
        indices_are_sorted: bool = False,
    ):
        return conv_ops.one_hot_exclusive_conv(
            dt=dt,
            times_out=times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            successor_kernel_channel_ids=successor_kernel_channel_ids,
            segment_ids_out=segment_ids_out,
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
        return self._finalize(
            self._conv(
                self.decay_rate_activation(self.decay_rate),
                self.kernel,
                dt,
                times_out,
                successor_kernel_channel_ids,
                segment_ids_out,
                indices_are_sorted=indices_are_sorted,
            )
        )


@register_stsc_serializable
class OneHotExclusiveDepthwiseConv(_ConvBase):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        **kwargs,
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            filters=self.filters,
            kernel_size=self.kernel_size,
        )
        return config

    def build(
        self,
        dt_shape,
        times_out_shape,
        segment_ids_out_shape,
        successor_kernel_channel_ids_shape,
    ):
        if self.built:
            return
        self.built = True
        decay_rate_shape = (self.filters,)
        kernel_shape = (self.kernel_size, self.filters)
        bias_shape = (self.filters,)
        self._build_base_conv_parameters(decay_rate_shape, kernel_shape, bias_shape)

    def _conv(
        self,
        decay_rate,
        kernel,
        dt,
        times_out,
        successor_kernel_channel_ids,
        segment_ids_out,
        indices_are_sorted: bool = False,
    ):
        return conv_ops.one_hot_exclusive_depthwise_conv(
            dt=dt,
            times_out=times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            segment_ids_out=segment_ids_out,
            successor_kernel_channel_ids=successor_kernel_channel_ids,
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
        return self._finalize(
            self._conv(
                self.decay_rate_activation(self.decay_rate),
                self.kernel,
                dt,
                times_out,
                successor_kernel_channel_ids,
                segment_ids_out,
                indices_are_sorted=indices_are_sorted,
            )
        )
