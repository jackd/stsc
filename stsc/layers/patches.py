from keras import activations, initializers, layers, ops

from ..ops import patches as patch_ops


class ExtractExclusivePatches(layers.Layer):
    def __init__(
        self,
        kernel_size: int,
        flatten: bool = False,
        decay_rate_initializer="zeros",
        decay_rate_activation="softplus",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.flatten = flatten
        self.decay_rate_activation = activations.get(decay_rate_activation)
        self.decay_rate_initializer = initializers.get(decay_rate_initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            kernel_size=self.kernel_size,
            flatten=self.flatten,
            decay_rate_activation=activations.serialize(self.decay_rate_activation),
            decay_rate_initializer=initializers.serialize(self.decay_rate_initializer),
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
        self.decay_rate = self.add_weight(
            name="decay_rate",
            shape=(filters_in,),
            initializer=self.decay_rate_initializer,
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
        out = patch_ops.get_exclusive_patches(
            features=features,
            dt=dt,
            times_out=times_out,
            decay_rate=self.decay_rate_activation(self.decay_rate),
            successor_kernel_ids=successor_kernel_ids,
            segment_ids_out=segment_ids_out,
            indices_are_sorted=indices_are_sorted,
            kernel_size=self.kernel_size,
        )
        if self.flatten:
            assert len(out.shape) == 3, out.shape
            out = ops.reshape(out, (-1, out.shape[1] * out.shape[2]))
        return out
