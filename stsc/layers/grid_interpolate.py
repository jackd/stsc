from keras import activations, initializers, layers

from ..ops import grid_interpolate as grid_interpolate_ops
from ..register import register_stsc_serializable


@register_stsc_serializable
class GridEmaInterpolate(layers.Layer):
    def __init__(
        self,
        num_frames: int,
        grid_size: int,
        decay_rate_initializer="zeros",
        decay_rate_activation="softplus",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.grid_size = grid_size
        self.decay_rate_activation = activations.get(decay_rate_activation)
        self.decay_rate_initializer = initializers.get(decay_rate_initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            num_frames=self.num_frames,
            grid_size=self.grid_size,
            decay_rate_activation=activations.serialize(self.decay_rate_activation),
            decay_rate_initializer=initializers.serialize(self.decay_rate_initializer),
        )
        return config

    def build(
        self,
        features_shape,
        times_shape,
        segment_ids_shape,
        batch_ids_shape,
        t_start_shape,
        t_stop_shape,
    ):
        if self.built:
            return

        self.decay_rate = self.add_weight(
            name="decay_rate",
            shape=(features_shape[-1],),
            initializer=self.decay_rate_initializer,
        )
        super().build(features_shape)

    def call(
        self,
        features,
        times,
        segment_ids,
        batch_ids,
        t_start,
        t_stop,
        indices_are_sorted: bool = False,
        normalize: bool = False,
    ):
        return grid_interpolate_ops.grid_ema_interpolate(
            features=features,
            times=times,
            decay_rate=self.decay_rate_activation(self.decay_rate),
            segment_ids=segment_ids,
            batch_ids=batch_ids,
            t_start=t_start,
            t_stop=t_stop,
            num_frames=self.num_frames,
            grid_size=self.grid_size,
            indices_are_sorted=indices_are_sorted,
            normalize=normalize,
        )
