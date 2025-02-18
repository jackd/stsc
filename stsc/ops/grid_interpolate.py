import keras

from ..backend import BackendTensor
from ..backend import grid_interpolate as grid_interpolate_ops

Tensor = BackendTensor | keras.KerasTensor


class GridEmaInterpolate(keras.Operation):
    def __init__(
        self,
        num_frames: int,
        grid_size: int,
        indices_are_sorted: bool = False,
        normalize: bool = False,
        name=None,
    ):
        self.num_frames = num_frames
        self.grid_size = grid_size
        self.indices_are_sorted = indices_are_sorted
        self.normalize = normalize
        super().__init__(name=name)

    def compute_output_spec(
        self,
        features: keras.KerasTensor,
        times: keras.KerasTensor,
        decay_rate: keras.KerasTensor,
        segment_ids: keras.KerasTensor,
        batch_ids: keras.KerasTensor,
        t_start: keras.KerasTensor,
        t_stop: keras.KerasTensor,
    ) -> keras.KerasTensor:
        (B,) = t_start.shape
        (C,) = decay_rate.shape
        dtype = features.dtype
        return keras.KerasTensor((B, self.grid_size, self.num_frames, C), dtype)

    def call(
        self,
        features: BackendTensor,
        times: BackendTensor,
        decay_rate: BackendTensor,
        segment_ids: BackendTensor,
        batch_ids: BackendTensor,
        t_start: BackendTensor,
        t_stop: BackendTensor,
    ) -> BackendTensor:
        return grid_interpolate_ops.grid_ema_interpolate(
            features=features,
            times=times,
            decay_rate=decay_rate,
            segment_ids=segment_ids,
            batch_ids=batch_ids,
            t_start=t_start,
            t_stop=t_stop,
            num_frames=self.num_frames,
            grid_size=self.grid_size,
            indices_are_sorted=self.indices_are_sorted,
            normalize=self.normalize,
        )


def grid_ema_interpolate(
    features: Tensor,
    times: Tensor,
    decay_rate: Tensor,
    segment_ids: Tensor,
    batch_ids: Tensor,
    t_start: Tensor,
    t_stop: Tensor,
    num_frames: int,
    grid_size: int,
    *,
    indices_are_sorted: bool = False,
    normalize: bool = False,
) -> Tensor:
    return GridEmaInterpolate(
        num_frames=num_frames,
        grid_size=grid_size,
        indices_are_sorted=indices_are_sorted,
        normalize=normalize,
    )(
        features=features,
        times=times,
        decay_rate=decay_rate,
        segment_ids=segment_ids,
        batch_ids=batch_ids,
        t_start=t_start,
        t_stop=t_stop,
    )


class GridFinalInterpolate(keras.Operation):
    def __init__(
        self,
        num_frames: int,
        grid_size: int,
        name=None,
    ):
        self.num_frames = num_frames
        self.grid_size = grid_size
        super().__init__(name=name)

    def compute_output_spec(
        self,
        features: keras.KerasTensor,
        times: keras.KerasTensor,
        segment_ids: keras.KerasTensor,
        batch_ids: keras.KerasTensor,
        t_start: keras.KerasTensor,
        t_stop: keras.KerasTensor,
    ) -> keras.KerasTensor:
        (B,) = t_start.shape
        C = features.shape[1]
        dtype = features.dtype
        features_out = keras.KerasTensor((B, self.grid_size, self.num_frames, C), dtype)
        mask = keras.KerasTensor((B, self.grid_size, self.num_frames), "bool")
        return features_out, mask

    def call(
        self,
        features: BackendTensor,
        times: BackendTensor,
        segment_ids: BackendTensor,
        batch_ids: BackendTensor,
        t_start: BackendTensor,
        t_stop: BackendTensor,
    ) -> BackendTensor:
        return grid_interpolate_ops.grid_final_interpolate(
            features=features,
            times=times,
            segment_ids=segment_ids,
            batch_ids=batch_ids,
            t_start=t_start,
            t_stop=t_stop,
            num_frames=self.num_frames,
            grid_size=self.grid_size,
        )


def grid_final_interpolate(
    features: Tensor,
    times: Tensor,
    segment_ids: Tensor,
    batch_ids: Tensor,
    t_start: Tensor,
    t_stop: Tensor,
    num_frames: int,
    grid_size: int,
) -> Tensor:
    return GridFinalInterpolate(
        num_frames=num_frames,
        grid_size=grid_size,
    )(
        features=features,
        times=times,
        segment_ids=segment_ids,
        batch_ids=batch_ids,
        t_start=t_start,
        t_stop=t_stop,
    )


class MeanPreviousInterpolate(keras.Operation):
    def __init__(
        self,
        num_frames: int,
        indices_are_sorted: bool = False,
        name=None,
    ):
        self.num_frames = num_frames
        self.indices_are_sorted = indices_are_sorted
        super().__init__(name=name)

    def compute_output_spec(
        self,
        features: keras.KerasTensor,
        times: keras.KerasTensor,
        batch_ids: keras.KerasTensor,
        t_start: keras.KerasTensor,
        t_stop: keras.KerasTensor,
    ) -> keras.KerasTensor:
        (B,) = t_start.shape
        C = features.shape[1]
        dtype = features.dtype
        return keras.KerasTensor((B, self.num_frames, C), dtype)

    def call(
        self,
        features: BackendTensor,
        times: BackendTensor,
        batch_ids: BackendTensor,
        t_start: BackendTensor,
        t_stop: BackendTensor,
    ) -> BackendTensor:
        return grid_interpolate_ops.mean_previous_interpolate(
            features=features,
            times=times,
            batch_ids=batch_ids,
            t_start=t_start,
            t_stop=t_stop,
            num_frames=self.num_frames,
            indices_are_sorted=self.indices_are_sorted,
        )


def mean_previous_interpolate(
    features: Tensor,
    times: Tensor,
    batch_ids: Tensor,
    t_start: Tensor,
    t_stop: Tensor,
    num_frames: int,
    *,
    indices_are_sorted: bool = False,
) -> Tensor:
    return MeanPreviousInterpolate(
        num_frames=num_frames, indices_are_sorted=indices_are_sorted
    )(
        features=features,
        times=times,
        batch_ids=batch_ids,
        t_start=t_start,
        t_stop=t_stop,
    )


class ReduceMeanFinal(keras.Operation):
    def __init__(self, grid_size: int, batch_size: int):
        self.grid_size = grid_size
        self.batch_size = batch_size
        super().__init__()

    def compute_output_spec(
        self, features: keras.KerasTensor, segment_ids: keras.KerasTensor
    ) -> keras.KerasTensor:
        C = features.shape[1]
        return keras.KerasTensor((self.batch_size, C), features.dtype)

    def call(
        self, features: BackendTensor, segment_ids: BackendTensor
    ) -> BackendTensor:
        return grid_interpolate_ops.reduce_mean_final(
            features=features,
            segment_ids=segment_ids,
            grid_size=self.grid_size,
            batch_size=self.batch_size,
        )


def reduce_mean_final(
    features: Tensor, segment_ids: Tensor, grid_size: int, batch_size: int
) -> Tensor:
    return ReduceMeanFinal(grid_size=grid_size, batch_size=batch_size)(
        features=features,
        segment_ids=segment_ids,
    )
