import typing as tp

from keras import KerasTensor, Operation

from ..backend import BackendTensor
from ..backend import sampling as _backend

Tensor = KerasTensor | BackendTensor


class ThrottledSample(Operation):
    def __init__(
        self, sample_rate: int, min_dt: int | float, grid_size: int, name=None
    ):
        self.sample_rate = sample_rate
        self.min_dt = min_dt
        self.grid_size = grid_size
        super().__init__(name=name)

    def compute_output_spec(
        self, pixel_ids: KerasTensor, times: KerasTensor, batch_splits: KerasTensor
    ):
        sample_ids = KerasTensor((pixel_ids.shape[0] // self.sample_rate,), "int32")
        batch_splits = KerasTensor(batch_splits.shape, batch_splits.dtype)
        return sample_ids, batch_splits

    def call(
        self,
        pixel_ids: BackendTensor,
        times: BackendTensor,
        batch_splits: KerasTensor,
    ):
        return _backend.throttled_sample(
            pixel_ids,
            times,
            batch_splits,
            sample_rate=self.sample_rate,
            min_dt=self.min_dt,
            grid_size=self.grid_size,
        )


def throttled_sample(
    pixel_ids: Tensor,
    times: Tensor,
    batch_splits: Tensor,
    sample_rate: int,
    min_dt: tp.Union[int, float],
    grid_size: int,
) -> tp.Tuple[Tensor, Tensor]:
    return ThrottledSample(sample_rate, min_dt, grid_size)(
        pixel_ids, times, batch_splits
    )
