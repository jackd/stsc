import typing as tp

from keras import KerasTensor, Operation

from ..backend import BackendTensor
from ..backend import patches as _patches_backend

Tensor = BackendTensor | KerasTensor


class GetPatches(Operation):
    def compute_output_spec(
        self,
        features: KerasTensor,
        times_in: KerasTensor,
        times_out: KerasTensor,
        decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
        segment_ids: KerasTensor,
        predecessor_ids: KerasTensor,
    ) -> KerasTensor:
        return KerasTensor(
            (predecessor_ids.shape[0], predecessor_ids.shape[1], features.shape[1]),
            features.dtype,
        )

    def call(
        self,
        features: BackendTensor,
        times_in: BackendTensor,
        times_out: BackendTensor,
        decay_rate: BackendTensor,
        segment_ids: BackendTensor,
        predecessor_ids: BackendTensor,
    ) -> BackendTensor:
        return _patches_backend.get_patches(
            features=features,
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            segment_ids=segment_ids,
            predecessor_ids=predecessor_ids,
        )


def get_patches(
    features: Tensor,
    times_in: Tensor,
    times_out: Tensor,
    decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
    segment_ids: Tensor,
    predecessor_ids: Tensor,
) -> Tensor:
    return GetPatches()(
        features=features,
        times_in=times_in,
        times_out=times_out,
        decay_rate=decay_rate,
        segment_ids=segment_ids,
        predecessor_ids=predecessor_ids,
    )


class GetOneHotPatches(Operation):
    def compute_output_spec(
        self,
        times_in: KerasTensor,
        times_out: KerasTensor,
        decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
        segment_filter_ids: KerasTensor,
        one_hot_predecessor_ids: KerasTensor,
    ) -> KerasTensor:
        if isinstance(decay_rate, Tensor):
            dtype = decay_rate.dtype
        else:
            dtype = decay_rate[0].dtype
        return KerasTensor(one_hot_predecessor_ids.shape, dtype)

    def call(
        self,
        times_in: BackendTensor,
        times_out: BackendTensor,
        decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
        segment_filter_ids: BackendTensor,
        one_hot_predecessor_ids: BackendTensor,
    ) -> BackendTensor:
        return _patches_backend.get_one_hot_patches(
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            segment_filter_ids=segment_filter_ids,
            one_hot_predecessor_ids=one_hot_predecessor_ids,
        )


def get_one_hot_patches(
    times_in: Tensor,
    times_out: Tensor,
    decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
    kernel: Tensor,
    segment_filter_ids: Tensor,
    one_hot_predecessor_ids: Tensor,
) -> Tensor:
    return GetOneHotPatches()(
        times_in=times_in,
        times_out=times_out,
        decay_rate=decay_rate,
        kernel=kernel,
        segment_filter_ids=segment_filter_ids,
        one_hot_predecessor_ids=one_hot_predecessor_ids,
    )


class GetExclusivePatches(Operation):
    def __init__(self, kernel_size: int, indices_are_sorted: int, name=None):
        self.kernel_size = kernel_size
        self.indices_are_sorted = indices_are_sorted
        super().__init__(name=name)

    def compute_output_spec(
        self,
        features: KerasTensor,
        times_in: KerasTensor,
        times_out: KerasTensor,
        decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
        successor_kernel_ids: KerasTensor,
        segment_ids_out: KerasTensor,
    ) -> KerasTensor:
        return KerasTensor(
            (segment_ids_out.shape[0], self.kernel_size, features.shape[1]),
            features.dtype,
        )

    def call(
        self,
        features: BackendTensor,
        dt: BackendTensor,
        times_out: BackendTensor,
        decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
        successor_kernel_ids: BackendTensor,
        segment_ids_out: BackendTensor,
    ) -> BackendTensor:
        return _patches_backend.get_exclusive_patches(
            features=features,
            dt=dt,
            times_out=times_out,
            decay_rate=decay_rate,
            successor_kernel_ids=successor_kernel_ids,
            segment_ids_out=segment_ids_out,
            kernel_size=self.kernel_size,
            indices_are_sorted=self.indices_are_sorted,
        )


def get_exclusive_patches(
    features: Tensor,
    dt: Tensor,
    times_out: Tensor,
    decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
    successor_kernel_ids: Tensor,
    segment_ids_out: Tensor,
    kernel_size: int,
    indices_are_sorted: bool = False,
) -> Tensor:
    return GetExclusivePatches(
        kernel_size=kernel_size, indices_are_sorted=indices_are_sorted
    )(
        features=features,
        dt=dt,
        times_out=times_out,
        decay_rate=decay_rate,
        successor_kernel_ids=successor_kernel_ids,
        segment_ids_out=segment_ids_out,
    )


class GetOneHotExclusivePatches(Operation):
    def __init__(self, kernel_size: int, indices_are_sorted: int, name=None):
        self.kernel_size = kernel_size
        self.indices_are_sorted = indices_are_sorted
        super().__init__(name=name)

    def compute_output_spec(
        self,
        dt: KerasTensor,
        times_out: KerasTensor,
        decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
        successor_kernel_channel_ids: KerasTensor,
        segment_ids_out: KerasTensor,
    ) -> KerasTensor:
        if isinstance(decay_rate, Tensor):
            filters = decay_rate.shape[0]
            dtype = filters.dtype
        else:
            filters = decay_rate[0].shape[0] * 2
            dtype = decay_rate[0].dtype
        return KerasTensor((segment_ids_out.shape[0], self.kernel_size, filters), dtype)

    def call(
        self,
        dt: BackendTensor,
        times_out: BackendTensor,
        decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
        successor_kernel_channel_ids: BackendTensor,
        segment_ids_out: BackendTensor,
    ) -> BackendTensor:
        return _patches_backend.get_one_hot_exclusive_patches(
            dt=dt,
            times_out=times_out,
            decay_rate=decay_rate,
            segment_ids_out=segment_ids_out,
            successor_kernel_channel_ids=successor_kernel_channel_ids,
            indices_are_sorted=self.indices_are_sorted,
            kernel_size=self.kernel_size,
        )


def get_one_hot_exclusive_patches(
    dt: Tensor,
    times_out: Tensor,
    decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
    successor_kernel_channel_ids: Tensor,
    segment_ids_out: Tensor,
    kernel_size: int,
    indices_are_sorted: bool = False,
) -> Tensor:
    return GetOneHotExclusivePatches(
        kernel_size=kernel_size, indices_are_sorted=indices_are_sorted
    )(
        dt=dt,
        times_out=times_out,
        decay_rate=decay_rate,
        segment_ids_out=segment_ids_out,
        successor_kernel_channel_ids=successor_kernel_channel_ids,
    )
