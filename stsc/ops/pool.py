import typing as tp

from keras import KerasTensor, ops

from ..backend import BackendTensor
from . import patches as patch_ops

Tensor = BackendTensor | KerasTensor


def pooling(
    features: BackendTensor,
    times_in: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    segment_ids: BackendTensor,
    predecessor_ids: BackendTensor,
    reduction: str = "mean",
) -> BackendTensor:
    return {
        "mean": ops.mean,
        "sum": ops.sum,
        "max": ops.max,
    }[reduction](
        patch_ops.get_patches(
            features, times_in, times_out, decay_rate, segment_ids, predecessor_ids
        ),
        axis=1,
    )


def one_hot_pooling(
    times_in: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    segment_filter_ids: BackendTensor,
    one_hot_predecessor_ids: BackendTensor,
    reduction: str = "mean",
) -> BackendTensor:
    return {
        "mean": ops.mean,
        "sum": ops.sum,
        "max": ops.max,
    }[reduction](
        patch_ops.get_one_hot_patches(
            times_in=times_in,
            times_out=times_out,
            decay_rate=decay_rate,
            segment_filter_ids=segment_filter_ids,
            one_hot_predecessor_ids=one_hot_predecessor_ids,
        ),
        axis=1,
    )


def exclusive_pooling(
    features: BackendTensor,
    dt: BackendTensor,
    times_out: BackendTensor,
    decay_rate: tp.Tuple[BackendTensor, BackendTensor],
    successor_kernel_ids: BackendTensor,
    segment_ids_out: BackendTensor,
    stride: int,
    indices_are_sorted: bool = False,
    reduction="mean",
) -> BackendTensor:
    """
    Args:
        features: [E_in, P]
        dt: [E_in]
        times_out: [E_out]
        decay_rate: [P] or ([P // 2], [P // 2]) complex components
        successor_kernel_ids: [E_in] in [0, E_out * K]
        segment_ids_out: [E_out] in [0, num_segments]

    Returns:
        [E_out, P]
    """
    x = patch_ops.get_exclusive_patches(
        features=features,
        dt=dt,
        times_out=times_out,
        decay_rate=decay_rate,
        successor_kernel_ids=successor_kernel_ids,
        segment_ids_out=segment_ids_out,
        indices_are_sorted=indices_are_sorted,
        kernel_size=stride,
    )
    return {
        "mean": ops.mean,
        "sum": ops.sum,
        "max": ops.max,
    }[
        reduction
    ](x, axis=1)


def one_hot_exclusive_pooling(
    dt: BackendTensor,
    times_out: BackendTensor,
    decay_rate: tp.Tuple[BackendTensor, BackendTensor],
    successor_kernel_channel_ids: BackendTensor,
    segment_ids_out: BackendTensor,
    stride: int,
    indices_are_sorted: bool = False,
    reduction: str = "mean",
) -> BackendTensor:
    """
    Args:
        dt: [E_in]
        times_out: [E_out]
        decay_rate: [P] or ([P // 2], [P // 2]) complex components
        successor_kernel_channel_ids: [E_in] in [0, E_out * K * P]
        segment_ids_out: [E_out]

    Returns:
        [E_out, P]
    """
    x = patch_ops.get_one_hot_exclusive_patches(
        dt=dt,
        times_out=times_out,
        decay_rate=decay_rate,
        successor_kernel_channel_ids=successor_kernel_channel_ids,
        segment_ids_out=segment_ids_out,
        indices_are_sorted=indices_are_sorted,
        kernel_size=stride,
    )
    return {
        "mean": ops.mean,
        "sum": ops.sum,
        "max": ops.max,
    }[
        reduction
    ](x, axis=1)
