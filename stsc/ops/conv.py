import typing as tp

from keras import KerasTensor, ops

from ..backend import BackendTensor
from . import patches as patch_ops

Tensor = BackendTensor | KerasTensor


def conv(
    features: BackendTensor,
    times_in: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    kernel: BackendTensor,
    segment_ids: BackendTensor,
    predecessor_ids: BackendTensor,
    normalize: bool = True,
) -> BackendTensor:
    """
    Args:
        features: [E_in, C_in]
        times_in: [E_in]
        times_out: [E_out]
        decay_rate: [C_in] or [C_in // 2] * 2 for complex decay
        kernel: [K, C_in, C_out]
        segment_ids: [E_in] in [0, grid_size_in * batch_size]
        predecessor_ids: [E_out, K] in [0, E_in]

    Returns:
        [E_out, C_out] output features
    """
    K, C_in, C_out = kernel.shape
    x = patch_ops.get_patches(
        features,
        times_in,
        times_out,
        decay_rate,
        segment_ids,
        predecessor_ids,
        normalize=normalize,
    )
    return ops.matmul(
        ops.reshape(x, (-1, K * C_in)), ops.reshape(kernel, (K * C_in, C_out))
    )


def depthwise_conv(
    features: BackendTensor,
    times_in: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    kernel: BackendTensor,
    segment_ids: BackendTensor,
    predecessor_ids: BackendTensor,
    normalize: bool = True,
) -> BackendTensor:
    """
    Args:
        features: [E_in, C]
        times_in: [E_in]
        times_out: [E_out]
        decay_rate: [C] or [C // 2] * 2 for complex decay
        kernel: [K, C]
        segment_ids: [E_in] in [0, grid_size_in * batch_size]
        predecessor_ids: [E_out, K] in [0, E_in]

    Returns:
        [E_out, C] output features
    """
    x = patch_ops.get_patches(
        features,
        times_in,
        times_out,
        decay_rate,
        segment_ids,
        predecessor_ids,
        normalize=normalize,
    )
    return ops.sum(x * kernel, axis=1)


def one_hot_conv(
    times_in: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    kernel: BackendTensor,
    segment_filter_ids: BackendTensor,
    one_hot_predecessor_ids: BackendTensor,
) -> BackendTensor:
    """
    Args:
        times_in: [E_in]
        times_out: [E_out]
        decay_rate: [C_in] or [C_in // 2] * 2 for complex decay
        kernel: [K, C_in, C_out]
        segment_filter_ids: [E_in] in [0, batch_size * grid_size_in * C_in]
        one_hot_predecessor_ids: [E_out, K, C_in] in [0, E_in]

    Returns:
        [E_out, C_out] output features
    """
    K, C_in, C_out = kernel.shape
    x = patch_ops.get_one_hot_patches(
        times_in, times_out, decay_rate, segment_filter_ids, one_hot_predecessor_ids
    )
    return ops.matmul(
        ops.reshape(x, (-1, K * C_in)), ops.reshape(kernel, (K * C_in, C_out))
    )


def one_hot_depthwise_conv(
    times_in: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    kernel: BackendTensor,
    segment_filter_ids: BackendTensor,
    one_hot_predecessor_ids: BackendTensor,
) -> BackendTensor:
    """
    Args:
        times_in: [E_in]
        times_out: [E_out]
        decay_rate: [C] or [C // 2] * 2 for complex decay
        kernel: [K, C]
        segment_filter_ids: [E_in] in [0, batch_size * grid_size_in * C]
        one_hot_predecessor_ids: [E_out, K, C] in [0, E_in]

    Returns:
        [E_out, C_out] output features
    """
    x = patch_ops.get_one_hot_patches(
        times_in=times_in,
        times_out=times_out,
        decay_rate=decay_rate,
        segment_filter_ids=segment_filter_ids,
        one_hot_predecessor_ids=one_hot_predecessor_ids,
    )
    return ops.sum(x * kernel, axis=1)


def exclusive_conv(
    features: BackendTensor,
    dt: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    kernel: BackendTensor,
    successor_kernel_ids: BackendTensor,
    segment_ids_out: BackendTensor,
    indices_are_sorted: bool = False,
    normalize: bool = True,
) -> BackendTensor:
    """
    Args:
        features: [E_in, C_in]
        dt: [E_in]
        times_out: [E_out]
        decay_rate: [C_in] or [C_in // 2] tuple (complex components)
        kernel: [K, C_in, C_out]
        successor_kernel_ids: [E_in] in [0, E_out * K]
        segment_ids_out: [E_out]

    Returns:
        [E_out, K, C_out]
    """
    K, C_in, C_out = kernel.shape
    features.shape[1:] == (C_in,), (features.shape, C_in)
    x = patch_ops.get_exclusive_patches(
        features=features,
        dt=dt,
        times_out=times_out,
        decay_rate=decay_rate,
        successor_kernel_ids=successor_kernel_ids,
        segment_ids_out=segment_ids_out,
        indices_are_sorted=indices_are_sorted,
        kernel_size=K,
        normalize=normalize,
    )
    return ops.matmul(
        ops.reshape(x, (-1, K * C_in)), ops.reshape(kernel, (K * C_in, C_out))
    )


def exclusive_depthwise_conv(
    features: BackendTensor,
    dt: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    kernel: BackendTensor,
    successor_kernel_ids: BackendTensor,
    segment_ids_out: BackendTensor,
    indices_are_sorted: bool = False,
    normalize: bool = True,
) -> BackendTensor:
    """
    Args:
        features: [E_in, C]
        dt: [E_in]
        times_out: [E_out]
        decay_rate: [C] of [C // 2] tuple (complex components)
        kernel: [K, C]
        successor_kernel_ids: [E_in] in [0, E_out * K]
        segment_ids_out: [E_out]

    Returns:
        [E_out, K, C]
    """
    K, C = kernel.shape
    assert features.shape[1:] == (C,), (features.shape, C)
    x = patch_ops.get_exclusive_patches(
        features=features,
        dt=dt,
        times_out=times_out,
        decay_rate=decay_rate,
        successor_kernel_ids=successor_kernel_ids,
        segment_ids_out=segment_ids_out,
        indices_are_sorted=indices_are_sorted,
        kernel_size=K,
        normalize=normalize,
    )
    return ops.sum(x * kernel, axis=1)


def one_hot_exclusive_conv(
    dt: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    kernel: BackendTensor,
    successor_kernel_channel_ids: BackendTensor,
    segment_ids_out: BackendTensor,
    indices_are_sorted: bool = False,
    normalize: bool = True,
) -> BackendTensor:
    """
    Args:
        dt: [E_in]
        times_out: [E_out]
        decay_rate: [C_in, M] or [C_in // 2, M] tuple (complex components)
        kernel[K, C_in*M, C_out]
        successor_kernel_channel_ids: [E_in] in [0, E_out * K * C_in]
        segment_ids_out: [E_out]

    Returns:
        [E_out, K, C_out]
    """
    K, C_in, C_out = kernel.shape
    x = patch_ops.get_one_hot_exclusive_patches(
        dt=dt,
        times_out=times_out,
        decay_rate=decay_rate,
        successor_kernel_channel_ids=successor_kernel_channel_ids,
        segment_ids_out=segment_ids_out,
        indices_are_sorted=indices_are_sorted,
        kernel_size=K,
        normalize=normalize,
    )
    return ops.matmul(
        ops.reshape(x, (-1, K * C_in)), ops.reshape(kernel, (K * C_in, C_out))
    )


def one_hot_exclusive_depthwise_conv(
    dt: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    kernel: BackendTensor,
    successor_kernel_channel_ids: BackendTensor,
    segment_ids_out: BackendTensor,
    indices_are_sorted: bool = False,
    normalize: bool = True,
) -> BackendTensor:
    """
    Args:
        dt: [E_in]
        times_out: [E_out]
        decay_rate: [C, M] of [C // 2, M] tuple (complex components)
        successor_kernel_channel_ids: [E_in] in [0, E_out * K * P]
        segment_ids_out: [E_out]

    Returns:
        [E_out, K, C*M]
    """
    K, _ = kernel.shape
    x = patch_ops.get_one_hot_exclusive_patches(
        dt=dt,
        times_out=times_out,
        decay_rate=decay_rate,
        successor_kernel_channel_ids=successor_kernel_channel_ids,
        segment_ids_out=segment_ids_out,
        indices_are_sorted=indices_are_sorted,
        kernel_size=K,
        normalize=normalize,
    )
    return ops.sum(x * kernel, axis=1)


# class Conv(Operation):
#     def compute_output_spec(
#         self,
#         features: KerasTensor,
#         times_in: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         kernel: KerasTensor,
#         segment_ids: KerasTensor,
#         predecessor_ids: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor((predecessor_ids.shape[0], kernel.shape[2]), features.dtype)

#     def call(
#         self,
#         features: BackendTensor,
#         times_in: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor,
#         kernel: BackendTensor,
#         segment_ids: BackendTensor,
#         predecessor_ids: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.conv(
#             features=features,
#             times_in=times_in,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             kernel=kernel,
#             segment_ids=segment_ids,
#             predecessor_ids=predecessor_ids,
#         )


# def conv(
#     features: Tensor,
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     kernel: Tensor,
#     segment_ids: Tensor,
#     predecessor_ids: Tensor,
# ) -> Tensor:
#     return Conv()(
#         features=features,
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         kernel=kernel,
#         segment_ids=segment_ids,
#         predecessor_ids=predecessor_ids,
#     )


# class DepthwiseConv(Operation):
#     def compute_output_spec(
#         self,
#         features: KerasTensor,
#         times_in: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         kernel: KerasTensor,
#         segment_ids: KerasTensor,
#         predecessor_ids: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor(
#             (predecessor_ids.shape[0], features.shape[-1]), features.dtype
#         )

#     def call(
#         self,
#         features: BackendTensor,
#         times_in: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
#         kernel: BackendTensor,
#         segment_ids: BackendTensor,
#         predecessor_ids: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.depthwise_conv(
#             features=features,
#             times_in=times_in,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             kernel=kernel,
#             segment_ids=segment_ids,
#             predecessor_ids=predecessor_ids,
#         )


# def depthwise_conv(
#     features: Tensor,
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     kernel: Tensor,
#     segment_ids: Tensor,
#     predecessor_ids: Tensor,
# ) -> Tensor:
#     return DepthwiseConv()(
#         features=features,
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         kernel=kernel,
#         segment_ids=segment_ids,
#         predecessor_ids=predecessor_ids,
#     )


# class Pooling(Operation):
#     def __init__(self, reduction: str = "mean", name=None):
#         self.reduction = reduction
#         super().__init__(name=name)

#     def compute_output_spec(
#         self,
#         features: KerasTensor,
#         times_in: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: tp.Tuple[KerasTensor, KerasTensor],
#         segment_ids: KerasTensor,
#         predecessor_ids: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor(
#             (predecessor_ids.shape[0], features.shape[-1]), features.dtype
#         )

#     def call(
#         self,
#         features: BackendTensor,
#         times_in: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
#         segment_ids: BackendTensor,
#         predecessor_ids: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.pooling(
#             features=features,
#             times_in=times_in,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             segment_ids=segment_ids,
#             predecessor_ids=predecessor_ids,
#             reduction=self.reduction,
#         )


# def pooling(
#     features: Tensor,
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     segment_ids: Tensor,
#     predecessor_ids: Tensor,
#     reduction: str = "mean",
# ):
#     return Pooling(reduction=reduction)(
#         features=features,
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         segment_ids=segment_ids,
#         predecessor_ids=predecessor_ids,
#     )


# def mean_pooling(
#     features: Tensor,
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     segment_ids: Tensor,
#     predecessor_ids: Tensor,
# ) -> Tensor:
#     return pooling(
#         features=features,
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         segment_ids=segment_ids,
#         predecessor_ids=predecessor_ids,
#         reduction="mean",
#     )


# def max_pooling(
#     features: Tensor,
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     segment_ids: Tensor,
#     predecessor_ids: Tensor,
# ) -> Tensor:
#     return pooling(
#         features=features,
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         segment_ids=segment_ids,
#         predecessor_ids=predecessor_ids,
#         reduction="max",
#     )


# class OneHotConv(Operation):
#     def compute_output_spec(
#         self,
#         times_in: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         kernel: KerasTensor,
#         segment_filter_ids: KerasTensor,
#         one_hot_predecessor_ids: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor(
#             (one_hot_predecessor_ids.shape[0], kernel.shape[2]), kernel.dtype
#         )

#     def call(
#         self,
#         times_in: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor,
#         kernel: BackendTensor,
#         segment_filter_ids: BackendTensor,
#         one_hot_predecessor_ids: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.one_hot_conv(
#             times_in=times_in,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             kernel=kernel,
#             segment_filter_ids=segment_filter_ids,
#             one_hot_predecessor_ids=one_hot_predecessor_ids,
#         )


# def one_hot_conv(
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     kernel: Tensor,
#     segment_filter_ids: Tensor,
#     one_hot_predecessor_ids: Tensor,
# ) -> Tensor:
#     return OneHotConv()(
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         kernel=kernel,
#         segment_filter_ids=segment_filter_ids,
#         one_hot_predecessor_ids=one_hot_predecessor_ids,
#     )


# class OneHotDepthwiseConv(Operation):
#     def compute_output_spec(
#         self,
#         times_in: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         kernel: KerasTensor,
#         segment_filter_ids: KerasTensor,
#         one_hot_predecessor_ids: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor(
#             (one_hot_predecessor_ids.shape[0], kernel.shape[-1]), kernel.dtype
#         )

#     def call(
#         self,
#         times_in: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
#         kernel: BackendTensor,
#         segment_filter_ids: BackendTensor,
#         one_hot_predecessor_ids: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.one_hot_depthwise_conv(
#             times_in=times_in,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             kernel=kernel,
#             segment_filter_ids=segment_filter_ids,
#             one_hot_predecessor_ids=one_hot_predecessor_ids,
#         )


# def one_hot_depthwise_conv(
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     kernel: Tensor,
#     segment_filter_ids: Tensor,
#     one_hot_predecessor_ids: Tensor,
# ) -> Tensor:
#     return OneHotDepthwiseConv()(
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         kernel=kernel,
#         segment_filter_ids=segment_filter_ids,
#         one_hot_predecessor_ids=one_hot_predecessor_ids,
#     )


# class OneHotPooling(Operation):
#     def __init__(self, reduction: str = "mean", name=None):
#         self.reduction = reduction
#         super().__init__(name=name)

#     def compute_output_spec(
#         self,
#         times_in: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: tp.Tuple[KerasTensor, KerasTensor],
#         segment_filter_ids: KerasTensor,
#         one_hot_predecessor_ids: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor(
#             (one_hot_predecessor_ids.shape[0], one_hot_predecessor_ids.shape[-1]),
#             (decay_rate if isinstance(decay_rate, Tensor) else decay_rate[0]).dtype,
#         )

#     def call(
#         self,
#         times_in: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
#         segment_filter_ids: BackendTensor,
#         one_hot_predecessor_ids: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.one_hot_pooling(
#             times_in=times_in,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             segment_filter_ids=segment_filter_ids,
#             one_hot_predecessor_ids=one_hot_predecessor_ids,
#             reduction=self.reduction,
#         )


# def one_hot_pooling(
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     segment_filter_ids: Tensor,
#     one_hot_predecessor_ids: Tensor,
#     reduction: str = "mean",
# ):
#     return OneHotPooling(reduction=reduction)(
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         segment_filter_ids=segment_filter_ids,
#         one_hot_predecessor_ids=one_hot_predecessor_ids,
#     )


# def one_hot_mean_pooling(
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     segment_filter_ids: Tensor,
#     one_hot_predecessor_ids: Tensor,
# ) -> Tensor:
#     return one_hot_pooling(
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         segment_filter_ids=segment_filter_ids,
#         one_hot_predecessor_ids=one_hot_predecessor_ids,
#         reduction="mean",
#     )


# def one_hot_max_pooling(
#     times_in: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     segment_filter_ids: Tensor,
#     one_hot_predecessor_ids: Tensor,
# ) -> Tensor:
#     return one_hot_pooling(
#         times_in=times_in,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         segment_filter_ids=segment_filter_ids,
#         one_hot_predecessor_ids=one_hot_predecessor_ids,
#         reduction="max",
#     )


# class ExclusiveConv(Operation):
#     def __init__(self, indices_are_sorted: int, name=None):
#         self.indices_are_sorted = indices_are_sorted
#         super().__init__(name=name)

#     def compute_output_spec(
#         self,
#         features: KerasTensor,
#         times_in: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         kernel: KerasTensor,
#         successor_kernel_ids: KerasTensor,
#         segment_ids_out: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor((segment_ids_out.shape[0], kernel.shape[2]), features.dtype)

#     def call(
#         self,
#         features: BackendTensor,
#         dt: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
#         kernel: BackendTensor,
#         successor_kernel_ids: BackendTensor,
#         segment_ids_out: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.exclusive_conv(
#             features=features,
#             dt=dt,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             kernel=kernel,
#             successor_kernel_ids=successor_kernel_ids,
#             segment_ids_out=segment_ids_out,
#             indices_are_sorted=self.indices_are_sorted,
#         )


# def exclusive_conv(
#     features: Tensor,
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     kernel: Tensor,
#     successor_kernel_ids: Tensor,
#     segment_ids_out: Tensor,
#     indices_are_sorted: bool = False,
# ) -> Tensor:
#     return ExclusiveConv(indices_are_sorted=indices_are_sorted)(
#         features=features,
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         kernel=kernel,
#         successor_kernel_ids=successor_kernel_ids,
#         segment_ids_out=segment_ids_out,
#     )


# class ExclusiveDepthwiseConv(Operation):
#     def __init__(self, indices_are_sorted: int, name=None):
#         self.indices_are_sorted = indices_are_sorted
#         super().__init__(name=name)

#     def compute_output_spec(
#         self,
#         features: KerasTensor,
#         dt: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         kernel: KerasTensor,
#         successor_kernel_ids: KerasTensor,
#         segment_ids_out: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor(
#             (segment_ids_out.shape[0], features.shape[1]), features.dtype
#         )

#     def call(
#         self,
#         features: BackendTensor,
#         dt: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
#         kernel: BackendTensor,
#         successor_kernel_ids: BackendTensor,
#         segment_ids_out: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.exclusive_depthwise_conv(
#             features=features,
#             dt=dt,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             kernel=kernel,
#             successor_kernel_ids=successor_kernel_ids,
#             segment_ids_out=segment_ids_out,
#             indices_are_sorted=self.indices_are_sorted,
#         )


# def exclusive_depthwise_conv(
#     features: Tensor,
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     kernel: Tensor,
#     segment_ids_out: Tensor,
#     successor_kernel_ids: Tensor,
#     indices_are_sorted: bool = False,
# ) -> Tensor:
#     return ExclusiveDepthwiseConv(indices_are_sorted=indices_are_sorted)(
#         features=features,
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         kernel=kernel,
#         successor_kernel_ids=successor_kernel_ids,
#         segment_ids_out=segment_ids_out,
#     )


# class ExclusivePooling(Operation):
#     def __init__(
#         self,
#         stride: int,
#         reduction: str = "mean",
#         indices_are_sorted: bool = False,
#         name=None,
#     ):
#         self.stride = stride
#         self.reduction = reduction
#         self.indices_are_sorted = indices_are_sorted
#         super().__init__(name=name)

#     def compute_output_spec(
#         self,
#         features: KerasTensor,
#         dt: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         exclusive_conv_indices: KerasTensor,
#         segment_ids_out: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor(
#             (segment_ids_out.shape[0], features.shape[-1]), features.dtype
#         )

#     def call(
#         self,
#         features: BackendTensor,
#         dt: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor,
#         successor_kernel_ids: BackendTensor,
#         segment_ids_out: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.exclusive_pooling(
#             features=features,
#             dt=dt,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             successor_kernel_ids=successor_kernel_ids,
#             segment_ids_out=segment_ids_out,
#             reduction=self.reduction,
#             stride=self.stride,
#             indices_are_sorted=self.indices_are_sorted,
#         )


# def exclusive_pooling(
#     features: Tensor,
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     segment_ids_out: Tensor,
#     successor_kernel_ids: Tensor,
#     stride: int,
#     reduction: str = "mean",
#     indices_are_sorted: bool = False,
# ):
#     return ExclusivePooling(
#         stride=stride, reduction=reduction, indices_are_sorted=indices_are_sorted
#     )(
#         features=features,
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         successor_kernel_ids=successor_kernel_ids,
#         segment_ids_out=segment_ids_out,
#     )


# def mean_exclusive_pooling(
#     features: Tensor,
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     successor_kernel_ids: Tensor,
#     segment_ids_out: Tensor,
#     stride: int,
#     indices_are_sorted: bool = False,
# ) -> Tensor:
#     return exclusive_pooling(
#         features=features,
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         successor_kernel_ids=successor_kernel_ids,
#         segment_ids_out=segment_ids_out,
#         stride=stride,
#         reduction="mean",
#         indices_are_sorted=indices_are_sorted,
#     )


# def max_exclusive_pooling(
#     features: Tensor,
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     successor_kernel_ids: Tensor,
#     segment_ids_out: Tensor,
#     indices_are_sorted: bool = False,
# ) -> Tensor:
#     return exclusive_pooling(
#         features=features,
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         successor_kernel_ids=successor_kernel_ids,
#         segment_ids_out=segment_ids_out,
#         reduction="max",
#         indices_are_sorted=indices_are_sorted,
#     )


# class OneHotExclusiveConv(Operation):
#     def __init__(self, indices_are_sorted: int, name=None):
#         self.indices_are_sorted = indices_are_sorted
#         super().__init__(name=name)

#     def compute_output_spec(
#         self,
#         dt: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         kernel: KerasTensor,
#         successor_kernel_filters_ids: KerasTensor,
#         segment_ids_out: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor((segment_ids_out.shape[0], kernel.shape[2]), kernel.dtype)

#     def call(
#         self,
#         dt: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
#         kernel: BackendTensor,
#         successor_kernel_filters_ids: BackendTensor,
#         segment_ids_out: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.one_hot_exclusive_conv(
#             dt=dt,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             kernel=kernel,
#             segment_ids_out=segment_ids_out,
#             successor_kernel_filters_ids=successor_kernel_filters_ids,
#             indices_are_sorted=self.indices_are_sorted,
#         )


# def one_hot_exclusive_conv(
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     kernel: Tensor,
#     successor_kernel_channel_ids: Tensor,
#     segment_ids_out: Tensor,
#     indices_are_sorted: bool = False,
# ) -> Tensor:
#     return OneHotExclusiveConv(indices_are_sorted=indices_are_sorted)(
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         kernel=kernel,
#         segment_ids_out=segment_ids_out,
#         successor_kernel_channel_ids=successor_kernel_channel_ids,
#     )


# class OneHotExclusiveDepthwiseConv(Operation):
#     def __init__(self, indices_are_sorted: int, name=None):
#         self.indices_are_sorted = indices_are_sorted
#         super().__init__(name=name)

#     def compute_output_spec(
#         self,
#         dt: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         kernel: KerasTensor,
#         successor_kernel_channel_ids: KerasTensor,
#         segment_ids_out: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor((segment_ids_out.shape[0], kernel.shape[1]), kernel.dtype)

#     def call(
#         self,
#         dt: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
#         kernel: BackendTensor,
#         segment_ids_out: BackendTensor,
#         successor_kernel_channel_ids: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.one_hot_exclusive_depthwise_conv(
#             dt=dt,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             kernel=kernel,
#             successor_kernel_channel_ids=successor_kernel_channel_ids,
#             segment_ids_out=segment_ids_out,
#             indices_are_sorted=self.indices_are_sorted,
#         )


# def one_hot_exclusive_depthwise_conv(
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     kernel: Tensor,
#     successor_kernel_channel_ids: Tensor,
#     segment_ids_out: Tensor,
#     indices_are_sorted: bool = False,
# ) -> Tensor:
#     return OneHotExclusiveDepthwiseConv(indices_are_sorted=indices_are_sorted)(
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         kernel=kernel,
#         successor_kernel_channel_ids=successor_kernel_channel_ids,
#         segment_ids_out=segment_ids_out,
#     )


# class OneHotExclusivePooling(Operation):
#     def __init__(
#         self,
#         stride: int,
#         reduction: str = "mean",
#         indices_are_sorted: bool = False,
#         name=None,
#     ):
#         self.stride = stride
#         self.reduction = reduction
#         self.indices_are_sorted = indices_are_sorted
#         super().__init__(name=name)

#     def compute_output_spec(
#         self,
#         dt: KerasTensor,
#         times_out: KerasTensor,
#         decay_rate: KerasTensor | tp.Tuple[KerasTensor, KerasTensor],
#         successor_kernel_channel_ids: KerasTensor,
#         segment_ids_out: KerasTensor,
#     ) -> KerasTensor:
#         if isinstance(decay_rate, (list, tuple)):
#             _assert_is_complex(decay_rate)
#             filters = decay_rate[0].shape[0] * 2
#             dtype = decay_rate[0].dtype
#         else:
#             filters = decay_rate.shape[0]
#             dtype = decay_rate.dtype
#         return KerasTensor((segment_ids_out.shape[0], filters), dtype)

#     def call(
#         self,
#         dt: BackendTensor,
#         times_out: BackendTensor,
#         decay_rate: BackendTensor,
#         successor_kernel_channel_ids: BackendTensor,
#         segment_ids_out: BackendTensor,
#     ) -> BackendTensor:
#         return _conv_backend.one_hot_exclusive_pooling(
#             dt=dt,
#             times_out=times_out,
#             decay_rate=decay_rate,
#             segment_ids_out=segment_ids_out,
#             successor_kernel_channel_ids=successor_kernel_channel_ids,
#             reduction=self.reduction,
#             stride=self.stride,
#             indices_are_sorted=self.indices_are_sorted,
#         )


# def one_hot_exclusive_pooling(
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     successor_kernel_channel_ids: Tensor,
#     segment_ids_out: Tensor,
#     stride: int,
#     reduction: str = "mean",
#     indices_are_sorted: bool = False,
# ):
#     return OneHotExclusivePooling(
#         stride=stride, reduction=reduction, indices_are_sorted=indices_are_sorted
#     )(
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         successor_kernel_channel_ids=successor_kernel_channel_ids,
#         segment_ids_out=segment_ids_out,
#     )


# def one_hot_mean_exclusive_pooling(
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     successor_kernel_channel_ids: Tensor,
#     segment_ids_out: Tensor,
#     stride: int,
#     indices_are_sorted: bool = False,
# ) -> Tensor:
#     return one_hot_exclusive_pooling(
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         successor_kernel_channel_ids=successor_kernel_channel_ids,
#         segment_ids_out=segment_ids_out,
#         stride=stride,
#         reduction="mean",
#         indices_are_sorted=indices_are_sorted,
#     )


# def one_hot_max_exclusive_pooling(
#     dt: Tensor,
#     times_out: Tensor,
#     decay_rate: Tensor | tp.Tuple[Tensor, Tensor],
#     successor_kernel_channel_ids: Tensor,
#     segment_ids_out: Tensor,
#     stride: int,
#     indices_are_sorted: bool = False,
# ) -> Tensor:
#     return one_hot_exclusive_pooling(
#         dt=dt,
#         times_out=times_out,
#         decay_rate=decay_rate,
#         successor_kernel_channel_ids=successor_kernel_channel_ids,
#         segment_ids_out=segment_ids_out,
#         stride=stride,
#         reduction="max",
#         indices_are_sorted=indices_are_sorted,
#     )
