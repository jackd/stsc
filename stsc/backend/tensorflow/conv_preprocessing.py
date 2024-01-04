import functools

import tensorflow as tf
from stsc_ops import get_stationary_predecessor_ids, get_successor_ids

from ..numba import conv_preprocessing as _impl


def get_predecessor_ids(
    pixel_ids_in: tf.Tensor,
    times_in: tf.Tensor,
    batch_splits_in: tf.Tensor,
    pixel_ids_out: tf.Tensor,
    times_out: tf.Tensor,
    batch_splits_out: tf.Tensor,
    kernel_offsets: tf.Tensor,
    grid_size: int,
) -> tf.Tensor:
    result = tf.numpy_function(
        functools.partial(_impl.get_predecessor_ids, grid_size=grid_size),
        (
            pixel_ids_in,
            times_in,
            batch_splits_in,
            pixel_ids_out,
            times_out,
            batch_splits_out,
            kernel_offsets,
        ),
        tf.int32,
        stateful=False,
    )
    result.set_shape((times_out.shape[0], kernel_offsets.shape[0]))
    return result


__all__ = ["get_stationary_predecessor_ids", "get_successor_ids", "get_predecessor_ids"]


# def get_successor_ids(
#     pixel_ids_in: tf.Tensor,
#     times_in: tf.Tensor,
#     batch_splits_in: tf.Tensor,
#     pixel_ids_out: tf.Tensor,
#     times_out: tf.Tensor,
#     batch_splits_out: tf.Tensor,
#     grid_size: int,
# ) -> tf.Tensor:
#     successor_ids = tf.numpy_function(
#         functools.partial(_impl.get_successor_ids, grid_size=grid_size),
#         (
#             pixel_ids_in,
#             times_in,
#             batch_splits_in,
#             pixel_ids_out,
#             times_out,
#             batch_splits_out,
#         ),
#         tf.int32,
#         stateful=False,
#     )
#     successor_ids.set_shape(pixel_ids_in.shape)
#     return successor_ids


# def get_stationary_predecessor_ids(
#     pixel_ids: tf.Tensor,
#     batch_splits: tf.Tensor,
#     kernel_offsets: tf.Tensor,
#     grid_size: int,
# ) -> tf.Tensor:
#     @tf.numpy_function(Tout=tf.int32, stateful=False)
#     def func(pixel_ids, batch_splits, kernel_offsets):
#         return _impl.get_stationary_predecessor_ids(
#             pixel_ids=pixel_ids,
#             batch_splits=batch_splits,
#             kernel_offsets=kernel_offsets,
#             grid_size=int(grid_size),
#         )

#     predecessor_ids = func(
#         pixel_ids,
#         batch_splits,
#         kernel_offsets,
#     )
#     predecessor_ids.set_shape((pixel_ids.shape[0], kernel_offsets.shape[0]))
#     return predecessor_ids
