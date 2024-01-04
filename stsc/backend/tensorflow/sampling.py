# import functools
# import typing as tp
# import tensorflow as tf
# from ..numba import sampling as _impl


# def throttled_sample(
#     pixel_ids: tf.Tensor,
#     times: tf.Tensor,
#     batch_splits: tf.Tensor,
#     sample_rate: int,
#     min_dt: tp.Union[int, float],
#     grid_size: int,
# ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
#     sample_ids, batch_splits_out = tf.numpy_function(
#         functools.partial(
#             _impl.throttled_sample,
#             sample_rate=sample_rate,
#             min_dt=min_dt,
#             grid_size=grid_size,
#         ),
#         (pixel_ids, times, batch_splits),
#         (tf.int32, tf.int32),
#         stateful=False,
#     )
#     sample_ids.set_shape((pixel_ids.shape[0] // sample_rate,))
#     batch_splits_out.set_shape(batch_splits.shape)
#     return sample_ids, batch_splits_out


from stsc_ops import throttled_sample

__all__ = ["throttled_sample"]
