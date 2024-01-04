# import tensorflow as tf
# from ..numba import counting_argsort as _impl


# def counting_argsort(segment_ids: tf.Tensor, splits: tf.Tensor) -> tf.Tensor:
#     order = tf.numpy_function(
#         _impl.counting_argsort,
#         (segment_ids, splits),
#         tf.int32,
#         stateful=False,
#     )
#     order.set_shape(segment_ids.shape)
#     assert segment_ids.shape is not None
#     return order

from stsc_ops import counting_argsort

__all__ = ["counting_argsort"]
