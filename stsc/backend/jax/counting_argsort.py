# __all__ = ["counting_argsort"]

# import jax.numpy as jnp
# from jax.core import ShapedArray
# from jk_utils.numba_utils import jambax

# from ..numba import counting_argsort as _impl


# def _counting_argsort_abstract(
#     segment_ids: ShapedArray, splits: ShapedArray
# ) -> ShapedArray:
#     return ShapedArray(segment_ids.shape, jnp.int32)


# counting_argsort = jambax.numba_to_jax(
#     "counting_argsort", _impl._counting_argsort_in_place, _counting_argsort_abstract
# )
