__all__ = ["throttled_sample"]
from jax_stsc_ops.sampling import throttled_sample

# import functools
# import typing as tp
# import jax.numpy as jnp
# from jax.core import ShapedArray
# from jk_utils.numba_utils import jambax
# import numba as nb

# from ..numba import sampling as _impl


# def _throttled_sample_abstract(
#     pixel_ids,
#     times,
#     batch_splits,
#     sample_rate: int,
# ) -> ShapedArray:
#     sample_ids = ShapedArray((pixel_ids.shape[0] // sample_rate,), jnp.int32)
#     splits_out = ShapedArray(batch_splits.shape, batch_splits.dtype)
#     return sample_ids, splits_out


# def throttled_sample(
#     pixel_ids: jnp.ndarray,
#     times: jnp.ndarray,
#     batch_splits: jnp.ndarray,
#     sample_rate: int,
#     min_dt: tp.Union[int, float],
#     grid_size: int,
# ) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
#     @nb.njit()
#     def numba_fn(args):
#         _impl._throttled_sample_in_place(
#             args, min_dt=min_dt, grid_size=grid_size, sample_rate=sample_rate
#         )

#     func = jambax.numba_to_jax(
#         "throttled_sampling",
#         # functools.partial(
#         #     _impl._throttled_sample_in_place,
#         #     sample_rate=sample_rate,
#         #     min_dt=min_dt,
#         #     grid_size=grid_size,
#         # ),
#         numba_fn,
#         functools.partial(_throttled_sample_abstract, sample_rate=sample_rate),
#     )
#     return func(pixel_ids, times, batch_splits)
