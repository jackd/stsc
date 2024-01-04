__all__ = [
    "get_predecessor_ids",
    "get_stationary_predecessor_ids",
    "get_successor_ids",
    "get_permuted_stationary_predecessor_ids",
    "get_permuted_successor_ids",
]

import jax.numpy as jnp
import numba as nb
from jax.core import ShapedArray
from jax_stsc_ops.conv_preprocessing import (
    get_permuted_stationary_predecessor_ids,
    get_permuted_successor_ids,
    get_stationary_predecessor_ids,
    get_successor_ids,
)
from jk_utils.numba_utils import jambax

from ..numba import conv_preprocessing as _impl


def _get_predecessor_ids_abstract(
    pixel_ids_in: ShapedArray,
    times_in: ShapedArray,
    batch_splits_in: ShapedArray,
    pixel_ids_out: ShapedArray,
    times_out: ShapedArray,
    batch_splits_out: ShapedArray,
    kernel_offsets: ShapedArray,
) -> ShapedArray:
    return ShapedArray((times_out.shape[0], kernel_offsets.shape[0]), "int32")


def get_predecessor_ids(
    pixel_ids_in: jnp.ndarray,
    times_in: jnp.ndarray,
    batch_splits_in: jnp.ndarray,
    pixel_ids_out: jnp.ndarray,
    times_out: jnp.ndarray,
    batch_splits_out: jnp.ndarray,
    kernel_offsets: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    @nb.njit()
    def numba_fn(args):
        _impl._get_predecessor_ids_in_place(args, grid_size=grid_size)

    fn = jambax.numba_to_jax(
        "get_predecessor_ids",
        # functools.partial(
        #     _impl._get_predecessor_ids_in_place,
        #     grid_size=grid_size,
        # ),
        numba_fn,
        _get_predecessor_ids_abstract,
    )
    return fn(
        pixel_ids_in,
        times_in,
        batch_splits_in,
        pixel_ids_out,
        times_out,
        batch_splits_out,
        kernel_offsets,
    )


# def _get_successor_ids_abstract(
#     pixel_ids_in: ShapedArray,
#     times_in: ShapedArray,
#     batch_splits_in: ShapedArray,
#     pixel_ids_out: ShapedArray,
#     times_out: ShapedArray,
#     batch_splits_out: ShapedArray,
# ):
#     return ShapedArray(pixel_ids_in.shape, jnp.int32)


# def get_successor_ids(
#     pixel_ids_in: jnp.ndarray,
#     times_in: jnp.ndarray,
#     batch_splits_in: jnp.ndarray,
#     pixel_ids_out: jnp.ndarray,
#     times_out: jnp.ndarray,
#     batch_splits_out: jnp.ndarray,
#     grid_size: int,
# ) -> jnp.ndarray:
#     @nb.njit()
#     def numba_fn(args):
#         _impl._get_successor_ids_in_place(args, grid_size=grid_size)

#     fn = jambax.numba_to_jax(
#         "get_successor_ids",
#         # numba_fn=functools.partial(
#         #     _impl._get_successor_ids_in_place, grid_size=grid_size
#         # ),
#         numba_fn,
#         abstract_eval_fn=_get_successor_ids_abstract,
#     )
#     return fn(
#         pixel_ids_in,
#         times_in,
#         batch_splits_in,
#         pixel_ids_out,
#         times_out,
#         batch_splits_out,
#     )


# def _get_stationary_predecessor_ids_abstract(
#     pixel_ids: ShapedArray,
#     batch_splits: ShapedArray,
#     kernel_offsets: ShapedArray,
# ) -> ShapedArray:
#     return ShapedArray((pixel_ids.shape[0], kernel_offsets.shape[0]), jnp.int32)


# def get_stationary_predecessor_ids(
#     pixel_ids: jnp.ndarray,
#     batch_splits: jnp.ndarray,
#     kernel_offsets: jnp.ndarray,
#     grid_size: int,
# ) -> jnp.ndarray:
#     @nb.njit()
#     def numba_fn(args):
#         _impl._get_stationary_predecessor_ids_in_place(args, grid_size=grid_size)

#     fn = jambax.numba_to_jax(
#         "get_stationary_predecessor_ids",
#         # functools.partial(
#         #     _impl._get_stationary_predecessor_ids_in_place,
#         #     grid_size=grid_size,
#         # ),
#         numba_fn,
#         _get_stationary_predecessor_ids_abstract,
#     )
#     return fn(
#         pixel_ids,
#         batch_splits,
#         kernel_offsets,
#     )
