from jk_utils import asserts
from keras import KerasTensor, ops

from ..backend import BackendTensor

Tensor = BackendTensor | KerasTensor


def ravel_multi_index(
    multi_index: Tensor,
    dims: Tensor,
    axis: int = 0,
) -> Tensor:
    """
    Convert multi-dimensional tensor indices into flat tensor indices.

    Args:
        multi_index: [N, D] (axis in (-1, 1)) or [D, N] int array indices.
        dims: [D] shape of multi-dimensional tensor.
        axis: axis to reduce over.

    Returns:
        [N]
    """
    if not isinstance(multi_index, Tensor):
        multi_index = ops.convert_to_tensor(multi_index, multi_index.dtype)
    if not isinstance(dims, Tensor):
        dims = ops.convert_to_tensor(dims, multi_index.dtype)
    asserts.assert_has_rank(dims, 1)
    shape = [1] * len(multi_index.shape)
    cum_dims = ops.flip(ops.cumprod(ops.flip(dims), axis=0))
    cum_dims = ops.cast(cum_dims, multi_index.dtype)
    cum_dims = ops.concatenate((cum_dims[1:], ops.ones((1,), dtype=cum_dims.dtype)))
    shape[axis] = -1
    cum_dims = ops.reshape(cum_dims, shape)
    return ops.sum(multi_index * cum_dims, axis=axis)


def inverse_perm(permutation: Tensor, dtype=None) -> Tensor:
    asserts.assert_has_rank(permutation, 1, "permutation")
    if dtype is None:
        dtype = permutation.dtype
    return ops.scatter(
        ops.expand_dims(permutation, axis=-1),
        ops.arange(permutation.shape[0], dtype=dtype),
        shape=(permutation.shape[0],),
    )
