import numba as nb
import numpy as np


@nb.njit()
def _get_predecessor_ids_in_place(args, grid_size: int):
    (
        predecessor_ids,
        pixel_ids_in,
        times_in,
        batch_splits_in,
        pixel_ids_out,
        times_out,
        batch_splits_out,
        kernel_offsets,
    ) = args
    batch_size = batch_splits_in.shape[0] - 1
    (n_in,) = pixel_ids_in.shape
    last = np.empty((grid_size,), dtype=np.int32)
    # last = np.full((batch_size, grid_size), n_in, dtype=np.int32)
    predecessor_ids[:] = n_in
    # for b in nb.prange(batch_size):
    for b in range(batch_size):
        i_in, i_in_stop = batch_splits_in[b : b + 2]
        if i_in == n_in:
            break
        s_in = pixel_ids_in[i_in]
        t_in = times_in[i_in]
        last[:] = n_in
        for i_out in range(batch_splits_out[b], batch_splits_out[b + 1]):
            t_out = times_out[i_out]
            s_out = pixel_ids_out[i_out]
            while t_in <= t_out and i_in < i_in_stop:
                last[s_in] = i_in
                i_in += 1
                if i_in < n_in:
                    t_in = times_in[i_in]
                    s_in = pixel_ids_in[i_in]
            for ik, k in enumerate(kernel_offsets):
                predecessor_ids[i_out, ik] = last[s_out + k]


@nb.njit()
def get_predecessor_ids(
    pixel_ids_in: np.ndarray,
    times_in: np.ndarray,
    batch_splits_in: np.ndarray,
    pixel_ids_out: np.ndarray,
    times_out: np.ndarray,
    batch_splits_out: np.ndarray,
    kernel_offsets: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """
    Get most recent predecessor ids as used in e.g. `conv`.

    If `predecessor_ids[e_in] == e_out`, it means that output event `e_out`
    is the earliest event at or after input event `e_in` with the same pixel_id.

    Args:
        pixel_ids_in: [E_in] in [0, grid_size)
        times_in: [E_in]
        batch_splits_in: [B+1] in [0, E_in]
        pixel_ids_out: [E_out] in [0, grid_size)
        times_out: [E_out]
        batch_splits_out: [B+1] in [0, E_out]
        kernel_offsets: [K]
        grid_size: total size of the (1D) grid

    Returns:
        [E_out, K] values in [0, E_in)
    """
    (E_in,) = pixel_ids_in.shape
    assert times_in.shape == (E_in,), (times_in.shape, E_in)
    (E_out,) = pixel_ids_out.shape
    assert times_out.shape == (E_out,), (times_out.shape, E_out)
    assert batch_splits_in.shape == batch_splits_out.shape, (
        batch_splits_in.shape,
        batch_splits_out.shape,
    )
    (K,) = kernel_offsets.shape
    result = np.empty((E_out, K), dtype=np.int32)
    _get_predecessor_ids_in_place(
        (
            result,
            pixel_ids_in,
            times_in,
            batch_splits_in,
            pixel_ids_out,
            times_out,
            batch_splits_out,
            kernel_offsets,
        ),
        grid_size=grid_size,
    )
    return result


@nb.njit()
def _get_stationary_predecessor_ids_in_place(
    args,
    grid_size: int,
):
    (predecessor_ids, pixel_ids, batch_splits, kernel_offsets) = args
    (n,) = pixel_ids.shape
    predecessor_ids[:] = n
    batch_size = batch_splits.shape[0] - 1
    last = np.empty((grid_size,), dtype=np.int32)
    # last = np.full((batch_size, grid_size), n, dtype=np.int32)
    # for b in nb.prange(batch_size):
    for b in range(batch_size):
        last[:] = n
        for i in range(batch_splits[b], batch_splits[b + 1]):
            pixel_id = pixel_ids[i]
            last[pixel_id] = i
            for ik, k in enumerate(kernel_offsets):
                predecessor_ids[i, ik] = last[pixel_id + k]


@nb.njit()
def get_stationary_predecessor_ids(
    pixel_ids: np.ndarray,
    batch_splits: np.ndarray,
    kernel_offsets: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """
    Get predecessor indices for a stationary chronological event stream.

    This is an optimization of `get_predecessor_ids` for the case where the input stream
    is the same as the output stream. The results should be identical so long as times
    are unique.

    Args:
        pixel_ids: [E] in [0, grid_size)
        batch_splits: [B+1] in [0, E]
        kernel_offsets: [K]
        grid_size:

    Returns:
        [E, K] values in [0, E)
    """
    (E,) = pixel_ids.shape
    (K,) = kernel_offsets.shape
    predecessor_ids = np.empty((E, K), dtype=np.int32)

    _get_stationary_predecessor_ids_in_place(
        (
            predecessor_ids,
            pixel_ids,
            batch_splits,
            kernel_offsets,
        ),
        grid_size=grid_size,
    )
    return predecessor_ids


@nb.njit()
def _get_successor_ids_in_place(args, grid_size: int):
    (
        predecessor_ids,
        pixel_ids_in,
        times_in,
        batch_splits_in,
        pixel_ids_out,
        times_out,
        batch_splits_out,
    ) = args
    batch_size = batch_splits_in.shape[0] - 1
    (n_out,) = times_out.shape
    last = np.empty((grid_size,), "int32")
    # last = np.full((batch_size, grid_size), n_out, "int32")
    predecessor_ids[batch_splits_in[-1] :] = n_out
    # for b in nb.prange(batch_size):
    for b in range(batch_size):
        e_in_start, e_in_end = batch_splits_in[b : b + 2]
        e_out_start, e_out_end = batch_splits_out[b : b + 2]
        last[:] = n_out
        e_out = e_out_end - 1
        for e_in in range(e_in_end - 1, e_in_start - 1, -1):
            t_in = times_in[e_in]
            while e_out >= e_out_start and t_in <= times_out[e_out]:
                last[pixel_ids_out[e_out]] = e_out
                e_out -= 1
            predecessor_ids[e_in] = last[pixel_ids_in[e_in]]


@nb.njit()
def get_successor_ids(
    pixel_ids_in: np.ndarray,
    times_in: np.ndarray,
    batch_splits_in: np.ndarray,
    pixel_ids_out: np.ndarray,
    times_out: np.ndarray,
    batch_splits_out: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """
    Get successor_ids as used in e.g. `exclusive_conv`.

    If `successor_ids[e_in] == e_out`, it means that output event `e_out`
    is the earliest event at or after input event `e_in` with the same pixel_id.

    Args:
        pixel_ids_in: [E_in] in [0, grid_size)
        times_in: [E_in]
        batch_splits_in: [B + 1] in [0, E_in]
        pixel_ids_out: [E_out] in [0, grid_size)
        times_out: [E_out]
        batch_splits_out: [B + 1] in [0, E_out]
        grid_size:

    Returns:
        [E_in] successor_ids in [0, E_out]
    """
    (E_in,) = pixel_ids_in.shape
    assert times_in.shape == (E_in,)
    (E_out,) = pixel_ids_out.shape
    assert times_out.shape == (E_out,)
    assert batch_splits_in.shape == batch_splits_out.shape
    assert len(batch_splits_in.shape) == 1
    predecessor_ids = np.empty((E_in,), "int32")
    _get_successor_ids_in_place(
        (
            predecessor_ids,
            pixel_ids_in,
            times_in,
            batch_splits_in,
            pixel_ids_out,
            times_out,
            batch_splits_out,
        ),
        grid_size=grid_size,
    )
    return predecessor_ids
