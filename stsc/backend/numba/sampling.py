import typing as tp

import numba as nb
import numpy as np


@nb.njit()
def _throttled_sample_in_place(
    args, sample_rate: int, min_dt: tp.Union[int, float], grid_size: int
):
    sample_indices, batch_splits_out, pixel_ids, times, batch_splits = args
    batch_size = batch_splits.shape[0] - 1
    counts = np.empty((grid_size,), dtype=np.uint32)
    earliest_possible = np.empty((grid_size,), dtype=times.dtype)
    # counts = np.zeros((batch_size, grid_size), dtype=np.uint32)
    # earliest_possible = np.full((batch_size, grid_size), -np.inf, dtype=times.dtype)
    size_out = 0
    batch_splits_out[0] = 0
    for b in range(batch_size):
        start, end = batch_splits[b : b + 2]
        counts[:] = 0
        earliest_possible[:] = -np.inf
        for e in range(start, end):
            s = pixel_ids[e]
            t = times[e]
            counts[s] += 1
            if counts[s] >= sample_rate and t >= earliest_possible[s]:
                counts[s] = 0
                earliest_possible[s] = t + min_dt
                sample_indices[size_out] = e
                size_out += 1
        batch_splits_out[b + 1] = size_out
    sample_indices[size_out:] = pixel_ids.shape[0]


@nb.njit()
def throttled_sample(
    pixel_ids: np.ndarray,
    times: np.ndarray,
    batch_splits: np.ndarray,
    sample_rate: int,
    min_dt: tp.Union[int, float],
    grid_size: int,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    (E,) = pixel_ids.shape
    assert times.shape == (E,)
    assert len(batch_splits.shape) == 1
    batch_size = batch_splits.shape[0] - 1
    sample_indices = np.empty((pixel_ids.shape[0] // sample_rate), dtype=np.int32)
    batch_splits_out = np.empty((batch_size + 1,), dtype=batch_splits.dtype)
    _throttled_sample_in_place(
        (sample_indices, batch_splits_out, pixel_ids, times, batch_splits),
        sample_rate=sample_rate,
        min_dt=min_dt,
        grid_size=grid_size,
    )
    return sample_indices, batch_splits_out
