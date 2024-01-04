import numba as nb
import numpy as np


@nb.njit()
def _counting_argsort_in_place(args):
    out, segment_ids, splits = args
    num_segments = splits.shape[0] - 1
    counts = np.zeros((num_segments,), dtype=np.int32)
    for i, segment_id in enumerate(segment_ids[: splits[-1]]):
        out[splits[segment_id] + counts[segment_id]] = i
        counts[segment_id] += 1


@nb.njit()
def counting_argsort(segment_ids: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """
    O(n) stable sort of segment ids.

    Faster than np.argsort.

    Args:
        segment_ids: [E] values in [0, S), presumably not sorted.
        splits: [S+1] values in [0, valid_size]. `valid_size <= E`.

    Returns:
        [E] indices, `indices[:valid_size]` sort `segment_ids[:valid_size]`.
    """
    (num_events,) = segment_ids.shape
    out = np.empty((num_events,), dtype=np.int32)
    _counting_argsort_in_place((out, segment_ids, splits))
    out[splits[-1] :] = np.arange(splits[-1], out.shape[0], dtype=np.int32)
    return out
