from jk_utils.backend import ema as ema_ops
from jk_utils.backend import segment_ops
from keras import ops

from ..backend_tensor import BackendTensor


def grid_ema_interpolate(
    features: BackendTensor,
    times: BackendTensor,
    decay_rate: BackendTensor,
    segment_ids: BackendTensor,
    batch_ids: BackendTensor,
    t_start: BackendTensor,
    t_stop: BackendTensor,
    num_frames: int,
    grid_size: int,
    *,
    indices_are_sorted: bool = False,
    normalize: bool = False,
) -> BackendTensor:
    """
    EMA interpolation at time steps uniformly distributed between t_start and t_stop.

    We skip the first time step, i.e. the first time value is `t_start + dt` and the
    last is `t_stop`, where `dt = (t_stop - t_step) / num_frames`.

    For dimensions, E = sum(e_b), where `e_s` is the number of events in segment `s`.

    Args:
        features: [E, C]
        times: [E]
        decay_rate: [C]
        segment_ids: [E] in [0, S] (S corresponds to padded examples)
        batch_ids: [E] in [0, B] (B corresponds to padded examples)
        t_start: [B]
        t_stop: [B]
        num_frames: number of divisions per segment
        indices_are_sorted: True if segment_ids is sorted and times are sorted
            within each segment.

    Returns:
        [B, grid_size, num_frames, C]
    """
    assert normalize
    (num_channels,) = decay_rate.shape
    assert len(features.shape) == 2, features.shape
    assert features.shape[1] == num_channels, (features.shape, num_channels)
    (batch_size,) = t_start.shape
    assert t_stop.shape == (batch_size,), (t_stop.shape, t_start.shape)

    times = times - ops.take(ops.pad(t_start, [[0, 1]]), batch_ids)  # [E]
    t_step_batch = (t_stop - t_start) / num_frames  # [B]
    t_step_event = ops.take(
        ops.pad(t_step_batch, [[0, 1]], constant_values=1.0), batch_ids
    )  # [E]
    normalized_times = times / t_step_event  # [E]
    time_indices = ops.cast(normalized_times, "int64")
    dt = (
        1 + ops.cast(time_indices, normalized_times.dtype) - normalized_times
    ) * t_step_event
    factors = ops.exp(-decay_rate * ops.expand_dims(dt, axis=-1))  # [E, C]
    features = segment_ops.segment_sum(
        factors * features,
        segment_ids=segment_ids * num_frames + time_indices,
        num_segments=batch_size * grid_size * num_frames,
        indices_are_sorted=indices_are_sorted,
    )  # [batch_size * grid_size * num_frames, C]
    features = ops.reshape(
        features, (batch_size * grid_size, num_frames, num_channels)
    )  # [S, num_frames, C]
    features = ops.transpose(features, (1, 0, 2))  # [num_frames, S, C]
    if normalize:
        norm_factor = segment_ops.segment_sum(
            factors,
            segment_ids=segment_ids * num_frames + time_indices,
            num_segments=batch_size * grid_size * num_frames,
            indices_are_sorted=indices_are_sorted,
        )
        norm_factor = ops.reshape(
            norm_factor, (batch_size * grid_size, num_frames, num_channels)
        )
        norm_factor = ops.transpose(norm_factor, (1, 0, 2))
    factors = ops.exp(-ops.expand_dims(t_step_batch, -1) * decay_rate)  # [B, C]
    factors = ops.expand_dims(ops.tile(factors, [grid_size, 1]), axis=0)  # [1, S, C]
    factors = ops.broadcast_to(factors, features.shape)
    features = ema_ops.ema(features, factors, axis=0)  # [num_frames, S, C]
    if normalize:
        norm_factor = ema_ops.ema(norm_factor, factors, axis=0)
        norm_factor = ops.maximum(norm_factor, 1e-3)
        features = features / norm_factor
    features = ops.transpose(features, (1, 0, 2))  # [S, num_frames, C]
    features = ops.reshape(features, (batch_size, grid_size, num_frames, num_channels))
    return features


def grid_final_interpolate(
    features: BackendTensor,
    times: BackendTensor,
    segment_ids: BackendTensor,
    batch_ids: BackendTensor,
    t_start: BackendTensor,
    t_stop: BackendTensor,
    num_frames: int,
    grid_size: int,
) -> BackendTensor:
    """
    Final interpolation at time steps uniformly distributed between t_start and t_stop.

    We skip the first time step, i.e. the first time value is `t_start + dt` and the
    last is `t_stop`, where `dt = (t_stop - t_step) / num_frames`.

    For dimensions, E = sum(e_b), where `e_s` is the number of events in segment `s`.

    Args:
        features: [E, C]
        times: [E]
        segment_ids: [E] in [0, S] (S corresponds to padded examples). Must be sorted
            by (segment_id, time).
        batch_ids: [E] in [0, B] (B corresponds to padded examples)
        t_start: [B]
        t_stop: [B]
        num_frames: number of divisions per segment
        grid_size: number of segments in each batch element.

    Returns:
        [B, grid_size, num_frames, C] features_out, features.dtype
        [B, grid_size, num_frames] valid_mask, bool
    """
    assert len(features.shape) == 2, features.shape
    features.shape[1]
    (batch_size,) = t_start.shape
    assert t_stop.shape == (batch_size,), (t_stop.shape, t_start.shape)

    times = times - ops.take(ops.pad(t_start, [[0, 1]]), batch_ids)  # [E]
    t_step_batch = (t_stop - t_start) / num_frames  # [B]
    t_step_event = ops.take(
        ops.pad(t_step_batch, [[0, 1]], constant_values=1.0), batch_ids
    )  # [E]
    normalized_times = times / t_step_event  # [E]
    time_indices = ops.cast(normalized_times, "int64")
    ids = segment_ops.segment_max(
        ops.arange(ops.shape(times)[0], dtype="int64"),
        segment_ids=segment_ids * num_frames + time_indices,
        num_segments=batch_size * grid_size * num_frames,
        indices_are_sorted=True,
    )  # [batch_size * grid_size * num_frames]
    ids = ops.reshape(ids, (batch_size, grid_size, num_frames))
    ids = ema_ops.cumulative_max(ids, axis=2)
    features_out = ops.take(features, ids, axis=0)
    mask = ids >= 0
    features_out = ops.where(
        ops.expand_dims(mask, axis=-1), features_out, ops.zeros_like(features_out)
    )
    return features_out, mask


def mean_previous_interpolate(
    features: BackendTensor,
    times: BackendTensor,
    batch_ids: BackendTensor,
    t_start: BackendTensor,
    t_stop: BackendTensor,
    num_frames: int,
    *,
    indices_are_sorted: bool = False,
) -> BackendTensor:
    """
    Running average of features interpolated uniformly between t_start and t_end.

    We skip the first time step, i.e. the first time value is `t_start + dt` and the
    last is `t_stop`, where `dt = (t_stop - t_step) / num_frames`.

    For dimensions, E = sum(e_b), where `e_s` is the number of events in segment `s`.

    Args:
        features: [E, C]
        times: [E]
        batch_ids: [E] in [0, B] (B corresponds to padded examples)
        t_start: [B]
        t_stop: [B]
        num_frames: number of divisions per segment
        indices_are_sorted: True if times are sorted within each batch.

    Returns:
        [B, num_frames, C]
    """
    (batch_size,) = t_start.shape
    assert t_stop.shape == (batch_size,), (t_start.shape, t_stop.shape)
    assert len(features.shape) == 2, features.shape
    C = features.shape[1]
    times = times - ops.take(ops.pad(t_start, [[0, 1]]), batch_ids)  # [E]
    t_step_batch = (t_stop - t_start) / num_frames  # [B]
    t_step_event = ops.take(
        ops.pad(t_step_batch, [[0, 1]], constant_values=1.0), batch_ids
    )  # [E]
    normalized_times = times / t_step_event  # [E]
    time_indices = ops.cast(normalized_times, "int64")

    segment_ids = batch_ids * num_frames + time_indices
    num_segments = batch_size * num_frames
    features = segment_ops.segment_sum(
        features,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )  # [batch_size * num_frames, C]
    features = ops.reshape(features, (batch_size, num_frames, C))
    features = ops.cumsum(features, axis=1)  # [B, num_frames, C]
    counts = segment_ops.segment_sum(
        ops.ones_like(times, dtype=features.dtype),
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )  # [batch_size * num_frames]
    counts = ops.reshape(counts, (batch_size, num_frames))
    counts = ops.cumsum(counts, axis=1)
    features = features / ops.maximum(ops.expand_dims(counts, axis=-1), 1)
    return features


def reduce_mean_final(
    features: BackendTensor,
    segment_ids: BackendTensor,
    grid_size: int,
    batch_size: int,
) -> BackendTensor:
    """
    Find the final feature on each segment and average over non-zero batch segments.

    Args:
        features: [E, C]. Must be sorted by (segment_id, time).
        segment_ids: [E] in [0, S] (S corresponds to padded examples).
        grid_size: number of pixels in each batch element.
        batch_size: batch size B.

    Returns:
        [B, C]
    """
    ids = segment_ops.segment_max(
        ops.arange(ops.shape(features)[0], dtype="int32"),
        segment_ids=segment_ids,
        num_segments=batch_size * grid_size,
        indices_are_sorted=True,
    )  # [B * grid_size]
    ids = ops.reshape(ids, (batch_size, grid_size))
    valid = ops.cast(ids >= 0, features.dtype)
    features = ops.take(features, ids, axis=0)  # [B, grid_size, C]
    features = ops.where(
        ops.expand_dims(valid, axis=-1), features, ops.zeros_like(features)
    )
    num_valid = ops.count_nonzero(valid, axis=1)
    features = ops.sum(features, axis=1) / ops.expand_dims(
        ops.maximum(ops.cast(num_valid, features.dtype), 1e-3), -1
    )
    return features
