import typing as tp

from jk_utils.backend import complex as complex_ops
from jk_utils.backend import ema as ema_ops
from jk_utils.backend import segment_ops
from keras import ops

from ..backend_tensor import BackendTensor


def _assert_is_complex(x):
    assert isinstance(x, (list, tuple))
    assert len(x) == 2, len(x)
    assert x[0].shape == x[1].shape
    assert x[0].dtype == x[1].dtype


def get_patches(
    features: BackendTensor,
    times_in: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    segment_ids: BackendTensor,
    predecessor_ids: BackendTensor,
    *,
    normalize: bool = True,
) -> BackendTensor:
    is_complex = isinstance(decay_rate, (list, tuple))
    if is_complex:
        _assert_is_complex(decay_rate)
        features = complex_ops.complex(*ops.split(features, 2, axis=-1))
        decay_rate_real, decay_rate_imag = decay_rate

        def scale_dt(dt):
            return complex_ops.complex(decay_rate_real * dt, decay_rate_imag * dt)

    else:

        def scale_dt(dt):
            return decay_rate * dt

    dt = -ops.diff(times_in, axis=0)
    same_segment = segment_ids[:-1] == segment_ids[1:]
    dt = ops.where(same_segment, dt, ops.zeros_like(dt))
    # tf.debugging.assert_non_positive(dt, "dt must be non_positive")  # DEBUG
    dt = ops.expand_dims(dt, axis=-1)
    factors = complex_ops.exp(scale_dt(dt))
    factors = ops.where(
        ops.expand_dims(same_segment, 1),
        factors,
        ops.zeros_like(factors),
    )
    factors = ops.pad(factors, [[1, 0], [0, 0]])  # [E_in, C_in]
    # if normalize:
    #     features = features * (ops.ones_like(factors) - factors)
    x = ema_ops.ema(features, factors, axis=0)  # [E_in, C_in]
    if normalize:
        denom = ema_ops.ema(ops.ones_like(features), factors, axis=0)
        x = ops.where(denom == 0, ops.zeros_like(x), x / denom)
    x = ops.where(ops.isfinite(x), x, ops.zeros_like(x))  # HACK

    # pad before take, so invalid indices take zeros
    x = ops.pad(x, [[0, 1], [0, 0]])

    x = ops.take(x, predecessor_ids, axis=0)  # [E_out, K, C_in]
    if not normalize:
        times_in = ops.pad(times_in, [[0, 1]])
        dt = ops.take(times_in, predecessor_ids, axis=0) - ops.expand_dims(
            times_out, -1
        )
        invalid = predecessor_ids == times_in.shape[0]
        dt = ops.where(invalid, ops.zeros_like(dt), dt)
        # tf.debugging.assert_non_positive(dt, "dt must be non_positive")  # DEBUG
        dt = ops.expand_dims(dt, axis=-1)
        factors = complex_ops.exp(scale_dt(dt))  # [E_out, K, C_in]
        factors = ops.where(
            ops.expand_dims(invalid, -1), ops.zeros_like(factors), factors
        )
        x = x * factors
    if is_complex:
        x = ops.concatenate((ops.real(x), ops.imag(x)), axis=-1)
    return x


def get_one_hot_patches(
    times_in: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    segment_filter_ids: BackendTensor,
    one_hot_predecessor_ids: BackendTensor,
    *,
    final_decay: bool = True,
) -> BackendTensor:
    """
    Args:
        times_in: [E_in]
        times_out: [E_out]
        decay_rate: [P]
        segment_filter_ids: [E_in] in [0, num_segments_in * P)
        one_hot_predecessor_ids: [E_out, K, P] in [0, E_in], E_in is dummy value

    Returns:
        [E_out, K, P]
    """
    is_complex = isinstance(decay_rate, (list, tuple))
    if is_complex:
        _assert_is_complex(decay_rate)
        decay_rate_real, decay_rate_imag = decay_rate
        (filters,) = decay_rate_real.shape

        def scale_take_dt(dt, filter_ids):
            real_features_mask = ops.cast(filter_ids // filters, "bool")
            filter_ids = filter_ids % filters
            real = ops.take(decay_rate_real, filter_ids, axis=0) * dt
            imag = ops.take(decay_rate_imag, filter_ids, axis=0) * dt

            return ops.where(
                real_features_mask,
                complex_ops.complex(real, imag),
                complex_ops.complex(-imag, real),  # multiplied by i
            )

        def scale_dt(dt):
            return complex_ops.complex(dt * decay_rate_real, dt * decay_rate_imag)

        filter_ids = segment_filter_ids % (2 * filters)
    else:
        assert isinstance(decay_rate, BackendTensor), decay_rate
        (filters,) = decay_rate.shape

        def scale_take_dt(dt, filter_ids):
            return ops.take(decay_rate, filter_ids, axis=0) * dt

        def scale_dt(dt):
            return dt * decay_rate

        filter_ids = segment_filter_ids % filters
    dt = ops.pad(-ops.diff(times_in, axis=0), [[1, 0]])  # [E_in]

    factors = complex_ops.exp(scale_take_dt(dt, filter_ids))  # [E_in]
    x = ema_ops.segment_ema(
        ops.ones_like(factors), factors, segment_filter_ids, axis=0
    )  # [E_in]
    x = ops.take(ops.pad(x, [[0, 1]]), one_hot_predecessor_ids)
    if final_decay:
        dt = ops.take(
            ops.pad(times_in, [[0, 1]]), one_hot_predecessor_ids, axis=0
        ) - ops.reshape(
            times_out, (-1, 1, 1)
        )  # [E_out, K, P]
        factors = complex_ops.exp(scale_dt(dt))  # [E_out, K, P]
        x = x * factors
    if is_complex:
        x = ops.concatenate((ops.real(x), ops.imag(x)), axis=-1)
    return x


def get_exclusive_patches(
    features: BackendTensor,
    dt: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    successor_kernel_ids: BackendTensor,
    segment_ids_out: BackendTensor,
    indices_are_sorted: bool,
    kernel_size: int,
    normalize: bool = True,
) -> BackendTensor:
    """
    Args:
        features: [E_in, P]
        dt: [E_in]
        times_out: [E_out]
        decay_rate: [P] or ([P // 2], [P // 2]) complex components
        successor_kernel_ids: [E_in] in [0, E_out * kernel_size]
        segment_ids_out: [E_out]

    Returns:
        [E_out, kernel_size, P]
    """
    is_complex = isinstance(decay_rate, (list, tuple))
    if is_complex:
        _assert_is_complex(decay_rate)
        decay_rate_real, decay_rate_imag = decay_rate
        features = complex_ops.complex(*ops.split(features, 2, axis=-1))

        def scale_dt(dt):
            return complex_ops.complex(decay_rate_real * dt, decay_rate_imag * dt)

    else:
        assert isinstance(decay_rate, BackendTensor), decay_rate
        assert isinstance(features, BackendTensor), features

        def scale_dt(dt):
            return decay_rate * dt

    E_in, P = features.shape
    assert dt.shape == (E_in,), (dt.shape, E_in)
    (E_out,) = times_out.shape
    assert segment_ids_out.shape == (E_out,)

    dt = ops.maximum(dt, ops.zeros_like(dt))  # only invalid events have negative dts
    dt = -ops.expand_dims(dt, -1)
    # tf.debugging.assert_non_positive(dt, "dt must be non_positive")  # DEBUG
    weights = complex_ops.exp(scale_dt(dt))  # [E_in, P]
    x = segment_ops.segment_sum(
        features * weights,
        successor_kernel_ids,
        num_segments=E_out * kernel_size,
        indices_are_sorted=indices_are_sorted,
    )  # [E_out * K, P]
    x = ops.reshape(x, (E_out, kernel_size, P))  # [E_out, K, P]
    dt_out = -ops.diff(times_out)
    same_segment = segment_ids_out[:-1] == segment_ids_out[1:]
    dt_out = ops.where(same_segment, dt_out, ops.zeros_like(dt_out))
    # tf.debugging.assert_non_positive(dt, "dt must be non_positive")  # DEBUG
    factors = complex_ops.exp(scale_dt(ops.expand_dims(dt_out, -1)))
    factors = ops.where(
        ops.expand_dims(same_segment, -1), factors, ops.zeros_like(factors)
    )
    factors = ops.pad(factors, [[1, 0], [0, 0]])  # [E_out, P]
    factors = ops.expand_dims(factors, axis=1)  # [E_out, 1, P]
    segment_ids_out = ops.reshape(segment_ids_out, (-1, 1, 1))  # [E_out, 1, 1]
    x = ema_ops.segment_ema(x, factors, segment_ids_out, axis=0)  # [E_out, K, P]
    if is_complex:
        x = ops.concatenate((ops.real(x), ops.imag(x)), axis=-1)
    if normalize:
        denom = segment_ops.segment_sum(
            weights,
            successor_kernel_ids,
            num_segments=E_out * kernel_size,
            indices_are_sorted=indices_are_sorted,
        )
        denom = ops.reshape(denom, (E_out, kernel_size, P))
        denom = ema_ops.segment_ema(denom, factors, segment_ids_out, axis=0)
        if is_complex:
            denom = ops.concatenate((ops.real(denom), ops.imag(denom)), axis=-1)
        x = x / (denom + 1e-5)
    return x


def get_one_hot_exclusive_patches(
    dt: BackendTensor,
    times_out: BackendTensor,
    decay_rate: BackendTensor | tp.Tuple[BackendTensor, BackendTensor],
    successor_kernel_channel_ids: BackendTensor,
    segment_ids_out: BackendTensor,
    indices_are_sorted: bool,
    kernel_size: int,
) -> BackendTensor:
    """
    Args:
        dt: [E_in]
        times_out: [E_out]
        decay_rate: [P, M] or ([P // 2, M], [P // 2, M]) complex components
        successor_kernel_channel_ids: [E_in] in [0, E_out * kernel_size * P]
        segment_ids_out: [E_out]

    Returns:
        [E_out, kernel_size, P * M]
    """
    is_complex = isinstance(decay_rate, (list, tuple))
    if is_complex:
        _assert_is_complex(decay_rate)
        decay_rate_real, decay_rate_imag = decay_rate
        (P, M) = decay_rate_real.shape

        def scale_take_dt(dt, filter_ids):
            real_features_mask = ops.cast(filter_ids // P, "bool")
            filter_ids = filter_ids % P
            real = ops.take(decay_rate_real, filter_ids, axis=0) * dt
            imag = ops.take(decay_rate_imag, filter_ids, axis=0) * dt

            return ops.where(
                real_features_mask,
                complex_ops.complex(real, imag),
                complex_ops.complex(-imag, real),  # multiplied by i
            )

        def scale_dt(dt):
            return complex_ops.complex(dt * decay_rate_real, dt * decay_rate_imag)

    else:
        assert isinstance(decay_rate, BackendTensor), decay_rate
        (P, M) = decay_rate.shape

        def scale_take_dt(dt, filter_ids):
            return ops.take(decay_rate, filter_ids, axis=0) * dt

        def scale_dt(dt):
            return dt * decay_rate

    (E_in,) = dt.shape
    (E_out,) = times_out.shape
    assert segment_ids_out.shape == (E_out,)
    dt = ops.expand_dims(dt, axis=1)
    filter_ids = successor_kernel_channel_ids % P
    # TODO: is the masking below necessary? Check jax particularly
    # dt = ops.pad(
    #     ops.where(
    #         successor_kernel_channel_ids[:-1] == successor_kernel_channel_ids[1:],
    #         dt[1:],
    #         ops.zeros_like(dt[1:]),
    #     ),
    #     [[1, 0]],
    # )
    dt = -dt
    scaled_dt = scale_take_dt(dt, filter_ids)  # [E_in, M]
    weights = complex_ops.exp(scaled_dt)  # [E_in, M]
    x = segment_ops.segment_sum(
        weights,
        successor_kernel_channel_ids,
        num_segments=E_out * kernel_size * P,
        indices_are_sorted=indices_are_sorted,
    )  # [E_out * K * P, M]
    x = ops.reshape(x, (E_out, kernel_size, P * M))  # [E_out, K, P * M]
    dt_out = ops.diff(times_out)
    dt_out = ops.reshape(dt_out, (-1, 1, 1))
    # TODO: is the masking below necessary? Check jax particularly
    # dt_out = ops.where(
    #     segment_ids_out[:-1] == segment_ids_out[1:], dt_out, ops.zeros_like(dt_out)
    # )
    factors = ops.pad(
        complex_ops.exp(scale_dt(-dt_out)), [[1, 0], [0, 0], [0, 0]]
    )  # [E_out, P, M]
    factors = ops.reshape(factors, (E_out, 1, P * M))  # [E_out, 1, P * M]
    segment_ids_out = ops.reshape(segment_ids_out, (-1, 1, 1))  # [E_out, 1, 1]
    x = ema_ops.segment_ema(x, factors, segment_ids_out, axis=0)  # [E_out, K, P * M]
    if is_complex:
        x = ops.concatenate((ops.real(x), ops.imag(x)), axis=-1)
    return x
