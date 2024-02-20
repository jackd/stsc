import typing as tp

import tensorflow as tf


def pad_to_length(
    x: tf.Tensor, target_length: int, axis: int = 0, **kwargs
) -> tf.Tensor:
    curr_length = tf.shape(x)[axis]
    padding = target_length - curr_length
    padding_arr = [[0, 0] for _ in x.shape]
    padding_arr[axis][1] = padding
    x = tf.pad(x, padding_arr, **kwargs)
    shape = list(x.shape)
    shape[axis] = target_length
    x.set_shape(shape)
    return x


def pad_features_to_max_events(
    times: tf.Tensor,
    coords: tf.Tensor,
    features: tf.Tensor,
    splits: tf.Tensor,
    max_events: int,
) -> tp.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    num_events = splits[-1]

    def if_longer():
        return (coords[:max_events], times[:max_events], features[:max_events])

    def if_shorter():
        return (
            pad_to_length(coords, max_events),
            pad_to_length(times, max_events),
            pad_to_length(features, max_events),
        )

    coords, times, features = tf.cond(num_events >= max_events, if_longer, if_shorter)
    coords.set_shape((max_events, coords.shape[1]))
    times.set_shape((max_events,))
    features.set_shape((max_events, *features.shape[1:]))

    splits = tf.minimum(splits, max_events)
    return times, coords, features, splits

    # row_ends = splits[1:]
    # batch_mask = row_ends <= max_events
    # num_valid_events = tf.reduce_max(
    #     tf.where(batch_mask, row_ends, tf.zeros_like(row_ends))
    # )
    # event_mask = tf.range(max_events) < num_valid_events
    # times = tf.where(event_mask, times, tf.zeros_like(times))
    # coords = tf.where(tf.expand_dims(event_mask, 1), coords, tf.zeros_like(coords))
    # features = tf.where(
    #     tf.reshape(event_mask, (-1,) + (1,) * (len(features.shape) - 1)),
    #     features,
    #     tf.zeros_like(features),
    # )
    # splits = tf.minimum(splits, num_valid_events)

    # return times, coords, features, splits, batch_mask


def batch_and_pad(
    dataset: tf.data.Dataset,
    batch_size: int,
    max_events: int,
    drop_remainder: bool = False,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    deterministic: bool = False,
    map_fun: tp.Callable | None = None,
    temporal_split: bool = False,
    dummy_temporal_split: bool = True,
) -> tf.data.Dataset:
    def full_map_fun(features, labels):
        times, coords, polarities = features
        splits = times.row_splits
        times = times.flat_values
        coords = coords.flat_values
        polarities = polarities.flat_values

        # times, coords, polarities, splits, batch_mask = pad_features_to_max_events(
        #     times, coords, polarities, splits, max_events
        # )
        times, coords, polarities, splits = pad_features_to_max_events(
            times, coords, polarities, splits, max_events
        )
        lengths = tf.experimental.numpy.diff(splits)

        # if splits.shape[0] is None:
        #     labels = pad_to_length(labels, batch_size)
        #     batch_mask = pad_to_length(batch_mask, batch_size)
        #     splits = pad_to_length(splits, batch_size + 1, constant_values=splits[-1])

        if temporal_split:
            labels = tf.reshape(tf.tile(tf.expand_dims(labels, 1), (1, 2)), (-1,))
            if dummy_temporal_split:
                lengths = tf.stack((lengths, tf.zeros_like(lengths)), axis=-1)
            else:
                example_lengths = tf.cast(
                    tf.random.uniform((batch_size,)) * tf.cast(lengths, tf.float32),
                    lengths.dtype,
                )
                lengths = tf.stack(
                    (example_lengths, lengths - example_lengths), axis=-1
                )
            lengths = tf.reshape(lengths, (-1,))
            splits = tf.pad(tf.cumsum(lengths, axis=0), [[1, 0]])

        batch_mask = lengths > 0
        sample_weight = tf.cast(batch_mask, tf.float32)

        inputs = (times, coords, polarities, splits)
        if map_fun is not None:
            return map_fun(inputs, labels, sample_weight)

        return inputs, labels, sample_weight

    dataset = dataset.ragged_batch(
        batch_size, drop_remainder=drop_remainder, row_splits_dtype=tf.int32
    )
    dataset = dataset.map(
        full_map_fun, num_parallel_calls=num_parallel_calls, deterministic=deterministic
    )
    return dataset
