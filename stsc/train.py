import tempfile
import typing as tp

import keras
import tensorflow as tf
from absl import logging

from .data import batching, transforms_tf
from .models import wrappers


def map_and_pack(
    example,
    grid_shape: tp.Tuple[int, int],
    time_scale: float | None = None,
    transform: transforms_tf.Transform | None = None,
    rezero_times: bool = True,
    expected_grid_shape: tp.Tuple[int, int] | None = None,
):
    """
    Unpack events in standard raw format, applies transform and repacks for model.

    Standard raw format is
    {"events": {"time": x, "coords": x, "polarity": x}, "label": x}

    Args:
        example: structure of tensors in standard raw format
        grid_shape: (height, width) of event stream (limits on coords)
        time_scale: if given, times are divided by this value before transform
        transform: `stsc.data.transforms_tf.Transform` to apply, e.g. for data
            augmentation
        rezero_times: if given, time origin is reset, i.e. times <- times - times[0]

    Returns:
        (times, coords, polarity), label
    """
    events = example["events"]

    times = events["time"]
    coords = events["coords"]
    polarity = events["polarity"]
    label = example["label"]

    coords = tf.cast(coords, "int32")

    times = tf.cast(times, tf.float32)
    if time_scale is not None:
        times = times / time_scale
    stream = transforms_tf.StreamData(coords, times, polarity, grid_shape)
    if transform is not None:
        stream, label = transform.transform_stream_example(stream, label)
        stream = transforms_tf.mask_valid_events(stream)
    if expected_grid_shape is not None:
        assert stream.grid_shape == expected_grid_shape, (
            stream.grid_shape,
            expected_grid_shape,
        )
    times = stream.times
    coords = stream.coords
    polarity = stream.polarity
    if rezero_times:
        times = times - times[0]
    return (times, coords, polarity), label


class LoggerCallback(keras.callbacks.Callback):
    def __init__(
        self, *, format_fn=lambda k, v: f"{k}: {v:.4f}", print_fn: tp.Callable = print
    ):
        self.print_fn = print_fn
        self.format_fn = format_fn

    def on_epoch_end(self, epoch, logs=None):
        logs_str = ", ".join(self.format_fn(k, logs[k]) for k in sorted(logs))
        self.print_fn(f"Epoch {epoch}: {logs_str}")


def build_and_fit(
    grid_shape: tp.Tuple[int, int],
    batch_size: int,
    num_frames: int,
    num_classes: int,
    events_per_example: int,
    epochs: int,
    backbone_func: tp.Callable,
    optimizer: keras.optimizers.Optimizer,
    loss: keras.losses.Loss,
    weighted_metrics: tp.Sequence[keras.metrics.Metric],
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset | None = None,
    test_data: tf.data.Dataset | None = None,
    examples_per_epoch: int | None = None,
    stream_filter: tp.Callable = lambda streams: streams[1:],
    dropout_rate: float = 0.0,
    normalize_heads: bool = False,
    callbacks: tp.Sequence[keras.callbacks.Callback] = [],
    use_example_loss: bool = False,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    verbose: bool = True,
    temporal_split: bool = False,
    cache_validation_data: bool = False,
):
    """
    Build and fit a model to data.

    Args:
        grid_shape: (height, width) of spatial shape (limits on (y, x) coords).
        batch_size:
        num_classes:
        events_per_example: approximate number of events per example. We use
            `events_per_batch = `events_per_example * batch_size - 1`. If the number
            of events in a given batch exceeds this during batching, trailing examples
            are discared until this is true.
        epochs: number of epochs to train for.
        backbone_func: function mapping a `stsc.components.StreamNode` to a sequence of
            output `stsc.components.StreamNode`s.
        optimizer: see `keras.Model.compile`.
        weighted_metrics: see `keras.Model.compile`.
        train_data: unbatched training dataset.
        validation_data: unbatched validation dataset.
        test_data: unbatched test dataset.
        examples_per_epoch: length of one epoch. If not given, `train_data` must have
            a known finite cardinality.
        stream_filter: function applied to the output of `backbone_func` to e.g. filter
            streams such that losses/metrics are only applied to a subset.
        dropout_rate: applied to features at each event stream prior to final
            classification layer.
        normalize_heads: if True, applies LayerNormalization to each head prior to
            dropout / final classification layer.
        callbacks: see `keras.Model.fit`
        use_example_loss: if True, the loss is calculated as an average of all stream
            inferences. If False, loss is calculated as the average loss over all
            stream inferences.
        num_parallel_calls: used in `tf.data.Dataset.map` calls.
        verbose: see `keras.Model.fit`.
        temporal_split: if True, each example is split into two at a random point during
            training and the batch size is doubled. During validation/testing examples
            are not split, but batches are padded to create the same doubled batch size,
            albeit with only the original `batch_size` examples having non-zero
            `sample_weight`.

    Returns:
        (model, history) or (model, history, test_result) if `test_data` is given.
    """
    if examples_per_epoch is None:
        examples_per_epoch = int(train_data.cardinality().numpy())
        if examples_per_epoch in (
            tf.data.UNKNOWN_CARDINALITY,
            tf.data.INFINITE_CARDINALITY,
        ):
            raise ValueError(
                "examples_per_epoch must be given if train_data has unknown or "
                "infinite cardinality"
            )

    max_events = batch_size * events_per_example - 1  # -1 so padding makes power of 2

    preprocessor_func, model = wrappers.per_event_model(
        num_frames=num_frames,
        loss=loss,
        weighted_metrics=weighted_metrics,
        optimizer=optimizer,
        dropout_rate=dropout_rate,
        normalize_heads=normalize_heads,
        use_example_loss=use_example_loss,
        max_events=max_events,
        batch_size=batch_size * 2 if temporal_split else batch_size,
        backbone_func=backbone_func,
        grid_shape=grid_shape,
        num_classes=num_classes,
        stream_filter=stream_filter,
    )

    def preprocess_dataset(
        dataset: tf.data.Dataset | None,
        num_parallel_calls: int,
        dummy_temporal_split: bool,
    ):
        if dataset is None:
            return None
        dataset = batching.batch_and_pad(
            dataset,
            batch_size=batch_size,
            max_events=max_events,
            drop_remainder=True,
            map_fun=preprocessor_func,
            num_parallel_calls=num_parallel_calls,
            temporal_split=temporal_split,
            dummy_temporal_split=dummy_temporal_split,
        )
        return dataset

    train_data = preprocess_dataset(
        train_data, num_parallel_calls, dummy_temporal_split=False
    )
    validation_data = preprocess_dataset(
        validation_data,
        1 if cache_validation_data else num_parallel_calls,
        dummy_temporal_split=True,
    )
    test_data = preprocess_dataset(test_data, 1, dummy_temporal_split=True)

    steps_per_epoch = examples_per_epoch // batch_size + int(
        bool(examples_per_epoch % batch_size)
    )
    model.summary()

    def run(validation_data):
        return model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
        )

    if cache_validation_data and validation_data is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logging.info("Caching validation data")
            validation_data.save(tmpdir)
            validation_data = tf.data.Dataset.load(tmpdir)
            history = run(validation_data)
    else:
        history = run(validation_data)

    if test_data is not None:
        test_result = model.evaluate(test_data, return_dict=True)
        return model, history, test_result
    return model, history


def get_dataset_info(dataset: tf.data.Dataset):
    num_examples = len(dataset)

    num_events = tf.zeros((), dtype=tf.int64)
    total_dt = tf.zeros((), dtype=tf.int64)

    print("Computing dataset info...")
    num_events, total_dt = dataset.reduce(
        (num_events, total_dt),
        lambda curr, t: (curr[0] + tf.shape(t, tf.int64)[0], curr[1] + t[-1] - t[0]),
    )
    mean_num_events = num_events / num_examples
    mean_dt = total_dt / num_examples
    print(f"mean_num_events: {mean_num_events}")
    print(f"mean_dt:         {mean_dt}")
    print(f"num_examples:    {num_examples}")

    return mean_num_events, mean_dt, num_examples
