import typing as tp

import keras
import tensorflow as tf
import tree
from jk_neuro.data import transforms_tf

from .data import batching
from .models import wrappers


def map_and_pack(
    example,
    grid_shape: tp.Tuple(int, int),
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
        transform: jk_neuro.data.transforms_tf.Transform to apply, e.g. for data
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


def _preprocessor_to_func(preprocessor):
    backend = keras.backend.backend()
    if backend == "jax":
        import jax

        jax_preprocessor = jax.jit(lambda *args: preprocessor(args), backend="cpu")

        def preprocessor_func(inputs, labels, sample_weight):
            preprocessor_inputs = tuple(tree.flatten((inputs, labels, sample_weight)))

            output = tf.numpy_function(
                jax_preprocessor,
                preprocessor_inputs,
                Tout=tuple(x.dtype for x in preprocessor.output),
                stateful=False,
            )
            output = tuple(output)
            tree.map_structure(
                lambda o, t: o.set_shape(t.shape), output, preprocessor.output
            )
            (
                *model_inputs,
                labels_broadcast,
                labels,
                sample_weight_broadcast,
                sample_weight,
            ) = output
            return (
                tuple(model_inputs),
                (labels_broadcast, labels),
                (sample_weight_broadcast, sample_weight),
            )

    else:
        assert backend == "tensorflow", backend

        @tf.function
        def preprocessor_func(inputs, labels, sample_weight):
            (
                *model_inputs,
                labels_broadcast,
                labels,
                sample_weight_broadcast,
                sample_weight,
            ) = preprocessor(tree.flatten((inputs, labels, sample_weight)))
            return (
                tuple(model_inputs),
                (labels_broadcast, labels),
                (sample_weight_broadcast, sample_weight),
            )

        return preprocessor_func


def build_and_fit(
    grid_shape: tp.Tuple[int, int],
    batch_size: int,
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
    callbacks: tp.Sequence[keras.callbacks.Callback] = [],
    use_example_loss: bool = False,
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
        callbacks: see `keras.Model.fit`
        use_example_loss: if True, the loss is calculated as an average of all stream
            inferences. If False, loss is calculated as the average loss over all
            stream inferences.

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

    preprocessor, model = wrappers.per_event_model(
        loss=loss,
        weighted_metrics=weighted_metrics,
        optimizer=optimizer,
        dropout_rate=dropout_rate,
        stream_filter=lambda streams: streams[1:],
        use_example_loss=use_example_loss,
        max_events=max_events,
        batch_size=batch_size,
        backbone_func=backbone_func,
        grid_shape=grid_shape,
        num_classes=num_classes,
        stream_filter=stream_filter,
    )
    preprocessor_func = _preprocessor_to_func(preprocessor)

    def preprocess_dataset(dataset: tf.data.Dataset | None):
        if dataset is None:
            return None
        dataset = batching.batch_and_pad(
            dataset,
            batch_size=batch_size,
            max_events=max_events,
            drop_remainder=True,
            map_fun=preprocessor_func,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset

    train_data = preprocess_dataset(train_data)
    validation_data = preprocess_dataset(validation_data)
    test_data = preprocess_dataset(test_data)

    steps_per_epoch = examples_per_epoch // batch_size + int(
        bool(examples_per_epoch % batch_size)
    )
    model.summary()
    history = model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        callbacks=callbacks,
    )
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
