import typing as tp

import keras
import tree
from jk_utils.layers.identity import Identity
from jk_utils.ops import segment_ops
from keras import ops

from ..components import StreamNode, input_stream


def get_inputs(max_events: int, batch_size: int | None, grid_shape: tp.Iterable[int]):
    grid_shape = tuple(grid_shape)
    polarity = keras.Input(batch_shape=(max_events,), dtype="bool")
    coords = keras.Input(batch_shape=(max_events, len(grid_shape)), dtype="int32")
    times = keras.Input(batch_shape=(max_events,), dtype="float32")
    batch_splits = keras.Input(
        batch_shape=(None if batch_size is None else batch_size + 1,), dtype="int32"
    )
    inputs = times, coords, polarity, batch_splits
    stream = input_stream(polarity, coords, times, batch_splits, grid_shape)
    return inputs, stream


def split(
    preprocessor_inputs, model_outputs, preprocessor_outputs=None
) -> tp.Tuple[keras.Model, keras.Model]:
    if preprocessor_outputs is None:
        preprocessor_outputs = []
    else:
        preprocessor_outputs = list(preprocessor_outputs)
        assert all(isinstance(x, keras.KerasTensor) for x in preprocessor_outputs)

    assert all(
        isinstance(x, keras.KerasTensor) for x in tree.flatten(preprocessor_inputs)
    )
    assert all(isinstance(x, keras.KerasTensor) for x in tree.flatten(model_outputs))

    function = keras.Function(
        preprocessor_inputs,
        preprocessor_outputs + tree.flatten(model_outputs),
    )
    model_inputs = set()
    model_tensors = set()
    model_outputs_set = set(tree.flatten(model_outputs))

    nodes_by_depth = function._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    for depth in depth_keys:
        nodes = nodes_by_depth[depth]
        for node in nodes:
            if not node.operation or node.is_input:
                continue  # Input tensors already exist.
            inputs = node.arguments.keras_tensors
            outputs = node.outputs
            op = node.operation
            is_model_layer = (
                isinstance(op, keras.layers.Layer) and len(op.trainable_weights) > 0
            ) or any(o in model_outputs_set for o in outputs)
            has_model_inputs = any(i in model_tensors for i in inputs)

            if is_model_layer or has_model_inputs:
                model_tensors.update(outputs)
                model_inputs.update(i for i in inputs if i not in model_tensors)

    assert all(m in model_tensors for m in tree.flatten(model_outputs))
    assert all(po not in model_tensors for po in preprocessor_outputs)
    model_inputs = sorted(model_inputs, key=lambda k: k.name)
    preprocessor = keras.Model(
        preprocessor_inputs, tuple(model_inputs + preprocessor_outputs)
    )
    assert len(preprocessor.trainable_weights) == 0
    model = keras.Model(tuple(model_inputs), model_outputs)

    return preprocessor, model


def clone_metric(metric, name):
    serialized = keras.metrics.serialize(metric)
    # serialized["config"]["name"] = f"{name}_{serialized['config']['name']}"
    return keras.metrics.deserialize(serialized)


def clone_metrics(metrics, name):
    return tree.map_structure(lambda m: clone_metric(m, name), metrics)


def per_event_model(
    num_classes: int,
    max_events: int,
    backbone_func: tp.Callable[
        [StreamNode],
        tp.Sequence[StreamNode],
    ],
    grid_shape: tp.Iterable[int],
    *,
    batch_size: int | None = None,
    loss=None,
    metrics=None,
    weighted_metrics=None,
    dropout_rate: float = 0.0,
    reduction="mean",
    # normalize: bool = True,
    stream_filter=lambda streams: streams[-1:],
    use_example_loss: bool = False,
    **compile_kwargs,
) -> tp.Tuple[keras.Model, keras.Model]:
    inputs, stream = get_inputs(max_events, batch_size, grid_shape)
    streams = backbone_func(stream)

    labels = keras.Input((), dtype="int32", batch_size=batch_size)
    example_sample_weight = keras.Input((), dtype="float32", batch_size=batch_size)

    def postprocess_stream(stream, example_sample_weight):
        stream = stream.map_features(keras.layers.Dropout(dropout_rate))
        stream = stream.map_features(keras.layers.Dense(num_classes))
        if stream.order is None:
            stream.force_chronological()
        batch_ids = stream.stream.batch_ids
        batch_lengths = stream.stream.batch_lengths
        event_features = stream.compute_features()
        batch_size = batch_lengths.shape[0]
        if reduction == "mean":
            example_features = segment_ops.segment_sum(
                event_features,
                batch_ids,
                num_segments=batch_size,
                indices_are_sorted=True,
            ) / (
                ops.expand_dims(ops.cast(batch_lengths, event_features.dtype), axis=-1)
                + keras.backend.epsilon()
            )
        elif reduction == "sum":
            example_features = segment_ops.segment_sum(
                event_features,
                batch_ids,
                num_segments=batch_size,
                indices_are_sorted=True,
            )
        elif reduction == "max":
            example_features = segment_ops.segment_max(
                event_features,
                batch_ids,
                num_segments=batch_size,
                indices_are_sorted=True,
            )
        else:
            raise NotImplementedError(f"reduction {reduction} not supported")

        valid_batch_count = ops.count_nonzero(batch_lengths)
        event_sample_weight = ops.take(
            ops.pad(
                ops.where(
                    batch_lengths == 0,
                    ops.zeros_like(example_sample_weight),
                    example_sample_weight
                    / (
                        ops.cast(
                            valid_batch_count * batch_lengths,
                            example_sample_weight.dtype,
                        )
                        + keras.backend.epsilon()
                    ),
                ),
                [[0, 1]],
            ),
            batch_ids,
        )
        return example_features, event_features, batch_ids, event_sample_weight

    streams = stream_filter(streams)
    if len(streams) == 1:
        (
            example_features,
            event_features,
            batch_ids,
            event_sample_weight,
        ) = postprocess_stream(streams[0], example_sample_weight)
    else:
        example_features, event_features, batch_ids, event_sample_weight = zip(
            *(postprocess_stream(stream, example_sample_weight) for stream in streams)
        )
        example_features = ops.mean(ops.stack(example_features, axis=0), axis=0)
        event_features = ops.concatenate(event_features, axis=0)
        batch_ids = ops.concatenate(batch_ids, axis=0)
        event_sample_weight = ops.concatenate(event_sample_weight, axis=0)
        event_sample_weight = event_sample_weight / len(streams)

    # rename
    event_features = Identity(name="event")(event_features)
    example_features = Identity(name="example")(example_features)

    preprocessor, model = split(
        preprocessor_inputs=inputs,
        model_outputs=(event_features, example_features),
        preprocessor_outputs=(batch_ids,),
    )

    # map_func model
    *model_inputs, batch_ids = preprocessor.outputs
    labels_broadcast = ops.take(ops.pad(labels, [[0, 1]]), batch_ids)

    event_sample_weight = event_sample_weight * ops.cast(
        ops.shape(event_sample_weight)[0], event_sample_weight.dtype
    )

    per_event_preprocessor = keras.Model(
        (*preprocessor.inputs, labels, example_sample_weight),
        (
            *model_inputs,
            labels_broadcast,
            labels,
            event_sample_weight,
            example_sample_weight,
        ),
    )

    # apply loss to unpooled model outputs / broadcast labels, sample_weight
    if loss is not None:
        if use_example_loss:
            compile_kwargs["loss"] = [None, loss]
        else:
            compile_kwargs["loss"] = [loss, None]
    # apply metrics to pooled model outputs / original labels, sample_weight
    if metrics is not None:
        compile_kwargs["metrics"] = [
            clone_metrics(metrics, "event"),
            clone_metrics(metrics, "example"),
        ]
    if weighted_metrics is not None:
        compile_kwargs["weighted_metrics"] = [
            clone_metrics(weighted_metrics, "event"),
            clone_metrics(weighted_metrics, "example"),
        ]

    # https://github.com/keras-team/keras/issues/18647
    # cloning trims the model of input inputs, making summaries cleaner
    model = keras.models.clone_model(
        model,
        tree.map_structure(
            lambda x: keras.Input(batch_shape=x.shape, dtype=x.dtype), model.input
        ),
        clone_function=lambda op: op,
    )

    model.compile(**compile_kwargs)
    return per_event_preprocessor, model
