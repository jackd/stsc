import typing as tp

import keras
import tensorflow as tf
import tree
from jk_utils.layers.identity import Identity
from jk_utils.ops import segment_ops
from keras import ops

from ..components import StreamNode, input_stream
from ..metrics.per_frame_metrics import PerFrameAccuracy
from ..ops.grid_interpolate import grid_final_interpolate


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


def clone_metric(metric):
    serialized = keras.metrics.serialize(metric)
    # serialized["config"]["name"] = f"{name}_{serialized['config']['name']}"
    return keras.metrics.deserialize(serialized)


def clone_loss(loss):
    serialized = keras.losses.serialize(loss)
    # HACK
    # for k in ("reduction", "ignore_class"):
    #     if k in serialized["config"]:
    #         del serialized["config"][k]
    return keras.metrics.deserialize(serialized)


def clone_metrics(metrics):
    return tree.map_structure(lambda m: clone_metric(m), metrics)


# def _preprocessor_to_func(preprocessor):
#     backend = keras.backend.backend()
#     if backend == "jax":
#         import jax

#         jax_preprocessor = jax.jit(lambda *args: preprocessor(args), backend="cpu")

#         def preprocessor_func(inputs, labels, sample_weight):
#             preprocessor_inputs = tuple(tree.flatten((inputs, labels, sample_weight)))

#             output = tf.numpy_function(
#                 jax_preprocessor,
#                 preprocessor_inputs,
#                 Tout=tuple(x.dtype for x in preprocessor.output),
#                 stateful=False,
#             )
#             output = tuple(output)
#             tree.map_structure(
#                 lambda o, t: o.set_shape(t.shape), output, preprocessor.output
#             )
#             (
#                 *model_inputs,
#                 labels_broadcast,
#                 labels,
#                 sample_weight_broadcast,
#                 sample_weight,
#             ) = output
#             return (
#                 tuple(model_inputs),
#                 (labels_broadcast, labels),
#                 (sample_weight_broadcast, sample_weight),
#             )

#     else:
#         assert backend == "tensorflow", backend

#         @tf.function
#         def preprocessor_func(inputs, labels, sample_weight):
#             (
#                 *model_inputs,
#                 labels_broadcast,
#                 labels,
#                 sample_weight_broadcast,
#                 sample_weight,
#             ) = preprocessor(tree.flatten((inputs, labels, sample_weight)))
#             return (
#                 tuple(model_inputs),
#                 (labels_broadcast, labels),
#                 (sample_weight_broadcast, sample_weight),
#             )

#     return preprocessor_func


# def per_event_model(
#     num_classes: int,
#     max_events: int,
#     backbone_func: tp.Callable[[StreamNode], tp.Sequence[StreamNode]],
#     grid_shape: tp.Iterable[int],
#     *,
#     batch_size: int | None = None,
#     loss=None,
#     metrics=None,
#     weighted_metrics=None,
#     dropout_rate: float = 0.0,
#     normalize_heads: bool = False,
#     reduction="mean",
#     stream_filter=lambda streams: streams[-1:],
#     use_example_loss: bool = False,
#     **compile_kwargs,
# ) -> tp.Tuple[tp.Callable, keras.Model]:
#     inputs, stream = get_inputs(max_events, batch_size, grid_shape)
#     streams = backbone_func(stream)

#     labels = keras.Input((), dtype="int32", batch_size=batch_size)
#     example_sample_weight = keras.Input((), dtype="float32", batch_size=batch_size)

#     def postprocess_stream(stream, example_sample_weight):
#         if normalize_heads:
#             stream = stream.map_features(keras.layers.LayerNormalization())
#         stream = stream.map_features(keras.layers.Dropout(dropout_rate))
#         stream = stream.map_features(keras.layers.Dense(num_classes))
#         if stream.order is None:
#             stream.force_chronological()
#         batch_ids = stream.stream.batch_ids
#         batch_lengths = stream.stream.batch_lengths
#         event_features = stream.compute_features()
#         batch_size = batch_lengths.shape[0]
#         if reduction == "mean":
#             example_features = segment_ops.segment_sum(
#                 event_features,
#                 batch_ids,
#                 num_segments=batch_size,
#                 indices_are_sorted=True,
#             ) / (
#                 ops.expand_dims(ops.cast(batch_lengths, event_features.dtype), axis=-1)
#                 + keras.backend.epsilon()
#             )
#         elif reduction == "sum":
#             example_features = segment_ops.segment_sum(
#                 event_features,
#                 batch_ids,
#                 num_segments=batch_size,
#                 indices_are_sorted=True,
#             )
#         elif reduction == "max":
#             example_features = segment_ops.segment_max(
#                 event_features,
#                 batch_ids,
#                 num_segments=batch_size,
#                 indices_are_sorted=True,
#             )
#         else:
#             raise NotImplementedError(f"reduction {reduction} not supported")

#         valid_batch_count = ops.count_nonzero(batch_lengths)
#         event_sample_weight = ops.take(
#             ops.pad(
#                 ops.where(
#                     batch_lengths == 0,
#                     ops.zeros_like(example_sample_weight),
#                     example_sample_weight
#                     / (
#                         ops.cast(
#                             valid_batch_count * batch_lengths,
#                             example_sample_weight.dtype,
#                         )
#                         + keras.backend.epsilon()
#                     ),
#                 ),
#                 [[0, 1]],
#             ),
#             batch_ids,
#         )
#         return example_features, event_features, batch_ids, event_sample_weight

#     streams = stream_filter(streams)
#     if len(streams) == 1:
#         (
#             example_features,
#             event_features,
#             batch_ids,
#             event_sample_weight,
#         ) = postprocess_stream(streams[0], example_sample_weight)
#     else:
#         example_features, event_features, batch_ids, event_sample_weight = zip(
#             *(postprocess_stream(stream, example_sample_weight) for stream in streams)
#         )
#         example_features = ops.mean(ops.stack(example_features, axis=0), axis=0)
#         event_features = ops.concatenate(event_features, axis=0)
#         batch_ids = ops.concatenate(batch_ids, axis=0)
#         event_sample_weight = ops.concatenate(event_sample_weight, axis=0)
#         event_sample_weight = event_sample_weight / len(streams)

#     # rename
#     event_features = Identity(name="event")(event_features)
#     example_features = Identity(name="example")(example_features)

#     preprocessor, model = split(
#         preprocessor_inputs=inputs,
#         model_outputs=(event_features, example_features),
#         preprocessor_outputs=(batch_ids,),
#     )

#     # map_func model
#     *model_inputs, batch_ids = preprocessor.outputs
#     labels_broadcast = ops.take(ops.pad(labels, [[0, 1]]), batch_ids)

#     event_sample_weight = event_sample_weight * ops.cast(
#         ops.shape(event_sample_weight)[0], event_sample_weight.dtype
#     )

#     per_event_preprocessor = keras.Model(
#         (*preprocessor.inputs, labels, example_sample_weight),
#         (
#             *model_inputs,
#             labels_broadcast,
#             labels,
#             event_sample_weight,
#             example_sample_weight,
#         ),
#     )
#     preprocessor_func = _preprocessor_to_func(per_event_preprocessor)

#     # apply loss to unpooled model outputs / broadcast labels, sample_weight
#     if loss is not None:
#         if use_example_loss:
#             compile_kwargs["loss"] = [None, loss]
#         else:
#             compile_kwargs["loss"] = [loss, None]
#     # apply metrics to pooled model outputs / original labels, sample_weight
#     if metrics is not None:
#         compile_kwargs["metrics"] = [
#             clone_metrics(metrics),
#             clone_metrics(metrics),
#         ]
#     if weighted_metrics is not None:
#         compile_kwargs["weighted_metrics"] = [
#             clone_metrics(weighted_metrics),
#             clone_metrics(weighted_metrics),
#         ]

#     # https://github.com/keras-team/keras/issues/18647
#     # cloning trims the model of input inputs, making summaries cleaner
#     model = keras.models.clone_model(
#         model,
#         tree.map_structure(
#             lambda x: keras.Input(batch_shape=x.shape, dtype=x.dtype), model.input
#         ),
#         clone_function=lambda op: op,
#     )

#     model.compile(**compile_kwargs)
#     return preprocessor_func, model


def _preprocessor_to_func(preprocessor, num_labels: int):
    def split_output(output):
        model_inputs = output[: len(output) - 2 * num_labels]
        labels = output[-2 * num_labels : -num_labels]
        sample_weight = output[-num_labels:]
        return (
            tuple(model_inputs),
            tuple(labels),
            tuple(sample_weight),
        )

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
            return split_output(output)

    else:
        assert backend == "tensorflow", backend

        @tf.function
        def preprocessor_func(inputs, labels, sample_weight):
            output = preprocessor(tree.flatten((inputs, labels, sample_weight)))
            return split_output(output)

    return preprocessor_func


def per_event_model(
    num_classes: int,
    max_events: int,
    backbone_func: tp.Callable[[StreamNode], tp.Sequence[StreamNode]],
    grid_shape: tp.Iterable[int],
    num_frames: int,
    *,
    batch_size: int | None = None,
    loss=None,
    metrics=None,
    weighted_metrics=None,
    dropout_rate: float = 0.0,
    normalize_heads: bool = False,
    stream_filter=lambda streams: streams[-1:],
    use_example_loss: bool = False,
    **compile_kwargs,
) -> tp.Tuple[tp.Callable, keras.Model]:
    inputs, stream = get_inputs(max_events, batch_size, grid_shape)
    t = stream.stream.times
    batch_splits = stream.stream.batch_splits
    t_start = ops.take(t, ops.slice(batch_splits, [0], [batch_size]))
    t_stop = ops.take(t, ops.slice(batch_splits, [1], [batch_size]) - 1)
    streams = backbone_func(stream)

    labels = keras.Input((), dtype="int32", batch_size=batch_size)
    example_sample_weight = keras.Input((), dtype="float32", batch_size=batch_size)

    def postprocess_stream(stream, example_sample_weight):
        if normalize_heads:
            stream = stream.map_features(keras.layers.LayerNormalization())
        stream = stream.map_features(keras.layers.Dropout(dropout_rate))
        stream = stream.map_features(keras.layers.Dense(num_classes))
        if stream.order is None:
            stream.order = stream.stream.get_contiguous_segments_order()
        assert stream.order.contiguous_segments
        batch_ids = stream.stream.batch_ids
        batch_lengths = stream.stream.batch_lengths
        event_features = stream.compute_features()
        segment_ids = stream.order.permute(stream.stream.segment_ids)
        frame_features, frame_features_mask = grid_final_interpolate(
            event_features,
            times=stream.order.permute(stream.stream.times),
            batch_ids=batch_ids,
            segment_ids=segment_ids,
            t_start=t_start,
            t_stop=t_stop,
            num_frames=num_frames,
            grid_size=stream.stream.grid_size,
        )  # [B, grid_size, num_frames, num_classes]
        frame_features = ops.sum(frame_features, axis=1) / ops.expand_dims(
            ops.maximum(
                ops.cast(
                    ops.count_nonzero(frame_features_mask, axis=1), frame_features.dtype
                ),
                1e-3,
            ),
            axis=-1,
        )  # [B, num_frames, num_classes]
        ##### example_features_thresh
        batch_size = batch_lengths.shape[0]

        t_threshold = t_start + (t_stop - t_start) * 0.5
        valid = stream.order.permute(stream.stream.times) > ops.take(
            t_threshold, batch_ids
        )
        example_features_thresh = segment_ops.segment_sum(
            ops.where(
                ops.expand_dims(valid, -1),
                event_features,
                ops.zeros_like(event_features),
            ),
            batch_ids,
            num_segments=batch_size,
            indices_are_sorted=True,
        )
        weight = segment_ops.segment_sum(
            ops.cast(valid, example_features_thresh.dtype),
            batch_ids,
            num_segments=batch_size,
            indices_are_sorted=True,
        )
        example_features_thresh = example_features_thresh / ops.maximum(
            ops.expand_dims(weight, -1), keras.backend.epsilon()
        )

        ##### example_features_final
        example_features_final = frame_features[
            :, -1
        ]  # same as example_features2 below
        # batch_size = batch_lengths.shape[0]
        # assert stream.order.contiguous_segments
        # example_features2 = reduce_mean_final(
        #     event_features,
        #     segment_ids=segment_ids,
        #     grid_size=stream.stream.grid_size,
        #     batch_size=batch_size,
        # )
        ##### example_features_mean
        example_features_mean = segment_ops.segment_sum(
            event_features,
            segment_ids=batch_ids,
            num_segments=batch_size,
            indices_are_sorted=True,
        ) / ops.expand_dims(
            ops.cast(stream.stream.batch_lengths, event_features.dtype), -1
        )

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
        return (
            example_features_final,
            example_features_mean,
            example_features_thresh,
            frame_features,
            event_features,
            batch_ids,
            event_sample_weight,
        )

    streams = stream_filter(streams)
    if len(streams) == 1:
        (
            example_features_final,
            example_features_mean,
            example_features_thresh,
            frame_features,
            event_features,
            batch_ids,
            event_sample_weight,
        ) = postprocess_stream(streams[0], example_sample_weight)
    else:
        (
            example_features_final,
            example_features_mean,
            example_features_thresh,
            frame_features,
            event_features,
            batch_ids,
            event_sample_weight,
        ) = zip(
            *(postprocess_stream(stream, example_sample_weight) for stream in streams)
        )
        example_features_final = ops.mean(
            ops.stack(example_features_final, axis=0), axis=0
        )
        example_features_mean = ops.mean(
            ops.stack(example_features_mean, axis=0), axis=0
        )
        example_features_thresh = ops.mean(
            ops.stack(example_features_thresh, axis=0), axis=0
        )
        frame_features = ops.mean(ops.stack(frame_features, axis=0), axis=0)
        event_features = ops.concatenate(event_features, axis=0)
        batch_ids = ops.concatenate(batch_ids, axis=0)
        event_sample_weight = ops.concatenate(event_sample_weight, axis=0)
        event_sample_weight = event_sample_weight / len(streams)

    # rename
    event_features = Identity(name="event")(event_features)
    frame_features = Identity(name="frame")(frame_features)
    example_features_mean = Identity(name="example_mean")(example_features_mean)
    example_features_final = Identity(name="example_final")(example_features_final)
    example_features_thresh = Identity(name="example_thresh")(example_features_thresh)

    preprocessor, model = split(
        preprocessor_inputs=inputs,
        model_outputs=(
            event_features,
            frame_features,
            example_features_mean,
            example_features_final,
            example_features_thresh,
        ),
        preprocessor_outputs=(batch_ids,),
    )

    # map_func model
    *model_inputs, batch_ids = preprocessor.outputs

    example_labels = labels
    frame_labels = ops.tile(ops.expand_dims(labels, axis=1), (1, num_frames))
    event_labels = ops.take(ops.pad(labels, [[0, 1]]), batch_ids)

    frame_sample_weight = ops.tile(
        ops.expand_dims(example_sample_weight, axis=1), (1, num_frames)
    )
    event_sample_weight = event_sample_weight * ops.cast(
        ops.shape(event_sample_weight)[0], event_sample_weight.dtype
    )

    per_event_preprocessor = keras.Model(
        (*preprocessor.inputs, example_labels, example_sample_weight),
        (
            *model_inputs,
            event_labels,
            frame_labels,
            example_labels,
            example_labels,
            example_labels,
            event_sample_weight,
            frame_sample_weight,
            example_sample_weight,
            example_sample_weight,
            example_sample_weight,
        ),
    )
    preprocessor_func = _preprocessor_to_func(
        per_event_preprocessor, num_labels=len(model.outputs)
    )

    # apply loss to unpooled model outputs / broadcast labels, sample_weight
    if loss is not None:
        if use_example_loss:
            compile_kwargs["loss"] = [
                None,
                None,
                loss,
                loss,
                loss,
            ]
        else:
            compile_kwargs["loss"] = [
                loss,
                None,
                None,
                None,
                None,
            ]
    # apply metrics to pooled model outputs / original labels, sample_weight
    if metrics is not None:
        compile_kwargs["metrics"] = [
            clone_metrics(metrics),
            [PerFrameAccuracy(num_frames)],
            clone_metrics(metrics),
            clone_metrics(metrics),
            clone_metrics(metrics),
        ]
    if weighted_metrics is not None:
        compile_kwargs["weighted_metrics"] = [
            clone_metrics(weighted_metrics),
            [PerFrameAccuracy(num_frames)],
            clone_metrics(weighted_metrics),
            clone_metrics(weighted_metrics),
            clone_metrics(weighted_metrics),
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
    return preprocessor_func, model


# def _preprocessor_to_func(preprocessor):
#     backend = keras.backend.backend()

#     def split_outputs(output):
#         model_inputs = output[:-6]
#         all_labels = output[-6:-3]
#         all_sample_weights = output[-3:]
#         return tuple(model_inputs), tuple(all_labels), tuple(all_sample_weights)

#     if backend == "jax":
#         import jax

#         jax_preprocessor = jax.jit(lambda *args: preprocessor(args), backend="cpu")

#         def preprocessor_func(inputs, labels, sample_weight):
#             preprocessor_inputs = tuple(tree.flatten((inputs, labels, sample_weight)))

#             output = tf.numpy_function(
#                 jax_preprocessor,
#                 preprocessor_inputs,
#                 Tout=tuple(x.dtype for x in preprocessor.output),
#                 stateful=False,
#             )
#             output = tuple(output)
#             tree.map_structure(
#                 lambda o, t: o.set_shape(t.shape), output, preprocessor.output
#             )
#             return split_outputs(output)

#     else:
#         assert backend == "tensorflow", backend

#         @tf.function
#         def preprocessor_func(inputs, labels, sample_weight):
#             output = preprocessor(tree.flatten((inputs, labels, sample_weight)))
#             return split_outputs(output)

#     return preprocessor_func


# def per_event_model(
#     num_frames: int,
#     num_classes: int,
#     max_events: int,
#     backbone_func: tp.Callable[[StreamNode], tp.Sequence[StreamNode]],
#     grid_shape: tp.Iterable[int],
#     *,
#     batch_size: int | None = None,
#     loss=None,
#     metrics=None,
#     weighted_metrics=None,
#     dropout_rate: float = 0.0,
#     normalize_heads: bool = False,
#     stream_filter=lambda streams: streams[-1:],
#     use_example_loss: bool = False,
#     **compile_kwargs,
# ) -> tp.Tuple[tp.Callable, keras.Model]:
#     inputs, stream = get_inputs(max_events, batch_size, grid_shape)
#     streams = backbone_func(stream)
#     streams = stream_filter(streams)

#     t = stream.stream.times
#     batch_splits = stream.stream.batch_splits
#     t_start = ops.take(t, ops.slice(batch_splits, [0], [batch_size]))
#     t_stop = ops.take(t, ops.slice(batch_splits, [1], [batch_size]) - 1)
#     stream_logits = []

#     def preprocess_stream(stream: StreamNode):
#         if stream.order is None:
#             stream.order = stream.stream.get_contiguous_segments_order()
#         x = GridEmaInterpolate(num_frames=num_frames, grid_size=stream.stream.grid_size)(
#             features=stream.compute_features(),
#             times=stream.order.permute(stream.stream.times),
#             segment_ids=stream.order.permute(stream.stream.segment_ids),
#             batch_ids=stream.stream.batch_ids,
#             t_start=t_start,
#             t_stop=t_stop,
#             indices_are_sorted=stream.order.contiguous_segments,
#             normalize=True,
#         )  # [B, grid_size, num_frames, C]

#         if normalize_heads:
#             x = keras.layers.LayerNormalization()(x)
#         x = keras.layers.Dropout(dropout_rate)(x)
#         # x = keras.layers.Dense(num_classes, name=f"stream{i}")(x)
#         x = keras.layers.Dense(num_classes)(x)
#         x = ops.mean(x, axis=1)  # [B, num_frames, num_classes]
#         # x = keras.layers.Identity(name=f"stream{i}")(x)
#         return x

#     num_streams = len(streams)
#     stream_logits = [preprocess_stream(s) for s in streams]
#     stream_frame_logits = keras.layers.Identity(name="stream_frame")(
#         ops.stack(stream_logits, axis=1)
#     )  # [B, num_streams, num_frames, num_classes]
#     example_logits = keras.layers.Identity(name="example")(
#         keras.ops.mean(stream_frame_logits, axis=(1, 2))
#     )  # [B, num_classes]
#     frame_logits = keras.layers.Identity(name="frame")(
#         keras.ops.mean(stream_frame_logits, axis=1)
#     )  # [B, num_frames, num_classes]

#     # frame_logits = keras.layers.Identity(name="frame")(
#     #     ops.mean(
#     #         ops.stack([ops.mean(sl, axis=1) for sl in stream_logits], axis=0), axis=0
#     #     )
#     # )  # [B, num_frames, num_classes]
#     # example_logits = keras.layers.Identity(name="example")(
#     #     ops.mean(frame_logits, axis=1)
#     # )  # [B, num_classes]

#     # apply loss to unpooled model outputs / broadcast labels, sample_weight
#     if loss is not None:
#         if use_example_loss:
#             compile_kwargs["loss"] = [
#                 None,
#                 None,
#                 loss,
#             ]
#         else:
#             compile_kwargs["loss"] = [
#                 loss,
#                 None,
#                 None,
#             ]
#     # apply metrics to pooled model outputs / original labels, sample_weight
#     if metrics is not None:
#         compile_kwargs["metrics"] = [
#             clone_metrics(metrics),
#             [PerFrameAccuracy(num_frames)],
#             clone_metrics(metrics),
#         ]
#     if weighted_metrics is not None:
#         compile_kwargs["weighted_metrics"] = [
#             clone_metrics(weighted_metrics),
#             [PerFrameAccuracy(num_frames)],
#             clone_metrics(weighted_metrics),
#         ]

#     preprocessor, model = split(
#         preprocessor_inputs=inputs,
#         model_outputs=(
#             stream_frame_logits,
#             frame_logits,
#             example_logits,
#         ),
#     )

#     # map_func model
#     model_inputs = preprocessor.outputs
#     # labels
#     example_labels = keras.Input((), dtype="int32", batch_size=batch_size)
#     frame_labels = ops.tile(ops.expand_dims(example_labels, axis=1), (1, num_frames))
#     stream_frame_labels = ops.tile(
#         ops.expand_dims(frame_labels, axis=1), (1, num_streams, 1)
#     )
#     # sample_weights
#     example_sample_weight = keras.Input((), dtype="float32", batch_size=batch_size)
#     frame_sample_weight = ops.tile(
#         ops.expand_dims(example_sample_weight, axis=1), (1, num_frames)
#     )
#     stream_frame_sample_weight = ops.tile(
#         ops.expand_dims(frame_sample_weight, axis=1), (1, num_streams, 1)
#     )

#     preprocessor = keras.Model(
#         (*preprocessor.inputs, example_labels, example_sample_weight),
#         (
#             *model_inputs,
#             stream_frame_labels,
#             frame_labels,
#             example_labels,
#             stream_frame_sample_weight,
#             frame_sample_weight,
#             example_sample_weight,
#         ),
#     )
#     preprocessor_func = _preprocessor_to_func(preprocessor)

#     # https://github.com/keras-team/keras/issues/18647
#     # cloning trims the model of input inputs, making summaries cleaner
#     model = keras.models.clone_model(
#         model,
#         tree.map_structure(
#             lambda x: keras.Input(batch_shape=x.shape, dtype=x.dtype), model.input
#         ),
#         clone_function=lambda op: op,
#     )

#     model.compile(**compile_kwargs)
#     return preprocessor_func, model
