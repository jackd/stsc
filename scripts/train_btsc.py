import functools
import tempfile
import typing as tp

import events_tfds.events  # pylint:disable=unused-import
import keras
import numpy as np
import tensorflow as tf
from absl import app, flags, logging

try:
    import jk_neuro.ops.heaviside  # pylint:disable=unused-import
except ImportError:
    pass

from stsc.btsc.layers import MaskedGlobalAveragePooling
from stsc.btsc.models import EmaMode, conv_next, vgg
from stsc.data import transforms_tf
from stsc.data.base import tfds_base_dataset, tfds_cardinality
from stsc.data.transforms_tf import (
    FlipHorizontal,
    FlipTime,
    Maybe,
    Pad,
    PadToSquare,
    RandomCrop,
    RandomRotate,
    RandomTemporalCropV2,
    RandomZoom,
    Recenter,
    Resize,
    SeriesTransform,
    TemporalCropV2,
    Transpose,
)
from stsc.metrics.per_frame_metrics import PerFrameAccuracy
from stsc.models.wrappers import clone_metrics
from stsc.train import LoggerCallback

assert keras.backend.backend() == "jax", keras.backend.backend()

keras.config.disable_traceback_filtering()


class TupleParser:
    def __init__(self, parser):
        self.parser = parser
        if callable(parser):
            self.parse_func = parser
        elif hasattr(parser, "parse"):
            self.parse_func = parser.parse
        else:
            raise ValueError(
                f"parser must be a callable or have a parse method, got {parser}"
            )

    def parse(self, x):
        if isinstance(x, str):
            x = x.split(",")
        return tuple(self.parse_func(i) for i in x)


class IterableSerializer:
    def serialize(self, x):
        return ",".join(str(i) for i in x)


flags.DEFINE(
    name="x",
    default=[1, 2],
    help="list of integers",
    parser=TupleParser(int),
    serializer=IterableSerializer(),
)


def DEFINE_tuple(
    name, default, help, parser, length=None, lower_bound=None, upper_bound=None, **args
):
    flags.DEFINE(
        TupleParser(parser),
        name=name,
        default=default,
        serializer=IterableSerializer(),
        help=help,
        **args,
    )
    if length:
        flags.register_validator(
            name, lambda x: len(x) == length, message=f"length must be {length}"
        )
    if lower_bound:
        flags.register_validator(
            name,
            lambda x: all(i >= lower_bound for i in x),
            message=f"must be >= {lower_bound}",
        )
    if lower_bound:
        flags.register_validator(
            name,
            lambda x: all(i >= lower_bound for i in x),
            message=f"must be >= {lower_bound}",
        )
    if upper_bound:
        flags.register_validator(
            name,
            lambda x: all(i <= upper_bound for i in x),
            message=f"must be <= {upper_bound}",
        )


def DEFINE_floats(name, default, help, length=None, **args):
    DEFINE_tuple(name, default, help, float, length=length, **args)


def DEFINE_integers(name, default, help, length=None, **args):
    DEFINE_tuple(name, default, help, int, length=length, **args)


flags.DEFINE_integer("seed", 54321, "random seed")

# environment
flags.DEFINE_bool("interactive", True, "environment is interactive. Used for logging")


# original dataset parameters
DEFINE_integers("grid_shape", None, "base dataset grid shape", length=2, required=True)
flags.DEFINE_integer(
    "num_classes", None, "number of classes", required=True, lower_bound=2
)
flags.DEFINE_integer(
    "num_frames",
    20,
    "number of buffered frames (before temporal cropping)",
    lower_bound=1,
)
flags.DEFINE_string(
    "dataset", None, "dataset name as used in events-tfds", required=True
)
flags.DEFINE_string("train_split", "train", "train split name")
flags.DEFINE_string("validation_split", "validation", "validation split name")
flags.DEFINE_string("test_split", "test", "test split name")
flags.DEFINE_bool(
    "transpose", False, "whether original dataset needs to have x-y flipped"
)

# data pipeline
DEFINE_integers(
    "rescaled_grid_shape",
    (128, 128),
    "preprocessed grid shape",
    length=2,
    lower_bound=1,
)
flags.DEFINE_bool("recenter", False, "move stream to center before any translation")
flags.DEFINE_integer(
    "translate", 10, "maximum augmentation pixel translate", lower_bound=0
)
flags.DEFINE_float(
    "rotate", 15.0, "maximum augmentation rotation (degrees)", lower_bound=0.0
)
DEFINE_floats("zoom", (0.9, 1.1), "augmentation zoom range", lower_bound=0.0, length=2)
flags.DEFINE_float("temporal_crop", 0.2, "maximum temporal crop fraction")
flags.DEFINE_bool(
    "flip_horizontal", False, "whether to horizontally flip half training examples"
)
DEFINE_integers("flip_horizontal_label_map", (), "label map for horizontal flip")
flags.DEFINE_bool(
    "flip_temporal", False, "whether to temporally flip half training examples"
)
DEFINE_integers("flip_temporal_label_map", (), "label map for temporal flip")
flags.DEFINE_integer("batch_size", 32, "batch size", lower_bound=1)
flags.DEFINE_bool("cache_validation_data", False, "cache validation dataset")


# optimizer
flags.DEFINE_float(
    "initial_lr", 1e-5, "initial learning rate (before warmup)", lower_bound=0
)
flags.DEFINE_float(
    "base_lr", 1e-3, "base learning rate (after warm up, before decay)", lower_bound=0
)
flags.DEFINE_integer("warmup_epochs", 1, "number of warmup epochs", lower_bound=0)
flags.DEFINE_float("weight_decay", 5e-4, "weight decay", lower_bound=0)
flags.DEFINE_bool(
    "exclude_decay_rate_from_weight_decay", True, "exclude decay rate from weight decay"
)

# model
flags.DEFINE_integer("filters0", 64, "Initial number of filters", lower_bound=8)
flags.DEFINE_enum(
    "backbone", "vgg", ["vgg", "vgg_cnn", "conv_next"], "backbone architecture"
)
# flags.DEFINE_float(
#     "min_dt", 0.0, "refactory period between sampled points", lower_bound=0
# )
# flags.DEFINE_bool("simple_pooling", True, "use simple pooling layers if pool=True")
flags.DEFINE_enum(
    "reduction", "mean", ["mean", "max"], "reduction method for pooling layers"
)
flags.DEFINE_bool("complex_conv", False, "use complex convolutions")
flags.DEFINE_integer(
    "initial_stride", 4, "stride for initial exclusive conv", lower_bound=1
)
flags.DEFINE_float(
    "dropout_rate", 0.5, "rate used in classifier heads", lower_bound=0.0
)
flags.DEFINE_bool(
    "normalize_heads",
    False,
    "use layer norm in heads before final dropout/classification layers",
)
flags.DEFINE_integer("start_head", 1, "start head index", lower_bound=0)
flags.DEFINE_bool("normalize_convs", True, "use normalized convolutions")
flags.DEFINE_string("activation", "default", "activation function")
flags.DEFINE_bool("mask", False, "Mask out voxels with no events.")
flags.DEFINE_enum_class(
    "ema",
    EmaMode.ALL,
    EmaMode,
    "Use EMA in convolutions.",
)
flags.DEFINE_bool(
    "use_example_loss", False, help="use example loss instead of event loss"
)

# fit
flags.DEFINE_integer("epochs", 200, "number of epochs to train for")

# unused below, but makes same flag files usable
# original dataset parameters
flags.DEFINE_float(
    "duration", None, "mean stream length", required=False, lower_bound=0.0
)
flags.DEFINE_integer(
    "events_per_example",
    None,
    "mean number of events per example, or over-estimate.",
    required=False,
)
flags.DEFINE_integer(
    "initial_sample_rate",
    16,
    "sample rate used for initial exclusive conv",
    lower_bound=1,
)


def map_frames(
    example,
    transform: transforms_tf.Transform | None = None,
):
    """
    Unpack events in standard raw format, integrate and applies transform.

    Standard raw format is
    {"events": {"time": x, "coords": x, "polarity": x}, "label": x}

    Args:
        example: structure of tensors in standard raw format
        grid_shape: (height, width) of event stream (limits on coords)
        transform: `stsc.data.transforms_tf.Transform` to apply, e.g. for data
            augmentation
        rezero_times: if given, time origin is reset, i.e. times <- times - times[0]

    Returns:
        (times, coords, polarity), label
    """
    frames = example["events"]  # [num_Frames, H, W, 2] int64
    label = example["label"]

    if transform is not None:
        frames, label = transform.transform_frames_example(frames, label)

    return frames, label


def per_frame_model(
    num_classes: int,
    backbone_func: tp.Callable[[keras.KerasTensor], tp.Sequence[keras.KerasTensor]],
    grid_shape: tp.Iterable[int],
    num_frames: int,
    batch_size: int,
    *,
    loss=None,
    metrics=None,
    weighted_metrics=None,
    dropout_rate: float = 0.0,
    normalize_heads: bool = False,
    reduction="mean",
    stream_filter=lambda streams: streams[-1:],
    use_example_loss: bool = False,
    **compile_kwargs,
) -> tp.Tuple[keras.Model, tp.Callable]:
    inp = keras.Input(
        (num_frames, *grid_shape, 2), batch_size=batch_size, dtype="int32"
    )
    x = keras.ops.cast(inp, "float32")
    outputs = backbone_func(x)
    outputs = stream_filter(outputs)
    num_streams = len(outputs)

    def map_func(x, example_label, sample_weight=None):
        frame_label = tf.broadcast_to(
            tf.reshape(example_label, (-1, 1)),
            (batch_size, num_frames),
        )
        stream_frame_label = tf.broadcast_to(
            tf.reshape(example_label, (-1, 1, 1)),
            (batch_size, num_streams, num_frames),
        )
        if sample_weight is not None:
            example_sample_weight = sample_weight
            frame_sample_weight = tf.broadcast_to(
                tf.reshape(example_sample_weight, (-1, 1)),
                (batch_size, num_frames),
            )
            stream_frame_sample_weight = tf.broadcast_to(
                tf.reshape(example_sample_weight, (-1, 1, 1)),
                (batch_size, num_streams, num_frames),
            )
            sample_weight = (
                stream_frame_sample_weight,
                frame_sample_weight,
                example_sample_weight,
            )
        labels = (
            stream_frame_label,
            frame_label,
            example_label,
        )
        return keras.utils.pack_x_y_sample_weight(x, labels, sample_weight)

    def postprocess_stream(stream: keras.KerasTensor):
        if normalize_heads:
            stream = keras.layers.LayerNormalization()(stream)
        # dropout before reduction slightly improves performance?
        stream = keras.layers.Dropout(dropout_rate)(stream)
        # spatial reduction
        if reduction == "mean":
            # stream = keras.ops.mean(stream, axis=(2, 3))
            stream = MaskedGlobalAveragePooling(axis=(2, 3))(stream)
        elif reduction == "sum":
            stream = keras.ops.sum(stream, axis=(2, 3))
        elif reduction == "max":
            stream = keras.ops.max(stream, axis=(2, 3))
        else:
            raise NotImplementedError(f"reduction {reduction} not supported")
        stream = keras.layers.Dense(num_classes)(stream)
        return stream

    logits = keras.ops.stack([postprocess_stream(stream) for stream in outputs], axis=1)
    # rename
    stream_frame_logits = keras.layers.Identity(name="stream_frame")(
        logits
    )  # [B, num_streams, num_frames, num_classes]
    frame_logits = keras.layers.Identity(name="frame")(
        keras.ops.mean(logits, axis=1)
    )  # [B, num_frames, num_classes]
    example_logits = keras.layers.Identity(name="example")(
        keras.ops.mean(frame_logits, axis=1)
    )  # [B, num_classes]

    # apply loss to unpooled model outputs / broadcast labels, sample_weight
    if loss is not None:
        if use_example_loss:
            compile_kwargs["loss"] = (
                None,
                None,
                loss,
            )
        else:
            compile_kwargs["loss"] = (
                loss,
                None,
                None,
            )
    # apply metrics to pooled model outputs / original labels, sample_weight
    if metrics is not None:
        compile_kwargs["metrics"] = [
            clone_metrics(metrics),
            [PerFrameAccuracy(num_frames)],
            clone_metrics(metrics),
        ]
    if weighted_metrics is not None:
        compile_kwargs["weighted_metrics"] = [
            clone_metrics(weighted_metrics),
            [PerFrameAccuracy(num_frames)],
            clone_metrics(weighted_metrics),
        ]

    model = keras.Model(
        inp,
        (
            stream_frame_logits,
            frame_logits,
            example_logits,
        ),
    )
    model.compile(**compile_kwargs)
    return model, map_func


def build_and_fit(
    batch_size: int,
    num_classes: int,
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
    verbose: bool = True,
    cache_validation_data: bool = False,
    num_parallel_calls=tf.data.AUTOTUNE,
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

    train_data = train_data.batch(batch_size, drop_remainder=True)
    if validation_data:
        validation_data = validation_data.batch(batch_size, drop_remainder=True)
    if test_data:
        test_data = test_data.batch(batch_size, drop_remainder=True)

    batch_size, num_frames, *grid_shape, num_channels = train_data.element_spec[0].shape
    assert batch_size is not None, batch_size
    steps_per_epoch = examples_per_epoch // batch_size + int(
        examples_per_epoch % batch_size > 0
    )
    assert num_channels == 2, num_channels
    # if validation_data is not None and validation_data.element_spec[0].shape[0] == None:
    #     batch_size = None

    model, map_func = per_frame_model(
        num_classes,
        backbone_func,
        grid_shape=grid_shape,
        num_frames=num_frames,
        batch_size=batch_size,
        loss=loss,
        weighted_metrics=weighted_metrics,
        dropout_rate=dropout_rate,
        normalize_heads=normalize_heads,
        reduction="mean",
        stream_filter=stream_filter,
        use_example_loss=use_example_loss,
        optimizer=optimizer,
    )
    train_data = train_data.map(map_func, num_parallel_calls=num_parallel_calls)
    if validation_data is not None:
        validation_data = validation_data.map(
            map_func, num_parallel_calls=num_parallel_calls
        )
    if test_data is not None:
        test_data = test_data.map(map_func, num_parallel_calls=num_parallel_calls)

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

    if cache_validation_data:
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


def main(_):
    FLAGS = flags.FLAGS
    seed = FLAGS.seed

    # environment
    interactive = FLAGS.interactive

    # data
    FLAGS.grid_shape
    num_classes = FLAGS.num_classes
    train_split = FLAGS.train_split
    validation_split = FLAGS.validation_split
    test_split = FLAGS.test_split
    dataset = f"{FLAGS.dataset}/frames-{FLAGS.num_frames:03d}"
    transpose = FLAGS.transpose
    recenter = FLAGS.recenter

    # data pipeline
    rescaled_grid_shape = FLAGS.rescaled_grid_shape
    translate = FLAGS.translate
    rotate = FLAGS.rotate * np.pi / 180  # convert to radians
    zoom = FLAGS.zoom
    temporal_crop = FLAGS.temporal_crop
    flip_horizontal = FLAGS.flip_horizontal
    flip_horizontal_label_map = FLAGS.flip_horizontal_label_map
    if flip_horizontal_label_map:
        assert len(flip_horizontal_label_map) == num_classes, (
            len(flip_horizontal_label_map),
            num_classes,
        )
        flip_horizontal_label_map = tf.convert_to_tensor(
            flip_horizontal_label_map, tf.int64
        )
    else:
        flip_horizontal_label_map = None
    flip_temporal = FLAGS.flip_temporal
    flip_temporal_label_map = FLAGS.flip_temporal_label_map
    if flip_temporal_label_map:
        assert len(flip_temporal_label_map) == num_classes, (
            len(flip_temporal_label_map),
            num_classes,
        )
        flip_temporal_label_map = tf.convert_to_tensor(
            flip_temporal_label_map, tf.int64
        )
    else:
        flip_temporal_label_map = None
    batch_size = FLAGS.batch_size
    cache_validation_data = FLAGS.cache_validation_data

    keras.utils.set_random_seed(seed)

    examples_per_epoch = tfds_cardinality(dataset, train_split)

    # optimizer
    initial_learning_rate = FLAGS.initial_lr
    base_lr = FLAGS.base_lr
    warmup_epochs = FLAGS.warmup_epochs
    weight_decay = FLAGS.weight_decay
    exclude_decay_rate_from_weight_decay = FLAGS.exclude_decay_rate_from_weight_decay

    # model
    filters0 = FLAGS.filters0
    backbone = FLAGS.backbone
    reduction = FLAGS.reduction
    complex_conv = FLAGS.complex_conv
    initial_stride = FLAGS.initial_stride
    dropout_rate = FLAGS.dropout_rate
    normalize_heads = FLAGS.normalize_heads
    start_head = FLAGS.start_head
    normalize_convs = FLAGS.normalize_convs
    activation = FLAGS.activation
    apply_mask = FLAGS.mask
    use_example_loss = FLAGS.use_example_loss
    ema_mode = FLAGS.ema

    if activation == "default":
        if backbone.startswith("vgg"):
            activation = "relu"
        elif backbone == "conv_next":
            activation = "gelu"
        else:
            raise NotImplementedError(f"Unrecognized backbone '{backbone}'")
    activation = keras.utils.deserialize_keras_object(activation)

    # fit
    epochs = FLAGS.epochs

    train_transforms = []

    if transpose:
        train_transforms.append(Transpose())
    if recenter:
        train_transforms.append(Recenter())
    train_transforms.append(PadToSquare())
    if rotate:
        train_transforms.append(RandomRotate(rotate))
    if zoom and zoom != (1, 1):
        train_transforms.append(RandomZoom(*zoom))
    train_transforms.append(Resize(rescaled_grid_shape))
    if translate:
        train_transforms.extend((Pad(translate), RandomCrop(rescaled_grid_shape)))
    if temporal_crop:
        train_transforms.append(RandomTemporalCropV2(temporal_crop))
    if flip_horizontal:
        train_transforms.append(
            Maybe(FlipHorizontal(label_map=flip_horizontal_label_map))
        )

    if flip_temporal:
        train_transforms.append(Maybe(FlipTime(label_map=flip_temporal_label_map)))

    test_transforms = []
    if transpose:
        test_transforms.append(Transpose())
    if recenter:
        test_transforms.append(Recenter())
    test_transforms.extend((PadToSquare(), Resize(rescaled_grid_shape)))
    if temporal_crop:
        test_transforms.append(TemporalCropV2(temporal_crop / 2, 1 - temporal_crop))

    train_transform = SeriesTransform(*train_transforms)
    test_transform = SeriesTransform(*test_transforms)

    train_data = tfds_base_dataset(
        dataset,
        train_split,
        map_fun=functools.partial(
            map_frames,
            transform=train_transform,
        ),
        shuffle=True,
        infinite=True,
        replace=True,
        seed=int(keras.random.randint((), 0, np.iinfo(np.int32).max)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if validation_split:
        validation_data = tfds_base_dataset(
            dataset,
            validation_split,
            map_fun=functools.partial(
                map_frames,
                transform=test_transform,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        validation_data = None
    if test_split:
        test_data = tfds_base_dataset(
            dataset,
            test_split,
            map_fun=functools.partial(
                map_frames,
                transform=test_transform,
            ),
            num_parallel_calls=1,  # only iterated over once at end, would suck to OOM
        )
    else:
        test_data = None

    steps_per_epoch = examples_per_epoch // batch_size + int(
        bool(examples_per_epoch % batch_size)
    )

    if backbone == "vgg":
        # standard vgg with pooling between different resolutions
        backbone_func = functools.partial(
            vgg.vgg_pool_backbone,
            activation=activation,
            filters0=filters0,
            reduction=reduction,
            complex_conv=complex_conv,
            initial_stride=initial_stride,
            normalize=normalize_convs,
            apply_mask=apply_mask,
            ema_mode=ema_mode,
        )
    elif backbone == "vgg_cnn":
        # vgg with final conv and pooling merged into a single strided conv
        backbone_func = functools.partial(
            vgg.vgg_cnn_backbone,
            activation=activation,
            filters0=filters0,
            complex_conv=complex_conv,
            initial_stride=initial_stride,
            normalize=normalize_convs,
            apply_mask=apply_mask,
            ema_mode=ema_mode,
        )
    elif backbone == "conv_next":
        backbone_func = functools.partial(
            conv_next.conv_next_backbone,
            filters0=filters0,
            complex_conv=complex_conv,
            initial_stride=initial_stride,
            normalize=normalize_convs,
            activation=activation,
            apply_mask=apply_mask,
            ema_mode=ema_mode,
        )
    else:
        raise NotImplementedError(f"backbone '{backbone}' not supported.")

    # optimizer
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        warmup_target=base_lr,
        decay_steps=(epochs - warmup_epochs) * steps_per_epoch,
        warmup_steps=warmup_epochs * steps_per_epoch,
    )
    optimizer = keras.optimizers.AdamW(lr, weight_decay=weight_decay)
    var_names = ["bias", "gamma", "beta"]
    if exclude_decay_rate_from_weight_decay:
        var_names.append("decay_rate")
    optimizer.exclude_from_weight_decay(var_names=var_names)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    weighted_metrics = [
        keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="x_entropy"),
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ]

    def stream_filter(streams):
        return streams[start_head:]

    callbacks = []

    if interactive:
        keras.config.enable_interactive_logging()
        verbose = True
    else:
        keras.config.disable_interactive_logging()
        callbacks.append(LoggerCallback(logging.info))
        verbose = False

    build_and_fit(
        batch_size=batch_size,
        num_classes=num_classes,
        examples_per_epoch=examples_per_epoch,
        epochs=epochs,
        backbone_func=backbone_func,
        optimizer=optimizer,
        loss=loss,
        weighted_metrics=weighted_metrics,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        stream_filter=stream_filter,
        dropout_rate=dropout_rate,
        normalize_heads=normalize_heads,
        callbacks=callbacks,
        verbose=verbose,
        use_example_loss=use_example_loss,
        cache_validation_data=cache_validation_data,
        num_parallel_calls=tf.data.AUTOTUNE,
    )


if __name__ == "__main__":
    app.run(main)
