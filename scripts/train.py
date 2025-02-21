import functools

import events_tfds.events  # pylint:disable=unused-import
import keras
import numpy as np
import tensorflow as tf
from absl import app, flags, logging

try:
    import jk_neuro.ops.heaviside  # pylint:disable=unused-import
except ImportError:
    pass

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
from stsc.models.backbones import conv_next, vgg
from stsc.train import LoggerCallback, build_and_fit, map_and_pack

assert keras.backend.backend() == "jax", keras.backend.backend()


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
flags.DEFINE_float(
    "duration", None, "mean stream length", required=True, lower_bound=0.0
)
DEFINE_integers("grid_shape", None, "base dataset grid shape", length=2, required=True)
flags.DEFINE_integer(
    "num_classes", None, "number of classes", required=True, lower_bound=2
)
flags.DEFINE_integer(
    "events_per_example",
    None,
    "mean number of events per example, or over-estimate.",
    required=True,
)
flags.DEFINE_integer(
    "num_frames",
    64,
    "number of frames for evaluation",
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
flags.DEFINE_float(
    "rescaled_duration",
    16.0,
    "mean stream length after temporal rescaling",
    lower_bound=0.0,
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
flags.DEFINE_bool(
    "cache_validation_data",
    False,
    "cache validation dataset. "
    "Might reduce memory usage but may also slow down training",
)
flags.DEFINE_bool("temporal_split", False, "split training examples in 2")


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
flags.DEFINE_float(
    "min_dt", 0.0, "refactory period between sampled points", lower_bound=0
)
flags.DEFINE_bool("simple_pooling", True, "use simple pooling layers if pool=True")
flags.DEFINE_enum(
    "reduction", "mean", ["mean", "max"], "reduction method for pooling layers"
)
flags.DEFINE_bool("complex_conv", False, "use complex convolutions")
flags.DEFINE_integer(
    "initial_stride", 4, "stride for initial exclusive conv", lower_bound=1
)
flags.DEFINE_integer(
    "initial_sample_rate",
    16,
    "sample rate used for initial exclusive conv",
    lower_bound=1,
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
flags.DEFINE_bool(
    "use_example_loss", False, help="use example loss instead of event loss"
)

# fit
flags.DEFINE_integer("epochs", 200, "number of epochs to train for")


def main(_):
    FLAGS = flags.FLAGS
    seed = FLAGS.seed

    # environment
    interactive = FLAGS.interactive

    # data
    duration = FLAGS.duration
    grid_shape = FLAGS.grid_shape
    num_classes = FLAGS.num_classes
    num_frames = FLAGS.num_frames
    events_per_example = FLAGS.events_per_example
    train_split = FLAGS.train_split
    validation_split = FLAGS.validation_split
    test_split = FLAGS.test_split
    dataset = FLAGS.dataset
    transpose = FLAGS.transpose
    recenter = FLAGS.recenter

    # data pipeline
    rescaled_duration = FLAGS.rescaled_duration
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
    temporal_split = FLAGS.temporal_split

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
    min_dt = FLAGS.min_dt
    simple_pooling = FLAGS.simple_pooling
    reduction = FLAGS.reduction
    complex_conv = FLAGS.complex_conv
    initial_stride = FLAGS.initial_stride
    initial_sample_rate = FLAGS.initial_sample_rate
    dropout_rate = FLAGS.dropout_rate
    normalize_heads = FLAGS.normalize_heads
    start_head = FLAGS.start_head
    normalize_convs = FLAGS.normalize_convs
    activation = FLAGS.activation
    use_example_loss = FLAGS.use_example_loss

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
        # train_transforms.append(RandomTemporalCropV3(temporal_crop))
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

    time_scale = duration / rescaled_duration
    data_kwargs = {
        "time_scale": time_scale,
        "grid_shape": grid_shape,
        "expected_grid_shape": rescaled_grid_shape,
    }

    train_data = tfds_base_dataset(
        dataset,
        train_split,
        map_fun=functools.partial(
            map_and_pack, transform=train_transform, **data_kwargs
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
                map_and_pack, transform=test_transform, **data_kwargs
            ),
            num_parallel_calls=1,  # we cache this in build_and_fit
        )
    else:
        validation_data = None
    if test_split:
        test_data = tfds_base_dataset(
            dataset,
            test_split,
            map_fun=functools.partial(
                map_and_pack, transform=test_transform, **data_kwargs
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
            min_dt=min_dt,
            simple_pooling=simple_pooling,
            reduction=reduction,
            complex_conv=complex_conv,
            initial_stride=initial_stride,
            initial_sample_rate=initial_sample_rate,
            normalize=normalize_convs,
        )
    elif backbone == "vgg_cnn":
        # vgg with final conv and pooling merged into a single strided conv
        backbone_func = functools.partial(
            vgg.vgg_cnn_backbone,
            activation=activation,
            filters0=filters0,
            min_dt=min_dt,
            complex_conv=complex_conv,
            initial_stride=initial_stride,
            initial_sample_rate=initial_sample_rate,
            normalize=normalize_convs,
        )
    elif backbone == "conv_next":
        backbone_func = functools.partial(
            conv_next.conv_next_backbone,
            filters0=filters0,
            min_dt=min_dt,
            complex_conv=complex_conv,
            initial_stride=initial_stride,
            initial_sample_rate=initial_sample_rate,
            normalize=normalize_convs,
            activation=activation,
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
        keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="xe"),
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ]

    def stream_filter(streams):
        return streams[start_head:]

    callbacks = []

    if interactive:
        keras.config.enable_interactive_logging()
        callbacks.append(
            LoggerCallback(format_fn=lambda k, v: f"{k}: {v}", print_fn=logging.info)
        )
        verbose = True
    else:
        keras.config.disable_interactive_logging()
        callbacks.append(LoggerCallback(print_fn=logging.info))
        verbose = False

    build_and_fit(
        rescaled_grid_shape,
        batch_size=batch_size,
        num_classes=num_classes,
        num_frames=num_frames,
        events_per_example=events_per_example,
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
        temporal_split=temporal_split,
        use_example_loss=use_example_loss,
        cache_validation_data=cache_validation_data,
        num_parallel_calls=tf.data.AUTOTUNE,
    )


if __name__ == "__main__":
    app.run(main)
