import functools

import keras
import numpy as np
from events_tfds.events.ncars import GRID_SHAPE, NUM_CLASSES
from jk_neuro.data.transforms_tf import (
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
)

from stsc.data.base import tfds_base_dataset
from stsc.models.backbones import vgg
from stsc.train import build_and_fit, map_and_pack

# data
time_scale = 1e5 / 16
grid_shape = GRID_SHAPE
num_classes = NUM_CLASSES
batch_size = 128
events_per_example = 4_096  # 2**12, mean 3_909
examples_per_epoch = 15_422
train_split = "train"
val_split = "test"
resized_grid_shape = (128, 128)
padding = 10

train_transform = SeriesTransform(
    Recenter(),
    PadToSquare(),
    RandomRotate(np.pi / 12),
    RandomZoom(0.9, 1.1),
    # RandomRotate(np.pi / 6),
    # RandomZoom(0.8, 1.2),
    Resize(resized_grid_shape),
    Pad(padding),
    RandomCrop(resized_grid_shape),
    # RandomTemporalCropV2(0.2),
)

validation_transform = SeriesTransform(
    Recenter(),
    PadToSquare(),
    Resize(resized_grid_shape),
    # TemporalCropV2(0.1, 0.8),
)

train_data = tfds_base_dataset(
    "ncars",
    train_split,
    map_fun=functools.partial(
        map_and_pack,
        time_scale=time_scale,
        expected_grid_shape=resized_grid_shape,
        transform=train_transform,
    ),
    shuffle=True,
    infinite=True,
    replace=True,
    num_parallel_calls=4,
)
validation_data = tfds_base_dataset(
    "ncars",
    val_split,
    map_fun=functools.partial(
        map_and_pack,
        time_scale=time_scale,
        expected_grid_shape=resized_grid_shape,
        transform=validation_transform,
    ),
    num_parallel_calls=4,
)


steps_per_epoch = examples_per_epoch // batch_size + int(
    bool(examples_per_epoch % batch_size)
)
epochs = 200

# model
filters0 = 64
pool = False  # HACK
min_dt = 0.0
simple_pooling = True
reduction = "mean"
complex_conv = False
initial_stride = 4
initial_sample_rate = 1  # default would be 4**2==16
if pool:
    backbone_func = functools.partial(
        vgg.vgg_pool_backbone,
        activation="relu",
        filters0=filters0,
        min_dt=min_dt,
        simple_pooling=simple_pooling,
        reduction=reduction,
        complex_conv=complex_conv,
        initial_stride=initial_stride,
        initial_sample_rate=initial_sample_rate,
    )
else:
    backbone_func = functools.partial(
        vgg.vgg_cnn_backbone,
        activation="relu",
        filters0=filters0,
        min_dt=min_dt,
        complex_conv=complex_conv,
        initial_stride=initial_stride,
        initial_sample_rate=initial_sample_rate,
    )


# optimizer
base_lr = 1e-3
warmup_epochs = 1
weight_decay = 5e-4
lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5,
    warmup_target=base_lr,
    decay_steps=(epochs - warmup_epochs) * steps_per_epoch,
    warmup_steps=warmup_epochs * steps_per_epoch,
)
optimizer = keras.optimizers.AdamW(lr, weight_decay=weight_decay)
optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "decay_rate"])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
weighted_metrics = [
    keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="x_entropy"),
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
]


def stream_filter(streams):
    return streams[1:]


dropout_rate = 0.5

build_and_fit(
    resized_grid_shape,
    batch_size=batch_size,
    num_classes=num_classes,
    events_per_example=events_per_example,
    examples_per_epoch=examples_per_epoch,
    epochs=epochs,
    backbone_func=backbone_func,
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=weighted_metrics,
    train_data=train_data,
    validation_data=validation_data,
    stream_filter=stream_filter,
    dropout_rate=dropout_rate,
)
