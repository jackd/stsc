import functools
import typing as tp

import keras
import tensorflow as tf
import tqdm
import tree
from events_tfds.events.cifar10_dvs import GRID_SHAPE, NUM_CLASSES
from jk_neuro.data.transforms_tf import (
    FlipHorizontal,
    FlipTime,
    Maybe,
    Pad,
    RandomCrop,
    Resize,
    SeriesTransform,
    StreamData,
    Transform,
    mask_valid_events,
)

from stsc.components import input_stream
from stsc.data.base import tfds_base_dataset
from stsc.data.batching import batch_and_pad
from stsc.models.backbones.vgg import vgg_backbone

keras.config.disable_traceback_filtering()  # DEBUG

time_scale = 1e5
grid_shape = GRID_SHAPE
num_classes = NUM_CLASSES
# batch_size = 1024
# batch_size = 32
# batch_size = 16
# batch_size = 8
batch_size = 2
events_per_example = 262_144  # 1_844_489_935 total in first 90% of examples
# events_per_example = 32_768  # 64 * 64 * 8
# epochs = 20
epochs = 100
examples_per_epoch = 9_000
train_split = "train[:90%]"
val_split = "train[90%:]"
resized_grid_shape = (128, 128)
padding = 4
filters0 = 32
base_lr = 1e-3
optimizer = "AdamW"
weight_decay = 5e-4
# base_lr = 1e-2
# optimizer = "SGD"
# weight_decay = 5e-3

batch_size = 32


base_transform = Resize(resized_grid_shape)

train_transform = SeriesTransform(
    Pad(padding),
    RandomCrop(resized_grid_shape),
    Maybe(FlipHorizontal()),
    Maybe(FlipTime()),
)

val_transform = None

steps_per_epoch = examples_per_epoch // batch_size
warmup_epochs = 0


def map_fun(example, transform: tp.Optional[Transform]):
    events = example["events"]

    times = events["time"]
    coords = events["coords"]
    polarity = events["polarity"]
    label = example["label"]

    coords = tf.cast(coords, "int32")

    times = tf.cast(times, tf.float32) / time_scale
    stream = StreamData(coords, times, polarity, GRID_SHAPE)
    stream, label = base_transform.transform_stream_example(stream, label)

    if transform is not None:
        stream, label = transform.transform_stream_example(stream, label)
        stream = mask_valid_events(stream)
    assert stream.grid_shape == resized_grid_shape, (
        stream.grid_shape,
        resized_grid_shape,
    )
    times = stream.times
    coords = stream.coords
    polarity = stream.polarity
    return (times, coords, polarity), label


train_ds = tfds_base_dataset(
    "cifar10_dvs",
    train_split,
    functools.partial(map_fun, transform=train_transform),
    shuffle=True,
    infinite=True,
    replace=True,
    num_parallel_calls=4,
)
val_ds = tfds_base_dataset(
    "cifar10_dvs",
    val_split,
    functools.partial(map_fun, transform=val_transform),
    num_parallel_calls=4,
)

backbone_func = functools.partial(
    vgg_backbone,
    activation="relu",
    filters0=filters0,
)


def preprocess_dataset(dataset: tf.data.Dataset):
    dataset = batch_and_pad(
        dataset,
        batch_size=batch_size,
        max_events=events_per_example * batch_size,
        drop_remainder=True,
    )
    return dataset


import jax
from jax.experimental import checkify


def f(times, coords, polarities, splits):
    stream = input_stream(polarities, coords, times, splits, resized_grid_shape)
    streams = backbone_func(stream)
    features = streams[-1].compute_features()
    return features


jitted_f = jax.jit(f)
checkified = checkify.checkify(jitted_f)

train_ds = preprocess_dataset(train_ds)
val_ds = preprocess_dataset(val_ds)

for el in tqdm.tqdm(train_ds):
    (times, coords, polarities, splits), labels, sample_weights = tree.map_structure(
        keras.ops.convert_to_tensor, el
    )
    # stream = input_stream(polarities, coords, times, splits, resized_grid_shape)
    # streams = backbone_func(stream)
    # features = streams[-1].compute_features()
    # print(features)
    # exit()
    err, value = checkified(times, coords, polarities, splits)
    err.throw()
exit()

model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    callbacks=[keras.callbacks.TerminateOnNaN()],
)
