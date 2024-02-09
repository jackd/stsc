import functools
import os
import shutil
import typing as tp
from pathlib import Path

import keras
import tensorflow as tf
import tree
from events_tfds.events.cifar10_dvs import GRID_SHAPE, NUM_CLASSES

from stsc.data.base import tfds_base_dataset
from stsc.data.batching import batch_and_pad
from stsc.data.transforms_tf import (
    FlipHorizontal,
    FlipTime,
    Maybe,
    Pad,
    RandomCrop,
    RandomTemporalCropV2,
    Resize,
    SeriesTransform,
    StreamData,
    TemporalCropV2,
    Transform,
    mask_valid_events,
)
from stsc.models import wrappers
from stsc.models.backbones import conv_next

# keras.config.disable_traceback_filtering()  # DEBUG

time_scale = 1e5
grid_shape = GRID_SHAPE
num_classes = NUM_CLASSES
# batch_size = 1024
# batch_size = 128
# batch_size = 64
batch_size = 32
# batch_size = 16
# batch_size = 8
# batch_size = 2
events_per_example = 262_144  # 1_844_489_935 total in first 90% of examples
# events_per_example = 32_768  # 64 * 64 * 8
# epochs = 20
# epochs = 100
epochs = 200
examples_per_epoch = 9_000
train_split = "train[:90%]"
val_split = "train[90%:]"
resized_grid_shape = (128, 128)
padding = 10
# filters0 = 128  # max w/ batch_size=32 before OOM
# filters0 = 32
filters0 = 16
# filters0 = 8
# filters0 = 4
base_lr = 1e-3
optimizer = "AdamW"
weight_decay = 5e-4
# base_lr = 1e-2
# optimizer = "SGD"
# weight_decay = 5e-3
# weight_decay = 1e-3
jit = True
# jit = False
# tf.debugging.enable_check_numerics()
# pool = False
# reduction = "max"
# complex_conv = True
# dropout_rate = 0.25
dropout_rate = 0.5

min_dt = 1.2 * 1e6 / time_scale / 32
# min_dt = 0
max_events = batch_size * events_per_example - 1  # -1 so padding makes power of 2

backend = keras.backend.backend()


base_transform = Resize(resized_grid_shape)

train_transform = SeriesTransform(
    Pad(padding),
    RandomCrop(resized_grid_shape),
    # RandomTemporalCropV2(0.2),
    Maybe(FlipHorizontal()),
    Maybe(FlipTime()),
)

# val_transform = TemporalCropV2(0.1, 0.8)
val_transform = None

steps_per_epoch = examples_per_epoch // batch_size
warmup_epochs = 1


def map_fun(example, transform: tp.Optional[Transform]):
    events = example["events"]

    times = events["time"]
    coords = events["coords"]
    polarity = events["polarity"]
    label = example["label"]

    coords = tf.cast(coords, "int32")

    times = times - times[0]

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
    map_fun=functools.partial(map_fun, transform=train_transform),
    shuffle=True,
    infinite=True,
    replace=True,
    num_parallel_calls=4,
)
val_ds = tfds_base_dataset(
    "cifar10_dvs",
    val_split,
    map_fun=functools.partial(map_fun, transform=val_transform),
    num_parallel_calls=4,
)

backbone_func = functools.partial(
    conv_next.conv_next_backbone,
    filters0=filters0,
    min_dt=min_dt,
    num_levels=5,
)


kwargs = {
    "max_events": max_events,
    "batch_size": batch_size,
    "backbone_func": backbone_func,
    "grid_shape": resized_grid_shape,
    "num_classes": num_classes,
}
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum")
lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5,
    warmup_target=base_lr,
    decay_steps=(epochs - warmup_epochs) * steps_per_epoch,
    warmup_steps=warmup_epochs * steps_per_epoch,
)
# lr = keras.optimizers.schedules.CosineDecay(base_lr, epochs * steps_per_epoch)
if optimizer == "SGD":
    optimizer = keras.optimizers.SGD(lr, momentum=0.9, weight_decay=weight_decay)
else:
    assert optimizer == "AdamW", optimizer
    optimizer = keras.optimizers.AdamW(lr, weight_decay=weight_decay)

optimizer.exclude_from_weight_decay(
    # var_names=["bias", "gamma", "beta", "scale", "tau", "decay", "decay_magnitude"]
    var_names=["bias", "gamma", "beta", "scale"]
)
metrics = None

weighted_metrics = [
    keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="cross_entropy"),
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
]
preprocessor, model = wrappers.per_event_model(
    loss=loss,
    metrics=metrics,
    weighted_metrics=weighted_metrics,
    optimizer=optimizer,
    dropout_rate=dropout_rate,
    normalize=False,
    jit_compile=jit,
    **kwargs,
)

# preprocessor, model = wrappers.pool_all_model(
#     **kwargs,
# )
# model.compile(
#     optimizer=optimizer,
#     weighted_metrics=weighted_metrics,
#     loss=loss,
# )
# preprocessor, model = wrappers.per_event_model(
#     per_event_loss=loss,
#     dropout_rate=0.25,
#     **kwargs,
# )
# # No loss in compile since it's already been implemented in per_event_model
# model.compile(
#     optimizer=optimizer,
#     weighted_metrics=weighted_metrics,
# )

model.summary()


if backend == "jax":
    import jax
    import jax.numpy as jnp

    def tf_to_jax(x: tf.Tensor) -> jnp.ndarray:
        dlpack = tf.experimental.dlpack.to_dlpack(x)
        jax_arr = jax.dlpack.from_dlpack(dlpack)
        return jax_arr

    def jax_to_tf(x: jnp.ndarray) -> tf.Tensor:
        dlpack = jax.dlpack.to_dlpack(x)
        tensor = tf.experimental.dlpack.from_dlpack(dlpack)
        return tensor

    # from jax.experimental import jax2tf
    # preprocessor = jax2tf.convert(preprocessor, with_gradient=False)

    jax_preprocessor = jax.jit(lambda *args: preprocessor(args), backend="cpu")

    # def jax_preprocessor(*args):
    #     return preprocessor(args)

    def preprocessor_fun(inputs, labels, sample_weight):
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
    def preprocessor_fun(inputs, labels, sample_weight):
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

    # preprocessor = tf.function(preprocessor, jit_compile=True)
    # base_preprocessor = preprocessor

    # def preprocessor(inputs, labels, sample_weight):
    #     with tf.xla.experimental.jit_scope():
    #         return base_preprocessor(inputs, labels, sample_weight)


def preprocess_dataset(dataset: tf.data.Dataset):
    dataset = batch_and_pad(
        dataset,
        batch_size=batch_size,
        max_events=max_events,
        drop_remainder=True,
        map_fun=preprocessor_fun,
        num_parallel_calls=tf.data.AUTOTUNE,
        # deterministic=False,
    )
    # dataset = dataset.prefetch(1)
    return dataset


train_ds = preprocess_dataset(train_ds)
# import tqdm

# for el in tqdm.tqdm(train_ds.take(1000).prefetch(tf.data.AUTOTUNE)):
#     pass
# exit()
train_ds = train_ds.prefetch(1)
val_ds = preprocess_dataset(val_ds)

print("Caching val_ds")
# cache val dataset to reduce RAM usage
tmp_dir = "/tmp/stsc/cache"
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
Path(tmp_dir).mkdir(parents=True)
val_ds.save(tmp_dir)
val_ds = tf.data.Dataset.load(tmp_dir)
print("Finished caching val_ds")

model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
)
