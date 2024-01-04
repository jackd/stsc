import dataclasses
import typing as tp

import events_tfds.events.cifar10_dvs  # pylint:disable=unused-import
import grain.python as pygrain
import jax
import jax.numpy as jnp
import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import tree


@dataclasses.dataclass
class Pack(pygrain.Operation):
    max_events: int
    batch_size: int

    def __call__(self, iterator: tp.Iterator) -> tp.Iterator:
        coords = np.zeros((self.max_events, 2), dtype="int32")
        times = np.zeros((self.max_events,), dtype="int32")
        polarity = np.zeros((self.max_events,), dtype="bool")
        labels = np.zeros((self.batch_size,), dtype="int32")
        batch_splits = np.zeros((self.batch_size + 1,), dtype="int32")
        batch_splits[0] = 0
        num_events = 0
        batch_size = 0
        index = 0
        try:
            while True:
                record: pygrain.Record = next(iterator)
                (c, t, p), l = record.data
                num_events_ = num_events + c.shape[0]
                if num_events_ <= self.max_events and batch_size < self.batch_size:
                    coords[num_events:num_events_] = c
                    times[num_events:num_events_] = t
                    polarity[num_events:num_events_] = p
                    labels[batch_size] = l
                    batch_size += 1
                    num_events = num_events_
                    batch_splits[batch_size] = num_events_
                else:
                    batch_splits[batch_size + 1 :] = batch_splits[batch_size]
                    data = (coords, times, polarity, batch_splits), labels
                    yield pygrain.Record(pygrain.RecordMetadata(index), data)
                    index += 1
                    batch_size = 0
                    num_events = 0
                    coords = np.zeros((self.max_events, 2), dtype="int32")
                    times = np.zeros((self.max_events,), dtype="int32")
                    polarity = np.zeros((self.max_events,), dtype="bool")
                    labels = np.zeros((self.batch_size,), dtype="int32")
                    batch_splits = np.zeros((self.batch_size + 1,), dtype="int32")

        except StopIteration:
            if batch_size > 0:
                # final batch
                batch_splits[batch_size + 1 :] = batch_splits[batch_size]
                data = (coords, times, polarity, batch_splits), labels
                yield pygrain.Record(pygrain.RecordMetadata(index), data)
                index += 1
                batch_size = 0
                num_events = 0


def tf_to_jax(x: tf.Tensor) -> jnp.ndarray:
    dlpack = tf.experimental.dlpack.to_dlpack(x)
    jax_arr = jax.dlpack.from_dlpack(dlpack)
    return jax_arr


class DeserializeTf(pygrain.MapTransform):
    def __init__(self, func):
        self.func = func

    def map(self, bytes):
        with tf.device("/cpu:0"):
            return self.func(bytes)


class Deserialize(pygrain.MapTransform):
    def __init__(self, func):
        self.func = func

    def map(self, bytes):
        return self.func(bytes)


class ToNumpy(pygrain.MapTransform):
    def map(self, element):
        return tree.map_structure(lambda x: x.numpy(), element)


class TfToJax(pygrain.MapTransform):
    def map(self, element):
        return tree.map_structure(tf_to_jax, element)


class ToJax(pygrain.MapTransform):
    def map(self, element):
        with jax.default_device(jax.devices("cpu")[0]):
            return tree.map_structure(jnp.asarray, element)


class Unpack(pygrain.MapTransform):
    def map(self, element):
        events = element["events"]
        coords = events["coords"].astype("int32")
        times = events["time"].astype("int32")
        polarity = events["polarity"]
        label = element["label"]
        return (coords, times, polarity), label


class KerasMap(pygrain.MapTransform):
    def __init__(self, model: keras.Model):
        self.model = model
        self.func = jax.jit(
            model,
            device=jax.devices("cpu")[0],
        )

    def map(self, element):
        inputs, label = element
        with jax.default_device(jax.devices("cpu")[0]):
            inputs = self.func(inputs)
        return inputs, label

    @staticmethod
    def from_json(json_string):
        return KerasMap(keras.models.model_from_json(json_string))

    def __reduce__(self):
        return KerasMap.from_json, (self.model.to_json(),)


class JaxMap(pygrain.MapTransform):
    def __init__(self, func):
        self.func = func

    def map(self, element):
        return self.func(element)


def main():
    # assert keras.backend.backend() == "jax"
    tfds_source = tfds.data_source("cifar10_dvs", split="train")
    deserialize = DeserializeTf(tfds_source.dataset_info.features.deserialize_example)
    # deserialize = Deserialize(deserialize_jax(tfds_source.dataset_info.features))
    # deserialize = Deserialize(tfds_source.dataset_info.features.deserialize_example_np)
    tfds_source = tfds_source.data_source

    coords = keras.Input((2,), dtype="int32")
    times = keras.Input((), dtype="int32")
    polarity = keras.Input((), dtype="bool")
    z = coords + keras.ops.expand_dims(times, axis=-1)
    keras.Model((coords, times, polarity), z)
    # path = "/tmp/model.keras"
    # model.save(path)
    # del model
    events_per_example = 262_144
    batch_size = 32
    examples_per_batch = events_per_example * batch_size

    length_struct = (
        (examples_per_batch, examples_per_batch, examples_per_batch),
        batch_size,
    )

    data_loader = pygrain.DataLoader(
        data_source=tfds_source,
        sampler=pygrain.IndexSampler(
            num_records=len(tfds_source),
            num_epochs=2,
            shard_options=pygrain.NoSharding(),
            shuffle=True,
            seed=0,
        ),
        worker_count=4,
        operations=[
            deserialize,
            ToNumpy(),
            # TfToJax(),
            # ToJax(),
            Unpack(),
            Pack(examples_per_batch, batch_size)
            # KerasMap(model),
            # JaxMap(
            #     jax.jit(
            #         lambda element: (model(element[0]), element[1]),
            #         device=jax.devices("cpu")[0],
            #     )
            # ),
        ],
        # read_options=pygrain.ReadOptions(prefetch_buffer_size=64),
    )

    for data in tqdm.tqdm(data_loader):
        # (coords, times, polarity, batch_splits), label = data
        # print(batch_splits)
        pass


if __name__ == "__main__":
    main()
