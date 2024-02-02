import events_tfds.events.asl_dvs  # pylint:disable=unused-import
import tensorflow as tf
import tensorflow_datasets as tfds

from stsc.data.base import tfds_base_dataset

ds = tfds.data_source("asl_dvs")
assert "train" in ds, ds
assert len(ds) == 1, ds

ds = ds["train"]

train_ds: tf.data.Dataset = tfds_base_dataset(
    "asl_dvs",
    "train",
    lambda el: el["events"]["time"],
)
num_examples = len(train_ds)

num_events = tf.zeros((), dtype=tf.int64)
total_dt = tf.zeros((), dtype=tf.int64)

num_events, total_dt = train_ds.reduce(
    (num_events, total_dt),
    lambda curr, t: (curr[0] + tf.shape(t, tf.int64)[0], curr[1] + t[-1] - t[0]),
)

print("num_events")
print(num_events)  # 2_837_416_554
print(num_events / num_examples)  # 28_148
print("total_dt")
print(total_dt)  # 11112895346
print(total_dt / num_examples)  # 110_246
print("num_examples")
print(num_examples)
