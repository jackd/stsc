import events_tfds.events.dvs_gesture  # pylint:disable=unused-import
import tensorflow as tf

from stsc.data.base import tfds_base_dataset

train_ds: tf.data.Dataset = tfds_base_dataset(
    "dvs_gesture",
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
print(num_events)  # 425_993_622
print(num_events / num_examples)  # 362_239
print("total_dt")
print(total_dt)  # 7_601_938_965
print(total_dt / num_examples)  # 6_464_233
print("num_examples")
print(num_examples)  # 1_176
