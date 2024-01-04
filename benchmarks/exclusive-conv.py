import functools

import google_benchmark as bm
import keras
from absl import flags

from stsc.ops import conv as conv_ops

flags.DEFINE_integer("e_in", int(2**18) * 32 - 1, "number of events")
flags.DEFINE_integer("g", 128 * 128, "grid_size")
flags.DEFINE_integer("f_in", 2, "number of input filters")
flags.DEFINE_integer("f_out", 32, "number of output filters")
flags.DEFINE_integer("k", 16, "kernel size (product over all dimensions)")
flags.DEFINE_boolean("sorted", False, "use indices_are_sorted implementation")

FLAGS = flags.FLAGS


def run_benchmark(state, func, args, **kwargs):
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        func = tf.function(functools.partial(func, **kwargs), jit_compile=True)
        func(*args)
        while state:
            func(*args)
    elif keras.backend.backend() == "jax":
        import jax

        func = jax.jit(functools.partial(func, **kwargs))
        jax.block_until_ready(func(*args))
        while state:
            jax.block_until_ready(func(*args))
    else:
        raise Exception(f"backend {keras.backend.backend()} not supported")


@bm.register
def conv(state):
    E_in = FLAGS.e_in
    f_in = FLAGS.f_in
    f_out = FLAGS.f_out
    k = FLAGS.k
    grid_size = FLAGS.g

    E_out = E_in // k

    features = keras.random.normal((E_in, f_in))
    times_in = keras.random.uniform((E_in,))
    times_out = keras.random.uniform((E_out,))
    decay_rate = keras.random.uniform((f_in,))
    kernel = keras.random.normal((k, f_in, f_out))
    conv_indices = keras.random.randint(
        (E_out, k), minval=0, maxval=E_in, dtype="int32"
    )
    segment_ids = keras.random.randint((E_in,), 0, grid_size, dtype="int32")
    segment_ids = keras.ops.sort(segment_ids)

    args = (
        features,
        times_in,
        times_out,
        decay_rate,
        kernel,
        segment_ids,
        conv_indices,
    )

    run_benchmark(state, conv_ops.conv, args)


@bm.register
def one_hot_conv(state):
    E_in = FLAGS.e_in
    f_in = FLAGS.f_in
    f_out = FLAGS.f_out
    k = FLAGS.k
    grid_size = FLAGS.g

    E_out = E_in // k

    times_in = keras.random.uniform((E_in,))
    times_out = keras.random.uniform((E_out,))
    decay_rate = keras.random.uniform((f_in,))
    kernel = keras.random.normal((k, f_in, f_out))
    segment_filter_ids = keras.random.randint(
        (E_in,), 0, grid_size * f_in, dtype="int32"
    )
    segment_filter_ids = keras.ops.sort(segment_filter_ids)
    one_hot_predecessor_ids = keras.random.randint(
        (E_out, k, f_in), minval=0, maxval=E_in, dtype="int32"
    )

    args = (
        times_in,
        times_out,
        decay_rate,
        kernel,
        segment_filter_ids,
        one_hot_predecessor_ids,
    )

    run_benchmark(state, conv_ops.one_hot_conv, args)


@bm.register
def exclusive_conv(state):
    E_in = FLAGS.e_in
    f_in = FLAGS.f_in
    f_out = FLAGS.f_out
    k = FLAGS.k
    grid_size = FLAGS.g

    E_out = E_in // k

    features = keras.random.normal((E_in, f_in))
    dt = keras.random.uniform((E_in,))
    times_out = keras.random.uniform((E_out,))
    decay_rate = keras.random.uniform((f_in,))
    kernel = keras.random.normal((k, f_in, f_out))
    exclusive_conv_indices = keras.random.randint(
        (E_in,), minval=0, maxval=(E_out * k), dtype="int32"
    )
    exclusive_conv_indices = keras.ops.sort(exclusive_conv_indices)
    segment_ids_out = keras.random.randint((E_out,), 0, grid_size, dtype="int32")
    segment_ids_out = keras.ops.sort(segment_ids_out)

    args = (
        features,
        dt,
        times_out,
        decay_rate,
        kernel,
        segment_ids_out,
        exclusive_conv_indices,
    )

    run_benchmark(state, conv_ops.exclusive_conv, args, indices_are_sorted=FLAGS.sorted)


@bm.register
def one_hot_exclusive_conv(state):
    E_in = FLAGS.e_in
    f_in = FLAGS.f_in
    f_out = FLAGS.f_out
    k = FLAGS.k
    grid_size = FLAGS.g

    E_out = E_in // k

    dt = keras.random.uniform((E_in,))
    times_out = keras.random.uniform((E_out,))
    decay_rate = keras.random.uniform((f_in,))
    kernel = keras.random.normal((k, f_in, f_out))
    exclusive_conv_indices = keras.random.randint(
        (E_in,), minval=0, maxval=(E_out * k * f_in), dtype="int32"
    )
    exclusive_conv_indices = keras.ops.sort(exclusive_conv_indices)
    segment_ids_out = keras.random.randint((E_out,), 0, grid_size, dtype="int32")
    segment_ids_out = keras.ops.sort(segment_ids_out)

    args = (
        dt,
        times_out,
        decay_rate,
        kernel,
        segment_ids_out,
        exclusive_conv_indices,
    )

    run_benchmark(
        state, conv_ops.one_hot_exclusive_conv, args, indices_are_sorted=FLAGS.sorted
    )


if __name__ == "__main__":
    bm.main()
