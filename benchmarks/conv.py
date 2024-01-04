import functools

import google_benchmark as bm
import keras
from absl import flags

from stsc.ops import conv as conv_ops

flags.DEFINE_integer("e", int(2**18), "number of events")
flags.DEFINE_integer("g", 128 * 128, "grid_size")
flags.DEFINE_integer("f_in", 32, "number of input filters")
flags.DEFINE_integer("f_out", 32, "number of output filters (conv only)")
flags.DEFINE_integer("k", 25, "kernel size (product over all dimensions)")


def get_args(depthwise: bool = False, one_hot: bool = False):
    FLAGS = flags.FLAGS
    E = FLAGS.e
    f_in = FLAGS.f_in
    f_out = FLAGS.f_out
    k = FLAGS.k
    grid_size = FLAGS.g

    times_in = keras.random.uniform((E,))
    times_out = keras.random.uniform((E,))
    decay_rate = keras.random.uniform((f_in,))
    kernel = keras.random.normal((k, f_in) if depthwise else (k, f_in, f_out))
    if one_hot:
        segment_filter_ids = keras.random.randint(
            (E,), 0, grid_size * f_in, dtype="int32"
        )
        segment_filter_ids = keras.ops.sort(segment_filter_ids)
        one_hot_predecessor_ids = keras.random.randint(
            (E, k, f_in), minval=0, maxval=E, dtype="int32"
        )
        return (
            times_in,
            times_out,
            decay_rate,
            kernel,
            segment_filter_ids,
            one_hot_predecessor_ids,
        )
    else:
        segment_ids = keras.random.randint((E,), 0, grid_size, dtype="int32")
        segment_ids = keras.ops.sort(segment_ids)
        features = keras.random.normal((E, f_in))
        conv_indices = keras.random.randint((E, k), minval=0, maxval=E, dtype="int32")
        return (
            features,
            times_in,
            times_out,
            decay_rate,
            kernel,
            segment_ids,
            conv_indices,
        )


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
    args = get_args(depthwise=False)
    run_benchmark(state, conv_ops.conv, args)


@bm.register
def depthwise_conv(state):
    args = get_args(depthwise=True)
    run_benchmark(state, conv_ops.depthwise_conv, args)


@bm.register
def one_hot_conv(state):
    args = get_args(one_hot=True, depthwise=False)
    run_benchmark(state, conv_ops.one_hot_conv, args)


@bm.register
def one_hot_depthwise_conv(state):
    args = get_args(one_hot=True, depthwise=True)
    run_benchmark(state, conv_ops.one_hot_depthwise_conv, args)


if __name__ == "__main__":
    bm.main()
