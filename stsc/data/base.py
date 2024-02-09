import typing as tp

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

T = tp.TypeVar("T")


def _shuffled_sequence(sequence: tp.Sequence, rng: np.random.Generator):
    perm = np.arange(len(sequence))
    rng.shuffle(perm)
    for p in perm:
        yield sequence[p]


def _spawn_rng(rng: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(rng.integers(0, np.iinfo(np.int64).max))


class Shuffler(tp.Iterable[T]):
    def __init__(self, sequence: tp.Sequence[T], seed: int = 0):
        self.sequence = sequence
        self.rng = np.random.default_rng(seed)

    def generate(self):
        return _shuffled_sequence(self.sequence, self.rng)

    def __iter__(self):
        return iter(self.generate())


class Sampler(tp.Iterable[T]):
    def __init__(self, sequence: tp.Sequence[T], seed: int = 0):
        self.sequence = sequence
        self.rng = np.random.default_rng(seed)

    def generate(self):
        sampled = self.rng.integers(0, len(self.sequence), size=len(self.sequence))
        for s in sampled:
            yield self.sequence[s]

    def __iter__(self):
        return iter(self.generate())


class InfiniteShuffler(tp.Iterable[T]):
    def __init__(self, sequence: tp.Sequence[T], seed: int = 0):
        self.sequence = sequence
        self.rng = np.random.default_rng(seed)

    def generate(self):
        rng = _spawn_rng(self.rng)
        while True:
            yield from _shuffled_sequence(self.sequence, rng)

    def __iter__(self):
        return iter(self.generate())


class InfiniteSampler(tp.Iterable[T]):
    def __init__(self, sequence: tp.Iterable[T], seed: int = 0):
        self.sequence = sequence
        self.rng = np.random.default_rng(seed)

    def generate(self):
        rng = _spawn_rng(self.rng)
        size = len(self.sequence)
        while True:
            yield self.sequence[rng.integers(0, size)]

    def __iter__(self):
        return iter(self.generate())


def _as_bytes_dataset(
    source: tp.Sequence[bytes],
    shuffle: bool,
    infinite: bool,
    replace: bool,
    seed: int = 0,
) -> tf.data.Dataset:
    if shuffle:
        if infinite:
            if replace:
                gen = InfiniteSampler(source, seed).generate
            else:
                gen = InfiniteShuffler(source, seed).generate
        else:
            if replace:
                gen = Sampler(source, seed).generate
            else:
                gen = Shuffler(source, seed).generate
    else:
        if infinite:

            def gen():
                while True:
                    yield from source

        else:

            def gen():
                return (source[i] for i in range(len(source)))

    dataset = tf.data.Dataset.from_generator(
        gen, output_signature=tf.TensorSpec((), dtype=tf.string)
    )
    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(
            tf.data.INFINITE_CARDINALITY if infinite else len(source)
        )
    )
    return dataset


def tfds_cardinality(name: str, split: str) -> int:
    return len(tfds.data_source(name, split=split))


def tfds_base_dataset(
    name: str,
    split: str,
    map_fun: tp.Optional[tp.Callable] = None,
    shuffle: bool = False,
    infinite: bool = False,
    replace: bool = False,
    seed: int = 0,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    deterministic: bool = True,
) -> tf.data.Dataset:
    source = tfds.data_source(name, split=split)
    ds = _as_bytes_dataset(
        source.data_source,
        shuffle=shuffle,
        infinite=infinite,
        replace=replace,
        seed=seed,
    )

    def deserialize_and_map(b):
        example = source.dataset_info.features.deserialize_example(b)
        if map_fun:
            return map_fun(example)
        return example

    ds = ds.map(
        deserialize_and_map,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
    )
    return ds
