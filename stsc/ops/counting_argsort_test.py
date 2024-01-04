import unittest

import numpy as np
from keras import ops
from keras.src import testing

from stsc.ops import counting_argsort as ca_lib


class CountingArgsortTest(testing.TestCase):
    def test_counting_argsort(
        self, seed: int = 0, num_segments: int = 10, num_elements: int = 100
    ):
        rng = np.random.default_rng(seed=seed)
        segment_ids = rng.integers(0, num_segments, size=num_elements)
        row_lengths = np.bincount(segment_ids, minlength=num_segments)
        splits = np.pad(np.cumsum(row_lengths), [[1, 0]])

        actual = ca_lib.counting_argsort(
            ops.convert_to_tensor(segment_ids), ops.convert_to_tensor(splits)
        )
        gathered = ops.take(ops.convert_to_tensor(segment_ids), actual)
        sorted = ops.sort(ops.convert_to_tensor(segment_ids))
        self.assertAllEqual(gathered, sorted)


if __name__ == "__main__":
    unittest.main()
