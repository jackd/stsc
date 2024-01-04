import unittest

import numpy as np
from jk_utils.ops import ragged as ragged_ops
from keras import ops
from keras.src import testing

from stsc.ops import conv_preprocessing as cp


class ConvPreprocessingTest(testing.TestCase):
    def test_get_successor_ids(self):
        pixel_ids_in = [0, 0, 1, 0, 2, 1, 3, 3]
        times_in = [1, 2, 3, 4, 6, 7, 0, 0]
        batch_splits_in = [0, 6]

        pixel_ids_out = [0, 0, 1, 0, 3]
        times_out = [2, 3, 5, 8, 0]
        batch_splits_out = [0, 4]

        grid_size = 3

        actual = cp.get_successor_ids(
            pixel_ids_in=np.array(pixel_ids_in, "int32"),
            times_in=np.array(times_in, "int32"),
            batch_splits_in=np.array(batch_splits_in, "int32"),
            pixel_ids_out=np.array(pixel_ids_out, "int32"),
            times_out=np.array(times_out, "int32"),
            batch_splits_out=np.array(batch_splits_out, "int32"),
            grid_size=grid_size,
        )
        n_out = len(times_out)
        expected = np.array([0, 0, 2, 3, n_out, n_out, n_out, n_out])
        self.assertAllEqual(actual, expected)

    def test_get_stationary_predecessor_ids_consistent_with_get_predecessor_ids(
        self,
        seed: int = 0,
        num_events: int = 71,
        grid_size: int = 11,
        kernel_size: int = 5,
        batch_size: int = 2,
        # num_events: int = 17,
        # grid_size: int = 3,
        # kernel_size: int = 2,
        # batch_size: int = 1,
    ):
        rng = np.random.default_rng(seed)

        batch_ids = (rng.uniform(size=num_events) * batch_size).astype("int32")
        batch_ids.sort()
        times = rng.uniform(size=num_events)
        times.sort()
        pixel_ids = rng.uniform(size=num_events) * grid_size

        pixel_ids = ops.convert_to_tensor(pixel_ids, "int32")
        times = ops.convert_to_tensor(times, "float32")
        batch_splits = ragged_ops.ids_to_splits(
            ops.convert_to_tensor(batch_ids, "int32"), batch_size
        )

        pad_left = (kernel_size - 1) // 2
        kernel_offsets = ops.arange(-pad_left, -pad_left + kernel_size)
        pixel_ids = pixel_ids + pad_left
        padded_grid_size = grid_size + kernel_size - 1
        actual = cp.get_stationary_predecessor_ids(
            pixel_ids,
            batch_splits,
            kernel_offsets,
            padded_grid_size,
        )
        expected = cp.get_predecessor_ids(
            pixel_ids,
            times,
            batch_splits,
            pixel_ids,
            times,
            batch_splits,
            kernel_offsets,
            padded_grid_size,
        )
        self.assertAllClose(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
    # ConvPreprocessingTest().test_get_stationary_predecessor_ids_consistent_with_get_predecessor_ids()
    # ConvPreprocessingTest().test_get_successor_ids()
