import unittest

import numpy as np
from jk_utils.ops import ragged as ragged_ops
from keras import ops
from keras.src import testing

from stsc.ops import conv as conv_ops
from stsc.ops import conv_preprocessing as conv_preprocessing_ops
from stsc.ops import utils as utils_ops
from stsc.ops.counting_argsort import counting_argsort


class ConvTestCase(testing.TestCase):
    def test_1d_conv_consistent_with_ops_conv(
        self,
        seed: int = 0,
        grid_size: int = 11,
        kernel_size: int = 5,
        batch_size: int = 2,
        filters_in: int = 3,
        filters_out: int = 7,
        sparsity: float = 0.25,
    ):
        rng = np.random.default_rng(seed=seed)
        features = rng.uniform(size=(batch_size, grid_size, filters_in)).astype(
            "float32"
        )
        mask_prob = rng.uniform(size=(batch_size, grid_size))
        mask = mask_prob > sparsity
        kernel = rng.uniform(size=(kernel_size, filters_in, filters_out)).astype(
            "float32"
        )
        decay_rate = rng.uniform(size=filters_in).astype("float32")

        features[~mask] = 0

        expected = ops.conv(
            features * np.exp(-decay_rate),
            kernel,
            1,
            padding="same",
            data_format="channels_last",
        )

        batch_ids, pixel_ids = np.where(mask)
        batch_ids = ops.convert_to_tensor(batch_ids)
        batch_splits = ragged_ops.ids_to_splits(batch_ids, batch_size)
        pixel_ids = ops.convert_to_tensor(pixel_ids)
        sparse_features = ops.convert_to_tensor(features[mask])
        times = ops.zeros(batch_ids.shape, dtype="float32")
        segment_ids = batch_ids * grid_size + pixel_ids

        segment_ids_out = ops.arange(batch_size * grid_size, dtype="int32")
        batch_ids_out = segment_ids_out // grid_size
        batch_splits_out = ragged_ops.ids_to_splits(batch_ids_out, batch_size)
        pixel_ids_out = segment_ids_out % grid_size
        times_out = ops.ones(batch_ids_out.shape, dtype="float32")

        pad_left = (kernel_size - 1) // 2
        kernel_offsets = ops.arange(-pad_left, -pad_left + kernel_size, dtype="int32")

        conv_indices = conv_preprocessing_ops.get_predecessor_ids(
            pixel_ids + pad_left,
            times,
            batch_splits,
            pixel_ids_out + pad_left,
            times_out,
            batch_splits_out,
            kernel_offsets,
            grid_size + kernel_size - 1,
        )

        actual = conv_ops.conv(
            sparse_features,
            times,
            times_out,
            decay_rate,
            kernel,
            segment_ids,
            conv_indices,
        )
        actual = ops.convert_to_numpy(actual).reshape(
            batch_size, grid_size, filters_out
        )
        self.assertAllClose(actual, expected)

    def test_1d_stationary_conv_consistent_with_ops_conv(
        self,
        seed: int = 0,
        grid_size: int = 11,
        kernel_size: int = 5,
        batch_size: int = 2,
        filters_in: int = 3,
        filters_out: int = 7,
        sparsity: float = 0.5,
        # grid_size=5,
        # kernel_size=3,
        # batch_size=1,
        # filters_in: int = 1,
        # filters_out: int = 1,
        # sparsity: float = 1.0,
    ):
        rng = np.random.default_rng(seed=seed)
        features = rng.uniform(size=(batch_size, grid_size, filters_in)).astype(
            "float32"
        )
        mask_prob = rng.uniform(size=(batch_size, grid_size))
        mask = mask_prob > sparsity
        kernel = rng.uniform(size=(kernel_size, filters_in, filters_out)).astype(
            "float32"
        )
        decay_rate = rng.uniform(size=filters_in).astype("float32")

        features[~mask] = 0

        expected = ops.conv(
            features * np.exp(-decay_rate),
            kernel,
            1,
            padding="same",
            data_format="channels_last",
        )

        sparse_features = []
        pixel_ids = []
        times = []
        batch_splits = [0]
        for b in range(batch_size):
            (i,) = np.where(mask[b])
            sparse_features.append(features[b][i])
            sparse_features.append(np.zeros((grid_size, filters_in), dtype="float32"))
            pixel_ids.append(i.astype("int32"))
            pixel_ids.append(np.arange(grid_size, dtype="int32"))
            times.append(ops.zeros(i.shape, dtype="float32"))
            times.append(ops.ones((grid_size,), dtype="float32"))
            batch_splits.append(batch_splits[-1] + i.shape[0] + grid_size)
        total = batch_splits[-1]
        sparse_features = ops.convert_to_tensor(np.concatenate(sparse_features, axis=0))
        pixel_ids = ops.convert_to_tensor(np.concatenate(pixel_ids, axis=0))
        times = ops.convert_to_tensor(np.concatenate(times, axis=0))
        batch_splits = ops.convert_to_tensor(batch_splits)

        batch_ids = ragged_ops.splits_to_ids(batch_splits, total=total)
        segment_ids = batch_ids * grid_size + pixel_ids
        segment_splits = ragged_ops.ids_to_splits(segment_ids, grid_size * batch_size)

        perm = counting_argsort(segment_ids, segment_splits)
        perm_inv = ops.scatter(
            ops.expand_dims(perm, 1), ops.arange(total, dtype="int32"), (total,)
        )
        pad_left = (kernel_size - 1) // 2
        predecessor_ids = conv_preprocessing_ops.get_stationary_predecessor_ids(
            pixel_ids=pixel_ids + pad_left,
            batch_splits=batch_splits,
            kernel_offsets=ops.arange(
                -pad_left, -pad_left + kernel_size, dtype="int32"
            ),
            grid_size=grid_size + kernel_size,
        )
        predecessor_ids = ops.take(
            ops.pad(perm_inv, [[0, 1]], constant_values=perm_inv.shape[0]),
            predecessor_ids,
        )
        permuted_times = ops.take(times, perm, axis=0)
        permuted_segment_ids = ops.take(segment_ids, perm, axis=0)
        permuted_sparse_features = ops.take(sparse_features, perm, axis=0)
        self.assertAllClose(permuted_segment_ids, ops.sort(segment_ids), rtol=0, atol=0)

        actual = conv_ops.conv(
            permuted_sparse_features,
            permuted_times,
            # permuted_times,
            times,
            decay_rate,
            kernel,
            permuted_segment_ids,
            predecessor_ids,
        )
        actual = ops.convert_to_numpy(actual)[ops.convert_to_numpy(times == 1)]
        actual = actual.reshape(batch_size, grid_size, filters_out)
        expected = ops.convert_to_numpy(expected)
        self.assertAllClose(actual, expected)

    def test_1d_exclusive_conv_consistent_with_ops_conv(
        self,
        seed: int = 0,
        grid_size: int = 12,
        kernel_size: int = 3,
        batch_size: int = 2,
        filters_in: int = 5,
        filters_out: int = 7,
        sparsity: float = 0.5,
        # grid_size=6,
        # kernel_size=3,
        # batch_size=1,
        # filters_in: int = 1,
        # filters_out: int = 1,
        # sparsity: float = 1.0,
    ):
        assert grid_size % kernel_size == 0, (grid_size, kernel_size)
        rng = np.random.default_rng(seed=seed)
        features = rng.uniform(size=(batch_size, grid_size, filters_in)).astype(
            "float32"
        )
        mask_prob = rng.uniform(size=(batch_size, grid_size))
        mask = mask_prob > sparsity
        kernel = rng.uniform(size=(kernel_size, filters_in, filters_out)).astype(
            "float32"
        )
        decay_rate = rng.uniform(size=filters_in).astype("float32")

        features[~mask] = 0

        expected = ops.conv(
            features * np.exp(-decay_rate),
            kernel,
            kernel_size,
            padding="valid",
            data_format="channels_last",
        )
        grid_size_out = grid_size // kernel_size

        batch_ids_in, pixel_ids_in = np.where(mask)
        batch_splits_in = ragged_ops.ids_to_splits(batch_ids_in, batch_size)
        pixel_ids_in = ops.convert_to_tensor(pixel_ids_in, "int32")
        times_in = ops.zeros(pixel_ids_in.shape, dtype="float32")
        sparse_features = ops.convert_to_tensor(features[mask])

        pixel_ids_out = ops.tile(
            ops.arange(grid_size_out, dtype="int32"), (batch_size,)
        )
        times_out = ops.ones((batch_size * grid_size_out,), dtype="float32")
        batch_splits_out = ops.arange(
            0, grid_size_out * (batch_size + 1), grid_size_out, dtype="int32"
        )
        batch_ids_out = ragged_ops.splits_to_ids(
            batch_splits_out, total=grid_size_out * batch_size
        )

        segment_ids_out = batch_ids_out * grid_size_out + pixel_ids_out

        successor_ids = conv_preprocessing_ops.get_successor_ids(
            pixel_ids_in=pixel_ids_in // kernel_size,
            times_in=times_in,
            batch_splits_in=batch_splits_in,
            pixel_ids_out=pixel_ids_out,
            times_out=times_out,
            batch_splits_out=batch_splits_out,
            grid_size=grid_size_out,
        )
        dt = ops.take(ops.pad(times_out, [[0, 1]]), successor_ids, axis=0) - times_in
        successor_kernel_ids = successor_ids * kernel_size + pixel_ids_in % kernel_size

        actual = conv_ops.exclusive_conv(
            sparse_features,
            dt,
            times_out,
            ops.convert_to_tensor(decay_rate),
            ops.convert_to_tensor(kernel),
            successor_kernel_ids=successor_kernel_ids,
            segment_ids_out=segment_ids_out,
            indices_are_sorted=True,
        )
        actual = ops.convert_to_numpy(actual)
        actual = actual.reshape(batch_size, grid_size_out, filters_out)
        expected = ops.convert_to_numpy(expected)
        self.assertAllClose(actual, expected, rtol=1e-3)

    def test_simple_1d_one_hot_exclusive_conv_consistent_with_exclusive_conv(
        self,
        seed: int = 0,
        grid_size: int = 12,
        kernel_size: int = 3,
        batch_size: int = 2,
        filters_out: int = 7,
        sparsity: float = 0.5,
        # grid_size=6,
        # kernel_size=3,
        # batch_size=1,
        # filters_out: int = 1,
        # sparsity: float = 1.0,
    ):
        # same data as test_1d_exclusive_conv_consistent_with_ops_conv
        # i.e. grid in, grid out, dt = 1
        # note this means chronological order == segmented order
        assert grid_size % kernel_size == 0, (grid_size, kernel_size)
        rng = np.random.default_rng(seed=seed)
        features_bool = rng.uniform(size=(batch_size, grid_size)) > 0.5
        mask_prob = rng.uniform(size=(batch_size, grid_size))
        mask = mask_prob > sparsity
        kernel = rng.uniform(size=(kernel_size, 2, filters_out)).astype("float32")
        decay_rate = rng.uniform(size=2).astype("float32")

        grid_size_out = grid_size // kernel_size

        batch_ids_in, pixel_ids_in = np.where(mask)
        batch_splits_in = ragged_ops.ids_to_splits(batch_ids_in, batch_size)
        pixel_ids_in = ops.convert_to_tensor(pixel_ids_in, "int32")
        times_in = ops.zeros(pixel_ids_in.shape, dtype="float32")
        sparse_features_bool = ops.convert_to_tensor(features_bool[mask])
        sparse_features_float = ops.one_hot(
            ops.cast(sparse_features_bool, "int32"), 2, dtype="float32"
        )

        pixel_ids_out = ops.tile(
            ops.arange(grid_size_out, dtype="int32"), (batch_size,)
        )
        times_out = ops.ones((batch_size * grid_size_out,), dtype="float32")
        batch_splits_out = ops.arange(
            0, grid_size_out * (batch_size + 1), grid_size_out, dtype="int32"
        )
        batch_ids_out = ragged_ops.splits_to_ids(
            batch_splits_out, total=grid_size_out * batch_size
        )

        segment_ids_out = batch_ids_out * grid_size_out + pixel_ids_out

        successor_ids = conv_preprocessing_ops.get_successor_ids(
            pixel_ids_in=pixel_ids_in // kernel_size,
            times_in=times_in,
            batch_splits_in=batch_splits_in,
            pixel_ids_out=pixel_ids_out,
            times_out=times_out,
            batch_splits_out=batch_splits_out,
            grid_size=grid_size_out,
        )
        dt = ops.take(ops.pad(times_out, [[0, 1]]), successor_ids, axis=0) - times_in
        successor_kernel_ids = successor_ids * kernel_size + pixel_ids_in % kernel_size
        decay_rate = ops.convert_to_tensor(decay_rate)
        kernel = ops.convert_to_tensor(kernel)

        one_hot_encoded = conv_ops.exclusive_conv(
            sparse_features_float,
            dt,
            times_out,
            decay_rate,
            kernel,
            successor_kernel_ids=successor_kernel_ids,
            segment_ids_out=segment_ids_out,
            indices_are_sorted=False,
        )
        successor_kernel_channel_ids = successor_kernel_ids * 2 + ops.cast(
            sparse_features_bool, "int32"
        )
        one_hot_implementation = conv_ops.one_hot_exclusive_conv(
            dt,
            times_out,
            decay_rate,
            kernel,
            successor_kernel_channel_ids=successor_kernel_channel_ids,
            segment_ids_out=segment_ids_out,
            indices_are_sorted=False,
        )
        self.assertAllClose(one_hot_encoded, one_hot_implementation)

    def test_1d_one_hot_exclusive_conv_consistent_with_exclusive_conv(
        self,
        seed: int = 0,
        grid_size: int = 12,
        kernel_size: int = 3,
        batch_size: int = 2,
        filters_out: int = 7,
        size_in: int = 131,
        size_out: int = 71,
        # grid_size=6,
        # kernel_size=3,
        # batch_size=1,
        # filters_out: int = 1,
        # size_in=17,
        # size_out=11,
    ):
        # random data
        assert grid_size % kernel_size == 0, (grid_size, kernel_size)
        grid_size_out = grid_size // kernel_size
        rng = np.random.default_rng(seed=seed)
        times_in = rng.uniform(size=(size_in,)).astype("float32")
        times_in.sort()
        pixel_ids_in = (rng.uniform(size=(size_in,)) * grid_size).astype("int32")
        batch_ids_in = (rng.uniform(size=(size_in,)) * batch_size).astype("int32")
        perm_in = np.argsort(batch_ids_in)
        times_in = times_in[perm_in]
        pixel_ids_in = pixel_ids_in[perm_in]
        batch_ids_in = batch_ids_in[perm_in]
        features_bool = rng.uniform(size=(size_in,)) > 0.5
        features_bool = features_bool[perm_in]
        del perm_in

        times_out = rng.uniform(size=(size_out,)).astype("float32")
        times_out.sort()
        pixel_ids_out = (rng.uniform(size=(size_out,)) * grid_size_out).astype("int32")
        batch_ids_out = (rng.uniform(size=(size_out,)) * batch_size).astype("int32")
        perm_out = np.argsort(batch_ids_out)
        times_out = times_out[perm_out]
        pixel_ids_out = pixel_ids_out[perm_out]
        batch_ids_out = batch_ids_out[perm_out]
        del perm_out

        decay_rate = rng.uniform(size=(2,))
        kernel = rng.uniform(size=(kernel_size, 2, filters_out)).astype("float32")

        times_in = ops.convert_to_tensor(times_in, "float32")
        pixel_ids_in = ops.convert_to_tensor(pixel_ids_in)
        batch_ids_in = ops.convert_to_tensor(batch_ids_in)
        features_bool = ops.convert_to_tensor(features_bool)

        times_out = ops.convert_to_tensor(times_out, "float32")
        pixel_ids_out = ops.convert_to_tensor(pixel_ids_out)
        batch_ids_out = ops.convert_to_tensor(batch_ids_out)

        decay_rate = ops.convert_to_tensor(decay_rate, "float32")
        kernel = ops.convert_to_tensor(kernel, "float32")

        segment_ids_out = batch_ids_out * grid_size_out + pixel_ids_out
        segment_splits_out = ragged_ops.ids_to_splits(
            segment_ids_out, batch_size * grid_size_out
        )
        seg_perm_out = counting_argsort(segment_ids_out, segment_splits_out)
        seg_perm_inv_out = utils_ops.inverse_perm(seg_perm_out)

        batch_splits_in = ragged_ops.ids_to_splits(batch_ids_in, batch_size)
        batch_splits_out = ragged_ops.ids_to_splits(batch_ids_out, batch_size)

        successor_ids = conv_preprocessing_ops.get_successor_ids(
            pixel_ids_in=pixel_ids_in // kernel_size,
            times_in=times_in,
            batch_splits_in=batch_splits_in,
            pixel_ids_out=pixel_ids_out,
            times_out=times_out,
            batch_splits_out=batch_splits_out,
            grid_size=grid_size_out,
        )
        dt = ops.take(ops.pad(times_out, [[0, 1]]), successor_ids, axis=0) - times_in
        reindexed_successor_ids = ops.take(
            ops.pad(seg_perm_inv_out, [[0, 1]], constant_values=size_out),
            successor_ids,
            axis=0,
        )
        reindexed_successor_kernel_ids = (
            reindexed_successor_ids * kernel_size + pixel_ids_in % kernel_size
        )
        permuted_times_out = ops.take(times_out, seg_perm_out)
        permuted_segment_ids_out = ops.take(segment_ids_out, seg_perm_out, axis=0)
        self.assertAllClose(
            permuted_segment_ids_out, ops.sort(permuted_segment_ids_out), atol=0, rtol=0
        )

        features_float = ops.one_hot(ops.cast(features_bool, "int32"), 2)
        one_hot_encoded = conv_ops.exclusive_conv(
            features_float,
            dt,
            permuted_times_out,
            decay_rate,
            kernel,
            reindexed_successor_kernel_ids,
            segment_ids_out,
            indices_are_sorted=False,
        )
        reindexed_successor_kernel_channel_ids = (
            reindexed_successor_kernel_ids * 2 + ops.cast(features_bool, "int32")
        )
        one_hot_implementation = conv_ops.one_hot_exclusive_conv(
            dt,
            permuted_times_out,
            decay_rate,
            kernel,
            reindexed_successor_kernel_channel_ids,
            segment_ids_out,
            indices_are_sorted=False,
        )
        self.assertAllClose(one_hot_encoded, one_hot_implementation)

    def test_1d_exclusive_conv_consistent_with_conv(
        self,
        seed: int = 0,
        grid_size: int = 12,
        kernel_size: int = 3,
        batch_size: int = 2,
        filters_in: int = 5,
        filters_out: int = 7,
        size_in: int = 131,
        size_out: int = 71,
        # grid_size=4,
        # kernel_size=2,
        # batch_size=2,
        # filters_in=1,
        # filters_out: int = 1,
        # size_in=9,
        # size_out=7,
    ):
        # random data
        assert grid_size % kernel_size == 0, (grid_size, kernel_size)
        grid_size_out = grid_size // kernel_size
        rng = np.random.default_rng(seed=seed)
        times_in = rng.uniform(size=(size_in,)).astype("float32")
        times_in.sort()
        pixel_ids_in = (rng.uniform(size=(size_in,)) * grid_size).astype("int32")
        batch_ids_in = (rng.uniform(size=(size_in,)) * batch_size).astype("int32")
        batch_ids_in.sort()
        features = rng.uniform(size=(size_in, filters_in)).astype("float32")

        times_out = rng.uniform(size=(size_out,)).astype("float32")
        times_out.sort()
        pixel_ids_out = (rng.uniform(size=(size_out,)) * grid_size_out).astype("int32")
        batch_ids_out = (rng.uniform(size=(size_out,)) * batch_size).astype("int32")
        batch_ids_out.sort()

        decay_rate = rng.uniform(size=(filters_in,))
        kernel = rng.uniform(size=(kernel_size, filters_in, filters_out)).astype(
            "float32"
        )

        times_in = ops.convert_to_tensor(times_in, "float32")
        pixel_ids_in = ops.convert_to_tensor(pixel_ids_in, "int32")
        batch_ids_in = ops.convert_to_tensor(batch_ids_in, "int32")
        features = ops.convert_to_tensor(features, "float32")

        times_out = ops.convert_to_tensor(times_out, "float32")
        pixel_ids_out = ops.convert_to_tensor(pixel_ids_out, "int32")
        batch_ids_out = ops.convert_to_tensor(batch_ids_out, "int32")

        decay_rate = ops.convert_to_tensor(decay_rate, "float32")
        kernel = ops.convert_to_tensor(kernel, "float32")

        segment_ids_out = batch_ids_out * grid_size_out + pixel_ids_out
        segment_splits_out = ragged_ops.ids_to_splits(
            segment_ids_out, batch_size * grid_size_out
        )
        seg_perm_out = counting_argsort(segment_ids_out, segment_splits_out)
        seg_perm_inv_out = utils_ops.inverse_perm(seg_perm_out)

        batch_splits_in = ragged_ops.ids_to_splits(batch_ids_in, batch_size)
        batch_splits_out = ragged_ops.ids_to_splits(batch_ids_out, batch_size)

        successor_ids = conv_preprocessing_ops.get_successor_ids(
            pixel_ids_in=pixel_ids_in // kernel_size,
            times_in=times_in,
            batch_splits_in=batch_splits_in,
            pixel_ids_out=pixel_ids_out,
            times_out=times_out,
            batch_splits_out=batch_splits_out,
            grid_size=grid_size_out,
        )

        dt = ops.take(ops.pad(times_out, [[0, 1]]), successor_ids, axis=0) - times_in
        successor_ids = ops.take(
            ops.pad(seg_perm_inv_out, [[0, 1]], constant_values=size_out),
            successor_ids,
            axis=0,
        )
        permuted_times_out = ops.take(times_out, seg_perm_out)
        successor_kernel_ids = successor_ids * kernel_size + pixel_ids_in % kernel_size
        permuted_segment_ids_out = ops.take(segment_ids_out, seg_perm_out, axis=0)
        self.assertAllClose(
            permuted_segment_ids_out, ops.sort(permuted_segment_ids_out), atol=0, rtol=0
        )
        exclusive_impl = conv_ops.exclusive_conv(
            features=features,
            dt=dt,
            times_out=permuted_times_out,
            decay_rate=decay_rate,
            kernel=kernel,
            successor_kernel_ids=successor_kernel_ids,
            segment_ids_out=permuted_segment_ids_out,
            indices_are_sorted=False,
        )
        # back to chronological order
        exclusive_impl = ops.take(exclusive_impl, seg_perm_inv_out, axis=0)

        kernel_offsets = ops.arange(kernel_size, dtype="int32")
        predecessor_ids = conv_preprocessing_ops.get_predecessor_ids(
            pixel_ids_in=pixel_ids_in,
            times_in=times_in,
            batch_splits_in=batch_splits_in,
            pixel_ids_out=pixel_ids_out * kernel_size,
            times_out=times_out,
            batch_splits_out=batch_splits_out,
            kernel_offsets=kernel_offsets,
            grid_size=grid_size,
        )
        segment_ids_in = batch_ids_in * grid_size + pixel_ids_in
        segment_splits = ragged_ops.ids_to_splits(
            segment_ids_in, grid_size * batch_size
        )
        perm_in = counting_argsort(segment_ids_in, segment_splits)
        perm_inv_in = utils_ops.inverse_perm(perm_in)
        predecessor_ids = ops.take(
            ops.pad(perm_inv_in, [[0, 1]], constant_values=size_in),
            predecessor_ids,
            axis=0,
        )
        permuted_segment_ids_in = ops.take(segment_ids_in, perm_in, axis=0)
        self.assertAllClose(permuted_segment_ids_in, ops.sort(segment_ids_in))

        conv_impl = conv_ops.conv(
            ops.take(features, perm_in, axis=0),
            ops.take(times_in, perm_in, axis=0),
            times_out,
            decay_rate,
            kernel,
            segment_ids=permuted_segment_ids_in,
            predecessor_ids=predecessor_ids,
        )
        self.assertAllClose(exclusive_impl, conv_impl)


if __name__ == "__main__":
    unittest.main()
    # ConvTestCase().test_1d_conv_consistent_with_ops_conv()
    # ConvTestCase().test_1d_exclusive_conv_consistent_with_ops_conv()
    # ConvTestCase().test_1d_one_hot_exclusive_conv_consistent_with_exclusive_conv()
    # ConvTestCase().test_1d_exclusive_conv_consistent_with_conv()
