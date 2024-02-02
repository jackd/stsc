import typing as tp
import unittest

import numpy as np
from absl.testing import parameterized
from keras import ops
from keras.src import testing

from stsc import components


def get_flash_stream_data(
    batch_size: int, grid_shape: tp.Sequence[int], t=0, mask=None
):
    grid_size = components.prod(*grid_shape)
    coords = np.tile(
        np.stack(
            np.meshgrid(
                *(np.arange(g, dtype=np.int32) for g in grid_shape), indexing="ij"
            ),
            axis=-1,
        ).reshape(1, -1, len(grid_shape)),
        (batch_size, 1, 1),
    )
    times = np.full((batch_size, grid_size), t, dtype=np.float32)
    if mask is None:
        times = times.reshape(-1)
        batch_splits = np.arange(0, grid_size * batch_size + 1, grid_size)
    else:
        coords = coords[mask]
        times = times[mask]
        batch_lengths = np.count_nonzero(mask, axis=1)
        batch_splits = np.pad(np.cumsum(batch_lengths), [[1, 0]])
    return (
        ops.convert_to_tensor(coords.reshape(-1, len(grid_shape)), "int32"),
        ops.convert_to_tensor(times, "float32"),
        ops.convert_to_tensor(batch_splits, "int32"),
    )


class ComponentsTest(testing.TestCase, parameterized.TestCase):
    @parameterized.product(filters_out=(None, 7), kernel_size=(3, 4))
    def test_1d_stationary_conv_consistent_with_rect_conv(
        self,
        seed: int = 0,
        grid_size: int = 11,
        filters_in: int = 5,
        filters_out: int | None = None,
        batch_size: int = 2,
        sparsity: float = 0.4,
        kernel_size: int = 3,
    ):
        dt = 1.0
        rng = np.random.default_rng(seed)
        features = rng.uniform(size=(batch_size, 2, grid_size, filters_in)).astype(
            np.float32
        )
        kernel_shape_ = (kernel_size, filters_in)
        if filters_out is not None:
            kernel_shape_ += (filters_out,)
        kernel = rng.uniform(size=kernel_shape_).astype(np.float32)
        bias = rng.uniform(size=filters_out or filters_in).astype(np.float32)
        decay_rate = rng.uniform(size=filters_in).astype(np.float32)

        features[:, 1] = 0
        times = np.zeros((batch_size, 2, grid_size), np.float32)
        times[:, 1] = dt
        coords = np.tile(
            np.arange(grid_size).reshape(1, 1, grid_size), (batch_size, 2, 1)
        )
        coords = np.expand_dims(coords, -1)

        mask = np.ones((batch_size, 2, grid_size), dtype=bool)
        mask[:, 0] = rng.uniform(size=(batch_size, grid_size)) >= sparsity
        features[~mask] = 0
        features_masked = features[mask]
        times = times[mask]
        coords = coords[mask]

        batch_lengths = np.count_nonzero(mask, axis=(1, 2))
        batch_splits = np.pad(np.cumsum(batch_lengths), [[1, 0]])

        stream_in = components.input_stream(
            ops.convert_to_tensor(features_masked, "float32"),
            ops.convert_to_tensor(coords, "int32"),
            ops.convert_to_tensor(times, "float32"),
            ops.convert_to_tensor(batch_splits, "int32"),
            (grid_size,),
        )
        stream_out = stream_in.stationary_conv(
            filters_out, kernel_size, normalize=False
        )
        stream_out.force_chronological()

        op = stream_out.source
        op.build()
        op.layer.kernel.assign(kernel)
        op.layer.bias.assign(bias)
        op.layer.decay_rate.assign(decay_rate)
        actual = stream_out.compute_features()
        actual = stream_out.order.unpermute(actual)
        actual = actual[stream_out.stream.times == dt]
        actual = ops.reshape(actual, (batch_size, grid_size, filters_out or filters_in))

        layer = stream_out.source.layer
        decay_rate = layer.decay_rate_activation(layer.decay_rate)

        conv_features = ops.exp(-dt * decay_rate) * features[:, 0]
        if filters_out is None:
            expected = ops.depthwise_conv(
                conv_features, ops.expand_dims(kernel, -1), padding="same"
            )
        else:
            expected = ops.conv(conv_features, kernel, padding="same")
        expected = expected + bias
        self.assertAllClose(actual, expected)

    @parameterized.product(filters_out=(None, 7), kernel_shape=((3, 4), (4, 5)))
    def test_2d_stationary_conv_consistent_with_rect_conv(
        self,
        seed: int = 0,
        grid_shape: tp.Sequence[int] = (7, 5),
        filters_in: int = 11,
        filters_out: int | None = None,
        batch_size: int = 2,
        sparsity: float = 0.4,
        kernel_shape: tp.Sequence[int] = (3, 3),
    ):
        dt = 1.0
        rng = np.random.default_rng(seed)
        features = rng.uniform(size=(batch_size, 2, *grid_shape, filters_in)).astype(
            np.float32
        )
        kernel_shape_ = (components.prod(*kernel_shape), filters_in)
        if filters_out is not None:
            kernel_shape_ += (filters_out,)
        kernel = rng.uniform(size=kernel_shape_).astype(np.float32)
        bias = rng.uniform(size=filters_out or filters_in).astype(np.float32)
        decay_rate = rng.uniform(size=filters_in).astype(np.float32)

        features[:, 1] = 0
        times = np.zeros((batch_size, 2, *grid_shape), np.float32)
        times[:, 1] = dt
        coords = np.tile(
            np.stack(
                np.meshgrid(*(np.arange(s) for s in grid_shape), indexing="ij"), axis=-1
            ).reshape(1, 1, *grid_shape, 2),
            (batch_size, 2, 1, 1, 1),
        )

        mask = np.ones((batch_size, 2, *grid_shape), dtype=bool)
        mask[:, 0] = rng.uniform(size=(batch_size, *grid_shape)) >= sparsity
        features[~mask] = 0
        features_masked = features[mask]
        times = times[mask]
        coords = coords[mask]

        batch_lengths = np.count_nonzero(mask, axis=(1, 2, 3))
        batch_splits = np.pad(np.cumsum(batch_lengths), [[1, 0]])

        stream_in = components.input_stream(
            ops.convert_to_tensor(features_masked, "float32"),
            ops.convert_to_tensor(coords, "int32"),
            ops.convert_to_tensor(times, "float32"),
            ops.convert_to_tensor(batch_splits, "int32"),
            grid_shape,
        )
        stream_out = stream_in.stationary_conv(
            filters_out, kernel_shape, normalize=False
        )
        stream_out.force_chronological()

        op = stream_out.source
        op.build()
        op.layer.kernel.assign(kernel)
        op.layer.bias.assign(bias)
        op.layer.decay_rate.assign(decay_rate)
        actual = stream_out.compute_features()
        actual = stream_out.order.unpermute(actual)
        actual = actual[stream_out.stream.times == dt]
        actual = ops.reshape(
            actual, (batch_size, *grid_shape, filters_out or filters_in)
        )

        layer = stream_out.source.layer
        decay_rate = layer.decay_rate_activation(layer.decay_rate)

        conv_features = ops.exp(-dt * decay_rate) * features[:, 0]
        if filters_out is None:
            expected = ops.depthwise_conv(
                conv_features,
                kernel.reshape(*kernel_shape, filters_in, 1),
                padding="same",
            )
        else:
            expected = ops.conv(
                conv_features,
                kernel.reshape(*kernel_shape, filters_in, filters_out),
                padding="same",
            )
        expected = expected + bias
        self.assertAllClose(actual, expected)

    @parameterized.product(
        filters_out=(None, 7),
        kernel_size=(3, 4),
    )
    def test_1d_exclusive_conv_consistent_with_rect_conv(
        self,
        seed: int = 0,
        grid_size: int = 12,
        filters_in: int = 5,
        filters_out: int | None = 2,
        batch_size: int = 2,
        kernel_size: int = 3,
        sparsity: float = 0.2,
    ):
        dt = 1.0
        rng = np.random.default_rng(seed)
        features = rng.uniform(size=(batch_size, grid_size, filters_in)).astype(
            np.float32
        )
        kernel_shape_ = (kernel_size, filters_in)
        if filters_out is not None:
            kernel_shape_ += (filters_out,)
        kernel = rng.uniform(size=kernel_shape_).astype(np.float32)
        bias = rng.uniform(size=filters_out or filters_in).astype(np.float32)
        decay_rate = rng.uniform(size=filters_in).astype(np.float32)

        mask = rng.uniform(size=(batch_size, grid_size)) >= sparsity

        features[~mask] = 0
        coords, times, batch_splits = get_flash_stream_data(
            batch_size, (grid_size,), mask=mask
        )

        stream = components.input_stream(
            ops.convert_to_tensor(features[mask], "float32"),
            coords,
            times,
            batch_splits,
            (grid_size,),
        )

        grid_size_out = grid_size // kernel_size
        coords_out, times_out, batch_splits_out = get_flash_stream_data(
            batch_size, (grid_size_out,), dt
        )

        out_data = components.StreamData(
            coords_out, times_out, batch_splits_out, (grid_size // kernel_size,)
        )
        stream_out = components.ExclusiveConv(
            stream, out_data, filters_out, normalize=False
        ).outputs
        stream_out.source.build()
        layer = stream_out.source.layer
        layer.kernel.assign(kernel)
        layer.bias.assign(bias)
        layer.decay_rate.assign(decay_rate)
        actual = stream_out.compute_features()
        actual = stream_out.order.unpermute(actual)

        decay_rate = layer.decay_rate_activation(layer.decay_rate)
        if filters_out is None:
            expected = ops.depthwise_conv(
                features * ops.exp(-decay_rate * dt),
                ops.expand_dims(kernel, -1),
                kernel_size,
            )
        else:
            expected = ops.conv(
                features * ops.exp(-decay_rate * dt), kernel, kernel_size
            )
        expected = ops.reshape(expected, (-1, expected.shape[-1])) + bias
        self.assertAllClose(actual, expected)

    @parameterized.product(
        filters_out=(None, 7),
        kernel_shape=((3, 3), (3, 4), (4, 3)),
    )
    def test_2d_exclusive_conv_consistent_with_rect_conv(
        self,
        seed: int = 0,
        grid_shape: int = (12, 12),
        filters_in: int = 5,
        filters_out: int | None = 2,
        batch_size: int = 2,
        kernel_shape: int = (3, 4),
        sparsity: float = 0.2,
    ):
        dt = 1.0
        rng = np.random.default_rng(seed)
        grid_size = components.prod(*grid_shape)
        kernel_size = components.prod(*kernel_shape)
        features = rng.uniform(size=(batch_size, grid_size, filters_in)).astype(
            np.float32
        )
        kernel_shape_ = (kernel_size, filters_in)
        if filters_out is not None:
            kernel_shape_ += (filters_out,)
        kernel = rng.uniform(size=kernel_shape_).astype(np.float32)
        bias = rng.uniform(size=filters_out or filters_in).astype(np.float32)
        decay_rate = rng.uniform(size=filters_in).astype(np.float32)

        mask = rng.uniform(size=(batch_size, grid_size)) >= sparsity

        features[~mask] = 0
        coords, times, batch_splits = get_flash_stream_data(
            batch_size, grid_shape, mask=mask
        )

        stream = components.input_stream(
            ops.convert_to_tensor(features[mask], "float32"),
            coords,
            times,
            batch_splits,
            grid_shape,
        )

        grid_shape_out = tuple(g // k for g, k in zip(grid_shape, kernel_shape))
        coords_out, times_out, batch_splits_out = get_flash_stream_data(
            batch_size, grid_shape_out, dt
        )

        out_data = components.StreamData(
            coords_out, times_out, batch_splits_out, grid_shape_out
        )
        stream_out = components.ExclusiveConv(
            stream, out_data, filters_out, normalize=False
        ).outputs
        stream_out.source.build()
        layer = stream_out.source.layer
        layer.kernel.assign(kernel)
        layer.bias.assign(bias)
        layer.decay_rate.assign(decay_rate)
        actual = stream_out.compute_features()
        actual = stream_out.order.unpermute(actual)

        decay_rate = layer.decay_rate_activation(layer.decay_rate)
        features = features.reshape(batch_size, *grid_shape, filters_in)
        if filters_out is None:
            expected = ops.depthwise_conv(
                features * ops.exp(-decay_rate * dt),
                kernel.reshape(*kernel_shape, filters_in, 1),
                kernel_shape,
            )
        else:
            expected = ops.conv(
                features * ops.exp(-decay_rate * dt),
                kernel.reshape(*kernel_shape, filters_in, filters_out),
                kernel_shape,
            )
        expected = ops.reshape(expected, (-1, expected.shape[-1])) + bias
        self.assertAllClose(actual, expected)

    @parameterized.product(
        kernel_size=(3, 4),
    )
    def test_1d_simple_pooling_consistent_with_rect_pool(
        self,
        seed: int = 0,
        grid_size: int = 12,
        filters: int = 5,
        batch_size: int = 2,
        kernel_size: int = 3,
    ):
        rng = np.random.default_rng(seed)
        features = rng.uniform(size=(batch_size, grid_size, filters)).astype(np.float32)

        coords, times, batch_splits = get_flash_stream_data(batch_size, (grid_size,))

        stream = components.input_stream(
            ops.convert_to_tensor(features.reshape(-1, filters), "float32"),
            coords,
            times,
            batch_splits,
            (grid_size,),
        )

        grid_size_out = grid_size // kernel_size
        coords_out, times_out, batch_splits_out = get_flash_stream_data(
            batch_size,
            (grid_size_out,),
        )

        out_data = components.StreamData(
            coords_out, times_out, batch_splits_out, (grid_size // kernel_size,)
        )
        stream_out = components.SimpleExclusivePooling(stream, out_data).outputs
        stream_out.force_chronological()
        stream_out.source.build()
        actual = stream_out.compute_features()
        actual = stream_out.order.unpermute(actual)

        expected = ops.average_pool(features, kernel_size, kernel_size)
        expected = ops.reshape(expected, (-1, expected.shape[-1]))
        self.assertAllClose(actual, expected)

    @parameterized.product(
        kernel_shape=((3, 3), (3, 4), (4, 3)),
    )
    def test_2d_simple_pooling_consistent_with_rect_pool(
        self,
        seed: int = 0,
        grid_shape: tp.Sequence[int] = (12, 12),
        filters: int = 5,
        batch_size: int = 2,
        kernel_shape: tp.Sequence[int] = (3, 3),
    ):
        grid_size = components.prod(*grid_shape)
        rng = np.random.default_rng(seed)
        features = rng.uniform(size=(batch_size, grid_size, filters)).astype(np.float32)

        coords, times, batch_splits = get_flash_stream_data(batch_size, grid_shape)

        stream = components.input_stream(
            ops.convert_to_tensor(features.reshape(-1, filters), "float32"),
            coords,
            times,
            batch_splits,
            grid_shape,
        )

        grid_shape_out = tuple(g // k for g, k in zip(grid_shape, kernel_shape))
        coords_out, times_out, batch_splits_out = get_flash_stream_data(
            batch_size, grid_shape_out
        )

        out_data = components.StreamData(
            coords_out, times_out, batch_splits_out, grid_shape_out
        )
        stream_out = components.SimpleExclusivePooling(stream, out_data).outputs
        stream_out.force_chronological()
        stream_out.source.build()
        actual = stream_out.compute_features()
        actual = stream_out.order.unpermute(actual)

        expected = ops.average_pool(
            features.reshape(batch_size, *grid_shape, filters),
            kernel_shape,
            kernel_shape,
        )
        expected = ops.reshape(expected, (-1, expected.shape[-1]))
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    unittest.main()
    # ComponentsTest().test_2d_exclusive_conv_consistent_with_rect_conv()
    # ComponentsTest().test_2d_simple_pooling_consistent_with_rect_pool()
    # print("Passed")
