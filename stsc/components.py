import functools
import typing as tp

import tree
from jk_utils import asserts
from jk_utils.layers.masked_batch_norm import MaskedBatchNormalization
from jk_utils.ops import ragged as ragged_ops
from jk_utils.ops import segment_ops
from keras import KerasTensor, Operation, layers, ops
from keras.backend import standardize_dtype

from .backend.backend_tensor import BackendTensor
from .layers import conv as conv_layers
from .layers import patches as patch_layers
from .layers import pool as pool_layers
from .ops import conv_preprocessing as conv_preprocessing_ops
from .ops import sampling as sampling_ops
from .ops import utils as utils_ops
from .ops.counting_argsort import counting_argsort

Tensor = BackendTensor | KerasTensor

T = tp.TypeVar("T")


class OneHotTensor(tp.NamedTuple):
    channel_ids: Tensor
    num_channels: int


def prod(*args: tp.Sequence[T], initial: T = 1) -> T:
    return functools.reduce(lambda a, b: a * b, args, initial)


def keyed_cache(key_func: tp.Callable):
    def ret_func(wrappeded):
        cache = {}

        def wrapped(*args, **kwargs):
            key = key_func(*args, **kwargs)
            if key not in cache:
                cache[key] = wrappeded(*args, **kwargs)
            return cache[key]

        functools.update_wrapper(wrapped, wrappeded)
        return wrapped

    return ret_func


class StreamData:
    """Chronological event stream data."""

    def __init__(
        self,
        coords: Tensor,
        times: Tensor,
        batch_splits: Tensor,
        grid_shape: tp.Sequence[int],
    ):
        asserts.assert_has_rank(coords, 2, "coords")
        assert standardize_dtype(coords.dtype) == "int32", coords.dtype
        asserts.assert_has_rank(times, 1, "times")
        assert standardize_dtype(times.dtype) == "float32", times.dtype
        asserts.assert_has_rank(batch_splits, 1, "batch_splits")
        assert standardize_dtype(batch_splits.dtype) == "int32", batch_splits.dtype

        assert coords.shape[0] == times.shape[0], (coords.shape, times.shape)
        self._coords = coords
        self._times = times
        self._grid_shape = tuple(grid_shape)
        assert len(self._grid_shape) == self._coords.shape[1], (
            self._grid_shape,
            self._coords.shape,
        )
        self._batch_splits = batch_splits

    @property
    def coords(self) -> Tensor:
        return self._coords

    @property
    def times(self) -> Tensor:
        return self._times

    @property
    def batch_splits(self) -> Tensor:
        return self._batch_splits

    @property
    def grid_shape(self) -> tp.Sequence[int]:
        return self._grid_shape

    @property
    def grid_shape_tensor(self) -> Tensor:
        return ops.convert_to_tensor(self.grid_shape, "int32")

    @property
    def spatial_dims(self) -> int:
        return len(self.grid_shape)

    @property
    def grid_size(self) -> int:
        return prod(*self.grid_shape)

    @property
    @functools.cache
    def batch_lengths(self) -> Tensor:
        return ragged_ops.splits_to_lengths(self.batch_splits)

    @property
    @functools.cache
    def batch_ids(self) -> Tensor:
        return ragged_ops.splits_to_ids(self.batch_splits, total=self.num_events)

    @property
    @functools.cache
    def pixel_ids(self) -> Tensor:
        return utils_ops.ravel_multi_index(self.coords, self.grid_shape, axis=1)

    @property
    @functools.cache
    def segment_ids(self) -> Tensor:
        return ops.minimum(
            self.batch_ids * self.grid_size + self.pixel_ids, self.num_segments
        )

    @property
    @functools.cache
    def segment_splits(self) -> Tensor:
        return ragged_ops.ids_to_splits(self.segment_ids, self.num_segments + 1)

    @property
    def num_segments(self) -> int:
        return self.batch_size * self.grid_size

    @property
    def batch_size(self) -> int:
        return self.batch_splits.shape[0] - 1

    @property
    def num_events(self) -> int:
        return self.times.shape[0]

    @property
    @functools.cache
    def num_valid_events(self) -> Tensor:
        return self.batch_splits[-1]

    @property
    @functools.cache
    def event_mask(self):
        return ops.less(ops.arange(self.num_events), self.num_valid_events)

    @property
    @functools.cache
    def batch_mask(self):
        return ops.greater(self.batch_lengths, ops.zeros_like(self.batch_lengths))

    @functools.cache
    def get_contiguous_segments_order(self) -> "Order":
        if self.grid_size == 1:
            return Chronological(self.num_events, True)
        perm = counting_argsort(self.segment_ids, self.segment_splits)
        return Permuted(perm, contiguous_segments=True)

    def _take(self, sample_ids: Tensor, batch_splits: Tensor) -> "StreamData":
        return StreamData(
            coords=ops.take(ops.pad(self.coords, [[0, 1], [0, 0]]), sample_ids, axis=0),
            times=ops.take(ops.pad(self.times, [[0, 1]]), sample_ids, axis=0),
            batch_splits=batch_splits,
            grid_shape=self.grid_shape,
        )

    def pool(self, strides: int | tp.Sequence[int]):
        if isinstance(strides, int):
            strides = (strides,) * self.spatial_dims
        else:
            assert len(strides) == self.spatial_dims
        if all(s == 1 for s in strides):
            return self
        return StreamData(
            coords=self.coords // ops.convert_to_tensor(strides, "int32"),
            times=self.times,
            batch_splits=self.batch_splits,
            grid_shape=tuple(g // s for g, s in zip(self.grid_shape, strides)),
        )

    def throttled_sample(self, sample_rate: int, min_dt: float = 0.0) -> "StreamData":
        sample_ids, batch_splits = sampling_ops.throttled_sample(
            self.pixel_ids,
            self.times,
            self.batch_splits,
            sample_rate=sample_rate,
            min_dt=min_dt,
            grid_size=self.grid_size,
        )
        return self._take(sample_ids, batch_splits)

    def pool_and_throttled_sample(
        self,
        strides: int | tp.Sequence[int],
        sample_rate: int | None = None,
        min_dt: float = 0.0,
    ) -> "StreamData":
        if isinstance(strides, int):
            strides = (strides,) * self.spatial_dims
        else:
            strides = tuple(strides)
            assert len(strides) == self.spatial_dims
        if sample_rate is None:
            sample_rate = prod(*strides)
        pooled = self.pool(strides)
        return pooled.throttled_sample(sample_rate=sample_rate, min_dt=min_dt)

    def pad(self, padding):
        assert all(len(p) == 2 for p in padding)
        assert all(isinstance(l, int) and isinstance(r, int) for l, r in padding)
        pad_left = tuple(p[0] for p in padding)
        return StreamData(
            coords=self.coords + ops.convert_to_tensor(pad_left, "int32"),
            times=self.times,
            batch_splits=self.batch_splits,
            grid_shape=tuple(g + l + r for g, (l, r) in zip(padding)),
        )

    def pool_features(self, features: Tensor, reduction: str = "mean"):
        if reduction in ("mean", "sum"):
            total = segment_ops.segment_sum(
                features, self.segment_ids, self.num_segments, indices_are_sorted=True
            )
            if reduction == "mean":
                batch_lengths = ops.cast(self.batch_lengths, total.dtype)
                return total / ops.expand_dims(
                    ops.where(
                        self.batch_mask, batch_lengths, ops.ones_like(batch_lengths)
                    ),
                    axis=1,
                )

            return total
        if reduction == "max":
            return segment_ops.segment_max(
                features, self.segment_ids, self.num_segments, indices_are_sorted=True
            )
        raise ValueError(
            f"Unsupported reduction '{reduction}'. Must be one of 'max', 'mean', 'sum'"
        )


class Order:
    def __init__(self, contiguous_segments: bool):
        self._contiguous_segments = contiguous_segments

    def permute(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Abstract method")

    def unpermute(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Abstract method")

    def reindex(self, x: Tensor, pad: bool = True) -> Tensor:
        raise NotImplementedError("Abstract method")

    @property
    def permutation(self) -> Tensor | None:
        raise NotImplementedError("Abstract property")

    @property
    def permutation_inverse(self) -> Tensor | None:
        raise NotImplementedError("Abstract property")

    @property
    def size(self) -> int:
        raise NotImplementedError("Abstract property")

    @property
    def contiguous_segments(self) -> bool:
        return self._contiguous_segments

    def is_compatible_with(self, data: StreamData) -> bool:
        size = self.size
        return size is None or data.num_events == size


class Permuted(Order):
    def __init__(self, permutation: Tensor, contiguous_segments: bool):
        self._permutation = permutation
        super().__init__(contiguous_segments=contiguous_segments)

    @keyed_cache(lambda self, x: (self, id(x)))
    def permute(self, x: Tensor) -> Tensor:
        assert x.shape[0] == self.size, (x.shape, self.size)
        return ops.take(x, self._permutation, axis=0)

    @keyed_cache(lambda self, x: (self, id(x)))
    def unpermute(self, x: Tensor) -> Tensor:
        assert x.shape[0] == self.size, (x.shape, self.size)
        return ops.take(x, self.permutation_inverse, axis=0)

    @keyed_cache(lambda self, x, pad=True: (self, id(x), pad))
    def reindex(self, x: Tensor, pad: bool = True) -> Tensor:
        perm = self.permutation_inverse
        if pad:
            perm = ops.pad(perm, [[0, 1]], constant_values=perm.shape[0])
        return ops.take(perm, x, axis=0)

    @property
    def permutation(self) -> Tensor:
        return self._permutation

    @property
    @functools.cache
    def permutation_inverse(self) -> Tensor:
        return utils_ops.inverse_perm(self.permutation)

    @property
    def size(self) -> int:
        return self._permutation.shape[0]


class Chronological(Order):
    def __init__(self, size: int, contiguous_segments: bool = False):
        self._size = size
        self._contiguous_segments = contiguous_segments
        self._arange = None

    def permute(self, x: Tensor) -> Tensor:
        assert x.shape[0] == self._size, (x.shape, self._size)
        return x

    def unpermute(self, x: Tensor) -> Tensor:
        assert x.shape[0] == self._size, (x.shape, self._size)
        return x

    def reindex(self, x: Tensor) -> Tensor:
        return x

    @property
    def permutation(self) -> Tensor:
        if self._arange is None:
            self._arange = ops.arange(self.size, dtype="int32")
        return self._arange

    @property
    def permutation_inverse(self) -> Tensor:
        if self._arange is None:
            self._arange = ops.arange(self.size, dtype="int32")
        return self._arange

    @property
    def size(self) -> int:
        return self._size


class StreamNode:
    def __init__(
        self,
        stream: StreamData,
        source: "OpNode",
        node_index: int,
        num_channels: int,
        order: tp.Optional[Order] = None,
    ):
        self._stream = stream
        self._source = source
        self._order = order
        if order is None:
            self._order_callbacks = []
        self._node_index = node_index
        self._num_channels = num_channels

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def stream(self) -> StreamData:
        return self._stream

    @property
    def node_index(self) -> int:
        """Index of `self.source.flat_outputs`."""
        return self._node_index

    def on_order_set(self, callback: tp.Callable[[Order], None]):
        if self._order is None:
            self._order_callbacks.append(callback)
        else:
            callback(self._order)

    @property
    def order(self) -> tp.Optional[Order]:
        return self._order

    @order.setter
    def order(self, order):
        if order is self._order:
            return
        if self._order is not None:
            raise RuntimeError("order is already set - cannot set again")
        if not isinstance(order, Order):
            raise ValueError(f"order must be an Order, got {order}")
        self._order = order
        for callback in self._order_callbacks:
            callback(order)
        del self._order_callbacks

    def compute_features(self):
        return tree.flatten(self.source.compute_features())[self.node_index]

    @property
    def source(self) -> "OpNode":
        return self._source

    def map_features(self, map_func: tp.Callable[[Tensor], Tensor]) -> "StreamNode":
        return FeatureMap(self, map_func).outputs

    def masked_batch_norm(self, **batch_norm_kwargs) -> "StreamNode":
        return MaskedBatchNormalizationNode(self, **batch_norm_kwargs).outputs

    def __add__(self, other: "StreamNode") -> "StreamNode":
        return Add(self, other).outputs

    def get_exclusive_patches(
        self,
        kernel_shape: int | tp.Sequence[int],
        sample_rate: int | None = None,
        min_dt: float = 0.0,
        **layer_kwargs,
    ) -> "StreamNode":
        stream_out = self.stream.pool_and_throttled_sample(
            kernel_shape, sample_rate=sample_rate, min_dt=min_dt
        )

        if isinstance(self.source, InputOp) and isinstance(
            self.source.input_features, OneHotTensor
        ):
            raise NotImplementedError
        else:
            op = ExclusivePatchExtractor(self, stream_out, **layer_kwargs)
        return op.outputs

    def exclusive_conv(
        self,
        filters: int | None,
        kernel_shape: int | tp.Sequence[int],
        sample_rate: int | None = None,
        min_dt: float = 0.0,
        **layer_kwargs,
    ) -> "StreamNode":
        stream_out = self.stream.pool_and_throttled_sample(
            kernel_shape, sample_rate=sample_rate, min_dt=min_dt
        )
        if isinstance(self.source, InputOp) and isinstance(
            self.source.input_features, OneHotTensor
        ):
            op = OneHotExclusiveConv(self, stream_out, filters=filters, **layer_kwargs)
        else:
            op = ExclusiveConv(self, stream_out, filters=filters, **layer_kwargs)
        return op.outputs

    def exclusive_pool(
        self,
        strides: int | tp.Sequence[int],
        sample_rate: int | None = None,
        min_dt: float = 0.0,
        reduction: str = "mean",
        simple: bool = False,
        **layer_kwargs,
    ) -> "StreamNode":
        stream_out = self.stream.pool_and_throttled_sample(
            strides, sample_rate=sample_rate, min_dt=min_dt
        )
        if isinstance(self.source, InputOp) and isinstance(
            self.source.input_features, OneHotTensor
        ):
            assert not simple, "not implemented"
            op = OneHotExclusivePooling(
                self, stream_out, reduction=reduction, **layer_kwargs
            )
        else:
            if simple:
                op = SimpleExclusivePooling(
                    self, stream_out, reduction=reduction, **layer_kwargs
                )
            else:
                op = ExclusivePooling(
                    self, stream_out, reduction=reduction, **layer_kwargs
                )
        return op.outputs

    def stationary_conv(
        self,
        filters: int | None,
        kernel_shape: int | tp.Sequence[int],
        **layer_kwargs,
    ) -> "StreamNode":
        op = StationaryConv(
            self, filters=filters, kernel_shape=kernel_shape, **layer_kwargs
        )
        return op.outputs

    def force_chronological(self) -> None:
        if self.order is None:
            self.order = Chronological(
                self.stream.num_events, self.stream.grid_size == 1
            )
        else:
            assert isinstance(self.order, Chronological), self.order


StreamNodeTree = (
    tp.Sequence["StreamNodeTree"] | tp.Mapping[str, "StreamNodeTree"] | StreamNode
)

TensorTree = tp.Sequence["TensorTree"] | tp.Mapping[str, "TensorTree"] | Tensor


class OpNode:
    def __init__(self, inputs: StreamNodeTree, outputs: StreamNodeTree):
        self._inputs = inputs
        self._outputs = outputs
        assert all(isinstance(i, StreamNode) for i in self.flat_inputs)
        assert all(isinstance(o, StreamNode) for o in self.flat_outputs)
        assert all(o.source is self for o in self.flat_outputs)
        assert all(o.node_index == i for i, o in enumerate(self.flat_outputs))

    @property
    def inputs(self) -> StreamNodeTree:
        return self._inputs

    @property
    def outputs(self) -> StreamNodeTree:
        return self._outputs

    @property
    def flat_inputs(self) -> tp.Sequence[StreamNode]:
        return tree.flatten(self._inputs)

    @property
    def flat_outputs(self) -> tp.Sequence[StreamNode]:
        return tree.flatten(self._outputs)

    @functools.cache
    def compute_features(self) -> TensorTree:
        for inp in tree.flatten(self.inputs):
            inp.source.build()
        return self._compute_features()

    def _compute_features(self) -> TensorTree:
        raise NotImplementedError("Abstract method")

    def build(self):
        pass


def _sync_orders(*streams: StreamNode):
    if len(streams) < 2:
        return

    def sync_func(order):
        for stream in streams:
            stream.order = order

    for stream in streams:
        stream.on_order_set(sync_func)


class Add(OpNode):
    def __init__(self, *inputs: StreamNode):
        assert len(inputs) > 0
        assert all(isinstance(i, StreamNode) for i in inputs), inputs
        stream = inputs[0].stream
        assert all(i.stream is stream for i in inputs[1:])
        num_channels = inputs[0].num_channels
        assert all(i.num_channels == num_channels for i in inputs[1:])
        output = StreamNode(stream, self, 0, num_channels)
        super().__init__(inputs, output)
        _sync_orders(*inputs, output)

    def _compute_features(self) -> Tensor:
        return sum([i.compute_features() for i in self.flat_inputs])


class FeatureMap(OpNode):
    def __init__(self, node: StreamNode, map_func: tp.Callable[[Tensor], Tensor]):
        self._map_func = map_func
        i = KerasTensor((node.stream.num_events, node.num_channels), dtype="float32")
        if isinstance(map_func, Operation):
            o = map_func.compute_output_spec(i)
        else:
            o = map_func(o)

        assert o.shape[0] == node.stream.num_events
        output = StreamNode(node.stream, self, 0, o.shape[1])
        super().__init__(node, output)
        _sync_orders(node, output)

    def _compute_features(self) -> Tensor:
        return self._map_func(self.inputs.compute_features())


class MaskedBatchNormalizationNode(OpNode):
    def __init__(self, node: StreamNode, **batch_norm_kwargs):
        output = StreamNode(node.stream, self, 0, node.num_channels)
        self._batch_norm_kwargs = batch_norm_kwargs
        self._layer = MaskedBatchNormalization(**self._batch_norm_kwargs)
        super().__init__(node, output)
        _sync_orders(node, output)

    @property
    def layer(self) -> layers.Layer:
        return self._layer

    def _compute_features(self) -> Tensor:
        x = self.inputs.compute_features()
        return self._layer(x, mask=self.inputs.stream.event_mask)


class InputOp(OpNode):
    def __init__(
        self,
        features: Tensor | OneHotTensor,
        stream: StreamData,
        order: tp.Optional[Order] = None,
    ):
        if isinstance(features, Tensor) and standardize_dtype(features.dtype) == "bool":
            features = OneHotTensor(ops.cast(features, "int32"), 2)
        if isinstance(features, OneHotTensor):
            asserts.assert_has_rank(features.channel_ids, 1)
            assert standardize_dtype(features.channel_ids.dtype) == "int32"
            num_channels = features.num_channels
        else:
            assert isinstance(features, Tensor), features
            asserts.assert_has_rank(features, 2)
            assert "float" in standardize_dtype(features.dtype)
            num_channels = features.shape[1]
        self._input_features = features
        node = StreamNode(stream, self, 0, num_channels, order=order)
        super().__init__((), node)

    @property
    def input_features(self) -> Tensor | OneHotTensor:
        return self._input_features

    def _compute_features(self):
        features = self.input_features
        if isinstance(features, Tensor):
            return self.outputs.order.permute(features)
        else:
            assert isinstance(features, OneHotTensor)
            indices, num_channels = features
            return OneHotTensor(self.outputs.order.permute(indices), num_channels)


def input_stream(
    features: Tensor | OneHotTensor,
    coords: Tensor,
    times: Tensor,
    batch_splits: Tensor,
    grid_shape: tp.Iterable[int],
) -> StreamNode:
    stream = StreamData(coords, times, batch_splits, grid_shape)
    return InputOp(features, stream).outputs


class OneHotExclusivePatchExtractor(OpNode):
    def __init__(self, node: StreamNode, stream_out: StreamData, num_channels: int):
        node_out = StreamNode(
            stream_out,
            self,
            0,
            num_channels,
            order=stream_out.get_contiguous_segments_order(),
        )
        super().__init__(node, node_out)
        self._built = False
        stream_in = node.stream
        assert all(
            i % o == 0 for i, o in zip(stream_in.grid_shape, stream_out.grid_shape)
        ), (stream_in.grid_shape, stream_out.grid_shape)
        self._strides = tuple(
            i // o for i, o in zip(stream_in.grid_shape, stream_out.grid_shape)
        )
        self._layer = self._create_layer()

    @property
    def strides(self) -> tp.Sequence[int]:
        return self._strides

    @property
    def kernel_size(self) -> int:
        return prod(*self.strides)

    def build(self):
        if self._built:
            return
        self._built = True

        node_in: StreamNode = self.inputs
        node_out: StreamNode = self.outputs
        stream_in = node_in.stream
        stream_out = node_out.stream
        kernel_size = self.kernel_size
        input_is_already_ordered = node_in.order is not None
        is_input = isinstance(node_in.source, InputOp)
        if is_input:
            channel_ids, num_channels = node_in.source.input_features
            if input_is_already_ordered:
                channel_ids = node_in.order.permute(channel_ids)
        else:
            channel_ids, num_channels = node_in.compute_features()
        assert num_channels == self.inputs.num_channels, (
            num_channels,
            self.inputs.num_channels,
        )

        strides_tensor = ops.convert_to_tensor(self.strides, "int32")
        # reindexed_successor_ids = conv_preprocessing_ops.get_permuted_successor_ids(
        #     pixel_ids_in=utils_ops.ravel_multi_index(
        #         stream_in.coords // strides_tensor, stream_out.grid_shape_tensor, axis=1
        #     ),
        #     times_in=stream_in.times,
        #     batch_splits_in=stream_in.batch_splits,
        #     pixel_ids_out=stream_out.pixel_ids,
        #     perm_in=perm_in_inverse,
        #     times_out=stream_out.times,
        #     batch_splits_out=stream_out.batch_splits,
        #     grid_size=stream_out.grid_size,
        #     perm_out=node_out.order.permutation_inverse,
        # )
        successor_ids = conv_preprocessing_ops.get_successor_ids(
            pixel_ids_in=utils_ops.ravel_multi_index(
                stream_in.coords // strides_tensor, stream_out.grid_shape_tensor, axis=1
            ),
            times_in=stream_in.times,
            batch_splits_in=stream_in.batch_splits,
            pixel_ids_out=stream_out.pixel_ids,
            times_out=stream_out.times,
            batch_splits_out=stream_out.batch_splits,
            grid_size=stream_out.grid_size,
        )
        reindexed_successor_ids = node_out.order.reindex(successor_ids)
        if input_is_already_ordered:
            reindexed_successor_ids = node_in.order.permute(reindexed_successor_ids)

        input_kernel_ids = utils_ops.ravel_multi_index(
            stream_in.coords % strides_tensor, strides_tensor, axis=1
        )
        permuted_times_in = stream_in.times
        if input_is_already_ordered:
            input_kernel_ids = node_in.order.permute(input_kernel_ids)
            permuted_times_in = node_in.order.permute(stream_in.times)
        permuted_times_out = node_out.order.permute(stream_out.times)
        dt = (
            ops.take(ops.pad(permuted_times_out, [[0, 1]]), reindexed_successor_ids)
            - permuted_times_in
        )
        dt = ops.where(
            reindexed_successor_ids == stream_out.num_events, ops.zeros_like(dt), dt
        )

        reindexed_successor_kernel_ids = (
            reindexed_successor_ids * kernel_size + input_kernel_ids
        )
        reindexed_successor_kernel_channel_ids = (
            reindexed_successor_kernel_ids * num_channels + channel_ids
        )
        reindexed_successor_kernel_channel_ids = ops.minimum(
            reindexed_successor_kernel_channel_ids,
            ops.full_like(
                reindexed_successor_kernel_channel_ids,
                node_out.stream.num_events * kernel_size * num_channels,
            ),
        )

        if input_is_already_ordered:
            self._indices_are_sorted = False
        else:
            splits = ragged_ops.ids_to_splits(
                reindexed_successor_kernel_channel_ids,
                node_out.stream.num_events * kernel_size * num_channels + 1,
            )
            node_in.order = Permuted(
                counting_argsort(reindexed_successor_kernel_channel_ids, splits),
                contiguous_segments=False,
            )
            self._indices_are_sorted = True
            dt = node_in.order.permute(dt)
            reindexed_successor_kernel_channel_ids = node_in.order.permute(
                reindexed_successor_kernel_channel_ids
            )
        self._dt = dt
        self._successor_kernel_channel_ids = reindexed_successor_kernel_channel_ids
        self._times_out = permuted_times_out
        self._segment_ids_out = node_out.order.permute(stream_out.segment_ids)
        self._layer.build(
            dt_shape=self._dt.shape,
            times_out_shape=self._times_out.shape,
            successor_kernel_channel_ids_shape=self._successor_kernel_channel_ids.shape,
            segment_ids_out_shape=self._segment_ids_out.shape,
        )

    def _create_layer(self) -> layers.Layer:
        raise NotImplementedError("Abstract method")

    def _compute_features(self):
        self.build()
        return self._layer(
            dt=self._dt,
            times_out=self._times_out,
            successor_kernel_channel_ids=self._successor_kernel_channel_ids,
            segment_ids_out=self._segment_ids_out,
            indices_are_sorted=self._indices_are_sorted,
        )


class OneHotExclusiveConv(OneHotExclusivePatchExtractor):
    def __init__(
        self,
        node: StreamNode,
        stream_out: StreamData,
        filters: int | None,
        **layer_kwargs,
    ):
        self._filters = filters
        self._layer_kwargs = layer_kwargs
        super().__init__(
            node, stream_out, node.num_channels if filters is None else filters
        )

    def _create_layer(self) -> layers.Layer:
        if self._filters is None:
            return conv_layers.OneHotExclusiveDepthwiseConv(
                filters=self.inputs.num_channels,
                kernel_size=self.kernel_size,
                **self._layer_kwargs,
            )

        return conv_layers.OneHotExclusiveConv(
            filters_in=self.inputs.num_channels,
            filters_out=self._filters,
            kernel_size=self.kernel_size,
            **self._layer_kwargs,
        )


class OneHotExclusivePooling(OneHotExclusivePatchExtractor):
    def __init__(
        self, node: StreamNode, stream_out: StreamData, reduction: str, **layer_kwargs
    ):
        self._reduction = reduction
        self._layer_kwargs = layer_kwargs
        super().__init__(node, stream_out, node.num_channels)

    def _create_layer(self) -> layers.Layer:
        return pool_layers.OneHotExclusivePooling(
            self.inputs.num_channels,
            self.kernel_size,
            reduction=self._reduction,
            **self._layer_kwargs,
        )


class SimpleExclusivePooling(OpNode):
    def __init__(
        self, node: StreamNode, stream_out: StreamData, reduction: str = "mean"
    ):
        node_out = StreamNode(stream_out, self, 0, node.num_channels)
        super().__init__(node, node_out)
        self._built = False
        stream_in = node.stream
        assert all(
            i % o == 0 for i, o in zip(stream_in.grid_shape, stream_out.grid_shape)
        ), (stream_in.grid_shape, stream_out.grid_shape)
        self._strides = tuple(
            i // o for i, o in zip(stream_in.grid_shape, stream_out.grid_shape)
        )
        self._reduction = reduction

    @property
    def reduction(self) -> str:
        return self._reduction

    @property
    def strides(self) -> tp.Sequence[int]:
        return self._strides

    def _compute_lengths(self, reindexed_successor_ids):
        return ragged_ops.ids_to_lengths(
            reindexed_successor_ids, self.outputs.stream.num_events + 1
        )

    def build(self):
        if self._built:
            return
        self._built = True
        node_in: StreamNode = self.inputs
        node_out: StreamNode = self.outputs
        stream_in = node_in.stream
        stream_out = node_out.stream
        assert node_out.order is not None  # can accomodate any order
        input_is_already_ordered = node_in.order is not None
        strides_tensor = ops.convert_to_tensor(self.strides, "int32")
        successor_ids = conv_preprocessing_ops.get_successor_ids(
            pixel_ids_in=utils_ops.ravel_multi_index(
                stream_in.coords // strides_tensor, stream_out.grid_shape_tensor, axis=1
            ),
            times_in=stream_in.times,
            batch_splits_in=stream_in.batch_splits,
            pixel_ids_out=stream_out.pixel_ids,
            times_out=stream_out.times,
            batch_splits_out=stream_out.batch_splits,
            grid_size=stream_out.grid_size,
        )
        reindexed_successor_ids = node_out.order.reindex(successor_ids)
        if self.reduction == "mean":
            self._lengths = self._compute_lengths(reindexed_successor_ids)

        if input_is_already_ordered:
            self._indices_are_sorted = False
        else:
            lengths = self._compute_lengths(reindexed_successor_ids)
            splits = ragged_ops.lengths_to_splits(lengths)
            perm = counting_argsort(reindexed_successor_ids, splits)
            node_in.order = Permuted(perm, contiguous_segments=False)
            self._indices_are_sorted = True
        self._successor_ids = node_in.order.permute(reindexed_successor_ids)

    def _compute_features(self):
        input_features = self.inputs.compute_features()
        num_segments = self.outputs.stream.num_events
        features = {
            "mean": segment_ops.segment_sum,
            "sum": segment_ops.segment_sum,
            "max": segment_ops.segment_max,
        }[self.reduction](
            input_features,
            self._successor_ids,
            num_segments=num_segments,
            indices_are_sorted=self._indices_are_sorted,
        )
        if self.reduction == "mean":
            lengths = self._lengths
            lengths = ops.slice(lengths, (0,), (num_segments,))
            lengths = ops.maximum(lengths, ops.ones_like(lengths))
            features = features / ops.expand_dims(ops.cast(lengths, features.dtype), -1)
        elif self.reduction == "max":
            # hacky work-around to get accurate gradients
            gathered_features = ops.take(
                ops.pad(features, [[0, 1], [0, 0]]), self._successor_ids
            )
            mask = input_features == gathered_features
            mask_count = segment_ops.segment_sum(
                ops.cast(mask, input_features.dtype),
                self._successor_ids,
                num_segments=num_segments + 1,
            )
            mask_count = ops.take(mask_count, self._successor_ids)
            factor = ops.where(
                ops.greater(mask_count, 0), 1 / mask_count, ops.zeros_like(mask_count)
            )
            features = segment_ops.segment_sum(
                input_features * ops.expand_dims(factor, -1),
                self._successor_ids,
                num_segments=num_segments,
                indices_are_sorted=self._indices_are_sorted,
            )
        return features


class ExclusivePatchExtractor(OpNode):
    def __init__(
        self,
        node: StreamNode,
        stream_out: StreamData,
        num_channels: int | None = None,
        **layer_kwargs,
    ):
        self._built = False
        stream_in = node.stream
        assert all(
            i % o == 0 for i, o in zip(stream_in.grid_shape, stream_out.grid_shape)
        ), (stream_in.grid_shape, stream_out.grid_shape)
        self._strides = tuple(
            i // o for i, o in zip(stream_in.grid_shape, stream_out.grid_shape)
        )
        if num_channels is None:
            num_channels = node.num_channels * self.kernel_size
        node_out = StreamNode(
            stream_out,
            self,
            0,
            num_channels,
            order=stream_out.get_contiguous_segments_order(),
        )
        super().__init__(node, node_out)
        self._layer_kwargs = layer_kwargs
        self._layer = self._create_layer()

    @property
    def strides(self) -> tp.Sequence[int]:
        return self._strides

    @property
    def kernel_size(self) -> int:
        return prod(*self.strides)

    def build(self):
        if self._built:
            return
        self._built = True
        node_in: StreamNode = self.inputs
        node_out: StreamNode = self.outputs
        stream_in = node_in.stream
        stream_out = node_out.stream
        kernel_size = self.kernel_size
        input_is_already_ordered = node_in.order is not None
        strides_tensor = ops.convert_to_tensor(self.strides, "int32")
        successor_ids = conv_preprocessing_ops.get_successor_ids(
            pixel_ids_in=utils_ops.ravel_multi_index(
                stream_in.coords // strides_tensor, stream_out.grid_shape_tensor, axis=1
            ),
            times_in=stream_in.times,
            batch_splits_in=stream_in.batch_splits,
            pixel_ids_out=stream_out.pixel_ids,
            times_out=stream_out.times,
            batch_splits_out=stream_out.batch_splits,
            grid_size=stream_out.grid_size,
        )
        reindexed_successor_ids = node_out.order.reindex(successor_ids)
        if input_is_already_ordered:
            reindexed_successor_ids = node_in.order.permute(reindexed_successor_ids)

        input_kernel_ids = utils_ops.ravel_multi_index(
            stream_in.coords % strides_tensor, strides_tensor, axis=1
        )
        permuted_times_in = stream_in.times
        if input_is_already_ordered:
            input_kernel_ids = node_in.order.permute(input_kernel_ids)
            permuted_times_in = node_in.order.permute(stream_in.times)
        permuted_times_out = node_out.order.permute(stream_out.times)
        dt = (
            ops.take(ops.pad(permuted_times_out, [[0, 1]]), reindexed_successor_ids)
            - permuted_times_in
        )
        dt = ops.where(
            reindexed_successor_ids == stream_out.num_events, ops.zeros_like(dt), dt
        )

        reindexed_successor_kernel_ids = (
            reindexed_successor_ids * kernel_size + input_kernel_ids
        )

        reindexed_successor_kernel_ids = ops.minimum(
            reindexed_successor_kernel_ids,
            ops.full_like(
                reindexed_successor_kernel_ids, node_out.stream.num_events * kernel_size
            ),
        )

        if input_is_already_ordered:
            self._indices_are_sorted = False
        else:
            splits = ragged_ops.ids_to_splits(
                reindexed_successor_kernel_ids,
                node_out.stream.num_events * kernel_size + 1,
            )
            perm = counting_argsort(reindexed_successor_kernel_ids, splits)
            node_in.order = Permuted(perm, contiguous_segments=False)
            self._indices_are_sorted = True
            dt = node_in.order.permute(dt)
            reindexed_successor_kernel_ids = node_in.order.permute(
                reindexed_successor_kernel_ids
            )
        self._dt = dt
        self._successor_kernel_ids = reindexed_successor_kernel_ids
        self._times_out = permuted_times_out
        self._segment_ids_out = node_out.order.permute(stream_out.segment_ids)
        self._layer.build(
            features_shape=(self.inputs.stream.num_events, self.inputs.num_channels),
            dt_shape=self._dt.shape,
            times_out_shape=self._times_out.shape,
            successor_kernel_ids_shape=self._successor_kernel_ids.shape,
            segment_ids_out_shape=self._segment_ids_out.shape,
        )

    def _create_layer(self) -> layers.Layer:
        return patch_layers.ExtractExclusivePatches(
            self.kernel_size, flatten=True, **self._layer_kwargs
        )

    @property
    def layer(self) -> layers.Layer:
        return self._layer

    def _compute_features(self):
        self.build()
        return self._layer(
            features=self.inputs.compute_features(),
            dt=self._dt,
            times_out=self._times_out,
            successor_kernel_ids=self._successor_kernel_ids,
            segment_ids_out=self._segment_ids_out,
            indices_are_sorted=self._indices_are_sorted,
        )


class ExclusiveConv(ExclusivePatchExtractor):
    def __init__(
        self,
        node: StreamNode,
        stream_out: StreamData,
        filters: int | None,
        **layer_kwargs,
    ):
        self._filters = filters
        self._layer_kwargs = layer_kwargs
        super().__init__(
            node,
            stream_out,
            node.num_channels if filters is None else filters,
        )

    def _create_layer(self) -> layers.Layer:
        if self._filters is None:
            return conv_layers.ExclusiveDepthwiseConv(
                self.kernel_size, **self._layer_kwargs
            )

        return conv_layers.ExclusiveConv(
            self._filters, self.kernel_size, **self._layer_kwargs
        )


class ExclusivePooling(ExclusivePatchExtractor):
    def __init__(
        self, node: StreamNode, stream_out: StreamData, reduction: str, **layer_kwargs
    ):
        self._layer_kwargs = layer_kwargs
        super().__init__(node, stream_out, node.num_channels, reduction=reduction)

    def _create_layer(self) -> layers.Layer:
        return pool_layers.ExclusivePooling(self.kernel_size, **self._layer_kwargs)


@functools.cache
def _get_stationary_predecessor_ids(
    stream: StreamData, kernel_shape: tp.Tuple[int, ...]
):
    assert len(kernel_shape) == stream.spatial_dims

    grid_shape = tuple(g + k - 1 for g, k in zip(stream.grid_shape, kernel_shape))
    pad_left = tuple((k - 1) // 2 for k in kernel_shape)
    grid_shape_tensor = ops.convert_to_tensor(grid_shape)
    kernel_offsets = ops.stack(
        ops.meshgrid(
            *(ops.arange(-l, -l + k) for l, k in zip(pad_left, kernel_shape)),
            indexing="ij",
        ),
        axis=0,
    )
    kernel_offsets = ops.reshape(
        kernel_offsets, (stream.spatial_dims, prod(*kernel_shape))
    )
    kernel_offsets = utils_ops.ravel_multi_index(kernel_offsets, grid_shape_tensor)
    predecessor_ids = conv_preprocessing_ops.get_stationary_predecessor_ids(
        utils_ops.ravel_multi_index(
            stream.coords + ops.convert_to_tensor(pad_left),
            grid_shape_tensor,
            axis=1,
        ),
        stream.batch_splits,
        kernel_offsets,
        grid_size=prod(*grid_shape),
    )
    return predecessor_ids


class StationaryPatchExtractor(OpNode):
    def __init__(
        self, node: StreamNode, kernel_shape: int | tp.Sequence[int], num_channels: int
    ):
        if node.order is None:
            node.order = node.stream.get_contiguous_segments_order()
        else:
            assert node.order.contiguous_segments
        output = StreamNode(node.stream, self, 0, num_channels)
        self._built = False
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape,) * node.stream.spatial_dims
        else:
            kernel_shape = tuple(kernel_shape)
            assert len(kernel_shape) == node.stream.spatial_dims
        self._kernel_shape = tuple(kernel_shape)
        super().__init__(node, output)
        self._layer = self._create_layer()

    @property
    def kernel_shape(self) -> tp.Sequence[int]:
        return self._kernel_shape

    @property
    def kernel_size(self) -> int:
        return prod(*self.kernel_shape)

    @property
    def spatial_dims(self) -> int:
        return len(self._kernel_shape)

    def _create_layer(self) -> layers.Layer:
        raise NotImplementedError("Abstract method")

    def build(self):
        if self._built:
            return
        self._built = True
        node_in: StreamNode = self.inputs
        node_out: StreamNode = self.outputs

        stream = node_in.stream
        order_in = node_in.order
        order_out = node_out.order
        assert order_out is not None

        self._times_in = order_in.permute(stream.times)
        self._times_out = order_out.permute(stream.times)
        self._segment_ids_in = order_in.permute(stream.segment_ids)

        self._predecessor_ids = _get_stationary_predecessor_ids(
            stream, self.kernel_shape
        )
        self._predecessor_ids = order_in.reindex(self._predecessor_ids)
        self._predecessor_ids = order_out.permute(self._predecessor_ids)
        self._layer.build(
            features_shape=(stream.num_events, self.inputs.num_channels),
            times_in_shape=(stream.num_events,),
            times_out_shape=(stream.num_events,),
            segment_ids_shape=(stream.num_events,),
            predecessor_ids_shape=(stream.num_events, self.kernel_size),
        )

    @property
    def layer(self) -> layers.Layer:
        return self._layer

    def _compute_features(self):
        self.build()
        return self._layer(
            features=self.inputs.compute_features(),
            times_in=self._times_in,
            times_out=self._times_out,
            segment_ids=self._segment_ids_in,
            predecessor_ids=self._predecessor_ids,
        )


class StationaryConv(StationaryPatchExtractor):
    def __init__(
        self,
        node: StreamNode,
        filters: tp.Optional[int],
        kernel_shape: int | tp.Sequence[int],
        **layer_kwargs,
    ):
        self._filters = filters
        self._layer_kwargs = layer_kwargs
        super().__init__(
            node, kernel_shape, node.num_channels if filters is None else filters
        )

    def _create_layer(self) -> layers.Layer:
        if self._filters is None:
            return conv_layers.DepthwiseConv(**self._layer_kwargs)
        return conv_layers.Conv(filters=self._filters, **self._layer_kwargs)


# class Connector:
#     def __init__(self, stream_in: StreamData, stream_out: StreamData):
#         assert isinstance(stream_in, StreamData)
#         assert isinstance(stream_out, StreamData)
#         self._stream_in = stream_in
#         self._stream_out = stream_out

#     @property
#     def stream_in(self) -> StreamData:
#         return self._stream_in

#     @property
#     def stream_out(self) -> StreamData:
#         return self._stream_out


# class ExclusiveConnector(Connector):
#     def __init__(self, stream_in: StreamData, stream_out: StreamData):
#         super().__init__(stream_in, stream_out)
#         assert all(
#             i % o == 0 for i, o in zip(stream_in.grid_shape, stream_out.grid_shape)
#         ), (stream_in.grid_shape, stream_out.grid_shape)
#         self._strides = tuple(
#             i // o for i, o in zip(stream_in.grid_shape, stream_out.grid_shape)
#         )

#         strides_tensor = ops.convert_to_tensor(self.strides, "int32")
#         self._successor_ids = conv_preprocessing_ops.get_successor_ids(
#             pixel_ids_in=utils_ops.ravel_multi_index(
#                 stream_in.coords // strides_tensor, stream_out.grid_shape_tensor, axis=1
#             ),
#             times_in=stream_in.times,
#             batch_splits_in=stream_in.batch_splits,
#             pixel_ids_out=stream_out.pixel_ids,
#             times_out=stream_out.times,
#             batch_splits_out=stream_out.batch_splits,
#             grid_size=stream_out.grid_size,
#         )
#         self._input_kernel_ids = utils_ops.ravel_multi_index(
#             stream_in.coords % strides_tensor, strides_tensor, axis=1
#         )
#         dt = (
#             ops.take(ops.pad(stream_out.times, [[0, 1]]), self._successor_ids)
#             - stream_in.times
#         )
#         self._dt = ops.where(
#             self._successor_ids == stream_out.num_events, ops.zeros_like(dt), dt
#         )

#     @property
#     def strides(self) -> tp.Sequence[int]:
#         return self._strides

#     @property
#     def kernel_size(self) -> int:
#         return prod(*self.strides)

#     @property
#     def successor_ids(self) -> Tensor:
#         return self._successor_ids

#     @property
#     def dt(self) -> Tensor:
#         return self._dt

#     @property
#     def input_kernel_ids(self) -> Tensor:
#         return self._input_kernel_ids


# @functools.cache
# def exclusive_connector(stream_in: StreamData, stream_out: StreamData):
#     return ExclusiveConnector(stream_in, stream_out)


# class StationaryConnector(Connector):
#     def __init__(self, kernel_shape: int | tp.Sequence[int], stream: StreamData):
#         if isinstance(kernel_shape, int):
#             kernel_shape = (kernel_shape,) * stream.spatial_dims
#         else:
#             kernel_shape = tuple(kernel_shape)
#             assert len(kernel_shape) == stream.spatial_dims, (
#                 kernel_shape,
#                 stream.spatial_dims,
#             )
#         self._kernel_shape = kernel_shape
#         super().__init__(stream, stream)

#     @property
#     def kernel_shape(self) -> tp.Sequence[int]:
#         return self._kernel_shape

#     @property
#     def kernel_size(self) -> int:
#         return prod(*self.kernel_shape)

#     @property
#     def spatial_dims(self) -> int:
#         return len(self.kernel_shape)

#     @property
#     @functools.cache
#     def predecessor_ids(self) -> Tensor:
#         stream = self.stream_in
#         kernel_shape = self.kernel_shape
#         grid_shape = tuple(g + k - 1 for g, k in zip(stream.grid_shape, kernel_shape))
#         pad_left = tuple((k - 1) // 2 for k in kernel_shape)
#         grid_shape_tensor = ops.convert_to_tensor(grid_shape)
#         kernel_offsets = ops.stack(
#             ops.meshgrid(
#                 *(ops.arange(-l, -l + k) for l, k in zip(pad_left, kernel_shape)),
#                 indexing="ij",
#             ),
#             axis=0,
#         )
#         kernel_offsets = ops.reshape(
#             kernel_offsets, (stream.spatial_dims, self.kernel_size)
#         )
#         kernel_offsets = utils_ops.ravel_multi_index(kernel_offsets, grid_shape_tensor)

#         return conv_preprocessing_ops.get_stationary_predecessor_ids(
#             stream.pixel_ids
#             + utils_ops.ravel_multi_index(
#                 ops.convert_to_tensor(pad_left), grid_shape_tensor
#             ),
#             stream.batch_splits,
#             kernel_offsets,
#             grid_size=prod(*grid_shape),
#         )


# @functools.cache
# def stationary_connector(
#     kernel_shape: int | tp.Sequence[int], stream: StreamData
# ) -> StationaryConnector:
#     return StationaryConnector(kernel_shape, stream)


# class ExclusiveConv(OpNode):
#     def __init__(
#         self,
#         node: StreamNode,
#         stream_out: StreamData,
#         filters: tp.Optional[int] = None,
#         **conv_kwargs,
#     ):
#         node_out = StreamNode(
#             stream_out, self, 0, order=stream_out.get_contiguous_segments_order()
#         )
#         super().__init__(node, node_out)
#         self._connector = ExclusiveConnector(node.stream, stream_out)
#         self._filters = filters
#         self._conv_kwargs = conv_kwargs

#     def _compute_features(self):
#         node_in: StreamNode = self.inputs
#         is_input = isinstance(node_in.source, InputOp)
#         if is_input:
#             assert node_in.order is None
#             features = node_in.source.input_features
#         else:
#             features = node_in.compute_features()
#         if isinstance(features, OneHotTensor):
#             return self._compute_one_hot(features)
#         assert isinstance(features, Tensor)
#         return self._compute_tensor(features)

#     def _compute_one_hot(
#         self,
#         features: OneHotTensor,
#     ):
#         assert isinstance(features, OneHotTensor)
#         channel_ids, num_channels = features
#         node_in = self.inputs
#         node_out = self.outputs

#         connector = self._connector
#         kernel_size = connector.kernel_size

#         dt = connector.dt

#         reindexed_successor_ids = node_out.order.reindex(connector.successor_ids)
#         reindexed_successor_kernel_ids = (
#             reindexed_successor_ids * kernel_size + connector.input_kernel_ids
#         )
#         reindexed_successor_kernel_channel_ids = (
#             reindexed_successor_kernel_ids * num_channels + channel_ids
#         )
#         reindexed_successor_kernel_channel_ids = ops.minimum(
#             reindexed_successor_kernel_channel_ids,
#             ops.full_like(
#                 reindexed_successor_kernel_channel_ids,
#                 node_out.stream.num_events * kernel_size * num_channels,
#             ),
#         )

#         layer_kwargs = dict(self._conv_kwargs)
#         layer_kwargs["kernel_size"] = kernel_size

#         if node_in.order is None:
#             splits = ragged_ops.ids_to_splits(
#                 reindexed_successor_kernel_channel_ids,
#                 node_out.stream.num_events * kernel_size * num_channels + 1,
#             )
#             node_in.order = Permuted(
#                 counting_argsort(reindexed_successor_kernel_channel_ids, splits),
#                 contiguous_segments=False,
#             )
#             indices_are_sorted = True
#         else:
#             indices_are_sorted = False

#         if self._filters is None:
#             layer = conv_layers.OneHotExclusiveDepthwiseConv(
#                 num_channels, **layer_kwargs
#             )
#         else:
#             layer = conv_layers.OneHotExclusiveConv(
#                 num_channels, self._filters, **layer_kwargs
#             )
#         order_in = node_in.order
#         assert order_in is not None
#         order_out = node_out.order
#         assert order_out is not None
#         return layer(
#             order_in.permute(dt),
#             order_out.permute(node_out.stream.times),
#             order_in.permute(reindexed_successor_kernel_channel_ids),
#             order_out.permute(node_out.stream.segment_ids),
#             indices_are_sorted=indices_are_sorted,
#         )

#     def _compute_tensor(self, features: Tensor):
#         assert isinstance(features, Tensor)
#         node_in = self.inputs
#         node_out = self.outputs

#         connector = self._connector
#         kernel_size = connector.kernel_size

#         reindexed_successor_ids = node_out.order.reindex(connector.successor_ids)
#         reindexed_successor_kernel_ids = (
#             reindexed_successor_ids * kernel_size + connector.input_kernel_ids
#         )
#         reindexed_successor_kernel_ids = ops.minimum(
#             reindexed_successor_kernel_ids,
#             ops.full_like(
#                 reindexed_successor_kernel_ids, node_out.stream.num_events * kernel_size
#             ),
#         )

#         dt = connector.dt
#         layer_kwargs = dict(self._conv_kwargs)
#         layer_kwargs["kernel_size"] = kernel_size

#         if node_in.order is None:
#             splits = ragged_ops.ids_to_splits(
#                 reindexed_successor_kernel_ids,
#                 node_out.stream.num_events * kernel_size + 1,
#             )
#             node_in.order = Permuted(
#                 counting_argsort(reindexed_successor_kernel_ids, splits),
#                 contiguous_segments=False,
#             )
#             indices_are_sorted = True
#         else:
#             indices_are_sorted = False

#         if self._filters is None:
#             layer = conv_layers.ExclusiveDepthwiseConv(**layer_kwargs)
#         else:
#             layer = conv_layers.ExclusiveConv(self._filters, **layer_kwargs)
#         order_in = node_in.order
#         assert order_in is not None
#         order_out = node_out.order
#         assert order_out is not None
#         return layer(
#             features=features,
#             dt=order_in.permute(dt),
#             times_out=order_out.permute(node_out.stream.times),
#             successor_kernel_ids=order_in.permute(reindexed_successor_kernel_ids),
#             segment_ids_out=order_out.permute(node_out.stream.segment_ids),
#             indices_are_sorted=indices_are_sorted,
#         )


# class StationaryConv(OpNode):
#     def __init__(
#         self,
#         node: StreamNode,
#         kernel_shape: int | tp.Sequence[int],
#         filters: tp.Optional[int] = None,
#         **conv_kwargs,
#     ):
#         if node.order is None:
#             node.order = node.stream.get_contiguous_segments_order()
#         else:
#             assert node.order.contiguous_segments
#         output = StreamNode(node.stream, self, 0)
#         super().__init__(node, output)
#         self._filters = filters
#         self._conv_kwargs = conv_kwargs
#         self._connector = stationary_connector(kernel_shape, node.stream)

#     def _compute_features(self) -> Tensor:
#         features = self.inputs.compute_features()
#         if self.outputs.order is None:
#             self.outputs.order = self.inputs.order
#         if isinstance(features, OneHotTensor):
#             return self._compute_one_hot(features)
#         else:
#             return self._compute_tensor(features)

#     def _compute_one_hot(self, features: OneHotTensor) -> Tensor:
#         channel_ids, num_channels = features
#         return self._compute_tensor(ops.one_hot(channel_ids, num_channels))

#     def _compute_tensor(self, features: Tensor) -> Tensor:
#         if self._filters is None:
#             layer = conv_layers.DepthwiseConv(**self._conv_kwargs)
#         else:
#             layer = conv_layers.Conv(self._filters, **self._conv_kwargs)
#         predecessor_ids = self._connector.predecessor_ids
#         node_in = self.inputs
#         node_out = self.outputs
#         return layer(
#             features=features,
#             times_in=node_in.order.permute(node_in.stream.times),
#             times_out=node_out.order.permute(node_out.stream.times),
#             segment_ids=node_in.order.permute(node_in.stream.segment_ids),
#             predecessor_ids=node_out.order.permute(
#                 node_in.order.reindex(predecessor_ids)
#             ),
#         )
