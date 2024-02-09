import typing as tp

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tfnp  # type: ignore
from keras.utils import pack_x_y_sample_weight


class StreamData(tp.NamedTuple):
    coords: tf.Tensor
    times: tf.Tensor
    polarity: tf.Tensor
    grid_shape: tp.Tuple[int, ...]

    def assert_valid(self):
        assert len(self.coords.shape) >= 2, self.coords.shape
        assert len(self.times.shape) >= 1, self.times.shape
        assert len(self.polarity.shape) >= 1, self.polarity.shape
        assert self.polarity.dtype == tf.bool, self.polarity.dtype
        assert len(self.grid_shape) == self.coords.shape[-1], (
            self.grid_shape,
            self.coords,
        )


def valid_mask(coords: tf.Tensor, grid_shape: tp.Tuple[int, ...]):
    grid_shape = tf.convert_to_tensor(grid_shape, coords.dtype)
    return tf.reduce_all(tf.logical_and(coords >= 0, coords < grid_shape), axis=-1)


def mask_valid_events(stream: StreamData) -> StreamData:
    mask = valid_mask(stream.coords, stream.grid_shape)
    return StreamData(
        stream.coords[mask],
        stream.times[mask],
        stream.polarity[mask],
        stream.grid_shape,
    )


T = tp.TypeVar("T")


class Transform(tp.Generic[T]):
    def transform_frames(self, frames: tf.Tensor, transformation: T) -> tf.Tensor:
        return frames

    def transform_stream(self, stream: StreamData, transformation: T) -> StreamData:
        return stream

    def transform_label(self, label, transformation: T):
        return label

    def transform_sample_weight(self, sample_weight, transformation: T):
        return sample_weight

    def get_transformation(
        self,
        frames_or_stream: tf.Tensor | StreamData,
        label=None,
        sample_weight=None,
    ) -> T:
        return None

    def transform_frames_example(
        self, frames: tf.Tensor, label=None, sample_weight=None
    ):
        transformation = self.get_transformation(frames, label, sample_weight)
        frames = self.transform_frames(frames, transformation)
        if label is not None:
            label = self.transform_label(label, transformation)
        if sample_weight is not None:
            sample_weight = self.transform_sample_weight(sample_weight, transformation)

        return pack_x_y_sample_weight(frames, label, sample_weight)

    def transform_stream_example(
        self, stream: StreamData, label=None, sample_weight=None
    ):
        transformation = self.get_transformation(stream, label, sample_weight)
        stream = self.transform_stream(stream, transformation)
        if label is not None:
            label = self.transform_label(label, transformation)
        if sample_weight is not None:
            sample_weight = self.transform_sample_weight(sample_weight)

        if label is None:
            return stream
        if sample_weight is None:
            return stream, label
        return stream, label, sample_weight


class SeriesTransform(Transform[tp.Sequence[T]]):
    def __init__(self, *transforms: Transform[T]):
        self.transforms = transforms
        assert all(
            isinstance(transform, Transform) for transform in transforms
        ), transforms

    def transform_frames(
        self, frames: tf.Tensor, transformation: tp.Sequence[T]
    ) -> tf.Tensor:
        if transformation is None:
            transformation = (None,) * len(self.transforms)
        else:
            assert len(transformation) == len(self.transforms), (
                len(transformation),
                len(self.transforms),
            )
        for transform, transformation_ in zip(self.transforms, transformation):
            frames = transform.transform_frames(frames, transformation_)
        return frames

    def transform_stream(
        self, stream: StreamData, transformation: tp.Sequence[T]
    ) -> StreamData:
        if transformation is None:
            transformation = (None,) * len(self.transforms)
        else:
            assert len(transformation) == len(self.transforms), (
                len(transformation),
                len(self.transforms),
            )
        for transform, transformation_ in zip(self.transforms, transformation):
            stream = transform.transform_stream(stream, transformation_)
        return stream

    def transform_label(
        self, label: tf.Tensor, transformation: tp.Sequence[T]
    ) -> tf.Tensor:
        if transformation is None:
            transformation = (None,) * len(self.transforms)
        else:
            assert len(transformation) == len(self.transforms), (
                len(transformation),
                len(self.transforms),
            )
        for transform, transformation_ in zip(self.transforms, transformation):
            label = transform.transform_label(label, transformation_)
        return label

    def transform_sample_weight(
        self, sample_weight: tf.Tensor, transformation: tp.Sequence[T]
    ) -> tf.Tensor:
        if transformation is None:
            transformation = (None,) * len(self.transforms)
        else:
            assert len(transformation) == len(self.transforms), (
                len(transformation),
                len(self.transforms),
            )
        for transform, transformation_ in zip(self.transforms, transformation):
            sample_weight = transform.transform_sample_weight(
                sample_weight, transformation_
            )
        return sample_weight

    def get_transformation(
        self,
        frames_or_stream: tf.Tensor | StreamData,
        label=None,
        sample_weight=None,
    ) -> tp.Sequence[T]:
        return tuple(
            transform.get_transformation(frames_or_stream, label, sample_weight)
            for transform in self.transforms
        )


class Resize(Transform[None]):
    def __init__(self, output_grid_shape: tp.Iterable[int]):
        if isinstance(output_grid_shape, int):
            output_grid_shape = (output_grid_shape,) * 2
        self.output_grid_shape = tuple(output_grid_shape)

    def transform_frames(self, frames: tf.Tensor, transformation: None) -> tf.Tensor:
        return tf.image.resize(frames, self.output_grid_shape)

    def transform_stream(self, stream: StreamData, transformation: None) -> StreamData:
        coords = stream.coords
        input_grid_shape = tf.convert_to_tensor(stream.grid_shape, "float32")
        output_grid_shape = tf.convert_to_tensor(self.output_grid_shape, "float32")
        coords = tf.cast(
            (
                tf.cast(coords, "float32")
                / (input_grid_shape - 1)
                * (output_grid_shape - 1)
            ),
            coords.dtype,
        )
        return StreamData(coords, stream.times, stream.polarity, self.output_grid_shape)


class Pad(Transform[None]):
    def __init__(
        self, padding: tp.Tuple[tp.Tuple[int, int], ...] | tp.Tuple[int, int] | int
    ):
        if isinstance(padding, int):
            padding = (padding, padding)
        else:
            padding = tuple(padding)
        assert len(padding) == 2, padding
        if all(isinstance(p, int) for p in padding):
            padding = tuple((p, p) for p in padding)
        else:
            padding = tuple(padding)
        self.padding = padding
        assert all(len(p) == 2 for p in self.padding), self.padding

    def transform_frames(self, frames: tf.Tensor, transformation) -> tf.Tensor:
        ndims = len(frames.shape)
        padding = np.zeros((ndims, 2), dtype=int)
        padding[-(len(self.padding) + 1) : -1] = self.padding
        return tf.pad(frames, padding)

    def transform_stream(self, stream: StreamData, transformation) -> tf.Tensor:
        pad_left, pad_right = zip(self.padding)
        coords = stream.coords + tf.convert_to_tensor(pad_left, stream.coords.dtype)
        grid_shape = tuple(g + sum(p) for g, p in zip(stream.grid_shape, self.padding))
        return StreamData(coords, stream.times, stream.polarity, grid_shape)


class PadToSquare(Transform[None]):
    def transform_frames(self, frames: tf.Tensor, transformation) -> tf.Tensor:
        ndims = len(frames.shape)
        shape = frames.shape[-3:-1]
        size = max(shape)
        padding = np.zeros((ndims, 2), dtype=int)
        p = size - shape[-3]
        padding[-3] = (p // 2, p - p // 2)
        p = size - shape[-2]
        padding[-2] = (p // 2, p - p // 2)
        return tf.pad(frames, padding)

    def transform_stream(self, stream: StreamData, transformation) -> tf.Tensor:
        shape = stream.grid_shape
        size = max(shape)
        coords = stream.coords + tf.convert_to_tensor(
            [(size - s) // 2 for s in shape], stream.coords.dtype
        )
        return StreamData(coords, stream.times, stream.polarity, (size,) * len(shape))


class Recenter(Transform[None]):
    def transform_frames(self, frames: tf.Tensor, transformation) -> tf.Tensor:
        raise NotImplementedError()

    def transform_stream(self, stream: StreamData, transformation) -> StreamData:
        maxes = tf.reduce_max(stream.coords, axis=-2, keepdims=True)
        mins = tf.reduce_min(stream.coords, axis=-2, keepdims=True)
        center = (maxes + mins) // 2
        new_center = tf.convert_to_tensor(
            [s // 2 for s in stream.grid_shape], stream.coords.dtype
        )
        coords = stream.coords + (new_center - center)
        return StreamData(coords, stream.times, stream.polarity, stream.grid_shape)


class _Crop(Transform[T]):
    def __init__(self, output_grid_shape: tp.Iterable[int]):
        self.output_grid_shape = tuple(output_grid_shape)

    def _get_left_offset(self, excess: tp.Sequence[int], transformation: T):
        raise NotImplementedError()

    def _get_excess(self, grid_shape):
        return tuple(g - og for g, og in zip(grid_shape, self.output_grid_shape))

    def transform_frames(self, frames: tf.Tensor, transformation: T) -> tf.Tensor:
        starts = self._get_left_offset(
            self._get_excess(frames.shape[-3:-1]), transformation
        )
        starts = tf.pad(starts, [[len(frames.shape) - 3, 1]])
        return tf.slice(
            frames, starts, (frames.shape[0], *self.output_grid_shape, frames.shape[-1])
        )

    def transform_stream(self, stream: StreamData, transformation: T) -> tf.Tensor:
        starts = self._get_left_offset(
            self._get_excess(stream.grid_shape), transformation
        )
        return StreamData(
            stream.coords - starts,
            stream.times,
            stream.polarity,
            self.output_grid_shape,
        )


class MultiCrop(Transform[None]):
    """
    Stack top left, top right, bottom left, bottom right and central crops.

    Stacking is performed on the leading dimension.
    """

    def __init__(self, output_grid_shape: tp.Iterable[int]):
        self.output_grid_shape = tuple(output_grid_shape)

    def _get_excess(self, grid_shape):
        return tuple(g - og for g, og in zip(grid_shape, self.output_grid_shape))

    def transform_frames(self, frames: tf.Tensor, transformation: None) -> tf.Tensor:
        excess = self._get_excess(frames.shape[-3:-1])
        batch_dims = len(frames.shape) - 3

        def get_cropped(left_offset: tp.Sequence[int]):
            starts = tf.pad(
                tf.convert_to_tensor(left_offset, "int32"), [[batch_dims, 1]]
            )
            return tf.slice(frames, starts, self.output_grid_shape)

        return tf.stack(
            [
                get_cropped(x)
                for x in (
                    (0, 0),
                    (0, excess[1]),
                    (excess[0], excess[1]),
                    (excess[0], 0),
                    (excess[0] // 2, excess[1] // 2),
                )
            ],
            axis=0,
        )

    def transform_stream(self, stream: StreamData, transformation: None) -> StreamData:
        def get_cropped(left_offset: tp.Sequence[int]):
            left_offset = tf.convert_to_tensor(left_offset, "int32")
            return stream.coords - left_offset

        excess = self._get_excess(stream.grid_shape)
        coords = tf.stack(
            [
                get_cropped(x)
                for x in (
                    (0, 0),
                    (0, excess[1]),
                    (excess[0], excess[1]),
                    (excess[0], 0),
                    (excess[0] // 2, excess[1] // 2),
                )
            ],
            axis=0,
        )
        batch_dims = len(stream.times.shape) - 1
        reps = (5,) + (1,) * (batch_dims + 1)
        times = tf.tile(tf.expand_dims(stream.times, 0), reps)
        polarity = tf.tile(tf.expand_dims(stream.polarity, 0), reps)
        return StreamData(coords, times, polarity, self.output_grid_shape)

    def transform_label(self, label, transform):
        return tf.tile(tf.expand_dims(label, 0), (5,) + (1,) * len(label.shape))

    def transform_sample_weight(self, sample_weight, transform):
        return tf.tile(
            tf.expand_dims(sample_weight, 0), (5,) + (1,) * len(sample_weight.shape)
        )


class CentralCrop(_Crop[None]):
    def _get_left_offset(self, excess: tp.Sequence[int], transformation=None):
        return tf.convert_to_tensor([e // 2 for e in excess], "int32")


class RandomCrop(_Crop[tf.Tensor]):
    def __init__(self, output_grid_shape: tp.Iterable[int]):
        if isinstance(output_grid_shape, int):
            output_grid_shape = (output_grid_shape,) * 2
        self.output_grid_shape = tuple(output_grid_shape)

    def _get_left_offset(self, excess: tp.Sequence[int], transformation: tf.Tensor):
        return tf.cast(
            transformation * tf.convert_to_tensor(excess, transformation.dtype),
            "int32",
        )

    def get_transformation(
        self, stream_or_frames, label=None, sample_weight=None
    ) -> tf.Tensor:
        if isinstance(stream_or_frames, StreamData):
            spatial_dims = len(stream_or_frames.grid_shape)
        else:
            spatial_dims = 2
        return tf.random.uniform((spatial_dims,))


class RandomRotate(Transform[tf.Tensor]):
    def __init__(self, max_angle: float):
        self._max_angle = max_angle

    def get_transformation(
        self, stream_or_frames, label=None, sample_weight=None
    ) -> tf.Tensor:
        return tf.random.uniform((), -self._max_angle, self._max_angle)

    def transform_frames(
        self, frames: tf.Tensor, transformation: tf.Tensor
    ) -> tf.Tensor:
        raise NotImplementedError("TODO")

    def transform_stream(
        self, stream: StreamData, transformation: tf.Tensor
    ) -> StreamData:
        c = tf.cos(transformation)
        s = tf.sin(transformation)
        mat = tf.stack([tf.stack([c, -s]), tf.stack([s, c])])
        coords = stream.coords
        dtype = coords.dtype
        coords = tf.cast(coords, tf.float32)
        center = tf.convert_to_tensor(
            [(s - 1) / 2 for s in stream.grid_shape], tf.float32
        )
        coords = coords - center
        coords = tf.linalg.matmul(coords, mat, adjoint_b=True)
        coords = coords + center
        coords = tf.cast(coords, dtype)
        return StreamData(coords, stream.times, stream.polarity, stream.grid_shape)


class RandomZoom(Transform[tf.Tensor]):
    def __init__(self, min_factor: float, max_factor: float):
        self._min_factor = min_factor
        self._max_factor = max_factor

    def get_transformation(
        self, stream_or_frames, label=None, sample_weight=None
    ) -> tf.Tensor:
        return tf.random.uniform((), self._min_factor, self._max_factor)

    def transform_frames(
        self, frames: tf.Tensor, transformation: tf.Tensor
    ) -> tf.Tensor:
        raise NotImplementedError("TODO")

    def transform_stream(
        self, stream: StreamData, transformation: tf.Tensor
    ) -> StreamData:
        coords = stream.coords
        dtype = coords.dtype
        coords = tf.cast(coords, tf.float32)
        center = tf.convert_to_tensor(
            [(s - 1) / 2 for s in stream.grid_shape], tf.float32
        )
        coords = (coords - center) * transformation + center
        coords = tf.cast(coords, dtype)
        return StreamData(coords, stream.times, stream.polarity, stream.grid_shape)


class _TemporalCrop(Transform[T]):
    def __init__(self, size: int, rezero_times: bool = True):
        self.size = size
        self.rezero_times = rezero_times

    def _get_temporal_offset(
        self, excess: int | tf.Tensor, transformation: T
    ) -> tf.Tensor:
        raise NotImplementedError()

    def transform_stream(self, stream: StreamData, transformation: T) -> StreamData:
        excess = tf.shape(stream.times)[-1] - self.size
        start = self._get_temporal_offset(excess, transformation)
        stop = start + self.size
        times = stream.times[..., start:stop]
        if self.rezero_times:
            times = times - times[..., 0]
        return StreamData(
            stream.coords[..., start:stop, :],
            times,
            stream.polarity[..., start:stop],
            stream.grid_shape,
        )


class RandomTemporalCrop(_TemporalCrop[tf.Tensor]):
    def get_transformation(
        self, frames_or_stream: tf.Tensor | StreamData, label=None, sample_weight=None
    ) -> tf.Tensor:
        return tf.random.uniform(())

    def _get_temporal_offset(
        self, excess: int | tf.Tensor, transformation: tf.Tensor
    ) -> tf.Tensor:
        return tf.cast(tf.cast(excess, "float32") * transformation, "int32")


class CentralTemporalCrop(_TemporalCrop[None]):
    def _get_temporal_offset(self, excess: int | tf.Tensor, transformation: None):
        return excess // 2


class _TemporalCropV2(Transform[T]):
    def __init__(self, rezero_times: bool = True):
        self.rezero_times = rezero_times

    def _get_temporal_limits(
        self, t_start: tf.Tensor, t_stop: tf.Tensor, transformation: T
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("Abstract method")

    def transform_stream(self, stream: StreamData, transformation: T) -> StreamData:
        t_start, t_stop = self._get_temporal_limits(
            stream.times[0], stream.times[-1], transformation
        )
        valid = tf.logical_and(stream.times >= t_start, stream.times <= t_stop)
        coords = tf.boolean_mask(stream.coords, valid)
        times = tf.boolean_mask(stream.times, valid)
        if self.rezero_times:
            times = times - t_start
        polarity = tf.boolean_mask(stream.polarity, valid)
        return StreamData(coords, times, polarity, stream.grid_shape)


class RandomTemporalCropV2(_TemporalCropV2[tf.Tensor]):
    def __init__(
        self,
        max_initial_skip_frac: float,
        total_frac: tp.Optional[float] = None,
        rezero_times: bool = True,
    ):
        assert 0 <= max_initial_skip_frac < 1, max_initial_skip_frac
        if total_frac is None:
            total_frac = 1 - max_initial_skip_frac
        else:
            assert max_initial_skip_frac + total_frac <= 1, (
                max_initial_skip_frac,
                total_frac,
            )
        self.max_initial_skip_frac = max_initial_skip_frac
        self.total_frac = total_frac
        super().__init__(rezero_times=rezero_times)

    def _get_temporal_limits(
        self, t_start: tf.Tensor, t_stop: tf.Tensor, transformation: tf.Tensor
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        dt = t_stop - t_start
        t_start = t_start + transformation * dt * self.max_initial_skip_frac
        t_stop = t_start + dt * self.total_frac
        return t_start, t_stop

    def get_transformation(
        self, frames_or_stream: tf.Tensor | StreamData, label=None, sample_weight=None
    ) -> tf.Tensor:
        return tf.random.uniform(())


class TemporalCropV2(_TemporalCropV2[None]):
    def __init__(
        self, initial_skip_frac: float, total_frac: float, rezero_times: bool = True
    ):
        self.initial_skip_frac = initial_skip_frac
        self.total_frac = total_frac
        super().__init__(rezero_times=rezero_times)

    def _get_temporal_limits(
        self, t_start: tf.Tensor, t_stop: tf.Tensor, transformation: None
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        dt = t_stop - t_start
        t_start = t_start + self.initial_skip_frac * dt
        t_stop = t_start + self.total_frac * dt
        return t_start, t_stop


class FlipHorizontal(Transform):
    def __init__(self, label_map: tp.Optional[tf.Tensor] = None):
        self.label_map = label_map

    def transform_frames(self, frames: tf.Tensor, transformation) -> tf.Tensor:
        return tf.reverse(frames, axis=[-2])

    def transform_stream(self, stream: StreamData, transformation) -> StreamData:
        y, x = tf.unstack(stream.coords, axis=-1)
        x = stream.grid_shape[1] - 1 - x
        coords = tf.stack((y, x), axis=-1)
        return StreamData(coords, stream.times, stream.polarity, stream.grid_shape)

    def transform_label(self, label, transformation):
        if self.label_map is None:
            return label
        return tf.gather(self.label_map, label)


class FlipVertical(Transform):
    def __init__(self, label_map: tp.Optional[tf.Tensor] = None):
        self.label_map = label_map

    def transform_frames(self, frames: tf.Tensor, transformation) -> tf.Tensor:
        return tf.reverse(frames, axis=[-3])

    def transform_stream(self, stream: StreamData, transformation) -> StreamData:
        def flip_coords(coords):
            y, x = tf.unstack(coords, axis=-1)
            y = stream.grid_shape[0] - 1 - y
            return tf.stack((y, x), axis=-1)

        coords = flip_coords(stream.coords)
        return StreamData(coords, stream.times, stream.polarity, stream.grid_shape)

    def transform_label(self, label, transformation):
        if self.label_map is None:
            return label
        return tf.gather(self.label_map, label)


class FlipTime(Transform):
    def __init__(self, label_map: tp.Optional[tf.Tensor] = None):
        self.label_map = label_map

    def transform_frames(self, frames: tf.Tensor, transformation) -> tf.Tensor:
        # return ops.flip(frames, axis=(-4, -1))  # flip time and polarity axes
        # flip time and polarity axes
        frames = tf.reverse(frames, axis=[-4])
        frames = tf.reverse(frames, axis=[-1])
        return frames

    def transform_stream(self, stream: StreamData, transformation) -> StreamData:
        # reverse order of events
        times = tf.reverse(stream.times, axis=[-1])
        coords = tf.reverse(stream.coords, axis=[-2])
        polarity = tf.reverse(stream.polarity, axis=[-1])

        # reverse time values and polarities
        times = times[..., :1] - times
        polarity = tf.logical_not(polarity)
        return StreamData(coords, times, polarity, stream.grid_shape)

    def transform_label(self, label, transformation):
        if self.label_map is None:
            return label
        return tf.gather(self.label_map, label)


class Transpose(Transform):
    def __init__(self, label_map: tp.Optional[tf.Tensor] = None):
        self.label_map = label_map

    def transform_frames(self, frames: tf.Tensor, transformation) -> tf.Tensor:
        return tfnp.swapaxes(frames, (-3, -2))

    def transform_stream(self, stream: StreamData, transformation) -> StreamData:
        coords = tf.reverse(stream.coords, axis=[1])
        return StreamData(coords, stream.times, stream.polarity, stream.grid_shape)

    def transform_label(self, label, transformation):
        if self.label_map is None:
            return label
        return tf.gather(self.label_map, label)


class Maybe(Transform):
    def __init__(self, transform: Transform, prob: bool = 0.5):
        self.transform = transform
        self.prob = prob

    def get_transformation(
        self, frames_or_stream: tf.Tensor | StreamData, label=None, sample_weight=None
    ):
        return tf.random.uniform(()) < self.prob, self.transform.get_transformation(
            frames_or_stream, label, sample_weight
        )

    def transform_frames(self, frames: tf.Tensor, transformation) -> tf.Tensor:
        predicate, transformation = transformation
        return tf.cond(
            predicate,
            lambda: self.transform.transform_frames(frames, transformation),
            lambda: frames,
        )

    def transform_stream(self, stream: StreamData, transformation) -> StreamData:
        predicate, transformation = transformation
        other = self.transform.transform_stream(stream, transformation)
        if other is stream:
            return stream
        assert other.grid_shape == stream.grid_shape, (
            other.grid_shape,
            stream.grid_shape,
        )
        coords, times, polarity = tf.cond(
            predicate,
            lambda: (other.coords, other.times, other.polarity),
            lambda: (stream.coords, stream.times, stream.polarity),
        )
        return StreamData(coords, times, polarity, stream.grid_shape)

    def transform_label(self, label: tf.Tensor, transformation) -> StreamData:
        predicate, transformation = transformation
        other = self.transform.transform_label(label, transformation)
        if other is label:
            return label
        label = tf.cond(
            predicate,
            lambda: other,
            lambda: label,
        )
        return label

    def transform_sample_weight(
        self, sample_weight: tf.Tensor, transformation
    ) -> StreamData:
        predicate, transformation = transformation
        other = self.transform.transform_sample_weight(sample_weight, transformation)
        if other is sample_weight:
            return sample_weight
        sample_weight = tf.cond(
            predicate,
            lambda: other,
            lambda: sample_weight,
        )
        return sample_weight


class Identity(Transform[None]):
    pass


class Stack(Transform[tp.Sequence[T]]):
    def __init__(self, *transforms: Transform[T]):
        self.transforms = transforms
        assert len(transforms) > 1, len(transforms)

    def transform_frames(
        self, frames: tf.Tensor, transformation: tp.Sequence[T]
    ) -> tf.Tensor:
        assert len(transformation) == len(self.transforms), (
            len(transformation),
            len(self.transforms),
        )
        frames = [
            transform.transform_frames(frames, t)
            for transform, t in zip(self.transforms, transformation)
        ]
        return tf.stack(frames, axis=0)

    def transform_stream(
        self, stream: StreamData, transformation: tp.Sequence[T]
    ) -> StreamData:
        streams = [
            transform.transform_stream(stream, t)
            for transform, t in zip(self.transforms, transformation)
        ]
        grid_shape = streams[0].grid_shape
        assert all(s.grid_shape == grid_shape for s in streams[1:])
        coords = tf.stack([s.coords for s in streams], axis=0)
        times = tf.stack([s.times for s in streams], axis=0)
        polarity = tf.stack([s.polarity for s in streams], axis=0)
        return StreamData(coords, times, polarity, grid_shape)

    def get_transformation(
        self,
        frames_or_stream: tf.Tensor | StreamData,
        label=None,
        sample_weight=None,
    ) -> tp.Sequence[T]:
        return tuple(
            t.get_transformation(frames_or_stream, label, sample_weight)
            for t in self.transforms
        )


class FlattenLeadingDims(Transform[None]):
    def transform_frames(self, frames: tf.Tensor, transformation: None) -> tf.Tensor:
        return tf.reshape(frames, (-1, *frames.shape[-4:]))

    def transform_stream(self, stream: StreamData, transformation: None) -> tf.Tensor:
        return StreamData(
            coords=tf.reshape(stream.coords, (-1, *stream.coords.shape[-2:])),
            times=tf.reshape(stream.times, (-1, *stream.times.shape[-1:])),
            polarity=tf.reshape(stream.polarity, (-1, *stream.polarity.shape[-1:])),
            grid_shape=stream.grid_shape,
        )
