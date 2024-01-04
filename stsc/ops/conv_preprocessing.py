from keras import KerasTensor, Operation

from ..backend import BackendTensor
from ..backend import conv_preprocessing as _backend

Tensor = KerasTensor | BackendTensor


class GetPredecessorIds(Operation):
    def __init__(self, grid_size: int, name=None):
        super().__init__(name=name)
        self.grid_size = grid_size

    def compute_output_spec(
        self,
        pixel_ids_in: KerasTensor,
        times_in: KerasTensor,
        batch_splits_in: KerasTensor,
        pixel_ids_out: KerasTensor,
        times_out: KerasTensor,
        batch_splits_out: KerasTensor,
        kernel_offsets: KerasTensor,
    ) -> KerasTensor:
        return KerasTensor((times_out.shape[0], kernel_offsets.shape[0]), dtype="int32")

    def call(
        self,
        pixel_ids_in: BackendTensor,
        times_in: BackendTensor,
        batch_splits_in: BackendTensor,
        pixel_ids_out: BackendTensor,
        times_out: BackendTensor,
        batch_splits_out: BackendTensor,
        kernel_offsets: BackendTensor,
    ) -> BackendTensor:
        return _backend.get_predecessor_ids(
            pixel_ids_in=pixel_ids_in,
            times_in=times_in,
            batch_splits_in=batch_splits_in,
            pixel_ids_out=pixel_ids_out,
            times_out=times_out,
            batch_splits_out=batch_splits_out,
            kernel_offsets=kernel_offsets,
            grid_size=self.grid_size,
        )


def get_predecessor_ids(
    pixel_ids_in: Tensor,
    times_in: Tensor,
    batch_splits_in: Tensor,
    pixel_ids_out: Tensor,
    times_out: Tensor,
    batch_splits_out: Tensor,
    kernel_offsets: Tensor,
    grid_size: int,
) -> Tensor:
    return GetPredecessorIds(grid_size=grid_size)(
        pixel_ids_in=pixel_ids_in,
        times_in=times_in,
        batch_splits_in=batch_splits_in,
        pixel_ids_out=pixel_ids_out,
        times_out=times_out,
        batch_splits_out=batch_splits_out,
        kernel_offsets=kernel_offsets,
    )


# class GetPermutedPredecessorIds(Operation):
#     def __init__(self, grid_size: int, name=None):
#         super().__init__(name=name)
#         self.grid_size = grid_size

#     def compute_output_spec(
#         self,
#         pixel_ids_in: KerasTensor,
#         times_in: KerasTensor,
#         batch_splits_in: KerasTensor,
#         perm_in: KerasTensor,
#         pixel_ids_out: KerasTensor,
#         times_out: KerasTensor,
#         batch_splits_out: KerasTensor,
#         perm_out: KerasTensor,
#         kernel_offsets: KerasTensor,
#     ) -> KerasTensor:
#         return KerasTensor((times_out.shape[0], kernel_offsets.shape[0]), dtype="int32")

#     def call(
#         self,
#         pixel_ids_in: BackendTensor,
#         times_in: BackendTensor,
#         batch_splits_in: BackendTensor,
#         perm_in: BackendTensor,
#         pixel_ids_out: BackendTensor,
#         times_out: BackendTensor,
#         batch_splits_out: BackendTensor,
#         permm_out: BackendTensor,
#         kernel_offsets: BackendTensor,
#     ) -> BackendTensor:
#         return _backend.get_permuted_predecessor_ids(
#             pixel_ids_in=pixel_ids_in,
#             times_in=times_in,
#             batch_splits_in=batch_splits_in,
#             perm_in=perm_in,
#             pixel_ids_out=pixel_ids_out,
#             times_out=times_out,
#             batch_splits_out=batch_splits_out,
#             perm_out=perm_out,
#             kernel_offsets=kernel_offsets,
#             grid_size=self.grid_size,
#         )


# def get_permuted_predecessor_ids(
#     pixel_ids_in: Tensor,
#     times_in: Tensor,
#     batch_splits_in: Tensor,
#     perm_in: Tensor,
#     pixel_ids_out: Tensor,
#     times_out: Tensor,
#     batch_splits_out: Tensor,
#     perm_out: Tensor,
#     kernel_offsets: Tensor,
#     grid_size: int,
# ) -> Tensor:
#     return GetPermutedPredecessorIds(grid_size=grid_size)(
#         pixel_ids_in=pixel_ids_in,
#         times_in=times_in,
#         batch_splits_in=batch_splits_in,
#         pixel_ids_out=pixel_ids_out,
#         perm_in=perm_in,
#         times_out=times_out,
#         batch_splits_out=batch_splits_out,
#         perm_out=perm_out,
#         kernel_offsets=kernel_offsets,
#     )


class GetSuccessorIds(Operation):
    def __init__(self, grid_size: int, name=None):
        self.grid_size = grid_size
        super().__init__(name=name)

    def compute_output_spec(
        self,
        pixel_ids_in: KerasTensor,
        times_in: KerasTensor,
        batch_splits_in: KerasTensor,
        pixel_ids_out: KerasTensor,
        times_out: KerasTensor,
        batch_splits_out: KerasTensor,
    ):
        return KerasTensor(shape=pixel_ids_in.shape, dtype="int32")

    def call(
        self,
        pixel_ids_in: BackendTensor,
        times_in: BackendTensor,
        batch_splits_in: BackendTensor,
        pixel_ids_out: BackendTensor,
        times_out: BackendTensor,
        batch_splits_out: BackendTensor,
    ) -> BackendTensor:
        return _backend.get_successor_ids(
            pixel_ids_in=pixel_ids_in,
            times_in=times_in,
            batch_splits_in=batch_splits_in,
            pixel_ids_out=pixel_ids_out,
            times_out=times_out,
            batch_splits_out=batch_splits_out,
            grid_size=self.grid_size,
        )


def get_successor_ids(
    pixel_ids_in: Tensor,
    times_in: Tensor,
    batch_splits_in: Tensor,
    pixel_ids_out: Tensor,
    times_out: Tensor,
    batch_splits_out: Tensor,
    grid_size: int,
) -> Tensor:
    return GetSuccessorIds(grid_size=grid_size)(
        pixel_ids_in=pixel_ids_in,
        times_in=times_in,
        batch_splits_in=batch_splits_in,
        pixel_ids_out=pixel_ids_out,
        times_out=times_out,
        batch_splits_out=batch_splits_out,
    )


class GetPermutedSuccessorIds(Operation):
    def __init__(self, grid_size: int, name=None):
        self.grid_size = grid_size
        super().__init__(name=name)

    def compute_output_spec(
        self,
        pixel_ids_in: KerasTensor,
        times_in: KerasTensor,
        batch_splits_in: KerasTensor,
        perm_in: KerasTensor,
        pixel_ids_out: KerasTensor,
        times_out: KerasTensor,
        batch_splits_out: KerasTensor,
        perm_out: KerasTensor,
    ):
        return KerasTensor(shape=pixel_ids_in.shape, dtype="int32")

    def call(
        self,
        pixel_ids_in: BackendTensor,
        times_in: BackendTensor,
        batch_splits_in: BackendTensor,
        perm_in: BackendTensor,
        pixel_ids_out: BackendTensor,
        times_out: BackendTensor,
        batch_splits_out: BackendTensor,
        perm_out: BackendTensor,
    ) -> BackendTensor:
        return _backend.get_permuted_successor_ids(
            pixel_ids_in=pixel_ids_in,
            times_in=times_in,
            batch_splits_in=batch_splits_in,
            perm_in=perm_in,
            pixel_ids_out=pixel_ids_out,
            times_out=times_out,
            batch_splits_out=batch_splits_out,
            perm_out=perm_out,
            grid_size=self.grid_size,
        )


def get_permuted_successor_ids(
    pixel_ids_in: Tensor,
    times_in: Tensor,
    batch_splits_in: Tensor,
    perm_in: Tensor,
    pixel_ids_out: Tensor,
    times_out: Tensor,
    batch_splits_out: Tensor,
    perm_out: Tensor,
    grid_size: int,
) -> Tensor:
    return GetPermutedSuccessorIds(grid_size=grid_size)(
        pixel_ids_in=pixel_ids_in,
        times_in=times_in,
        batch_splits_in=batch_splits_in,
        perm_in=perm_in,
        pixel_ids_out=pixel_ids_out,
        times_out=times_out,
        batch_splits_out=batch_splits_out,
        perm_out=perm_out,
    )


class GetStationaryPredecessorIds(Operation):
    def __init__(self, grid_size: int, name=None):
        super().__init__(name=name)
        self.grid_size = grid_size

    def compute_output_spec(
        self,
        pixel_ids: KerasTensor,
        batch_splits: KerasTensor,
        kernel_offsets: KerasTensor,
    ) -> KerasTensor:
        return KerasTensor(
            (pixel_ids.shape[0], kernel_offsets.shape[0]),
            "int32",
            name="conv_indices",
        )

    def call(
        self,
        pixel_ids: BackendTensor,
        batch_splits: BackendTensor,
        kernel_offsets: BackendTensor,
    ) -> BackendTensor:
        return _backend.get_stationary_predecessor_ids(
            pixel_ids=pixel_ids,
            batch_splits=batch_splits,
            kernel_offsets=kernel_offsets,
            grid_size=self.grid_size,
        )


def get_stationary_predecessor_ids(
    pixel_ids: Tensor,
    batch_splits: Tensor,
    kernel_offsets: Tensor,
    grid_size: int,
) -> Tensor:
    """
    Get convolution indices for chronological event streams.

    Args:
        pixel_ids: [E] in [0, grid_size)
        batch_splits: [B+1] in [0, E]
        kernel_offsets: [K]

    Returns:
        [E, K] values in [0, E)
    """
    return GetStationaryPredecessorIds(grid_size)(
        pixel_ids=pixel_ids,
        batch_splits=batch_splits,
        kernel_offsets=kernel_offsets,
    )


class GetPermutedStationaryPredecessorIds(Operation):
    def __init__(self, grid_size: int, name=None):
        super().__init__(name=name)
        self.grid_size = grid_size

    def compute_output_spec(
        self,
        pixel_ids: KerasTensor,
        batch_splits: KerasTensor,
        kernel_offsets: KerasTensor,
        perm_in: KerasTensor,
        perm_out: KerasTensor,
    ) -> KerasTensor:
        return KerasTensor(
            (pixel_ids.shape[0], kernel_offsets.shape[0]),
            "int32",
            name="conv_indices",
        )

    def call(
        self,
        pixel_ids: BackendTensor,
        batch_splits: BackendTensor,
        kernel_offsets: BackendTensor,
        perm_in: BackendTensor,
        perm_out: BackendTensor,
    ) -> BackendTensor:
        return _backend.get_permuted_stationary_predecessor_ids(
            pixel_ids=pixel_ids,
            batch_splits=batch_splits,
            kernel_offsets=kernel_offsets,
            perm_in=perm_in,
            perm_out=perm_out,
            grid_size=self.grid_size,
        )


def get_permuted_stationary_predecessor_ids(
    pixel_ids: Tensor,
    batch_splits: Tensor,
    kernel_offsets: Tensor,
    perm_in: Tensor,
    perm_out: Tensor,
    grid_size: int,
) -> Tensor:
    """
    Get convolution indices for chronological event streams.

    Args:
        pixel_ids: [E] in [0, grid_size)
        batch_splits: [B+1] in [0, E]
        kernel_offsets: [K]

    Returns:
        [E, K] values in [0, E)
    """
    return GetPermutedStationaryPredecessorIds(grid_size)(
        pixel_ids=pixel_ids,
        batch_splits=batch_splits,
        kernel_offsets=kernel_offsets,
        perm_in=perm_in,
        perm_out=perm_out,
    )


# def get_padded_stationary_conv_indices(
#     coords: Tensor,
#     batch_splits: Tensor,
#     kernel_shape: tp.Union[int, tp.Sequence[int]],
#     perm_in: tp.Optional[Tensor],
#     perm_out: tp.Optional[Tensor],
#     grid_shape: tp.Sequence[int],
# ) -> Tensor:
#     asserts.assert_has_rank(coords, 2, "coords")
#     asserts.assert_has_rank(batch_splits, 1, "batch_splits")
#     spatial_dims = coords.shape[1]
#     assert len(grid_shape) == spatial_dims, (grid_shape, spatial_dims)
#     if isinstance(kernel_shape, int):
#         pad_left = ((kernel_shape - 1) // 2,) * spatial_dims
#         grid_shape = tuple(g + kernel_shape - 1 for g in grid_shape)
#         kernel_shape = (kernel_shape,) * spatial_dims
#     else:
#         assert len(kernel_shape) == spatial_dims, (kernel_shape, spatial_dims)
#         pad_left = tuple((k - 1) // 2 for k in kernel_shape)
#         grid_shape = tuple(g + k - 1 for g, k in zip(grid_shape, kernel_shape))
#     coords = coords + ops.convert_to_tensor(pad_left, coords.dtype)
#     dtype = standardize_dtype(coords.dtype)
#     kernel_offsets = np.ravel_multi_index(
#         np.meshgrid(
#             *(np.arange(-l, l + k, dtype) for l, k in zip(pad_left, kernel_shape)),
#             indexing="ij"
#         ),
#         grid_shape,
#     )
#     kernel_offsets = ops.convert_to_tensor(kernel_offsets.flatten(), dtype)
#     pixel_ids = ravel_multi_index(coords, grid_shape, axis=1)
#     grid_size = functools.reduce(lambda a, b: a * b, grid_shape)
#     return get_stationary_predecessor_ids(
#         pixel_ids=pixel_ids,
#         batch_splits=batch_splits,
#         kernel_offsets=kernel_offsets,
#         perm_in=perm_in,
#         perm_out=perm_out,
#         grid_size=grid_size,
#     )
