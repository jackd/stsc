from keras import KerasTensor, Operation

from ..backend import BackendTensor
from ..backend import counting_argsort as _backend

Tensor = BackendTensor | KerasTensor


class CountingArgsort(Operation):
    def compute_output_spec(self, segment_ids: KerasTensor, splits: KerasTensor):
        return KerasTensor(segment_ids.shape, "int32")

    def call(self, segment_ids: BackendTensor, splits: BackendTensor) -> BackendTensor:
        return _backend.counting_argsort(segment_ids, splits)


def counting_argsort(segment_ids: Tensor, splits: Tensor) -> Tensor:
    return CountingArgsort()(segment_ids, splits)
