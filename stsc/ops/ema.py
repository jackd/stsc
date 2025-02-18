import keras

from ..backend import BackendTensor
from ..backend import ema as ema_ops

Tensor = BackendTensor | keras.KerasTensor


class Ema(keras.Operation):
    def __init__(self, axis: int = 0, reverse: bool = False, name=None):
        self.axis = axis
        self.reverse = reverse
        super().__init__(name=name)

    def compute_output_spec(
        self, x: keras.KerasTensor, f: keras.KerasTensor
    ) -> keras.KerasTensor:
        return keras.KerasTensor(x.shape, x.dtype)

    def call(self, x: BackendTensor, f: BackendTensor) -> BackendTensor:
        return ema_ops.ema(x, f, axis=self.axis, reverse=self.reverse)


def ema(x: Tensor, f: Tensor, axis: int = 0, reverse: bool = False) -> Tensor:
    return Ema(axis=axis, reverse=reverse)(x, f)
