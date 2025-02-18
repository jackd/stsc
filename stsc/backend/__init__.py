from keras import backend

if backend.backend() == "tensorflow":
    from .tensorflow import conv_preprocessing, counting_argsort, sampling
elif backend.backend() == "jax":
    from .jax import conv_preprocessing, counting_argsort, ema, sampling
else:
    raise RuntimeError(
        f"keras backend {backend.backend()} not supported. "
        "Must be one of 'tensorflow' or 'jax'."
    )
from .backend_tensor import BackendTensor
from .common import grid_interpolate, patches

__all__ = [
    "BackendTensor",
    "conv_preprocessing",
    "counting_argsort",
    "grid_interpolate",
    "patches",
    "sampling",
    "ema",
]
