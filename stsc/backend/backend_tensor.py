from keras import backend

if backend.backend() == "tensorflow":
    from tensorflow import Tensor as BackendTensor
elif backend.backend() == "jax":
    from jax.numpy import ndarray as BackendTensor

elif backend.backend() == "torch":
    from torch import Tensor as BackendTensor
else:
    raise RuntimeError(
        f"keras backend {backend.backend()} not supported. "
        "Must be one of 'tensorflow' or 'jax'."
    )


__all__ = ["BackendTensor"]
