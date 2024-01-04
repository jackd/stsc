from keras import backend

if backend.backend() == "tensorflow":
    import tensorflow as tf

    BackendTensor = tf.Tensor
elif backend.backend() == "jax":
    import jax.numpy as jnp

    BackendTensor = jnp.ndarray
elif backend.backend() == "torch":
    import torch

    BackendTensor = torch.Tensor
else:
    raise RuntimeError(
        f"keras backend {backend.backend()} not supported. "
        "Must be one of 'tensorflow' or 'jax'."
    )
