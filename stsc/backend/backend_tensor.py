from keras import backend

if backend.backend() == "tensorflow":
    pass
elif backend.backend() == "jax":
    pass

elif backend.backend() == "torch":
    pass
else:
    raise RuntimeError(
        f"keras backend {backend.backend()} not supported. "
        "Must be one of 'tensorflow' or 'jax'."
    )
