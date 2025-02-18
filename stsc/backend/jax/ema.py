import jax
import jax.numpy as jnp


def _ema_add_op(a, b):
    x_a, f_a = a
    x_b, f_b = b
    return x_a * f_b + x_b, f_a * f_b


def ema(
    x: jnp.ndarray, f: jnp.ndarray, axis: int = 0, reverse: bool = False
) -> jnp.ndarray:
    return jax.lax.associative_scan(_ema_add_op, (x, f), reverse=reverse, axis=axis)[0]
