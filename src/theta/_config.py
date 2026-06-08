"""Global numerical configuration.

IRT estimation (and especially standard errors derived from the information
matrix) needs double precision. JAX defaults to float32 and only allows
float64 arrays when the x64 flag is set *before* any array is created, so we
flip it here at import time.
"""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)
