"""Optional MAP priors that regularize item parameters during the M-step.

By default the slope ``a`` and difficulty ``b`` are *unpenalized* (their scales
are ``None``), so 1PL/2PL calibration is pure Marginal Maximum Likelihood and
the 3PL/4PL slope/difficulty are MML as well. Only the weakly-identified
asymptotes carry light default priors:

* ``c``        ~ Beta(c_a, c_b)            (pull guessing toward a small value)
* slip ``1-d`` ~ Beta(slip_a, slip_b)      (pull the upper asymptote toward 1)

These are the standard fix (used by ``mirt`` and friends) for the near-flat
asymptote likelihood; without them the 3PL/4PL EM wanders to the boundary. Set
the corresponding shapes to ``None`` for a fully unpenalized fit, or set
``la_sd``/``beta_sd`` to a finite value to regularize the slope/difficulty.

Normalizing constants are dropped (irrelevant to the argmax).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Priors:
    la_sd: float | None = None      # Normal(0, la_sd) on log a; None => unpenalized
    beta_sd: float | None = None    # Normal(0, beta_sd) on the intercept; None => unpenalized
    c_a: float | None = 2.0         # Beta(c_a, c_b) on guessing c
    c_b: float | None = 8.0
    slip_a: float | None = 2.0      # Beta(slip_a, slip_b) on slip (1 - d)
    slip_b: float | None = 10.0

    @property
    def has_la(self) -> bool:
        return self.la_sd is not None

    @property
    def has_beta(self) -> bool:
        return self.beta_sd is not None

    @property
    def has_c(self) -> bool:
        return self.c_a is not None and self.c_b is not None

    @property
    def has_d(self) -> bool:
        return self.slip_a is not None and self.slip_b is not None

    def log_la(self, la):
        return -0.5 * (la / self.la_sd) ** 2

    def dlog_la(self, la):
        """First/second derivative of ``log_la`` w.r.t. ``la`` (for the shared-a step)."""
        if not self.has_la:
            return 0.0, 0.0
        return -la / self.la_sd**2, -1.0 / self.la_sd**2

    def log_beta(self, beta):
        return -0.5 * (beta / self.beta_sd) ** 2

    def log_c(self, c):
        return (self.c_a - 1.0) * jnp.log(c) + (self.c_b - 1.0) * jnp.log1p(-c)

    def log_d(self, d):
        slip = 1.0 - d
        return (self.slip_a - 1.0) * jnp.log(slip) + (self.slip_b - 1.0) * jnp.log1p(-slip)
