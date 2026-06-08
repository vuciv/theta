"""Logistic IRT item-response models (1PL / 2PL / 3PL / 4PL) and the
parametrizations used internally.

Two parameter spaces are used:

* **natural / reporting space** ``(a, b, c, d)`` per item, in the usual IRT
  difficulty form

      P(theta) = c + (d - c) * sigmoid(a * (theta - b))

  with ``a`` discrimination, ``b`` difficulty, ``c`` lower asymptote
  (pseudo-guessing) and ``d`` upper asymptote (1 - slip).

* **unconstrained space** ``psi = (la, beta, gc, gd)`` per item, used for the
  Newton M-step so that the optimizer is unconstrained yet every bound is
  respected exactly:

      a    = exp(la)                      (a > 0)
      beta = -a * b           (slope-intercept; beta is the intercept, free)
      c    = sigmoid(gc)                  (0 < c < 1)
      d    = c + (1 - c) * sigmoid(gd)    (c < d < 1, ordering enforced)

  Internally the model evaluates ``P`` from the intercept form
  ``z = a * theta + beta`` which equals ``a * (theta - b)``.

A :class:`ModelSpec` records, per model, which of the four coordinates are free
and what the frozen asymptotes are.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

# psi coordinate indices
LA, BETA, GC, GD = 0, 1, 2, 3


def sigmoid(x):
    return 0.5 * (jnp.tanh(0.5 * x) + 1.0)


def logit(p):
    return jnp.log(p) - jnp.log1p(-p)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    # which of (la, beta, gc, gd) the per-item M-step optimizes
    free: tuple[bool, bool, bool, bool]
    # 1PL: a single discrimination shared across items (coord LA is updated
    # globally, not per item)
    shared_a: bool
    c_fixed: float  # value of c when coord GC is frozen
    d_fixed: float  # value of d when coord GD is frozen
    report_names: tuple[str, ...]  # reporting params actually estimated

    @property
    def free_mask(self):
        return jnp.array(self.free, dtype=jnp.float64)


MODELS: dict[str, ModelSpec] = {
    "1PL": ModelSpec("1PL", (True, True, False, False), True, 0.0, 1.0, ("a", "b")),
    "2PL": ModelSpec("2PL", (True, True, False, False), False, 0.0, 1.0, ("a", "b")),
    "3PL": ModelSpec("3PL", (True, True, True, False), False, 0.0, 1.0, ("a", "b", "c")),
    "4PL": ModelSpec("4PL", (True, True, True, True), False, 0.0, 1.0, ("a", "b", "c", "d")),
}


def get_spec(model: str) -> ModelSpec:
    key = model.upper()
    if key not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    return MODELS[key]


# --------------------------------------------------------------------------
# psi (unconstrained)  <->  natural (a, beta, c, d)
# --------------------------------------------------------------------------

def psi_to_natural(psi, spec: ModelSpec):
    """Map ``psi[..., 4]`` to natural item parameters ``(a, beta, c, d)``.

    Frozen asymptotes take their fixed values; the ordering ``d > c`` is
    built into the ``d`` map so it can never be violated.
    """
    la = psi[..., LA]
    beta = psi[..., BETA]
    a = jnp.exp(la)
    c = jnp.where(spec.free[GC], sigmoid(psi[..., GC]), spec.c_fixed)
    d = jnp.where(spec.free[GD], c + (1.0 - c) * sigmoid(psi[..., GD]), spec.d_fixed)
    return a, beta, c, d


def natural_to_report(a, beta, c, d):
    """``(a, beta, c, d)`` (intercept form) -> ``(a, b, c, d)`` (difficulty form)."""
    b = -beta / a
    return a, b, c, d


def report_to_psi(a, b, c, d, spec: ModelSpec):
    """Inverse map ``(a, b, c, d)`` -> ``psi[..., 4]`` for init / simulation."""
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.broadcast_to(jnp.asarray(b, dtype=jnp.float64), a.shape)
    c = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float64), a.shape)
    d = jnp.broadcast_to(jnp.asarray(d, dtype=jnp.float64), a.shape)
    la = jnp.log(a)
    beta = -a * b
    # clip asymptotes away from the open-interval boundaries before logit
    c_s = jnp.clip(c, 1e-6, 1 - 1e-6)
    gc = logit(c_s)
    frac = jnp.clip((d - c) / (1.0 - c), 1e-6, 1 - 1e-6)
    gd = logit(frac)
    return jnp.stack([la, beta, gc, gd], axis=-1)


# --------------------------------------------------------------------------
# probabilities
# --------------------------------------------------------------------------

def prob_natural(theta, a, beta, c, d):
    """P(correct) for scalars/broadcastable arrays in intercept form."""
    z = a * theta + beta
    return c + (d - c) * sigmoid(z)


def prob_matrix(theta_nodes, psi, spec: ModelSpec):
    """Probability matrix ``P[J, Q]`` for items ``psi[J, 4]`` at ability nodes
    ``theta_nodes[Q]``. This is the workhorse feeding the E-step matmuls."""
    a, beta, c, d = psi_to_natural(psi, spec)  # each [J]
    z = a[:, None] * theta_nodes[None, :] + beta[:, None]  # [J, Q]
    return c[:, None] + (d[:, None] - c[:, None]) * sigmoid(z)


def prob_report(theta, a, b, c, d):
    """P(correct) in the difficulty parametrization (user-facing)."""
    z = a * (theta - b)
    return c + (d - c) * sigmoid(z)
