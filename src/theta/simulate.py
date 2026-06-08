"""Synthetic IRT data generation, for tests and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from theta import models as M


@dataclass
class SimData:
    responses: np.ndarray  # [N, J] of 0/1
    theta: np.ndarray      # [N] true abilities
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray


def simulate(
    n_persons: int,
    n_items: int,
    model: str = "2PL",
    *,
    seed: int = 0,
    theta=None,
):
    """Generate true item parameters, abilities, and a 0/1 response matrix.

    Item parameters are drawn from conventional ranges; ``c``/``d`` are only
    active for the models that use them.
    """
    spec = M.get_spec(model)
    rng = np.random.default_rng(seed)

    if theta is None:
        theta = rng.standard_normal(n_persons)
    theta = np.asarray(theta, dtype=np.float64)

    a = np.ones(n_items) if spec.shared_a else rng.uniform(0.7, 2.0, n_items)
    if spec.shared_a:
        a[:] = rng.uniform(0.8, 1.5)  # one shared slope
    b = rng.normal(0.0, 1.0, n_items)
    c = rng.uniform(0.10, 0.30, n_items) if spec.free[M.GC] else np.zeros(n_items)
    d = rng.uniform(0.90, 0.99, n_items) if spec.free[M.GD] else np.ones(n_items)

    P = np.asarray(M.prob_report(theta[:, None], a[None], b[None], c[None], d[None]))
    u = rng.uniform(size=P.shape)
    responses = (u < P).astype(np.float64)

    return SimData(responses=responses, theta=theta, a=a, b=b, c=c, d=d)


def sample_responses(theta, a, b, c, d, key):
    """JAX-native Bernoulli sampling given abilities and item params."""
    P = M.prob_report(theta[:, None], a[None], b[None], c[None], d[None])
    return (jax.random.uniform(key, P.shape) < P).astype(jnp.float64)
