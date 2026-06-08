"""Gauss-Hermite quadrature for marginalizing over the latent ability prior.

The marginal likelihood integrates the latent N(mu, sigma) trait out of the
person likelihood. We approximate

    integral f(theta) * phi(theta) d theta  ~  sum_q  W_q * f(theta_q)

with Gauss-Hermite nodes. Using the substitution ``theta = mu + sigma*sqrt(2)*x``
the standard Gauss-Hermite rule (which integrates ``exp(-x^2) g(x)``) gives nodes
``theta_q`` and normalized prior masses ``W_q`` (summing to 1).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def gauss_hermite(n_points: int = 61, mu: float = 0.0, sigma: float = 1.0):
    """Return ``(nodes, weights, log_weights)`` for a N(mu, sigma) prior.

    Parameters
    ----------
    n_points : number of quadrature nodes (more = more accurate, slower).
    mu, sigma : mean and sd of the latent trait prior (fixed to (0, 1) for
        identification in standard calibration).

    Returns
    -------
    nodes : jnp.ndarray [Q]      ability values
    weights : jnp.ndarray [Q]    prior probability mass at each node (sum to 1)
    log_weights : jnp.ndarray [Q]
    """
    x, w = np.polynomial.hermite.hermgauss(n_points)  # integrates exp(-x^2)
    nodes = mu + sigma * math.sqrt(2.0) * x
    weights = w / math.sqrt(math.pi)
    weights = weights / weights.sum()  # guard against tiny normalization drift
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    return nodes, weights, jnp.log(weights)
