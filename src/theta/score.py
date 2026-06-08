"""Ability (theta) estimation given calibrated item parameters.

Three estimators:

* ``EAP`` (expected a posteriori) - posterior mean over the quadrature grid.
  Vectorized, no iteration, always finite; the default.
* ``MAP`` (maximum a posteriori) - mode of the posterior, 1-D Newton per person
  with a N(0,1) prior.
* ``ML`` (maximum likelihood) - mode of the likelihood (no prior); undefined for
  all-correct / all-incorrect response patterns, which are returned as NaN.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from theta import models as M
from theta.quadrature import gauss_hermite

_EPS = 1e-9


def _prob_item_node(theta_nodes, a, b, c, d):
    """P[J, Q] in the difficulty parametrization."""
    z = a[:, None] * (theta_nodes[None, :] - b[:, None])
    return c[:, None] + (d[:, None] - c[:, None]) * M.sigmoid(z)


@partial(jax.jit, static_argnums=())
def _eap(Xm, Om, a, b, c, d, nodes, log_weights):
    P = jnp.clip(_prob_item_node(nodes, a, b, c, d), _EPS, 1 - _EPS)
    loglik = Xm @ jnp.log(P) + Om @ jnp.log1p(-P)           # [N, Q]
    logpost = loglik + log_weights[None, :]
    post = jnp.exp(logpost - jax.scipy.special.logsumexp(logpost, axis=1, keepdims=True))
    mean = post @ nodes
    var = post @ (nodes**2) - mean**2
    return mean, jnp.sqrt(jnp.clip(var, 0.0, None))


def _person_negll(theta, x, mask, a, b, c, d, prior_w, prior_mean, prior_sd):
    """Negative (log-likelihood + optional log-prior) for one person.

    ``prior_w`` is 1 for MAP (a N(prior_mean, prior_sd) penalty) and 0 for ML.
    """
    P = jnp.clip(M.prob_report(theta, a, b, c, d), _EPS, 1 - _EPS)
    ll = jnp.sum(mask * (x * jnp.log(P) + (1 - x) * jnp.log1p(-P)))
    penalty = 0.5 * prior_w * ((theta - prior_mean) / prior_sd) ** 2
    return -(ll - penalty)


_THETA_BOUND = 6.0  # ML trust region; ML on extreme patterns would otherwise diverge
_MODE_AXES = (0, 0, 0, None, None, None, None, None, None, None)


@partial(jax.jit, static_argnums=(11,))
def _mode(theta0, X, Mobs, a, b, c, d, prior_w, prior_mean, prior_sd, bound, n_steps=12):
    """Backtracking 1-D Newton from EAP start, vmapped over persons.

    The Newton direction is forced downhill (positive curvature). ``bound``
    confines theta to ``[-bound, bound]``; it is finite only for ML (whose
    likelihood is monotone on degenerate patterns and would otherwise run to
    +/-inf). For MAP ``bound`` is +inf, so the proper prior - not an arbitrary
    box - determines the mode (e.g. an all-missing pattern returns prior_mean).
    """
    g_fn = jax.vmap(jax.grad(_person_negll), in_axes=_MODE_AXES)
    h_fn = jax.vmap(jax.hessian(_person_negll), in_axes=_MODE_AXES)
    f_fn = jax.vmap(_person_negll, in_axes=_MODE_AXES)
    args = (a, b, c, d, prior_w, prior_mean, prior_sd)
    bt = jnp.array([1.0, 0.5, 0.25, 0.1, 0.0])

    def step(theta, _):
        g = g_fn(theta, X, Mobs, *args)
        h = h_fn(theta, X, Mobs, *args)
        h = jnp.maximum(h, 1e-3)  # ensure a descent direction on the negll
        delta = g / h
        cand = jnp.clip(theta[None, :] - bt[:, None] * delta[None, :],
                        -bound, bound)                              # [T, N]
        fvals = jax.vmap(lambda th: f_fn(th, X, Mobs, *args))(cand)
        fvals = jnp.where(jnp.isfinite(fvals), fvals, jnp.inf)
        theta = cand[jnp.argmin(fvals, axis=0), jnp.arange(theta.shape[0])]
        return theta, None

    theta, _ = jax.lax.scan(step, theta0, None, length=n_steps)
    # standard error from observed information at the mode
    h = h_fn(theta, X, Mobs, *args)
    se = 1.0 / jnp.sqrt(jnp.clip(h, 1e-8, None))
    return theta, se


def score(responses, a, b, c, d, *, method="EAP", n_points=61,
          prior_mean=0.0, prior_sd=1.0):
    """Estimate person abilities ``theta`` and their standard errors.

    Parameters
    ----------
    responses : [N, J] 0/1 matrix, NaN for missing.
    a, b, c, d : item parameters (length J).
    method : ``"EAP"`` (default), ``"MAP"``, or ``"ML"``.

    Returns
    -------
    theta : np.ndarray [N]
    se : np.ndarray [N]
    """
    method = method.upper()
    if method not in ("EAP", "MAP", "ML"):
        raise ValueError(f"unknown scoring method {method!r}; use 'EAP', 'MAP', or 'ML'")
    a, b, c, d = (jnp.asarray(v, dtype=jnp.float64) for v in (a, b, c, d))
    X = np.asarray(responses, dtype=np.float64)
    Mobs = (~np.isnan(X)).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    Xm = jnp.asarray(X * Mobs)
    Om = jnp.asarray((1.0 - X) * Mobs)
    Xj = jnp.asarray(X)
    Mj = jnp.asarray(Mobs)

    nodes, weights, log_weights = gauss_hermite(n_points, prior_mean, prior_sd)
    eap_mean, eap_se = _eap(Xm, Om, a, b, c, d, nodes, log_weights)

    if method == "EAP":
        return np.asarray(eap_mean), np.asarray(eap_se)

    prior_w = 1.0 if method == "MAP" else 0.0
    # ML is bounded (its likelihood diverges on degenerate patterns); MAP is not
    # (its prior keeps the posterior proper, so the mode can sit anywhere).
    bound = _THETA_BOUND if method == "ML" else jnp.inf
    theta, se = _mode(eap_mean, Xj, Mj, a, b, c, d, prior_w, prior_mean, prior_sd, bound)
    theta = np.asarray(theta)
    se = np.asarray(se)

    if method == "ML":
        # ML is undefined for perfect / zero scores
        n_obs = Mobs.sum(axis=1)
        n_correct = (X * Mobs).sum(axis=1)
        degenerate = (n_correct == 0) | (n_correct == n_obs)
        theta = np.where(degenerate, np.nan, theta)
        se = np.where(degenerate, np.nan, se)
    return theta, se
