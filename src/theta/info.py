"""Item-parameter standard errors from the observed information matrix.

We form the marginal (incomplete-data) log-likelihood as a function of the free
item parameters in reporting space ``(a, b, c, d)``, take its Hessian by
automatic differentiation at the converged estimate, and read SEs off the
inverse. Because the marginal likelihood couples items through the person
integral, this is the full (non-block-diagonal) information matrix.

Note: point estimates are MAP (priors regularize the asymptotes), but SEs are
reported from the likelihood information by default, which is the conventional
frequentist quantity. The asymptotes of weakly identified items will simply
show large SEs - that is the honest answer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from theta import models as M
from theta.models import GC, GD, ModelSpec

_EPS = 1e-9


def _unpack(vec, spec: ModelSpec, J: int):
    """Flat free-parameter vector -> (a, b, c, d) arrays of length J."""
    i = 0
    if spec.shared_a:
        a = jnp.broadcast_to(vec[i], (J,)); i += 1
    else:
        a = vec[i:i + J]; i += J
    b = vec[i:i + J]; i += J
    if spec.free[GC]:
        c = vec[i:i + J]; i += J
    else:
        c = jnp.full((J,), spec.c_fixed)
    if spec.free[GD]:
        d = vec[i:i + J]; i += J
    else:
        d = jnp.full((J,), spec.d_fixed)
    return a, b, c, d


def _pack(a, b, c, d, spec: ModelSpec):
    blocks = [a[:1] if spec.shared_a else a, b]
    if spec.free[GC]:
        blocks.append(c)
    if spec.free[GD]:
        blocks.append(d)
    return jnp.concatenate(blocks)


def _neg_marginal_ll(vec, Xm, Om, nodes, log_weights, spec, J):
    a, b, c, d = _unpack(vec, spec, J)
    z = a[:, None] * (nodes[None, :] - b[:, None])           # [J, Q]
    P = jnp.clip(c[:, None] + (d[:, None] - c[:, None]) * M.sigmoid(z), _EPS, 1 - _EPS)
    loglik = Xm @ jnp.log(P) + Om @ jnp.log1p(-P)            # [N, Q]
    logpost = loglik + log_weights[None, :]
    return -jnp.sum(jax.scipy.special.logsumexp(logpost, axis=1))


def standard_errors(result, responses):
    """Compute item-parameter SEs for a :class:`CalibrateResult`.

    Returns a dict with one array per estimated reporting parameter
    (``a``, ``b``, and ``c``/``d`` where applicable). Non-estimated entries are
    NaN. Also returns the full covariance matrix and the parameter order.
    """
    spec = result.spec
    J = len(result.b)
    X = np.asarray(responses, dtype=np.float64)
    Mobs = (~np.isnan(X)).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    Xm = jnp.asarray(X * Mobs)
    Om = jnp.asarray((1.0 - X) * Mobs)
    log_weights = jnp.log(result.weights)

    vec = _pack(jnp.asarray(result.a), jnp.asarray(result.b),
                jnp.asarray(result.c), jnp.asarray(result.d), spec)
    H = jax.hessian(_neg_marginal_ll)(vec, Xm, Om, result.nodes, log_weights, spec, J)
    # observed information = Hessian of negative log-likelihood
    try:
        cov = jnp.linalg.inv(H)
    except Exception:
        cov = jnp.linalg.pinv(H)
    se_vec = np.asarray(jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0, None)))

    out = {k: np.full(J, np.nan) for k in ("a", "b", "c", "d")}
    i = 0
    if spec.shared_a:
        out["a"][:] = se_vec[i]; i += 1
    else:
        out["a"] = se_vec[i:i + J]; i += J
    out["b"] = se_vec[i:i + J]; i += J
    if spec.free[GC]:
        out["c"] = se_vec[i:i + J]; i += J
    if spec.free[GD]:
        out["d"] = se_vec[i:i + J]; i += J
    out["cov"] = np.asarray(cov)
    return out
